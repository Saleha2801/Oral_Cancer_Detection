import os
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# Project root directory detection
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_CLINICAL_DIR = PROJECT_ROOT / "Clinical oral images"
DEFAULT_NDU_UFES_IMAGES_DIR = PROJECT_ROOT / "NDU-UFES" / "histopathological images"
DEFAULT_NDU_UFES_CSV = PROJECT_ROOT / "NDU-UFES" / "histopathological img dataset" / "ndb-ufes.csv"
DEFAULT_EXTERNAL_DIR = PROJECT_ROOT / "datasets" / "external"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transforms(image_size: int = IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_test_transform


def compute_file_hash(filepath: str, block_size: int = 65536) -> str:
    """Computes MD5 hash to identify duplicate images."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


class OralImageFolderDataset(Dataset):
    """PyTorch Dataset for standard folder-organized oral images."""
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


class DatasetManager:
    """
    Manages loading, validation, patient-level splitting, and statistics
    for all oral cancer modalities.
    """
    def __init__(
        self,
        clinical_dir: Optional[Path] = None,
        ndu_ufes_images_dir: Optional[Path] = None,
        ndu_ufes_csv: Optional[Path] = None,
        external_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        self.clinical_dir = Path(clinical_dir) if clinical_dir else DEFAULT_CLINICAL_DIR
        self.ndu_ufes_images_dir = Path(ndu_ufes_images_dir) if ndu_ufes_images_dir else DEFAULT_NDU_UFES_IMAGES_DIR
        self.ndu_ufes_csv = Path(ndu_ufes_csv) if ndu_ufes_csv else DEFAULT_NDU_UFES_CSV
        self.external_dir = Path(external_dir) if external_dir else DEFAULT_EXTERNAL_DIR
        self.seed = seed

    def load_clinical_oral_dataframe(self, remove_duplicates: bool = True) -> Tuple[pd.DataFrame, int]:
        """
        Scans Clinical oral images/cancer and Clinical oral images/non-cancer.
        Detects duplicates and returns verified DataFrame.
        """
        records = []
        seen_hashes = {}
        duplicates_count = 0

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

        classes = [
            ("cancer", 0, "Cancer"),
            ("non-cancer", 1, "Non-cancer")
        ]

        for folder_name, label_id, class_name in classes:
            folder_path = self.clinical_dir / folder_name
            if not folder_path.exists():
                continue

            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(valid_exts):
                    fpath = folder_path / fname
                    if not fpath.is_file():
                        continue

                    fhash = compute_file_hash(str(fpath))
                    if fhash in seen_hashes:
                        duplicates_count += 1
                        if remove_duplicates:
                            continue
                    else:
                        seen_hashes[fhash] = str(fpath)

                    records.append({
                        "image_path": str(fpath),
                        "filename": fname,
                        "class_name": class_name,
                        "label": label_id,
                        "modality": "Clinical Oral Photography",
                        "hash": fhash,
                    })

        df = pd.DataFrame(records)
        return df, duplicates_count

    def load_ndu_ufes_dataframe(self, task: str = "TaskII") -> pd.DataFrame:
        """
        Loads NDU-UFES histopathology images with paired patient IDs.
        Task II: OSCC (0) vs Leukoplakia (1)
        Task IV: OSCC (0), Leukoplakia with dysplasia (1), Leukoplakia without dysplasia (2)
        """
        if not self.ndu_ufes_csv.exists() or not self.ndu_ufes_images_dir.exists():
            raise FileNotFoundError("NDU-UFES dataset files not found.")

        meta_df = pd.read_csv(self.ndu_ufes_csv)
        records = []

        for _, row in meta_df.iterrows():
            public_id = str(row["public_id"]).zfill(4)
            img_name = f"{public_id}.png"
            img_path = self.ndu_ufes_images_dir / img_name

            if not img_path.exists():
                # Check direct filename
                alt_name = str(row.get("path", "")).split("/")[-1]
                if alt_name and (self.ndu_ufes_images_dir / alt_name).exists():
                    img_path = self.ndu_ufes_images_dir / alt_name
                else:
                    continue

            if task == "TaskII":
                # Binary: OSCC (0), Leukoplakia (1)
                task_val = str(row.get("TaskII", "")).strip()
                label_id = 0 if task_val == "OSCC" else 1
                class_name = "OSCC (Malignant)" if label_id == 0 else "Leukoplakia (Non-cancer)"
            elif task == "TaskIV":
                task_val = str(row.get("TaskIV", "")).strip()
                if "OSCC" in task_val:
                    label_id = 0
                    class_name = "OSCC (Malignant)"
                elif "with dysplasia" in task_val:
                    label_id = 1
                    class_name = "Leukoplakia with Dysplasia"
                else:
                    label_id = 2
                    class_name = "Leukoplakia without Dysplasia"
            else:
                label_id = 0 if "OSCC" in str(row.get("diagnosis", "")) else 1
                class_name = "Cancer" if label_id == 0 else "Non-cancer"

            records.append({
                "image_path": str(img_path),
                "filename": img_name,
                "patient_id": str(row["patient_id"]),
                "class_name": class_name,
                "label": label_id,
                "localization": row.get("localization", "Tongue"),
                "larger_size": row.get("larger_size", 1.5),
                "tobacco_use": row.get("tobacco_use", "Not informed"),
                "alcohol_consumption": row.get("alcohol_consumption", "Not informed"),
                "gender": row.get("gender", "M"),
                "age_group": row.get("age_group", "1"),
                "modality": "Histopathology Biopsy",
            })

        return pd.DataFrame(records)

    def split_clinical_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs stratified train/validation/test split for photography."""
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=df["label"],
            random_state=self.seed
        )

        val_rel_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_rel_size),
            stratify=temp_df["label"],
            random_state=self.seed
        )

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def split_patient_grouped_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        CRITICAL: Splits at patient_id level to guarantee ZERO patient overlap
        between training, validation, and testing.
        """
        if "patient_id" not in df.columns:
            return self.split_clinical_dataset(df, train_ratio, val_ratio, test_ratio)

        patients = df["patient_id"].values
        gss_train = GroupShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=self.seed)
        train_idx, temp_idx = next(gss_train.split(df, groups=patients))

        train_df = df.iloc[train_idx]
        temp_df = df.iloc[temp_idx]

        temp_patients = temp_df["patient_id"].values
        val_rel_size = val_ratio / (val_ratio + test_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_rel_size), random_state=self.seed)
        val_sub_idx, test_sub_idx = next(gss_val.split(temp_df, groups=temp_patients))

        val_df = temp_df.iloc[val_sub_idx]
        test_df = temp_df.iloc[test_sub_idx]

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def get_dataset_statistics(self, dataset_name: str = "clinical_oral") -> Dict[str, Any]:
        """
        Returns comprehensive statistics:
        Total images, class distribution, train/val/test counts, percentages.
        """
        if dataset_name == "clinical_oral":
            df, dupes = self.load_clinical_oral_dataframe()
            train_df, val_df, test_df = self.split_clinical_dataset(df)
            modality = "Clinical Oral Photography"
        elif dataset_name == "ndu_ufes_histo":
            df = self.load_ndu_ufes_dataframe(task="TaskII")
            dupes = 0
            train_df, val_df, test_df = self.split_patient_grouped_dataset(df)
            modality = "Histopathological Biopsy"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        classes = sorted(df["class_name"].unique().tolist())
        total_images = len(df)
        class_counts = df["class_name"].value_counts().to_dict()
        class_percentages = {k: round((v / total_images) * 100, 2) for k, v in class_counts.items()}

        return {
            "dataset_name": dataset_name,
            "modality": modality,
            "total_images": total_images,
            "classes": classes,
            "num_classes": len(classes),
            "class_counts": class_counts,
            "class_percentages": class_percentages,
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "training_distribution": train_df["class_name"].value_counts().to_dict(),
            "validation_distribution": val_df["class_name"].value_counts().to_dict(),
            "test_distribution": test_df["class_name"].value_counts().to_dict(),
            "duplicates_filtered": dupes,
            "patient_level_splitting_enforced": "patient_id" in df.columns,
        }

    def create_dataloaders(
        self,
        dataset_name: str = "clinical_oral",
        batch_size: int = 32,
        image_size: int = IMAGE_SIZE,
        num_workers: int = 0,
        subsample: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, Any]]:
        """
        Creates ready-to-train PyTorch DataLoaders.
        """
        train_tf, val_tf = get_default_transforms(image_size)

        if dataset_name == "clinical_oral":
            df, _ = self.load_clinical_oral_dataframe()
            train_df, val_df, test_df = self.split_clinical_dataset(df)
            class_names = ["Cancer", "Non-cancer"]
        elif dataset_name == "ndu_ufes_histo":
            df = self.load_ndu_ufes_dataframe(task="TaskII")
            train_df, val_df, test_df = self.split_patient_grouped_dataset(df)
            class_names = ["OSCC (Malignant)", "Leukoplakia (Non-cancer)"]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if subsample is not None and subsample < len(train_df):
            train_df = train_df.sample(n=subsample, random_state=self.seed).reset_index(drop=True)
            val_df = val_df.sample(n=min(len(val_df), max(10, subsample // 4)), random_state=self.seed).reset_index(drop=True)
            test_df = test_df.sample(n=min(len(test_df), max(10, subsample // 4)), random_state=self.seed).reset_index(drop=True)

        train_dataset = OralImageFolderDataset(train_df, transform=train_tf)
        val_dataset = OralImageFolderDataset(val_df, transform=val_tf)
        test_dataset = OralImageFolderDataset(test_df, transform=val_tf)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        split_info = {
            "total_images": len(train_df) + len(val_df) + len(test_df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "class_names": class_names,
        }

        return train_loader, val_loader, test_loader, class_names, split_info


# Global singleton instance
dataset_manager = DatasetManager()

if __name__ == "__main__":
    print("Testing DatasetManager...")
    stats_oral = dataset_manager.get_dataset_statistics("clinical_oral")
    print(f"\n[Clinical Oral] Total: {stats_oral['total_images']}, Train: {stats_oral['training_samples']}, Val: {stats_oral['validation_samples']}, Test: {stats_oral['test_samples']}")
    print(f"Classes: {stats_oral['class_counts']}")

    stats_histo = dataset_manager.get_dataset_statistics("ndu_ufes_histo")
    print(f"\n[NDU-UFES Histo] Total: {stats_histo['total_images']}, Train: {stats_histo['training_samples']}, Val: {stats_histo['validation_samples']}, Test: {stats_histo['test_samples']}")
    print(f"Classes: {stats_histo['class_counts']}")
    print(f"Patient-level splitting enforced: {stats_histo['patient_level_splitting_enforced']}")
