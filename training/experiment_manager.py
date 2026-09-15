import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Default project directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


class ExperimentManager:
    """
    Academic experiment management system for oral cancer deep learning models.
    Persists experiments, histories, confusion matrices, and metrics.
    """
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        self.csv_path = self.results_dir / "experiments.csv"
        self.histories_dir = self.results_dir / "histories"
        self.metrics_dir = self.results_dir / "metrics"
        self.cm_dir = self.results_dir / "confusion_matrices"
        self.plots_dir = self.results_dir / "plots"

        self._ensure_directories()

    def _ensure_directories(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.histories_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.cm_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def get_next_experiment_id(self) -> str:
        """Generates sequential ID: EXP_001, EXP_002, etc."""
        if not self.csv_path.exists():
            return "EXP_001"
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty or "experiment_id" not in df.columns:
                return "EXP_001"
            ids = df["experiment_id"].dropna().tolist()
            nums = []
            for eid in ids:
                if isinstance(eid, str) and eid.startswith("EXP_"):
                    try:
                        nums.append(int(eid.split("_")[1]))
                    except (ValueError, IndexError):
                        pass
            next_num = max(nums, default=0) + 1
            return f"EXP_{next_num:03d}"
        except Exception:
            return "EXP_001"

    def record_experiment(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        history: Dict[str, List[float]],
        eval_metrics: Dict[str, Any],
        confusion_matrix: np.ndarray,
        class_names: List[str],
        trainable_params: int,
        training_time_sec: float,
    ) -> Dict[str, Any]:
        """
        Saves full experiment record across CSV, history JSON, metrics JSON, and plot PNG.
        """
        self._ensure_directories()
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Prepare flat summary row
        summary_row = {
            "experiment_id": experiment_id,
            "model": config.get("model", "Unknown"),
            "dataset": config.get("dataset", "clinical_oral"),
            "epochs": int(config.get("epochs", 10)),
            "actual_epochs": len(history.get("train_loss", [])),
            "batch_size": int(config.get("batch_size", 32)),
            "learning_rate": float(config.get("learning_rate", 0.001)),
            "optimizer": config.get("optimizer", "Adam"),
            "train_loss": round(float(history["train_loss"][-1]), 4) if history.get("train_loss") else None,
            "val_loss": round(float(history["val_loss"][-1]), 4) if history.get("val_loss") else None,
            "test_loss": round(float(eval_metrics.get("test_loss", 0.0)), 4),
            "train_accuracy": round(float(history["train_acc"][-1]), 4) if history.get("train_acc") else None,
            "val_accuracy": round(float(history["val_acc"][-1]), 4) if history.get("val_acc") else None,
            "test_accuracy": round(float(eval_metrics.get("test_accuracy", 0.0)), 4),
            "precision": round(float(eval_metrics.get("precision", 0.0)), 4),
            "recall": round(float(eval_metrics.get("recall", 0.0)), 4),
            "f1_score": round(float(eval_metrics.get("f1_score", 0.0)), 4),
            "auc": round(float(eval_metrics.get("auc", 0.0)), 4),
            "trainable_parameters": trainable_params,
            "training_time_seconds": round(training_time_sec, 2),
            "timestamp": now_str,
        }

        # 2. Append to experiments.csv
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            # Replace if already exists with same experiment_id, else append
            df = df[df["experiment_id"] != experiment_id]
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            df = pd.DataFrame([summary_row])
        df.to_csv(self.csv_path, index=False)

        # 3. Save History JSON
        history_file = self.histories_dir / f"{experiment_id}.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_id": experiment_id,
                "model": config.get("model"),
                "epochs_configured": config.get("epochs"),
                "actual_epochs": summary_row["actual_epochs"],
                "history": history
            }, f, indent=2)

        # 4. Save Metrics & Classification Report JSON
        metrics_file = self.metrics_dir / f"{experiment_id}.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_id": experiment_id,
                "config": config,
                "metrics": eval_metrics,
                "class_names": class_names,
                "timestamp": now_str
            }, f, indent=2)

        # 5. Save Confusion Matrix JSON & Plot
        cm_file = self.cm_dir / f"{experiment_id}.json"
        cm_list = confusion_matrix.tolist()
        cm_norm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]).tolist()
        with open(cm_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_id": experiment_id,
                "confusion_matrix": cm_list,
                "normalized": cm_norm,
                "class_names": class_names,
            }, f, indent=2)

        # 6. Generate Training Curve Plot
        plot_file = self.plots_dir / f"{experiment_id}_curves.png"
        self._generate_training_plot(experiment_id, config.get("model", "Model"), history, plot_file)

        return summary_row

    def _generate_training_plot(self, exp_id: str, model_name: str, history: Dict[str, List[float]], output_path: Path):
        """Generates and saves the loss and accuracy curves."""
        epochs_range = range(1, len(history.get("train_loss", [])) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy
        ax1.plot(epochs_range, [acc * 100 for acc in history.get("train_acc", [])], 'b-o', label='Train Accuracy', linewidth=2)
        ax1.plot(epochs_range, [acc * 100 for acc in history.get("val_acc", [])], 'g--s', label='Val Accuracy', linewidth=2)
        ax1.set_title(f"{exp_id}: {model_name} Accuracy", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy (%)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # Loss
        ax2.plot(epochs_range, history.get("train_loss", []), 'r-o', label='Train Loss', linewidth=2)
        ax2.plot(epochs_range, history.get("val_loss", []), 'm--s', label='Val Loss', linewidth=2)
        ax2.set_title(f"{exp_id}: {model_name} Loss", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Returns all recorded experiments as a list of dicts."""
        if not self.csv_path.exists():
            return []
        df = pd.read_csv(self.csv_path)
        return df.to_dict(orient="records")

    def get_history(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Reads stored history JSON."""
        path = self.histories_dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_confusion_matrix(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Reads stored confusion matrix JSON."""
        path = self.cm_dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates dynamic best model and summary for frontend dashboard."""
        exps = self.get_all_experiments()
        if not exps:
            return {
                "total_experiments": 0,
                "models_tested": [],
                "best_experiment": None,
            }

        # Find best by test_accuracy, then f1_score
        best_exp = max(exps, key=lambda x: (x.get("test_accuracy") or 0.0, x.get("f1_score") or 0.0))

        models_tested = sorted(list(set(x["model"] for x in exps if "model" in x)))

        return {
            "total_experiments": len(exps),
            "models_tested": models_tested,
            "num_models_tested": len(models_tested),
            "best_experiment": best_exp,
            "latest_experiment": exps[-1],
        }

    def get_comparison_metrics(self) -> Dict[str, Any]:
        """
        Organizes recorded results into structured comparisons for charts:
        - Model Comparison (Custom CNN, ResNet50, EfficientNetB0, DenseNet121)
        - Epoch Experiment (10 vs 20 vs 30 vs 50)
        - Batch Size Experiment (16 vs 32 vs 64)
        - Learning Rate Experiment (0.0001 vs 0.001 vs 0.01)
        """
        exps = self.get_all_experiments()
        if not exps:
            return {}

        df = pd.DataFrame(exps)

        # 1. Best per model architecture
        model_comparison = []
        for model_name, group in df.groupby("model"):
            best_row = group.sort_values(by="test_accuracy", ascending=False).iloc[0]
            model_comparison.append(best_row.to_dict())

        # 2. Epoch comparison
        epoch_comparison = []
        if "epochs" in df.columns:
            for ep_val, group in df.groupby("epochs"):
                # Take average or best for this epoch setting
                best_row = group.sort_values(by="test_accuracy", ascending=False).iloc[0]
                epoch_comparison.append({
                    "epochs": int(ep_val),
                    "test_accuracy": best_row["test_accuracy"],
                    "val_accuracy": best_row["val_accuracy"],
                    "val_loss": best_row["val_loss"],
                    "train_accuracy": best_row["train_accuracy"],
                    "model": best_row["model"],
                    "experiment_id": best_row["experiment_id"]
                })
            epoch_comparison = sorted(epoch_comparison, key=lambda x: x["epochs"])

        # 3. Batch size comparison
        batch_comparison = []
        if "batch_size" in df.columns:
            for bs_val, group in df.groupby("batch_size"):
                best_row = group.sort_values(by="test_accuracy", ascending=False).iloc[0]
                batch_comparison.append({
                    "batch_size": int(bs_val),
                    "test_accuracy": best_row["test_accuracy"],
                    "val_accuracy": best_row["val_accuracy"],
                    "val_loss": best_row["val_loss"],
                    "training_time": best_row.get("training_time_seconds"),
                    "model": best_row["model"],
                    "experiment_id": best_row["experiment_id"]
                })
            batch_comparison = sorted(batch_comparison, key=lambda x: x["batch_size"])

        # 4. Learning rate comparison
        lr_comparison = []
        if "learning_rate" in df.columns:
            for lr_val, group in df.groupby("learning_rate"):
                best_row = group.sort_values(by="test_accuracy", ascending=False).iloc[0]
                lr_comparison.append({
                    "learning_rate": float(lr_val),
                    "test_accuracy": best_row["test_accuracy"],
                    "val_accuracy": best_row["val_accuracy"],
                    "val_loss": best_row["val_loss"],
                    "model": best_row["model"],
                    "experiment_id": best_row["experiment_id"]
                })
            lr_comparison = sorted(lr_comparison, key=lambda x: x["learning_rate"])

        return {
            "model_comparison": model_comparison,
            "epoch_comparison": epoch_comparison,
            "batch_comparison": batch_comparison,
            "learning_rate_comparison": lr_comparison,
        }


# Global singleton instance
experiment_manager = ExperimentManager()
