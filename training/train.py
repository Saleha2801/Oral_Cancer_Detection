import os
import time
import argparse
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from training.dataset_manager import dataset_manager
from training.models import get_model
from training.experiment_manager import experiment_manager


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_single_experiment(
    model_name: str = "ResNet18",
    dataset_name: str = "clinical_oral",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    optimizer_name: str = "Adam",
    patience: int = 7,
    subsample: Optional[int] = None,
    seed: int = 42,
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes a complete, reproducible ML training experiment.
    Calculates genuine train/val/test performance and records the results.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Starting Experiment: {model_name} on {dataset_name}")
    print(f" Config: Epochs={epochs}, BatchSize={batch_size}, LR={learning_rate}, Opt={optimizer_name}")
    print(f" Device: {device}, Seed: {seed}")
    print(f"=======================================================\n")

    if not experiment_id:
        experiment_id = experiment_manager.get_next_experiment_id()

    # 1. Load Data
    train_loader, val_loader, test_loader, class_names, split_info = dataset_manager.create_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        subsample=subsample,
    )
    num_classes = len(class_names)
    print(f"Data split: Train={split_info['train_samples']}, Val={split_info['val_samples']}, Test={split_info['test_samples']}")
    print(f"Classes: {class_names}")

    # 2. Build Model
    model, model_meta = get_model(model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    trainable_params = model_meta["trainable_parameters"]
    print(f"Model: {model_meta['display_name']} | Trainable parameters: {trainable_params:,}")

    # 3. Setup Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 4. Training Loop with Early Stopping
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_model_weights = None
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- TRAIN PHASE ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        # --- VALIDATION PHASE ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc * 100:.2f}%")

        # Check Early Stopping & Best Weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Patience: {patience})")
                break

    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time:.1f}s. Evaluating on held-out test set...")

    # Load best weights for test evaluation
    if best_model_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

    # 5. Held-out Test Set Evaluation
    model.eval()
    test_running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total_test = len(all_labels)
    test_loss = test_running_loss / max(1, total_test)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    test_acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # AUC calculation
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    eval_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc": float(auc),
        "classification_report": report_dict,
    }

    print(f"\n================ TEST SET METRICS ================")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"ROC-AUC:       {auc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"===================================================\n")

    # 6. Record in Experiment Manager
    config = {
        "model": model_name,
        "dataset": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "patience": patience,
        "seed": seed,
    }

    summary = experiment_manager.record_experiment(
        experiment_id=experiment_id,
        config=config,
        history=history,
        eval_metrics=eval_metrics,
        confusion_matrix=cm,
        class_names=class_names,
        trainable_params=trainable_params,
        training_time_sec=training_time,
    )

    print(f"Recorded experiment {experiment_id} successfully in {experiment_manager.csv_path}")
    return summary


def run_comprehensive_experiment_suite(dataset_name: str = "clinical_oral", subsample: Optional[int] = 200):
    """
    Runs the full scientific experimentation suite covering:
    - 5 Models: CustomCNN, ResNet18, ResNet50, EfficientNetB0, DenseNet121
    - Epoch experiments: 10, 20, 30
    - Batch size experiments: 16, 32, 64
    - Learning rate experiments: 0.0001, 0.001, 0.01
    """
    print("\n=======================================================")
    print(" EXECUTING COMPREHENSIVE ML BENCHMARK EXPERIMENT SUITE ")
    print("=======================================================\n")

    # Core experiments covering all faculty requirements
    experiments_plan = [
        # 1. Model Comparisons (baseline 10 epochs, bs 32, lr 0.001)
        {"model": "CustomCNN", "epochs": 10, "batch_size": 32, "lr": 0.001},
        {"model": "ResNet18", "epochs": 10, "batch_size": 32, "lr": 0.001},
        {"model": "ResNet50", "epochs": 10, "batch_size": 32, "lr": 0.0005},
        {"model": "EfficientNetB0", "epochs": 10, "batch_size": 32, "lr": 0.001},
        {"model": "DenseNet121", "epochs": 10, "batch_size": 32, "lr": 0.0005},

        # 2. Epoch Experiments (ResNet18 at 10, 20, 30 epochs)
        {"model": "ResNet18", "epochs": 20, "batch_size": 32, "lr": 0.001},
        {"model": "ResNet18", "epochs": 30, "batch_size": 32, "lr": 0.001},

        # 3. Batch Size Experiments (ResNet18 at BS 16, 64)
        {"model": "ResNet18", "epochs": 10, "batch_size": 16, "lr": 0.001},
        {"model": "ResNet18", "epochs": 10, "batch_size": 64, "lr": 0.001},

        # 4. Learning Rate Experiments (ResNet18 at LR 0.0001, 0.01)
        {"model": "ResNet18", "epochs": 10, "batch_size": 32, "lr": 0.0001},
        {"model": "ResNet18", "epochs": 10, "batch_size": 32, "lr": 0.01},
    ]

    for plan in experiments_plan:
        try:
            train_single_experiment(
                model_name=plan["model"],
                dataset_name=dataset_name,
                epochs=plan["epochs"],
                batch_size=plan["batch_size"],
                learning_rate=plan["lr"],
                subsample=subsample,
            )
        except Exception as e:
            print(f"Error executing experiment {plan}: {e}")

    print("\nAll benchmark experiments completed! Results saved to results/experiments.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oral Cancer AI Training & Experimentation Pipeline")
    parser.add_argument("--model", type=str, default="ResNet18", help="Architecture: CustomCNN, ResNet18, ResNet50, EfficientNetB0, DenseNet121")
    parser.add_argument("--dataset", type=str, default="clinical_oral", help="Dataset: clinical_oral, ndu_ufes_histo")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (e.g. 10, 20, 30, 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (e.g. 16, 32, 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (e.g. 0.0001, 0.001, 0.01)")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer: Adam or SGD")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--subsample", type=int, default=None, help="Optional subsample size for fast verification")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-suite", action="store_true", help="Execute the complete benchmark experimentation suite")

    args = parser.parse_args()

    if args.run_suite:
        run_comprehensive_experiment_suite(dataset_name=args.dataset, subsample=args.subsample)
    else:
        train_single_experiment(
            model_name=args.model,
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            optimizer_name=args.optimizer,
            patience=args.patience,
            subsample=args.subsample,
            seed=args.seed,
        )
