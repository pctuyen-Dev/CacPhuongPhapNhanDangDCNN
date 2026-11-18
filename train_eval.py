"""
Bước 3 & Bước 4: Huấn luyện mô hình và Đánh giá hiệu năng nhận dạng
- Chỉ sử dụng tập Training để cập nhật trọng số.
- Sử dụng tập Validation để chọn mô hình tốt nhất (Best Checkpoint) và Early Stopping.
- Tuyệt đối KHÔNG sử dụng tập Testing trong quá trình huấn luyện.
- Đánh giá trên tập Testing với các chỉ số:
  + Accuracy (%)
  + Macro-Precision (%)
  + Macro-Recall (%)
  + Macro-F1 (%)
  + Confusion Matrix (Ma trận nhầm lẫn)
"""
import os
import time
import copy
import random
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int):
    """Cố định hạt giống ngẫu nhiên cho Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Huấn luyện 1 epoch trên tập Training."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Đánh giá nhanh trên tập Validation để chọn Best Checkpoint."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

    val_loss = running_loss / max(total, 1)
    val_acc = (correct / max(total, 1)) * 100.0
    return val_loss, val_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    checkpoint_path: str = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Huấn luyện mô hình với AdamW + CosineAnnealingLR + Early Stopping dựa trên Validation Accuracy.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_weights = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history_rows = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Lưu trọng số tốt nhất theo Validation Accuracy (ưu tiên val_loss nhỏ hơn nếu bằng nhau)
        improved = (val_acc > best_val_acc) or (
            np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_path:
                torch.save(best_weights, checkpoint_path)
        else:
            epochs_no_improve += 1

        history_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "is_best": improved,
        })

        print(
            f"    Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}%"
            + (" [*]" if improved else "")
        )

        if patience > 0 and epochs_no_improve >= patience:
            print(f"    [Early Stopping] Dừng sớm tại epoch {epoch} (Best epoch: {best_epoch}).")
            break

    train_duration_sec = time.time() - start_time
    model.load_state_dict(best_weights)

    train_info = {
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "train_time_sec": round(train_duration_sec, 2),
        "history": history_rows,
    }
    return model, train_info


@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    cm_save_prefix: str = None,
    title_info: str = "",
) -> Dict[str, Any]:
    """
    Bước 4: Đánh giá mô hình trên tập Testing Set (hoàn toàn độc lập).
    Tính toán:
    - Accuracy (%)
    - Macro-Precision (%)
    - Macro-Recall (%)
    - Macro-F1 (%)
    - Confusion Matrix (lưu file CSV và ảnh PNG)
    - Thời gian suy luận trung bình trên mỗi ảnh (ms/image)
    """
    model.eval()
    model = model.to(device)

    all_preds: List[int] = []
    all_targets: List[int] = []

    start_infer = time.perf_counter()
    for inputs, targets in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.numpy().tolist())
    total_infer_sec = time.perf_counter() - start_infer

    num_test_samples = max(len(all_targets), 1)
    infer_ms_per_img = (total_infer_sec * 1000.0) / num_test_samples

    # Tính các chỉ số đánh giá theo thang phần trăm (%)
    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0

    labels_indices = list(range(len(class_names)))
    cm = confusion_matrix(all_targets, all_preds, labels=labels_indices)

    if cm_save_prefix:
        save_confusion_matrix(cm, class_names, cm_save_prefix, title_info)

    return {
        "accuracy": round(acc, 2),
        "precision": round(macro_prec, 2),
        "recall": round(macro_rec, 2),
        "f1_score": round(macro_f1, 2),
        "infer_ms_per_img": round(infer_ms_per_img, 3),
        "confusion_matrix": cm,
    }


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_prefix: str,
    title_info: str = "",
):
    """Lưu ma trận nhầm lẫn dưới dạng CSV và biểu đồ Heatmap PNG."""
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{save_prefix}.csv", encoding="utf-8-sig")

    num_classes = len(class_names)
    fig_size = max(6, min(14, int(num_classes * 0.8) + 3))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix {title_info}".strip(), fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=10)
    ax.set_ylabel("Nhãn thực tế (True Label)", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=200)
    plt.close(fig)
