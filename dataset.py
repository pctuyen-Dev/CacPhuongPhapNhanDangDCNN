"""
Bước 1 & Bước 2: Chuẩn bị dữ liệu và Phân chia có phân tầng (Stratified Sampling)
Đảm bảo:
- Cùng tập dữ liệu đầu vào (SLIF database) và cùng cách tiền xử lý cho mọi mô hình.
- Phân chia ngẫu nhiên có phân tầng (Stratified) theo 4 tỷ lệ:
  + TN1: 40% Train - 10% Val - 50% Test
  + TN2: 50% Train - 10% Val - 40% Test
  + TN3: 60% Train - 10% Val - 30% Test
  + TN4: 70% Train - 10% Val - 20% Test
- Tuyệt đối không rò rỉ dữ liệu (Data Leakage): Tập Test tách biệt hoàn toàn.
"""
import os
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class SLIFSubsetDataset(Dataset):
    """Dataset PyTorch đọc danh sách đường dẫn ảnh và nhãn tương ứng."""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_transforms(img_size: int = IMAGE_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Thiết lập tiền xử lý dữ liệu đồng nhất cho tất cả 6 phương pháp DCNN:
    - Tập Train: Resize, lật ngang ngẫu nhiên, xoay nhẹ, chuẩn hóa theo chuẩn ImageNet.
    - Tập Val / Test: Chỉ Resize và chuẩn hóa theo chuẩn ImageNet (không tăng cường dữ liệu).
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def load_slif_image_paths(
    data_dir: str, max_samples_per_class: int = 0
) -> Tuple[List[str], List[int], List[str]]:
    """
    Quét thư mục dữ liệu SLIF theo cấu trúc thư mục con:
        data_dir/
            class_0/
                img1.jpg
                ...
            class_1/
                ...
    Nếu `max_samples_per_class > 0`, lấy tối đa `max_samples_per_class` ảnh mỗi lớp
    để tăng tốc độ huấn luyện khi cần thiết.
    Trả về: (image_paths, labels, class_names)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu tại: '{data_dir}'.\n"
            f"Vui lòng đặt bộ dữ liệu vào '{data_dir}' (mỗi lớp 1 thư mục con) "
            f"hoặc truyền đường dẫn qua tham số --data_dir."
        )

    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ])

    if len(class_names) < 2:
        raise ValueError(
            f"Thư mục '{data_dir}' cần chứa ít nhất 2 thư mục con tương ứng với các lớp (classes). "
            f"Hiện tìm thấy: {class_names}"
        )

    image_paths: List[str] = []
    labels: List[int] = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        cls_files: List[str] = []
        for root, _, files in os.walk(class_dir):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in VALID_EXTENSIONS:
                    cls_files.append(os.path.join(root, fname))
        if max_samples_per_class and max_samples_per_class > 0:
            cls_files = cls_files[:max_samples_per_class]
        image_paths.extend(cls_files)
        labels.extend([class_idx] * len(cls_files))

    if len(image_paths) == 0:
        raise ValueError(f"Không tìm thấy file ảnh hợp lệ nào trong '{data_dir}'.")

    return image_paths, labels, class_names


def stratified_split_three_way(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Chia ngẫu nhiên có phân tầng (Stratified Sampling) thành 3 tập Train / Val / Test
    theo đúng tỷ lệ quy định và hạt giống ngẫu nhiên `seed`.

    Quy trình:
    1. Tách tập Training (`train_ratio`) và phần còn lại (`val_ratio + test_ratio`) với `stratify=labels`.
    2. Tách phần còn lại thành Validation và Testing theo tỷ lệ tương đối `test_ratio / (val_ratio + test_ratio)`
       với `stratify=temp_labels`.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), (
        f"Tổng tỷ lệ phải bằng 1.0, nhận được: {train_ratio + val_ratio + test_ratio}"
    )

    paths_arr = np.array(image_paths)
    labels_arr = np.array(labels)

    # Bước 1: Tách tập Train và tập tạm (Val + Test)
    rem_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths_arr,
        labels_arr,
        test_size=rem_ratio,
        random_state=seed,
        stratify=labels_arr,
    )

    # Bước 2: Tách tập tạm thành tập Val và tập Test
    relative_test_ratio = test_ratio / rem_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_labels,
    )

    return {
        "train": (train_paths.tolist(), train_labels.tolist()),
        "val": (val_paths.tolist(), val_labels.tolist()),
        "test": (test_paths.tolist(), test_labels.tolist()),
    }


def create_dataloaders(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = IMAGE_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Tạo DataLoader cho Train, Validation và Testing từ dữ liệu đã chia Stratified.
    """
    splits = stratified_split_three_way(
        image_paths, labels, train_ratio, val_ratio, test_ratio, seed
    )
    train_transform, eval_transform = get_transforms(img_size=img_size)

    train_dataset = SLIFSubsetDataset(*splits["train"], transform=train_transform)
    val_dataset = SLIFSubsetDataset(*splits["val"], transform=eval_transform)
    test_dataset = SLIFSubsetDataset(*splits["test"], transform=eval_transform)

    # Đảm bảo tính tái lập của DataLoader shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    split_counts = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }
    return train_loader, val_loader, test_loader, split_counts


def create_dummy_slif_dataset(data_dir: str, num_classes: int = 5, samples_per_class: int = 40):
    """
    Hàm tiện ích tạo dữ liệu ảnh giả lập (dummy dataset) để kiểm tra nhanh luồng code
    trước khi chạy trên bộ dữ liệu SLIF thật.
    """
    print(f"[INFO] Đang tạo bộ dữ liệu SLIF giả lập tại '{data_dir}' ({num_classes} lớp, {samples_per_class} ảnh/lớp)...")
    rng = np.random.default_rng(42)
    for c in range(num_classes):
        class_dir = os.path.join(data_dir, f"Class_{c+1:02d}")
        os.makedirs(class_dir, exist_ok=True)
        for i in range(samples_per_class):
            img_path = os.path.join(class_dir, f"sample_{i+1:03d}.jpg")
            if not os.path.exists(img_path):
                # Tạo ảnh màu với đặc trưng màu khác nhau cho từng lớp
                base_color = rng.integers(30, 220, size=(1, 1, 3), dtype=np.uint8)
                noise = rng.integers(0, 35, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
                arr = np.clip(base_color + noise, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(img_path)
    print("[INFO] Hoàn tất tạo dữ liệu giả lập.")
