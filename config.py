"""
Cấu hình trung tâm cho bài thí nghiệm:
LẬP TRÌNH AI CHO RASPBERRY Pi - SO SÁNH CÁC PHƯƠNG PHÁP NHẬN DẠNG DCNN
"""
import os

# ==============================================================================
# 1. ĐƯỜNG DẪN DỮ LIỆU & KẾT QUẢ
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục chứa cơ sở dữ liệu SLIF (cấu trúc chuẩn: data/SLIF/<ten_lop>/<anh.jpg>)
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "SLIF")

# Thư mục lưu kết quả, biểu đồ, bảng tổng hợp và trọng số mô hình
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")

RAW_RESULTS_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")

# ==============================================================================
# 2. THIẾT LẬP 4 THÍ NGHIỆM PHÂN CHIA DỮ LIỆU (Bước 2)
# ==============================================================================
# Mỗi thí nghiệm gồm: (train_ratio, val_ratio, test_ratio, label_str)
EXPERIMENTS = {
    "TN1": {"train": 0.40, "val": 0.10, "test": 0.50, "ratio_str": "40/10/50"},
    "TN2": {"train": 0.50, "val": 0.10, "test": 0.40, "ratio_str": "50/10/40"},
    "TN3": {"train": 0.60, "val": 0.10, "test": 0.30, "ratio_str": "60/10/30"},
    "TN4": {"train": 0.70, "val": 0.10, "test": 0.20, "ratio_str": "70/10/20"},
}

# Số lần chạy độc lập cho mỗi thiết lập thí nghiệm với hạt giống ngẫu nhiên khác nhau
NUM_RUNS = 3
RANDOM_SEEDS = [42, 1024, 2025]  # 3 seed độc lập cho Lần 1, Lần 2, Lần 3

# ==============================================================================
# 3. DANH SÁCH CÁC MÔ HÌNH DCNN (Bước 3)
# ==============================================================================
MODELS_LIST = [
    "DenseNet121",       # DenseNet12 trong đề bài (DenseNet-121 chuẩn trong torchvision)
    "ResNet50",
    "VGG16",
    "MobileNetV2",
    "MobileNetV3-Small",
    "ShuffleNetV2",
]

# ==============================================================================
# 4. THAM SỐ TIỀN XỬ LÝ & HUẤN LUYỆN MÔ HÌNH (Bước 3)
# ==============================================================================
IMAGE_SIZE = 224  # Kích thước đầu vào chuẩn cho các mạng DCNN (224x224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_HYPERPARAMS = {
    "epochs": 15,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "pretrained": True,        # Sử dụng Transfer Learning từ ImageNet
    "early_stopping_patience": 5,
    "num_workers": 2,
    "max_samples_per_class": 500,  # 500 ảnh/lớp (5.500 ảnh cho 11 lớp); đặt = 0 nếu muốn chạy toàn bộ 32.534 ảnh
}


def ensure_dirs():
    """Tạo đầy đủ các thư mục lưu trữ kết quả nếu chưa tồn tại."""
    for d in [RESULTS_DIR, TABLES_DIR, CHARTS_DIR, CM_DIR, CHECKPOINTS_DIR]:
        os.makedirs(d, exist_ok=True)
