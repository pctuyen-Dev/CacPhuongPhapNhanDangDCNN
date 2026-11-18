"""
CHƯƠNG TRÌNH CHÍNH (MAIN PIPELINE):
LẬP TRÌNH AI CHO RASPBERRY Pi - SO SÁNH CÁC PHƯƠNG PHÁP NHẬN DẠNG DCNN

Thực hiện tự động từ Bước 1 đến Bước 5:
- Bước 1: Chuẩn bị dữ liệu SLIF & tiền xử lý chuẩn hóa đồng nhất.
- Bước 2: Thiết lập 4 thí nghiệm phân chia dữ liệu có phân tầng (Stratified Sampling):
  + TN1: 40% Train - 10% Val - 50% Test
  + TN2: 50% Train - 10% Val - 40% Test
  + TN3: 60% Train - 10% Val - 30% Test
  + TN4: 70% Train - 10% Val - 20% Test
  Mỗi thí nghiệm lặp lại độc lập 3 lần với 3 hạt giống ngẫu nhiên (seeds) khác nhau.
- Bước 3: Huấn luyện 6 mô hình DCNN:
  DenseNet121, ResNet50, VGG16, MobileNetV2, MobileNetV3-Small, ShuffleNetV2.
- Bước 4: Đánh giá trên tập Testing độc lập (Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Confusion Matrix).
- Bước 5: Tổng hợp 4 bảng kết quả, vẽ 4 biểu đồ so sánh (kèm Error Bar) và xuất báo cáo phân tích.
"""
import os
import sys
import argparse
import pandas as pd
import torch

# Đảm bảo hiển thị tiếng Việt UTF-8 trên cửa sổ Terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import (
    DEFAULT_DATA_DIR,
    EXPERIMENTS,
    MODELS_LIST,
    NUM_RUNS,
    RANDOM_SEEDS,
    IMAGE_SIZE,
    DEFAULT_HYPERPARAMS,
    RAW_RESULTS_CSV,
    CM_DIR,
    CHECKPOINTS_DIR,
    ensure_dirs,
)
from dataset import (
    load_slif_image_paths,
    create_dataloaders,
    create_dummy_slif_dataset,
)
from models import build_dcnn_model, get_model_complexity
from train_eval import set_seed, train_model, evaluate_on_test
from visualize_and_report import run_all_reporting


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện và so sánh 6 phương pháp DCNN trên cơ sở dữ liệu SLIF"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Đường dẫn thư mục dữ liệu SLIF (mặc định: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_HYPERPARAMS["epochs"],
        help=f"Số epoch huấn luyện tối đa cho mỗi lần chạy (mặc định: {DEFAULT_HYPERPARAMS['epochs']})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_HYPERPARAMS["batch_size"],
        help=f"Kích thước batch (mặc định: {DEFAULT_HYPERPARAMS['batch_size']})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_HYPERPARAMS["learning_rate"],
        help=f"Tốc độ học Learning Rate (mặc định: {DEFAULT_HYPERPARAMS['learning_rate']})",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_HYPERPARAMS["weight_decay"],
        help=f"Hệ số Weight Decay (mặc định: {DEFAULT_HYPERPARAMS['weight_decay']})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_HYPERPARAMS["early_stopping_patience"],
        help=f"Số epoch chờ dừng sớm Early Stopping (mặc định: {DEFAULT_HYPERPARAMS['early_stopping_patience']})",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=IMAGE_SIZE,
        help=f"Kích thước ảnh đầu vào (mặc định: {IMAGE_SIZE}x{IMAGE_SIZE})",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_HYPERPARAMS["num_workers"],
        help="Số luồng đọc dữ liệu DataLoader (mặc định: 2)",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Huấn luyện từ đầu (không dùng trọng số Pretrained ImageNet)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS_LIST,
        choices=MODELS_LIST,
        help="Danh sách mô hình cần chạy (mặc định: chạy đủ cả 6 mô hình)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help="Danh sách thí nghiệm cần chạy (TN1 TN2 TN3 TN4)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=NUM_RUNS,
        help="Số lần chạy độc lập cho mỗi thí nghiệm (mặc định: 3)",
    )
    parser.add_argument(
        "--no_save_weights",
        action="store_true",
        help="Không lưu file trọng số .pth (mặc định CÓ lưu vào results/checkpoints/)",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Chạy lại toàn bộ từ đầu kể cả khi đã có kết quả lưu trong raw_results.csv",
    )
    parser.add_argument(
        "--create_dummy",
        action="store_true",
        help="Tạo dữ liệu ảnh giả lập SLIF để kiểm tra nhanh luồng chương trình",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=DEFAULT_HYPERPARAMS["max_samples_per_class"],
        help="Giới hạn số ảnh tối đa mỗi lớp (mặc định: 500 ảnh/lớp; truyền 0 để dùng toàn bộ 32.534 ảnh)",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Chỉ tổng hợp bảng và vẽ biểu đồ từ file raw_results.csv đã có sẵn",
    )
    return parser.parse_args()


def print_experiment_configuration(args, device, num_classes, total_images, class_names):
    """In rõ cấu hình tham số mô hình và thiết lập thí nghiệm (Yêu cầu Bước 3)."""
    gpu_str = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    pretrained_str = (
        "ImageNet Pretrained (Transfer Learning)"
        if not args.no_pretrained
        else "Khởi tạo ngẫu nhiên (From Scratch)"
    )
    exp_list_str = ", ".join(
        [f"{k} ({EXPERIMENTS[k]['ratio_str']})" for k in args.experiments]
    )

    print("=" * 80)
    print("CẤU HÌNH THỰC NGHIỆM NHẬN DẠNG DCNN TRÊN CƠ SỞ DỮ LIỆU SLIF")
    print("=" * 80)
    print(f"  - Thiết bị tính toán (Device) : {device}{gpu_str}")
    print(f"  - Thư mục dữ liệu (SLIF)      : {args.data_dir}")
    print(f"  - Tổng số mẫu ảnh sử dụng     : {total_images} ảnh")
    print(f"  - Số lượng lớp (Classes)      : {num_classes} lớp -> {class_names}")
    print(f"  - Kích thước ảnh đầu vào      : {args.img_size} x {args.img_size} x 3 (RGB)")
    print("  - Chuẩn hóa (Normalization)   : ImageNet Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]")
    print(f"  - Trọng số khởi tạo           : {pretrained_str}")
    print(f"  - Thuật toán tối ưu           : AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  - Lịch giảm tốc độ học        : CosineAnnealingLR (T_max={args.epochs})")
    print(f"  - Số Epochs tối đa / Batch    : {args.epochs} epochs | Batch size = {args.batch_size} | Early Stopping Patience = {args.patience}")
    print(f"  - Danh sách mô hình ({len(args.models)})       : {', '.join(args.models)}")
    print(f"  - Thí nghiệm phân chia ({len(args.experiments)})    : {exp_list_str}")
    print(f"  - Số lần chạy độc lập / TN    : {args.runs} lần (Seeds: {RANDOM_SEEDS[:args.runs]})")
    print("=" * 80)


def main():
    args = parse_args()
    ensure_dirs()

    if args.report_only:
        run_all_reporting(RAW_RESULTS_CSV)
        return

    if args.create_dummy and not os.path.exists(args.data_dir):
        create_dummy_slif_dataset(args.data_dir)

    # Bước 1: Đọc dữ liệu SLIF
    image_paths, labels, class_names = load_slif_image_paths(
        args.data_dir, max_samples_per_class=args.max_samples_per_class
    )
    num_classes = len(class_names)
    total_images = len(image_paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = not args.no_pretrained

    print_experiment_configuration(args, device, num_classes, total_images, class_names)

    # Đọc lịch sử kết quả nếu đã chạy dở trước đó (Resume support)
    if os.path.exists(RAW_RESULTS_CSV) and not args.force_rerun:
        results_df = pd.read_csv(RAW_RESULTS_CSV)
        print(f"[RESUME] Tìm thấy {len(results_df)} kết quả đã hoàn thành trong '{RAW_RESULTS_CSV}'.")
    else:
        results_df = pd.DataFrame()

    total_combinations = len(args.experiments) * len(args.models) * args.runs
    current_step = 0

    # Bước 2, 3, 4: Vòng lặp qua 4 Thí nghiệm (TN1..TN4) x 3 Lần chạy (Seeds) x 6 Mô hình
    for exp_key in args.experiments:
        exp_cfg = EXPERIMENTS[exp_key]
        train_r = exp_cfg["train"]
        val_r = exp_cfg["val"]
        test_r = exp_cfg["test"]
        ratio_str = exp_cfg["ratio_str"]

        for run_idx in range(1, args.runs + 1):
            seed = RANDOM_SEEDS[run_idx - 1] if run_idx <= len(RANDOM_SEEDS) else (1000 + run_idx)

            # Tạo DataLoader phân tầng (Stratified) dùng chung cho tất cả 6 mô hình trong cùng (exp_key, run_idx)
            # Đảm bảo tính công bằng tuyệt đối: cả 6 mô hình nhận cùng tập Train/Val/Test ở mỗi lần chạy!
            train_loader, val_loader, test_loader, split_counts = create_dataloaders(
                image_paths=image_paths,
                labels=labels,
                train_ratio=train_r,
                val_ratio=val_r,
                test_ratio=test_r,
                seed=seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                img_size=args.img_size,
            )

            for model_name in args.models:
                current_step += 1

                # Kiểm tra xem tổ hợp (model_name, exp_key, run_idx) đã chạy chưa
                if not results_df.empty:
                    already_done = (
                        (results_df["model"] == model_name)
                        & (results_df["experiment"] == exp_key)
                        & (results_df["run_idx"] == run_idx)
                    ).any()
                    if already_done:
                        print(
                            f"[{current_step:02d}/{total_combinations:02d}] [BỎ QUA - ĐÃ CÓ KẾT QUẢ] "
                            f"{model_name} | {exp_key} ({ratio_str}) | Lần {run_idx}"
                        )
                        continue

                print(
                    f"\n[{current_step:02d}/{total_combinations:02d}] ĐANG CHẠY: "
                    f"Mô hình = {model_name} | {exp_key} ({ratio_str}) | Lần {run_idx}/{args.runs} (Seed={seed})"
                )
                print(
                    f"    Số mẫu Stratified -> Train: {split_counts['train']} | "
                    f"Val: {split_counts['val']} | Test: {split_counts['test']}"
                )

                # Cố định seed trước khi khởi tạo mô hình và huấn luyện
                set_seed(seed)

                # Xây dựng mô hình DCNN
                model = build_dcnn_model(
                    model_name=model_name,
                    num_classes=num_classes,
                    pretrained=pretrained,
                )
                comp_info = get_model_complexity(model)
                print(
                    f"    Thông số mô hình  -> Tổng tham số: {comp_info['total_params']:,} "
                    f"({comp_info['size_mb']:.2f} MB)"
                )

                ckpt_path = None
                if not args.no_save_weights:
                    ckpt_name = f"{model_name}_{exp_key}_run{run_idx}.pth"
                    ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_name)

                # Huấn luyện mô hình trên tập Train, chọn trọng số tốt nhất trên tập Val
                trained_model, train_info = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    checkpoint_path=ckpt_path,
                )

                # Lưu lịch sử huấn luyện từng epoch (loss, acc) ra CSV
                if train_info.get("history"):
                    hist_csv = os.path.join(
                        CHECKPOINTS_DIR, f"history_{model_name}_{exp_key}_run{run_idx}.csv"
                    )
                    pd.DataFrame(train_info["history"]).to_csv(
                        hist_csv, index=False, encoding="utf-8-sig"
                    )

                # Đánh giá trên tập Test (Bước 4)
                cm_prefix = os.path.join(CM_DIR, f"cm_{model_name}_{exp_key}_run{run_idx}")
                title_info = f"- {model_name} ({exp_key}: {ratio_str} - Lần {run_idx})"
                eval_metrics = evaluate_on_test(
                    model=trained_model,
                    test_loader=test_loader,
                    device=device,
                    class_names=class_names,
                    cm_save_prefix=cm_prefix,
                    title_info=title_info,
                )

                print(
                    f"    ==> KẾT QUẢ TẬP TEST: "
                    f"Accuracy = {eval_metrics['accuracy']:.2f}% | "
                    f"Precision = {eval_metrics['precision']:.2f}% | "
                    f"Recall = {eval_metrics['recall']:.2f}% | "
                    f"F1 = {eval_metrics['f1_score']:.2f}% | "
                    f"Suy luận = {eval_metrics['infer_ms_per_img']:.2f} ms/ảnh"
                )

                # Lưu ngay kết quả của lần chạy này vào CSV để chống mất dữ liệu giữa chừng
                new_row = {
                    "model": model_name,
                    "experiment": exp_key,
                    "ratio_str": ratio_str,
                    "run_idx": run_idx,
                    "seed": seed,
                    "train_samples": split_counts["train"],
                    "val_samples": split_counts["val"],
                    "test_samples": split_counts["test"],
                    "best_epoch": train_info["best_epoch"],
                    "best_val_acc": train_info["best_val_acc"],
                    "accuracy": eval_metrics["accuracy"],
                    "precision": eval_metrics["precision"],
                    "recall": eval_metrics["recall"],
                    "f1_score": eval_metrics["f1_score"],
                    "total_params": comp_info["total_params"],
                    "size_mb": comp_info["size_mb"],
                    "infer_ms_per_img": eval_metrics["infer_ms_per_img"],
                    "train_time_sec": train_info["train_time_sec"],
                }
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                results_df.to_csv(RAW_RESULTS_CSV, index=False, encoding="utf-8-sig")

                # Giải phóng bộ nhớ GPU sau mỗi mô hình
                del trained_model, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Bước 5: Tổng hợp bảng kết quả, vẽ biểu đồ và sinh báo cáo phân tích
    run_all_reporting(RAW_RESULTS_CSV)


if __name__ == "__main__":
    main()
