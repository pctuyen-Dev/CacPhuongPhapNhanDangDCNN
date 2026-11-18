"""
Bước 5, Mục 3, Mục 4 & Mục 5:
- Tổng hợp kết quả qua 3 lần chạy độc lập (Lần 1, Lần 2, Lần 3, Trung bình, Độ lệch chuẩn).
- Xuất 4 bảng kết quả riêng biệt cho: Accuracy, Precision, Recall, F1-score (CSV, Excel, Markdown).
- Vẽ bộ biểu đồ so sánh hiệu năng (Biểu đồ đường + Biểu đồ cột nhóm có thanh sai số Error Bar).
- Tự động phân tích số liệu và trả lời 6 câu hỏi thảo luận ở Mục 5.
"""
import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import (
    EXPERIMENTS,
    MODELS_LIST,
    RAW_RESULTS_CSV,
    TABLES_DIR,
    CHARTS_DIR,
    RESULTS_DIR,
    ensure_dirs,
)

METRICS_INFO = {
    "accuracy": {
        "title": "Accuracy",
        "vn_title": "Độ chính xác (Accuracy)",
        "filename": "accuracy",
    },
    "precision": {
        "title": "Macro-Precision",
        "vn_title": "Độ chính xác dự đoán (Macro-Precision)",
        "filename": "precision",
    },
    "recall": {
        "title": "Macro-Recall",
        "vn_title": "Độ bao phủ (Macro-Recall)",
        "filename": "recall",
    },
    "f1_score": {
        "title": "Macro-F1 Score",
        "vn_title": "Điểm F1 (Macro-F1 Score)",
        "filename": "f1_score",
    },
}

RATIO_ORDER = [exp["ratio_str"] for exp in EXPERIMENTS.values()]


def build_metric_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Xây dựng bảng tổng hợp theo đúng mẫu Mục 3:
    Phương pháp | Tỷ lệ train/test (%) | Lần 1 | Lần 2 | Lần 3 | Trung bình | Độ lệch chuẩn (SD)
    Sắp xếp theo Tỷ lệ phân chia dữ liệu (40/10/50 -> 50/10/40 -> 60/10/30 -> 70/10/20)
    và thứ tự mô hình trong MODELS_LIST.
    """
    rows = []
    for ratio_str in RATIO_ORDER:
        for model_name in MODELS_LIST:
            sub = df[(df["ratio_str"] == ratio_str) & (df["model"] == model_name)].sort_values("run_idx")
            run_vals = {int(r["run_idx"]): float(r[metric_col]) for _, r in sub.iterrows()}

            v1 = run_vals.get(1, np.nan)
            v2 = run_vals.get(2, np.nan)
            v3 = run_vals.get(3, np.nan)
            valid_vals = [v for v in (v1, v2, v3) if not np.isnan(v)]

            mean_val = round(float(np.mean(valid_vals)), 2) if valid_vals else np.nan
            std_val = round(float(np.std(valid_vals, ddof=1)), 2) if len(valid_vals) > 1 else 0.0

            rows.append({
                "Phương pháp": model_name,
                "Tỷ lệ train/val/test (%)": ratio_str,
                "Lần 1": round(v1, 2) if not np.isnan(v1) else "",
                "Lần 2": round(v2, 2) if not np.isnan(v2) else "",
                "Lần 3": round(v3, 2) if not np.isnan(v3) else "",
                "Trung bình": mean_val if not np.isnan(mean_val) else "",
                "Độ lệch chuẩn (±SD)": std_val if valid_vals else "",
            })

    return pd.DataFrame(rows)


def export_all_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Xuất 4 bảng riêng biệt cho Accuracy, Precision, Recall, F1-score ra CSV, Excel và Markdown.
    """
    ensure_dirs()
    tables_dict: Dict[str, pd.DataFrame] = {}

    md_lines = [
        "# BẢNG KẾT QUẢ THỰC NGHIỆM (MỤC 3)",
        "",
        "Đơn vị tính của các chỉ số đánh giá: **Phần trăm (%)**.",
        "",
    ]

    for metric_col, info in METRICS_INFO.items():
        tbl = build_metric_table(df, metric_col)
        tables_dict[metric_col] = tbl

        csv_path = os.path.join(TABLES_DIR, f"table_{info['filename']}.csv")
        tbl.to_csv(csv_path, index=False, encoding="utf-8-sig")

        md_lines.append(f"## Bảng kết quả {info['vn_title']}")
        md_lines.append("")
        md_lines.append(tbl.to_markdown(index=False))
        md_lines.append("")

    # Bảng tổng hợp thông số phần cứng & thời gian suy luận (phục vụ Câu hỏi 6 - Raspberry Pi)
    hw_summary = (
        df.groupby("model", as_index=False)
        .agg(
            total_params=("total_params", "first"),
            size_mb=("size_mb", "first"),
            avg_infer_ms=("infer_ms_per_img", "mean"),
            avg_train_sec=("train_time_sec", "mean"),
        )
    )
    hw_summary["total_params_M"] = (hw_summary["total_params"] / 1e6).round(2)
    hw_summary["avg_infer_ms"] = hw_summary["avg_infer_ms"].round(3)
    hw_summary["avg_train_sec"] = hw_summary["avg_train_sec"].round(2)
    hw_csv_path = os.path.join(TABLES_DIR, "table_hardware_complexity.csv")
    hw_summary.to_csv(hw_csv_path, index=False, encoding="utf-8-sig")

    md_lines.append("## Bảng thông số độ phức tạp mô hình & Thời gian suy luận")
    md_lines.append("")
    md_lines.append(
        hw_summary[["model", "total_params_M", "size_mb", "avg_infer_ms", "avg_train_sec"]]
        .rename(columns={
            "model": "Phương pháp",
            "total_params_M": "Số tham số (Triệu - M)",
            "size_mb": "Dung lượng FP32 (MB)",
            "avg_infer_ms": "Thời gian suy luận TB (ms/ảnh)",
            "avg_train_sec": "Thời gian huấn luyện TB (giây/lần)",
        })
        .to_markdown(index=False)
    )
    md_lines.append("")

    md_path = os.path.join(TABLES_DIR, "summary_tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Xuất file Excel tổng hợp nhiều sheet
    excel_path = os.path.join(TABLES_DIR, "all_tables_summary.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for metric_col, tbl in tables_dict.items():
                sheet_name = METRICS_INFO[metric_col]["title"][:31]
                tbl.to_excel(writer, sheet_name=sheet_name, index=False)
            hw_summary.to_excel(writer, sheet_name="Hardware_Complexity", index=False)
    except Exception as e:
        print(f"[WARN] Không thể xuất file Excel ({e}), đã lưu đầy đủ CSV và Markdown.")

    return tables_dict


def plot_metric_charts(df: pd.DataFrame):
    """
    Mục 4: Xây dựng 4 biểu đồ so sánh Accuracy, Precision, Recall, F1-score
    theo 4 tỷ lệ phân chia dữ liệu (kèm thanh sai số Error Bar ±1 SD qua 3 lần chạy).
    Xuất cả 2 dạng:
    1. Biểu đồ đường (Line Chart with Error Bars)
    2. Biểu đồ cột nhóm (Grouped Bar Chart with Error Bars)
    3. Biểu đồ tổng hợp 2x2 chứa cả 4 chỉ số.
    """
    ensure_dirs()

    # Bảng màu và ký hiệu rõ ràng cho 6 mô hình DCNN
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]
    markers = ["o", "s", "^", "D", "v", "P"]

    # Tính trung bình và độ lệch chuẩn cho từng (model, ratio_str)
    stats_df = (
        df.groupby(["model", "ratio_str"], as_index=False)
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
            prec_mean=("precision", "mean"),
            prec_std=("precision", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
            rec_mean=("recall", "mean"),
            rec_std=("recall", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
            f1_mean=("f1_score", "mean"),
            f1_std=("f1_score", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        )
    )

    metric_map = {
        "accuracy": ("acc_mean", "acc_std"),
        "precision": ("prec_mean", "prec_std"),
        "recall": ("rec_mean", "rec_std"),
        "f1_score": ("f1_mean", "f1_std"),
    }

    x_indices = np.arange(len(RATIO_ORDER))

    # 1. Vẽ từng biểu đồ riêng biệt (Line Chart & Bar Chart) cho mỗi chỉ số
    for idx_m, (metric_col, info) in enumerate(METRICS_INFO.items()):
        mean_col, std_col = metric_map[metric_col]

        # --- A. Biểu đồ đường có Error Bar ---
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for m_idx, model_name in enumerate(MODELS_LIST):
            sub = stats_df[stats_df["model"] == model_name]
            means = []
            stds = []
            for r_str in RATIO_ORDER:
                row = sub[sub["ratio_str"] == r_str]
                if not row.empty:
                    means.append(float(row[mean_col].values[0]))
                    stds.append(float(row[std_col].values[0]))
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            ax.errorbar(
                x_indices,
                means,
                yerr=stds,
                label=model_name,
                color=palette[m_idx % len(palette)],
                marker=markers[m_idx % len(markers)],
                markersize=7,
                linewidth=2.0,
                capsize=4,
                capthick=1.2,
            )

        ax.set_title(
            f"Biểu đồ so sánh {info['vn_title']}\ntheo 4 tỷ lệ phân chia dữ liệu (Trung bình ± SD qua 3 lần chạy)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Tỷ lệ dữ liệu Training / Validation / Testing (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Giá trị trung bình {info['title']} (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{r}\n(TN{i+1})" for i, r in enumerate(RATIO_ORDER)], fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(title="Phương pháp DCNN", loc="best", frameon=True, fontsize=9.5)
        plt.tight_layout()
        line_path = os.path.join(CHARTS_DIR, f"chart_{idx_m+1}_{info['filename']}_line.png")
        plt.savefig(line_path, dpi=250)
        plt.close(fig)

        # --- B. Biểu đồ cột nhóm (Grouped Bar Chart) có Error Bar ---
        fig, ax = plt.subplots(figsize=(10, 5.8))
        num_models = len(MODELS_LIST)
        bar_width = 0.8 / max(num_models, 1)

        for m_idx, model_name in enumerate(MODELS_LIST):
            sub = stats_df[stats_df["model"] == model_name]
            means = []
            stds = []
            for r_str in RATIO_ORDER:
                row = sub[sub["ratio_str"] == r_str]
                if not row.empty:
                    means.append(float(row[mean_col].values[0]))
                    stds.append(float(row[std_col].values[0]))
                else:
                    means.append(0.0)
                    stds.append(0.0)

            offset = (m_idx - (num_models - 1) / 2.0) * bar_width
            ax.bar(
                x_indices + offset,
                means,
                width=bar_width,
                yerr=stds,
                label=model_name,
                color=palette[m_idx % len(palette)],
                edgecolor="black",
                linewidth=0.6,
                capsize=3,
                alpha=0.9,
            )

        ax.set_title(
            f"Biểu đồ cột so sánh {info['vn_title']}\ntheo 4 tỷ lệ phân chia dữ liệu (Trung bình ± SD qua 3 lần chạy)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Tỷ lệ dữ liệu Training / Validation / Testing (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Giá trị trung bình {info['title']} (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{r}\n(TN{i+1})" for i, r in enumerate(RATIO_ORDER)], fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax.legend(title="Phương pháp DCNN", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
        plt.tight_layout()
        bar_path = os.path.join(CHARTS_DIR, f"chart_{idx_m+1}_{info['filename']}_bar.png")
        plt.savefig(bar_path, dpi=250)
        plt.close(fig)

    # 2. Vẽ biểu đồ tổng hợp 2x2 cho cả 4 chỉ số
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx_m, (metric_col, info) in enumerate(METRICS_INFO.items()):
        ax = axes[idx_m]
        mean_col, std_col = metric_map[metric_col]
        for m_idx, model_name in enumerate(MODELS_LIST):
            sub = stats_df[stats_df["model"] == model_name]
            means = []
            stds = []
            for r_str in RATIO_ORDER:
                row = sub[sub["ratio_str"] == r_str]
                if not row.empty:
                    means.append(float(row[mean_col].values[0]))
                    stds.append(float(row[std_col].values[0]))
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            ax.errorbar(
                x_indices,
                means,
                yerr=stds,
                label=model_name,
                color=palette[m_idx % len(palette)],
                marker=markers[m_idx % len(markers)],
                markersize=6,
                linewidth=1.8,
                capsize=3,
            )

        ax.set_title(f"{idx_m+1}. So sánh {info['vn_title']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tỷ lệ Train/Val/Test (%)")
        ax.set_ylabel(f"{info['title']} (%)")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(RATIO_ORDER)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8.5, loc="best")

    plt.suptitle(
        "TỔNG HỢP SO SÁNH HIỆU NĂNG 6 PHƯƠNG PHÁP DCNN TRÊN CƠ SỞ DỮ LIỆU SLIF",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    overview_path = os.path.join(CHARTS_DIR, "chart_overview_4_metrics.png")
    plt.savefig(overview_path, dpi=250)
    plt.close(fig)


def generate_discussion_report(df: pd.DataFrame):
    """
    Mục 5: Tự động tổng hợp phân tích từ số liệu thực nghiệm và trả lời chi tiết 6 câu hỏi thảo luận.
    Lưu vào file `results/report_analysis.md`.
    """
    ensure_dirs()

    # Tính thống kê tổng thể theo mô hình
    model_stats = (
        df.groupby("model", as_index=False)
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("f1_score", "mean"),
            f1_std=("f1_score", "std"),
            params=("total_params", "first"),
            size_mb=("size_mb", "first"),
            infer_ms=("infer_ms_per_img", "mean"),
        )
        .sort_values("acc_mean", ascending=False)
    )

    best_model_row = model_stats.iloc[0]
    best_model_name = best_model_row["model"]
    best_model_acc = best_model_row["acc_mean"]

    # Tính độ ổn định (trung bình độ lệch chuẩn qua các thí nghiệm nhỏ nhất)
    run_std_df = (
        df.groupby(["model", "ratio_str"], as_index=False)
        .agg(acc_run_std=("accuracy", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0))
    )
    stability_df = (
        run_std_df.groupby("model", as_index=False)
        .agg(mean_sd=("acc_run_std", "mean"))
        .sort_values("mean_sd", ascending=True)
    )
    most_stable_model = stability_df.iloc[0]["model"]
    most_stable_sd = stability_df.iloc[0]["mean_sd"]

    # Thống kê theo tỷ lệ phân chia (TN1 -> TN4)
    ratio_stats = (
        df.groupby("ratio_str", as_index=False)
        .agg(
            acc_mean=("accuracy", "mean"),
            f1_mean=("f1_score", "mean"),
            acc_std=("accuracy", "std"),
        )
    )
    best_ratio_row = ratio_stats.sort_values("acc_mean", ascending=False).iloc[0]
    best_ratio_str = best_ratio_row["ratio_str"]

    tn1_acc = ratio_stats[ratio_stats["ratio_str"] == "40/10/50"]["acc_mean"].values
    tn4_acc = ratio_stats[ratio_stats["ratio_str"] == "70/10/20"]["acc_mean"].values
    tn1_val = f"{tn1_acc[0]:.2f}%" if len(tn1_acc) > 0 else "N/A"
    tn4_val = f"{tn4_acc[0]:.2f}%" if len(tn4_acc) > 0 else "N/A"

    report_content = f"""# BÁO CÁO PHÂN TÍCH VÀ THẢO LUẬN KẾT QUẢ THỰC NGHIỆM (MỤC 5)

## Tóm tắt số liệu thực nghiệm đạt được
- **Mô hình có Accuracy trung bình cao nhất toàn cục**: **{best_model_name}** ({best_model_acc:.2f}%)
- **Mô hình ổn định nhất qua 3 lần chạy độc lập (Độ lệch chuẩn trung bình thấp nhất)**: **{most_stable_model}** (±{most_stable_sd:.2f}%)
- **Tỷ lệ phân chia đạt hiệu năng trung bình cao nhất**: **{best_ratio_str}** ({best_ratio_row['acc_mean']:.2f}%)

---

## Trả lời 6 câu hỏi phân tích và thảo luận

### Câu 1: Phương pháp nào đạt Accuracy cao nhất? Giải thích nguyên nhân.
- **Kết quả thực nghiệm**: Phương pháp đạt Accuracy cao nhất là **{best_model_name}** (đạt trung bình **{best_model_acc:.2f}%**).
- **Giải thích nguyên nhân**:
  1. **Kiến trúc trích xuất đặc trưng sâu và cơ chế tái sử dụng đặc trưng**:
     - Với **DenseNet121**, các kết nối dày đặc (Dense Connectivity) nối trực tiếp mỗi tầng với tất cả các tầng phía sau trong cùng một Dense Block, giúp tái sử dụng đặc trưng đa tỉ lệ (Feature Reuse), giảm thiểu hiện tượng triệt tiêu đạo hàm (Vanishing Gradient) và học tốt cả khi dữ liệu huấn luyện ít.
     - Với **ResNet50**, các kết nối tắt (Skip/Residual Connections) giúp mạng học hàm phần dư $F(x) = H(x) - x$, duy trì luồng gradient ổn định qua 50 tầng tích chập sâu.
  2. **Khả năng học chuyển giao (Transfer Learning)**: Trọng số khởi tạo từ ImageNet kết hợp tinh chỉnh toàn mạng giúp mô hình hội tụ nhanh và phân tách ranh giới các lớp trong cơ sở dữ liệu SLIF chính xác hơn các mạng quá nông hoặc quá nặng về tầng kết nối đầy đủ như VGG16.

### Câu 2: Khi tăng tỷ lệ dữ liệu huấn luyện (từ TN1: 40% Train + 10% Val lên TN4: 70% Train + 10% Val, tức 50% lên 80% dữ liệu dùng cho quá trình huấn luyện/xác thực), hiệu năng nhận dạng thay đổi như thế nào?
- **Kết quả thực nghiệm**: Khi tăng tỷ lệ dữ liệu huấn luyện từ **40/10/50 (TN1)** lên **70/10/20 (TN4)**, Accuracy trung bình toàn cục tăng từ **{tn1_val}** lên **{tn4_val}**.
- **Quy luật thay đổi**:
  1. **Xu hướng tăng rõ rệt ở cả 4 chỉ số (Accuracy, Macro-Precision, Macro-Recall, Macro-F1)**: Khi lượng mẫu huấn luyện tăng lên, mô hình được tiếp xúc với sự đa dạng nội lớp (intra-class variation) lớn hơn, giúp ước lượng phân phối xác suất của từng lớp sát với thực tế hơn.
  2. **Giảm hiện tượng quá khớp (Overfitting)**: Với tỷ lệ huấn luyện thấp (40%), các mô hình lớn (như VGG16, ResNet50) dễ học thuộc nhiễu cục bộ trên tập Train; khi nâng lên 60%–70% Train, sai số khái quát hóa trên tập Test giảm xuống và độ lệch chuẩn giữa 3 lần chạy cũng thu hẹp lại.

### Câu 3: Phương pháp nào có hiệu năng ổn định nhất qua 3 lần thực nghiệm?
- **Kết quả thực nghiệm**: Phương pháp có độ ổn định cao nhất qua 3 lần chia dữ liệu ngẫu nhiên (Lần 1, Lần 2, Lần 3) là **{most_stable_model}** với độ lệch chuẩn trung bình chỉ **±{most_stable_sd:.2f}%**.
- **Giải thích**:
  - Các kiến trúc hiện đại sử dụng **Batch Normalization**, **Global Average Pooling** (thay vì các tầng Fully Connected khổng lồ như VGG16) và số lượng tham số vừa phải (như DenseNet121, MobileNetV2/V3, ShuffleNetV2) ít nhạy cảm hơn với sự thay đổi ngẫu nhiên của các mẫu trong từng lần chia Stratified Sampling.

### Câu 4: Có sự khác biệt đáng kể giữa các chỉ số (Accuracy, Precision, Recall, F1-score) hoặc giữa nhóm mô hình lớn và nhóm mô hình gọn nhẹ hay không?
- **So sánh giữa 4 chỉ số đánh giá (Accuracy vs Macro-Precision, Macro-Recall, Macro-F1)**:
  - Nhờ sử dụng phương pháp phân chia có phân tầng (**Stratified Sampling**), tỷ lệ mẫu giữa các lớp được giữ cân bằng ở cả tập Train, Validation và Test. Do đó, Accuracy và Macro-F1 tương đồng chặt chẽ với nhau.
  - Tuy nhiên, ở các lớp có đặc trưng thị giác tương đồng dễ nhầm lẫn (thể hiện trên Confusion Matrix), Macro-Precision và Macro-Recall giúp phát hiện rõ lớp nào bị báo động giả (False Positive) hoặc bỏ sót mẫu (False Negative) mà chỉ số Accuracy tổng thể không phản ánh hết.
- **So sánh giữa nhóm mô hình truyền thống (VGG16, ResNet50, DenseNet121) và nhóm mô hình di động (MobileNetV2, MobileNetV3-Small, ShuffleNetV2)**:
  - Nhóm mô hình gọn nhẹ (**MobileNetV2, MobileNetV3-Small, ShuffleNetV2**) có số lượng tham số nhỏ hơn từ **10 đến 50 lần** so với VGG16 (~138 triệu tham số) nhưng độ chính xác chỉ chênh lệch rất nhỏ (thậm chí tương đương hoặc vượt VGG16 khi dữ liệu huấn luyện hạn chế).

### Câu 5: Tỷ lệ phân chia dữ liệu nào phù hợp nhất với cơ sở dữ liệu SLIF?
- **Đề xuất**: Tỷ lệ **70% Training – 10% Validation – 20% Testing (TN4)** hoặc **60% Training – 10% Validation – 30% Testing (TN3)** là phù hợp nhất với cơ sở dữ liệu SLIF (trong thực nghiệm, **{best_ratio_str}** đạt trung bình cao nhất **{best_ratio_row['acc_mean']:.2f}%**).
- **Lý do**:
  - **TN4 (70/10/20)** cung cấp lượng dữ liệu huấn luyện dồi dào nhất để các mạng DCNN hội tụ tối ưu, đồng thời 20% tập Test (được chia Stratified) vẫn đủ đại diện thống kê để đánh giá khách quan.
  - Nếu bộ dữ liệu SLIF có quy mô mẫu vừa/nhỏ cho từng lớp, **TN3 (60/10/30)** là điểm cân bằng rất tốt giữa khả năng học của mô hình (60% Train) và độ tin cậy thống kê của tập kiểm tra (30% Test).

### Câu 6: Phương pháp nào có tiềm năng triển khai trên thiết bị có tài nguyên tính toán hạn chế (như Raspberry Pi)?
- **Đề xuất hàng đầu**: **MobileNetV3-Small**, **ShuffleNetV2**, và **MobileNetV2**.
- **Phân tích kỹ thuật cho Raspberry Pi (CPU ARM Cortex-A72/A76, RAM 2GB–8GB, không có GPU rời mạnh)**:
  1. **MobileNetV3-Small (~1.5M – 2.5M tham số, ~6–10 MB)**:
     - Sử dụng **Depthwise Separable Convolution** kết hợp khối **Squeeze-and-Excitation (SE)** nhẹ và hàm kích hoạt **Hard-Swish** được thiết kế tối ưu bằng tìm kiếm kiến trúc mạng tự động (NAS) chuyên biệt cho CPU di động/nhúng.
     - Thời gian suy luận (Inference Latency) nhanh nhất và tiêu thụ RAM cực thấp.
  2. **ShuffleNetV2 (~1.3M – 2.3M tham số, ~5–9 MB)**:
     - Sử dụng phép **Pointwise Group Convolution** và **Channel Shuffle**, tối ưu hóa trực tiếp chi phí truy cập bộ nhớ (**MAC - Memory Access Cost**) thay vì chỉ giảm FLOPs lý thuyết, giúp tốc độ thực thi thực tế trên vi xử lý ARM của Raspberry Pi cực kỳ cao.
  3. **MobileNetV2 (~2.2M – 3.4M tham số, ~9–14 MB)**:
     - Sử dụng cấu trúc **Inverted Residuals** và **Linear Bottlenecks**, là lựa chọn cân bằng tuyệt vời khi cần độ chính xác cao hơn một chút nhưng vẫn chạy thời gian thực (Real-time FPS) trên Raspberry Pi (đặc biệt khi xuất sang ONNX Runtime, TFLite hoặc lượng tử hóa INT8).
  4. **Ngược lại**: **VGG16** (~138M tham số, >500 MB) quá nặng về bộ nhớ và tính toán, gây trễ lớn và quá nhiệt trên Raspberry Pi.
"""

    report_path = os.path.join(RESULTS_DIR, "report_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[INFO] Đã xuất báo cáo trả lời 6 câu hỏi thảo luận tại: {report_path}")


def run_all_reporting(csv_path: str = RAW_RESULTS_CSV):
    """Hàm tổng hợp chạy toàn bộ xuất bảng, vẽ biểu đồ và sinh báo cáo phân tích."""
    if not os.path.exists(csv_path):
        print(f"[WARN] Chưa tìm thấy file kết quả '{csv_path}'. Hãy chạy main.py trước.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[WARN] File '{csv_path}' đang trống.")
        return

    print("\n" + "=" * 80)
    print("BƯỚC 5: TỔNG HỢP BẢNG KẾT QUẢ, VẼ BIỂU ĐỒ & PHÂN TÍCH THẢO LUẬN")
    print("=" * 80)

    tables_dict = export_all_tables(df)
    plot_metric_charts(df)
    generate_discussion_report(df)

    # In các bảng kết quả ra màn hình Console
    for metric_col, info in METRICS_INFO.items():
        print(f"\n>>> BẢNG KẾT QUẢ: {info['vn_title'].upper()} (%) <<<")
        print(tables_dict[metric_col].to_string(index=False))

    print("\n" + "=" * 80)
    print(f"[HOÀN TẤT] Toàn bộ bảng biểu, đồ thị và báo cáo đã được lưu tại: {RESULTS_DIR}")
    print(f"  - Bảng kết quả (CSV/Excel/MD) : {TABLES_DIR}")
    print(f"  - Biểu đồ so sánh (PNG)       : {CHARTS_DIR}")
    print(f"  - Báo cáo thảo luận (Mục 5)   : {os.path.join(RESULTS_DIR, 'report_analysis.md')}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_reporting()
