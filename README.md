# LẬP TRÌNH AI CHO RASPBERRY Pi: SO SÁNH CÁC PHƯƠNG PHÁP NHẬN DẠNG DCNN

Bộ mã nguồn thực hiện đầy đủ 5 bước thực nghiệm so sánh **6 mô hình DCNN** (`DenseNet121`, `ResNet50`, `VGG16`, `MobileNetV2`, `MobileNetV3-Small`, `ShuffleNetV2`) trên cơ sở dữ liệu **SLIF** theo **4 tỷ lệ phân chia Stratified Sampling** (`40/10/50`, `50/10/40`, `60/10/30`, `70/10/20`), mỗi tổ hợp chạy độc lập **3 lần**.

---

## 1. Cấu trúc thư mục dự án

```text
Train/
├── config.py                 # Cấu hình 4 thí nghiệm (TN1-TN4), 6 mô hình, tham số huấn luyện & seeds
├── dataset.py                # Bước 1 & 2: Đọc dữ liệu SLIF, tiền xử lý và chia Stratified 3 tập Train/Val/Test
├── models.py                 # Bước 3: Khởi tạo 6 kiến trúc DCNN và tính số tham số / kích thước (MB)
├── train_eval.py             # Bước 3 & 4: Vòng lặp huấn luyện, Early Stopping và đánh giá trên tập Test
├── visualize_and_report.py   # Bước 5, Mục 3, 4, 5: Xuất 4 bảng kết quả, vẽ biểu đồ Error Bar & trả lời 6 câu hỏi
├── main.py                   # Chương trình chính điều phối toàn bộ 72 lần chạy (có hỗ trợ chạy tiếp Resume)
├── requirements.txt          # Các thư viện Python cần thiết
└── data/
    └── SLIF/                 # Đặt thư mục dữ liệu SLIF tại đây (mỗi lớp 1 thư mục con)
        ├── class_A/
        ├── class_B/
        └── ...
```

---

## 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 3. Chuẩn bị cơ sở dữ liệu SLIF (Bước 1)

Đặt dữ liệu ảnh **SLIF** theo cấu trúc thư mục phân lớp chuẩn vào `data/SLIF/` (hoặc bất kỳ đường dẫn nào trên máy và truyền qua `--data_dir`):

```text
data/SLIF/
├── Lop_1/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── Lop_2/
│   ├── img001.jpg
│   └── ...
└── ...
```

---

## 4. Hướng dẫn chạy huấn luyện (Bạn tự chạy lệnh dưới đây)

### Chạy toàn bộ thực nghiệm chuẩn (6 mô hình × 4 tỷ lệ × 3 lần = 72 lượt)
```bash
python main.py --data_dir "data/SLIF" --epochs 25 --batch_size 32
```
> **Lưu ý tính năng tự động lưu & chạy tiếp (Resume)**:
> Sau mỗi lượt chạy, kết quả được ghi ngay lập tức vào `results/raw_results.csv`. Nếu quá trình train bị ngắt giữa chừng, khi chạy lại lệnh trên chương trình sẽ **tự động bỏ qua các lượt đã xong** và chạy tiếp phần còn lại.

### Các tùy chọn chạy linh hoạt khác:
- **Kiểm tra nhanh luồng code bằng dữ liệu giả lập (Dummy test)**:
  ```bash
  python main.py --create_dummy --epochs 2 --batch_size 16
  ```
- **Chạy riêng một vài mô hình hoặc thí nghiệm cụ thể**:
  ```bash
  python main.py --data_dir "data/SLIF" --models MobileNetV2 MobileNetV3-Small ShuffleNetV2 --experiments TN1 TN4
  ```
- **Chỉ xuất lại Bảng kết quả, Biểu đồ và Báo cáo từ file `raw_results.csv` đã có**:
  ```bash
  python visualize_and_report.py
  ```

---

## 5. Đầu ra tự động sau khi chạy xong (`results/`)

1. **Bảng kết quả thực nghiệm (Mục 3)** tại `results/tables/`:
   - `table_accuracy.csv`: Bảng tổng hợp Accuracy (Lần 1, Lần 2, Lần 3, Trung bình, Độ lệch chuẩn).
   - `table_precision.csv`: Bảng tổng hợp Macro-Precision.
   - `table_recall.csv`: Bảng tổng hợp Macro-Recall.
   - `table_f1_score.csv`: Bảng tổng hợp Macro-F1.
   - `table_hardware_complexity.csv`: Bảng so sánh số lượng tham số (M), dung lượng mô hình (MB) và thời gian suy luận (ms/ảnh) phục vụ đánh giá triển khai trên **Raspberry Pi**.
   - `all_tables_summary.xlsx` & `summary_tables.md`: File Excel nhiều sheet và file Markdown định dạng sẵn để đưa vào báo cáo.

2. **Biểu đồ trực quan hóa (Mục 4)** tại `results/charts/`:
   - `chart_1_accuracy_line.png` & `chart_1_accuracy_bar.png` (kèm Error Bar độ lệch chuẩn 3 lần chạy)
   - `chart_2_precision_line.png` & `chart_2_precision_bar.png`
   - `chart_3_recall_line.png` & `chart_3_recall_bar.png`
   - `chart_4_f1_score_line.png` & `chart_4_f1_score_bar.png`
   - `chart_overview_4_metrics.png`: Biểu đồ tổng hợp 2×2 cho cả 4 chỉ số.

3. **Ma trận nhầm lẫn (Confusion Matrix - Bước 4)** tại `results/confusion_matrices/`:
   - Lưu cả file `.csv` và ảnh Heatmap `.png` cho từng mô hình và thí nghiệm.

4. **Trả lời 6 câu hỏi phân tích & thảo luận (Mục 5)** tại:
   - `results/report_analysis.md`
