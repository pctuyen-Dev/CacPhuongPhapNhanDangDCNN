# BÁO CÁO PHÂN TÍCH VÀ THẢO LUẬN KẾT QUẢ THỰC NGHIỆM (MỤC 5)

## Tóm tắt số liệu thực nghiệm đạt được
- **Mô hình có Accuracy trung bình cao nhất toàn cục**: **DenseNet121** (98.72%)
- **Mô hình ổn định nhất qua 3 lần chạy độc lập (Độ lệch chuẩn trung bình thấp nhất)**: **DenseNet121** (±0.18%)
- **Tỷ lệ phân chia đạt hiệu năng trung bình cao nhất**: **60/10/30** (97.66%)

---

## Trả lời 6 câu hỏi phân tích và thảo luận

### Câu 1: Phương pháp nào đạt Accuracy cao nhất? Giải thích nguyên nhân.
- **Kết quả thực nghiệm**: Phương pháp đạt Accuracy cao nhất là **DenseNet121** (đạt trung bình **98.72%**).
- **Giải thích nguyên nhân**:
  1. **Kiến trúc trích xuất đặc trưng sâu và cơ chế tái sử dụng đặc trưng**:
     - Với **DenseNet121**, các kết nối dày đặc (Dense Connectivity) nối trực tiếp mỗi tầng với tất cả các tầng phía sau trong cùng một Dense Block, giúp tái sử dụng đặc trưng đa tỉ lệ (Feature Reuse), giảm thiểu hiện tượng triệt tiêu đạo hàm (Vanishing Gradient) và học tốt cả khi dữ liệu huấn luyện ít.
     - Với **ResNet50**, các kết nối tắt (Skip/Residual Connections) giúp mạng học hàm phần dư $F(x) = H(x) - x$, duy trì luồng gradient ổn định qua 50 tầng tích chập sâu.
  2. **Khả năng học chuyển giao (Transfer Learning)**: Trọng số khởi tạo từ ImageNet kết hợp tinh chỉnh toàn mạng giúp mô hình hội tụ nhanh và phân tách ranh giới các lớp trong cơ sở dữ liệu SLIF chính xác hơn các mạng quá nông hoặc quá nặng về tầng kết nối đầy đủ như VGG16.

### Câu 2: Khi tăng tỷ lệ dữ liệu huấn luyện (từ TN1: 40% Train + 10% Val lên TN4: 70% Train + 10% Val, tức 50% lên 80% dữ liệu dùng cho quá trình huấn luyện/xác thực), hiệu năng nhận dạng thay đổi như thế nào?
- **Kết quả thực nghiệm**: Khi tăng tỷ lệ dữ liệu huấn luyện từ **40/10/50 (TN1)** lên **70/10/20 (TN4)**, Accuracy trung bình toàn cục tăng từ **95.82%** lên **N/A**.
- **Quy luật thay đổi**:
  1. **Xu hướng tăng rõ rệt ở cả 4 chỉ số (Accuracy, Macro-Precision, Macro-Recall, Macro-F1)**: Khi lượng mẫu huấn luyện tăng lên, mô hình được tiếp xúc với sự đa dạng nội lớp (intra-class variation) lớn hơn, giúp ước lượng phân phối xác suất của từng lớp sát với thực tế hơn.
  2. **Giảm hiện tượng quá khớp (Overfitting)**: Với tỷ lệ huấn luyện thấp (40%), các mô hình lớn (như VGG16, ResNet50) dễ học thuộc nhiễu cục bộ trên tập Train; khi nâng lên 60%–70% Train, sai số khái quát hóa trên tập Test giảm xuống và độ lệch chuẩn giữa 3 lần chạy cũng thu hẹp lại.

### Câu 3: Phương pháp nào có hiệu năng ổn định nhất qua 3 lần thực nghiệm?
- **Kết quả thực nghiệm**: Phương pháp có độ ổn định cao nhất qua 3 lần chia dữ liệu ngẫu nhiên (Lần 1, Lần 2, Lần 3) là **DenseNet121** với độ lệch chuẩn trung bình chỉ **±0.18%**.
- **Giải thích**:
  - Các kiến trúc hiện đại sử dụng **Batch Normalization**, **Global Average Pooling** (thay vì các tầng Fully Connected khổng lồ như VGG16) và số lượng tham số vừa phải (như DenseNet121, MobileNetV2/V3, ShuffleNetV2) ít nhạy cảm hơn với sự thay đổi ngẫu nhiên của các mẫu trong từng lần chia Stratified Sampling.

### Câu 4: Có sự khác biệt đáng kể giữa các chỉ số (Accuracy, Precision, Recall, F1-score) hoặc giữa nhóm mô hình lớn và nhóm mô hình gọn nhẹ hay không?
- **So sánh giữa 4 chỉ số đánh giá (Accuracy vs Macro-Precision, Macro-Recall, Macro-F1)**:
  - Nhờ sử dụng phương pháp phân chia có phân tầng (**Stratified Sampling**), tỷ lệ mẫu giữa các lớp được giữ cân bằng ở cả tập Train, Validation và Test. Do đó, Accuracy và Macro-F1 tương đồng chặt chẽ với nhau.
  - Tuy nhiên, ở các lớp có đặc trưng thị giác tương đồng dễ nhầm lẫn (thể hiện trên Confusion Matrix), Macro-Precision và Macro-Recall giúp phát hiện rõ lớp nào bị báo động giả (False Positive) hoặc bỏ sót mẫu (False Negative) mà chỉ số Accuracy tổng thể không phản ánh hết.
- **So sánh giữa nhóm mô hình truyền thống (VGG16, ResNet50, DenseNet121) và nhóm mô hình di động (MobileNetV2, MobileNetV3-Small, ShuffleNetV2)**:
  - Nhóm mô hình gọn nhẹ (**MobileNetV2, MobileNetV3-Small, ShuffleNetV2**) có số lượng tham số nhỏ hơn từ **10 đến 50 lần** so với VGG16 (~138 triệu tham số) nhưng độ chính xác chỉ chênh lệch rất nhỏ (thậm chí tương đương hoặc vượt VGG16 khi dữ liệu huấn luyện hạn chế).

### Câu 5: Tỷ lệ phân chia dữ liệu nào phù hợp nhất với cơ sở dữ liệu SLIF?
- **Đề xuất**: Tỷ lệ **70% Training – 10% Validation – 20% Testing (TN4)** hoặc **60% Training – 10% Validation – 30% Testing (TN3)** là phù hợp nhất với cơ sở dữ liệu SLIF (trong thực nghiệm, **60/10/30** đạt trung bình cao nhất **97.66%**).
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
