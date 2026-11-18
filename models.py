"""
Bước 3: Xây dựng các mô hình nhận dạng DCNN
Bao gồm 6 kiến trúc:
1. DenseNet121 (tương ứng DenseNet12 trong đề bài)
2. ResNet50
3. VGG16
4. MobileNetV2
5. MobileNetV3-Small
6. ShuffleNetV2 (shufflenet_v2_x1_0)
"""
from typing import Dict, Any
import torch
import torch.nn as nn
from torchvision import models


def build_dcnn_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Khởi tạo mô hình DCNN và thay thế tầng phân loại cuối (Classifier Head)
    để phù hợp với số lượng lớp `num_classes` của bộ dữ liệu SLIF.
    """
    name_key = model_name.strip().lower()

    if name_key in ("densenet121", "densenet12"):
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif name_key == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name_key == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif name_key == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif name_key in ("mobilenetv3-small", "mobilenetv3_small"):
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif name_key in ("shufflenetv2", "shufflenet_v2"):
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Tên mô hình không hợp lệ: '{model_name}'. "
            f"Hỗ trợ: DenseNet121, ResNet50, VGG16, MobileNetV2, MobileNetV3-Small, ShuffleNetV2"
        )

    return model


def get_model_complexity(model: nn.Module) -> Dict[str, Any]:
    """
    Tính toán tổng số tham số (Parameters) và dung lượng bộ nhớ ước tính (MB)
    để phục vụ phân tích khả năng triển khai trên Raspberry Pi (Câu hỏi 6).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Ước tính kích thước FP32 (4 bytes mỗi tham số + buffer)
    param_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size_bytes + buffer_size_bytes) / (1024 ** 2)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": round(size_mb, 2),
    }
