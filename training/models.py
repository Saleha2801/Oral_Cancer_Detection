import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Dict, Any


class CustomCNN(nn.Module):
    """
    Academic baseline custom Convolutional Neural Network for oral lesion classification.
    Consists of 4 feature extraction stages with Batch Normalization, ReLU, Max Pooling,
    followed by Adaptive Average Pooling, Dropout, and a Dense linear classifier.
    """
    def __init__(self, num_classes: int = 2):
        super(CustomCNN, self).__init__()

        # Block 1: 3 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112
        )

        # Block 2: 32 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56
        )

        # Block 3: 64 -> 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
        )

        # Block 4: 128 -> 256
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (trainable_parameters, total_parameters)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Factory function for all requested CNN models:
    - CustomCNN
    - ResNet18
    - ResNet50
    - EfficientNetB0
    - DenseNet121
    """
    name_clean = model_name.strip().lower()

    if name_clean in ["customcnn", "custom_cnn", "custom"]:
        model = CustomCNN(num_classes=num_classes)
        display_name = "Custom CNN Baseline"
        arch_family = "Custom CNN"

    elif name_clean in ["resnet18", "resnet-18"]:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        display_name = "ResNet-18"
        arch_family = "Residual Networks (ResNet)"

    elif name_clean in ["resnet50", "resnet-50"]:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        display_name = "ResNet-50"
        arch_family = "Residual Networks (ResNet)"

    elif name_clean in ["efficientnetb0", "efficientnet_b0", "efficientnet"]:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        display_name = "EfficientNet-B0"
        arch_family = "Compound Scaling (EfficientNet)"

    elif name_clean in ["densenet121", "densenet-121", "densenet"]:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        display_name = "DenseNet-121"
        arch_family = "Densely Connected Networks (DenseNet)"

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Choose from CustomCNN, ResNet18, ResNet50, EfficientNetB0, DenseNet121.")

    trainable_p, total_p = count_parameters(model)

    meta = {
        "model_key": model_name,
        "display_name": display_name,
        "architecture_family": arch_family,
        "trainable_parameters": trainable_p,
        "total_parameters": total_p,
        "parameters_formatted": f"{trainable_p / 1e6:.2f}M" if trainable_p >= 1e6 else f"{trainable_p / 1e3:.1f}K",
        "pretrained": pretrained,
        "num_classes": num_classes,
    }

    return model, meta


if __name__ == "__main__":
    print("Testing CNN Architecture Suite...")
    sample_input = torch.randn(2, 3, 224, 224)

    test_models = ["CustomCNN", "ResNet18", "ResNet50", "EfficientNetB0", "DenseNet121"]
    for mname in test_models:
        m, meta = get_model(mname, num_classes=2, pretrained=False)
        out = m(sample_input)
        print(f"[{meta['display_name']}] Output shape: {out.shape}, Params: {meta['parameters_formatted']} ({meta['trainable_parameters']:,})")
