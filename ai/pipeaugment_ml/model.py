from typing import Tuple, Any, Callable
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def build_classifier(
    model_name: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True
) -> Tuple[nn.Module, str, Callable]:
    """
    Builds a torchvision classifier adapted for two-class corrosion prediction.
    Uses official pretrained weights through torchvision's official weights API.
    """
    if model_name.lower() != "resnet18":
        raise ValueError(f"Unsupported model architecture: {model_name}. Only resnet18 is authorized.")

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        weights_id = str(weights)
        model = models.resnet18(weights=weights)
        transforms = weights.transforms()
    else:
        weights_id = "none"
        model = models.resnet18(weights=None)
        transforms = None

    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model, weights_id, transforms


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all feature extraction parameters, leaving only the classifier head trainable."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all model parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
