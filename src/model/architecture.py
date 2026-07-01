import torch.nn as nn
import torchvision.models as models

def get_model_architecture(model_name: str = "resnet18", num_classes: int = 10, pretrained: bool = True):
    """Factory function for model architectures."""
    if model_name in ["resnet18"]:
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name in ["resnet50"]:
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name in ["mobilenet_v3", "mobilenet_v3_large"]:
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported: resnet18, resnet50, mobilenet_v3_large, efficientnet_b0")
    
    return model
