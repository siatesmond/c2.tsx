"""
model.py

Defines the AIGC detector model: a pretrained backbone (via timm) with a
binary classification head (real vs. fake).

Well under the challenge's <2B parameter constraint -- EfficientNet-B0 is
~5.3M parameters, ResNet50 is ~25.6M. Both are far smaller than the limit,
leaving compute budget for actually training with heavy augmentation rather
than fighting a huge model on limited hardware.

Usage:
    from src.model import build_model, count_parameters

    model = build_model("efficientnet_b0")
    print(f"Parameters: {count_parameters(model):,}")
"""

import torch
import torch.nn as nn
import timm


# Backbones known to work well here, all far under the 2B param cap.
# Feel free to try others from timm.list_models(pretrained=True).
SUPPORTED_BACKBONES = {
    "efficientnet_b0": "efficientnet_b0",   # ~5.3M params -- fastest, good default
    "resnet50": "resnet50",                  # ~25.6M params -- classic, reliable
    "vit_small": "vit_small_patch16_224",    # ~22M params -- attention-based option
}


def build_model(backbone: str = "efficientnet_b0", pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    """
    Builds a binary AIGC classifier.

    num_classes=1 because we treat this as binary classification with a
    single logit + BCEWithLogitsLoss (label 0 = real, 1 = fake), rather than
    2-way softmax -- simpler and gives a natural "confidence score" via
    sigmoid, which is exactly the `pred` field the required infer.py script
    needs to output.

    Args:
        backbone: one of SUPPORTED_BACKBONES keys, or any timm model name.
        pretrained: use ImageNet-pretrained weights (recommended -- speeds
            up convergence a lot given our limited training time).
        num_classes: output logits. Leave at 1 for binary sigmoid setup.

    Returns:
        nn.Module that outputs raw logits of shape (batch, num_classes).
    """
    model_name = SUPPORTED_BACKBONES.get(backbone, backbone)

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count -- use this to confirm you're under the 2B cap."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    # Quick sanity check -- run `python3 -m src.model` from project root.
    device = get_device()
    print(f"Using device: {device}")

    for name in SUPPORTED_BACKBONES:
        model = build_model(name, pretrained=False)  # pretrained=False for a fast local test
        n_params = count_parameters(model)
        print(f"{name}: {n_params:,} parameters ({n_params / 1e9:.4f}B)")

    # Forward pass sanity check with the default backbone
    model = build_model("efficientnet_b0", pretrained=False).to(device)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)  # batch of 4
    output = model(dummy_input)
    print(f"\nForward pass OK. Output shape: {output.shape}  (expected: [4, 1])")

    probs = torch.sigmoid(output)
    print(f"Sigmoid probabilities (fake confidence): {probs.squeeze().tolist()}")