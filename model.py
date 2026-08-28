"""Model definition: an EfficientNet backbone with a binary real-vs-AI head.

The network outputs a single logit; a sigmoid converts it to the probability
that an image is AI-generated. Helper functions build the model and load a
saved checkpoint.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from config import MODEL_CFG


class EfficientNetDetector(nn.Module):
    def __init__(self, name=MODEL_CFG.name, pretrained=MODEL_CFG.pretrained,
                 num_classes=MODEL_CFG.num_classes, dropout=MODEL_CFG.dropout):
        super().__init__()
        self.name = name
        # "DEFAULT" pulls the recommended ImageNet-pretrained weights; None = random init.
        weights = "DEFAULT" if pretrained else None
        # Look up the requested torchvision architecture by name.
        try:
            builder = getattr(models, name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported model: {name}") from exc
        self.backbone = builder(weights=weights)
        # Replace the stock classifier with: dropout -> linear head (num_classes logits).
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        # Standard forward pass through the (modified) backbone.
        return self.backbone(x)

    def predict_proba(self, x):
        # Inference helper: returns AI probabilities in [0, 1] for a batch.
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(-1)


def build_model(cfg=MODEL_CFG, device=None):
    # Construct the detector and optionally move it to the target device.
    model = EfficientNetDetector(cfg.name, cfg.pretrained,
                                 cfg.num_classes, cfg.dropout)
    if device is not None:
        model.to(device)
    return model


def load_best_weights(model, path, device=None):
    # Load weights from a checkpoint produced by train.py.
    # Accepts either the full dict (with "model_state") or a raw state_dict.
    state = torch.load(path, map_location=device or "cpu")
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()  # set to evaluation mode for inference
    return model
