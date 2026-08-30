"""Model definitions.

Two detectors, both emitting a single logit (sigmoid -> P(AI-generated)):

  * EfficientNetDetector -- a torchvision EfficientNet backbone with a binary
    head. The original baseline; still selected by any `--model efficientnet_*`.

  * HybridDetector -- selected by `--model hybrid_clip`. Combines a frozen CLIP
    vision encoder (high-level semantics, transformation-robust, can't memorise
    generator fingerprints because it never trains) with a small low-level
    branch made of two fused sub-streams: a high-pass spatial residual CNN and
    a per-patch FFT spectrogram CNN. See config.HybridConfig for the rationale.

`build_model` dispatches on the configured name; `load_best_weights` restores a
checkpoint saved by train.py for either architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import HYBRID_CFG, HYBRID_MODEL_NAME, MODEL_CFG


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


# ---------------------------------------------------------------------------
# Hybrid detector: frozen CLIP semantics + low-level spatial/frequency branch.
# ---------------------------------------------------------------------------


def _gaussian_kernel2d(kernel_size, sigma):
    """A normalised 2D Gaussian kernel (kernel_size x kernel_size)."""
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g1d = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()
    return torch.outer(g1d, g1d)


class _SmallCNN(nn.Module):
    """Compact (Conv-BN-ReLU-MaxPool) stack -> global avg pool -> linear.

    Used for both low-level sub-streams. Kept tiny on purpose: the low-level
    signal is texture/noise statistics, not object semantics, so a few hundred
    thousand parameters is plenty and keeps overfitting (hence dataset-shortcut
    learning) in check.
    """

    def __init__(self, in_ch, out_dim, widths=(32, 64, 128)):
        super().__init__()
        layers = []
        c = in_ch
        for w in widths:
            layers += [
                nn.Conv2d(c, w, 3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c, out_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class HybridDetector(nn.Module):
    """CLIP semantic branch + low-level (spatial residual + FFT) branch.

    Input `x` is a (B, 4, H, W) tensor: channels 0-2 are the RGB image in
    [0, 1] (un-normalised), channel 3 is the per-patch FFT spectrogram from
    frequency.compute_freq_spectrogram. Packing the spectrogram as a 4th
    channel means every `model(x)` call site in the codebase stays unchanged;
    this module splits it back out internally.
    """

    # OpenAI CLIP image preprocessing statistics.
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, cfg=HYBRID_CFG, dropout=None):
        super().__init__()
        # Imported lazily so the rest of the codebase (and the EfficientNet
        # path) doesn't hard-depend on `transformers` being installed.
        from transformers import CLIPVisionModel

        self.cfg = cfg
        self.name = HYBRID_MODEL_NAME
        dropout = cfg.dropout if dropout is None else dropout

        # --- semantic branch: frozen CLIP vision tower -------------------
        try:
            self.clip = CLIPVisionModel.from_pretrained(cfg.clip_model_id)
        except Exception as exc:  # network error, missing cache, bad id, ...
            raise RuntimeError(
                f"Could not load CLIP checkpoint '{cfg.clip_model_id}'. Weights "
                "are fetched from the HuggingFace Hub on first use -- run once "
                "with internet access to populate the HF cache, or set "
                "HybridConfig.clip_model_id to a local directory."
            ) from exc
        if cfg.freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad_(False)
            self.clip.eval()
        self.clip_proj = nn.Sequential(
            nn.Linear(cfg.clip_feature_dim, cfg.clip_proj_dim),
            nn.LayerNorm(cfg.clip_proj_dim),
            nn.ReLU(inplace=True),
        )

        # --- low-level branch: spatial sub-stream ----------------------
        # A fixed high-pass (image minus its own blur) removes low-frequency
        # content -- scene layout, colour cast, exposure -- which are exactly
        # the cues a classifier uses as a dataset shortcut. What's left is
        # sensor/generator texture and noise.
        k = cfg.residual_kernel
        kern = _gaussian_kernel2d(k, cfg.residual_blur_sigma)  # (k, k)
        # depthwise kernel: one (1, k, k) filter per RGB channel -> (3, 1, k, k)
        self.register_buffer(
            "blur_kernel", kern.view(1, 1, k, k).expand(3, 1, k, k).contiguous())
        self.spatial_cnn = _SmallCNN(3, cfg.spatial_feat_dim)

        # --- low-level branch: frequency sub-stream ------------------
        self.freq_cnn = _SmallCNN(1, cfg.freq_feat_dim)

        # --- fuse the two low-level sub-streams ---------------------
        self.lowlevel_fuse = nn.Sequential(
            nn.Linear(cfg.spatial_feat_dim + cfg.freq_feat_dim, cfg.lowlevel_dim),
            nn.LayerNorm(cfg.lowlevel_dim),
            nn.ReLU(inplace=True),
        )

        # --- final fusion head ------------------------------------
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cfg.clip_proj_dim + cfg.lowlevel_dim, cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

        self.register_buffer("clip_mean", torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(self.CLIP_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        # Keep the frozen encoder in eval mode even when the detector is put in
        # train mode, so its LayerNorm/dropout behaviour never changes.
        super().train(mode)
        if self.cfg.freeze_clip:
            self.clip.eval()
        return self

    def _high_pass(self, rgb01):
        k = self.cfg.residual_kernel
        blurred = F.conv2d(rgb01, self.blur_kernel, padding=k // 2, groups=3)
        return rgb01 - blurred

    def _clip_features(self, rgb01):
        x = (rgb01 - self.clip_mean) / self.clip_std
        if self.cfg.freeze_clip:
            with torch.no_grad():
                out = self.clip(pixel_values=x)
        else:
            out = self.clip(pixel_values=x)
        return out.pooler_output

    def forward(self, x):
        rgb01 = x[:, :3]        # (B, 3, H, W) in [0, 1]
        spec = x[:, 3:4]        # (B, 1, H, W) FFT spectrogram

        clip_feat = self.clip_proj(self._clip_features(rgb01))

        spatial_feat = self.spatial_cnn(self._high_pass(rgb01))
        freq_feat = self.freq_cnn(spec)
        lowlevel = self.lowlevel_fuse(torch.cat([spatial_feat, freq_feat], dim=1))

        return self.head(torch.cat([clip_feat, lowlevel], dim=1))

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).squeeze(-1)


def build_model(cfg=MODEL_CFG, device=None):
    # Construct the detector and optionally move it to the target device.
    name = getattr(cfg, "name", cfg)
    if name == HYBRID_MODEL_NAME:
        model = HybridDetector(HYBRID_CFG)
    else:
        model = EfficientNetDetector(cfg.name, cfg.pretrained,
                                     cfg.num_classes, cfg.dropout)
    if device is not None:
        model.to(device)
    return model


def load_best_weights(model, path, device=None):
    # Load weights from a checkpoint produced by train.py.
    # Accepts either the full dict (with "model_state") or a raw state_dict.
    state = torch.load(path, map_location=device or "cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()  # set to evaluation mode for inference
    return model
