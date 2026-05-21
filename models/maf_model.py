"""
Fusion module and full MAF (Multimodal Unlearning Framework) model.

Architecture:
    Text  → TextEncoder       → z_t (B, D) ─┐
    Image → ImageEncoder      → z_i (B, D) ─┤
    Audio → AudioEncoder      → z_a (B, D) ─┤→ Attention Fusion → z_s (B, D) → Classifier
    Video → VisualFacetEncoder→ z_v (B, D) ─┘

Supports:
    - memotion7k: text + image
    - crema-d: audio
    - cmu-mosi: text + audio
    - cmu-mosei: text + audio
    - cmu-mosei-3modal: text + audio proxy/COVAREP-style vector + video FACET42
    - meld: text + audio
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.encoders import build_text_encoder, ImageEncoder, build_audio_encoder


# ─────────────────────────────────────────────────────────────────────────────
# Attention-based Fusion
# ─────────────────────────────────────────────────────────────────────────────

class AttentionFusion(nn.Module):
    """
    Learnable attention over available modality embeddings,
    projected into a shared fusion_dim space z_s.
    """

    def __init__(
        self,
        fusion_dim: int = 512,
        modalities: Tuple[str, ...] = ("text", "image", "audio"),
    ):
        super().__init__()
        self.modalities = modalities
        self.fusion_dim = fusion_dim
        self.n_modalities = len(modalities)

        # Per-modality attention score.
        self.attn = nn.Linear(fusion_dim * self.n_modalities, self.n_modalities)

        # Final projection.
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim * self.n_modalities, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Tanh(),
        )

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: dict mapping modality name -> (B, D) tensor

        Returns:
            z_s: (B, D) shared representation
        """
        missing = [m for m in self.modalities if m not in embeddings]
        if missing:
            raise KeyError(
                f"Missing modality embeddings for {missing}. "
                f"Expected modalities={self.modalities}, got={list(embeddings.keys())}"
            )

        vecs = [embeddings[m] for m in self.modalities]
        concat = torch.cat(vecs, dim=-1)                    # (B, D*n)
        weights = torch.softmax(self.attn(concat), dim=-1)  # (B, n)

        stack = torch.stack(vecs, dim=1)                    # (B, n, D)
        attended = (weights.unsqueeze(-1) * stack).sum(1)   # (B, D)

        z_s = self.proj(concat) + attended
        return z_s


# ─────────────────────────────────────────────────────────────────────────────
# Feature-vector encoders
# ─────────────────────────────────────────────────────────────────────────────

class CovarEPEncoder(nn.Module):
    """
    Lightweight encoder for 74-dim COVAREP-style acoustic/proxy features.

    Input:
        (B, 74) or (B, 1, 74)

    Output:
        (B, out_dim)
    """

    def __init__(self, input_dim: int = 74, hidden_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x.float())


class VisualFacetEncoder(nn.Module):
    """
    Encoder for FACET42 35-dim facial action-unit features.

    These are video-derived facial features extracted from video frames.

    Input:
        (B, 35) or (B, 1, 35)

    Output:
        (B, out_dim)
    """

    def __init__(self, input_dim: int = 35, hidden_dim: int = 128, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x.float())


# ─────────────────────────────────────────────────────────────────────────────
# Full MAF Model
# ─────────────────────────────────────────────────────────────────────────────

class MAFModel(nn.Module):
    """
    Complete multimodal model with separate encoders, attention fusion,
    and classification head.

    Supports unimodal, bimodal, and trimodal inputs.
    """

    def __init__(
        self,
        num_classes: int,
        fusion_dim: int = 512,
        modalities: Tuple[str, ...] = ("text", "image", "audio"),
        text_model_name: str = "bert-base-uncased",
        audio_encoder_type: str = "crnn",
        n_mels: int = 128,
        tiny: bool = False,
        video_encoder_type: str = None,
    ):
        super().__init__()
        self.modalities = modalities
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

        # Text encoder.
        if "text" in modalities:
            self.text_encoder = build_text_encoder(text_model_name, fusion_dim)

        # Image encoder.
        if "image" in modalities:
            self.image_encoder = ImageEncoder(fusion_dim, tiny=tiny)

        # Audio encoder.
        if "audio" in modalities:
            if audio_encoder_type == "covarep":
                self.audio_encoder = CovarEPEncoder(
                    input_dim=74,
                    hidden_dim=256,
                    out_dim=fusion_dim,
                )
            else:
                self.audio_encoder = build_audio_encoder(
                    audio_encoder_type,
                    n_mels,
                    fusion_dim,
                )

        # Video encoder for FACET42 feature vectors.
        if "video" in modalities:
            if video_encoder_type in (None, "facet42"):
                self.video_encoder = VisualFacetEncoder(
                    input_dim=35,
                    hidden_dim=128,
                    out_dim=fusion_dim,
                )
            else:
                raise ValueError(f"Unknown video_encoder_type: {video_encoder_type}")

        self.fusion = AttentionFusion(fusion_dim, modalities)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    # ── Forward ──────────────────────────────────────────────────────────────

    def encode(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run each modality encoder and return dict of embeddings.

        Expected batch keys:
            text  -> input_ids, attention_mask
            image -> image
            audio -> audio
            video -> video
        """
        embs = {}

        if "text" in self.modalities:
            if "input_ids" not in batch or "attention_mask" not in batch:
                raise KeyError("Text modality requires 'input_ids' and 'attention_mask'.")
            embs["text"] = self.text_encoder(
                batch["input_ids"],
                batch["attention_mask"],
            )

        if "image" in self.modalities:
            if "image" not in batch:
                raise KeyError("Image modality requires 'image'.")
            embs["image"] = self.image_encoder(batch["image"])

        if "audio" in self.modalities:
            if "audio" not in batch:
                raise KeyError("Audio modality requires 'audio'.")
            embs["audio"] = self.audio_encoder(batch["audio"])

        if "video" in self.modalities:
            if "video" not in batch:
                raise KeyError("Video modality requires 'video'.")
            embs["video"] = self.video_encoder(batch["video"])

        return embs

    def fuse(self, embs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse embeddings into shared representation z_s."""
        return self.fusion(embs)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, C)
            z_s:    (B, D)
        """
        embs = self.encode(batch)
        z_s = self.fuse(embs)
        logits = self.classifier(z_s)
        return logits, z_s

    def get_embeddings(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return per-modality embeddings for shared-space loss."""
        return self.encode(batch)

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits, _ = self.forward(batch)
        return logits.argmax(dim=-1)

    def predict_proba(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits, _ = self.forward(batch)
        return torch.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configs
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "memotion7k": {
        "num_classes": 3,
        "modalities": ("text", "image"),
        "chance": 1.0 / 3.0,
    },
    "crema-d": {
        "num_classes": 6,
        "modalities": ("audio",),
        "chance": 1.0 / 6.0,
    },
    "cmu-mosi": {
        "num_classes": 3,
        "modalities": ("text", "audio"),
        "audio_encoder": "covarep",
        "audio_dim": 74,
        "chance": 1.0 / 3.0,
    },
    "cmu-mosei": {
        "num_classes": 3,
        "modalities": ("text", "audio"),
        "audio_encoder": "covarep",
        "audio_dim": 74,
        "chance": 1.0 / 3.0,
    },
    "cmu-mosei-3modal": {
        "num_classes": 3,
        "modalities": ("text", "audio", "video"),
        "audio_encoder": "covarep",
        "video_encoder": "facet42",
        "audio_dim": 74,
        "video_dim": 35,
        "chance": 1.0 / 3.0,
    },
    "meld": {
        "num_classes": 7,
        "modalities": ("text", "audio"),
        "chance": 1.0 / 7.0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(dataset: str, config, tiny: bool = False) -> MAFModel:
    if dataset not in DATASET_CONFIGS:
        raise KeyError(
            f"Unknown dataset '{dataset}'. "
            f"Available datasets: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset]

    audio_encoder_type = cfg.get("audio_encoder", config.audio_encoder)
    video_encoder_type = cfg.get("video_encoder", None)

    return MAFModel(
        num_classes=cfg["num_classes"],
        fusion_dim=config.fusion_dim,
        modalities=cfg["modalities"],
        text_model_name=config.text_model_name,
        audio_encoder_type=audio_encoder_type,
        n_mels=config.n_mels,
        tiny=tiny,
        video_encoder_type=video_encoder_type,
    )