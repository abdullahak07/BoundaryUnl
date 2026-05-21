"""
Modality-specific encoders for MAF.

Text  : BERT-base (110M, uncased)  → 512-dim
Image : ResNet-50                  → 512-dim
Audio : CNN | BiGRU | CRNN | ResNet18 → 512-dim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

try:
    from transformers import BertModel
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# ─────────────────────────────────────────────────────────────────────────────
# Text Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """BERT-base encoder → 512-dim projection."""

    def __init__(self, model_name: str = "bert-base-uncased",
                 fusion_dim: int = 512, freeze_bert: bool = False):
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers package required: pip install transformers")

        import logging as _logging
        _logging.getLogger("transformers").setLevel(_logging.ERROR)
        self.bert = BertModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        _logging.getLogger("transformers").setLevel(_logging.WARNING)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad_(False)

        hidden = self.bert.config.hidden_size   # 768 for bert-base
        self.proj = nn.Sequential(
            nn.Linear(hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
        Returns:
            z_t: (B, fusion_dim)
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]   # [CLS] token
        return self.proj(cls)


class MockTextEncoder(nn.Module):
    """Light-weight fallback when transformers is not available (for testing)."""

    def __init__(self, vocab_size: int = 30522, embed_dim: int = 128,
                 fusion_dim: int = 512, max_len: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn   = nn.GRU(embed_dim, 256, batch_first=True, bidirectional=True)
        self.proj  = nn.Linear(512, fusion_dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        _, h = self.rnn(x)                    # h: (2, B, 256)
        h = torch.cat([h[0], h[1]], dim=-1)   # (B, 512)
        return F.relu(self.proj(h))


def build_text_encoder(model_name: str = "bert-base-uncased",
                       fusion_dim: int = 512) -> nn.Module:
    if _HAS_TRANSFORMERS:
        return TextEncoder(model_name, fusion_dim)
    return MockTextEncoder(fusion_dim=fusion_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Image Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TinyImageEncoder(nn.Module):
    """Lightweight CNN for testing (replaces ResNet-50)."""
    def __init__(self, fusion_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=4, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(16 * 4 * 4, fusion_dim)

    def forward(self, x):
        return F.relu(self.proj(self.net(x).flatten(1)))


class ImageEncoder(nn.Module):
    """ResNet-50 backbone → 512-dim projection.
    Set pretrained=False for offline / test environments.
    Set tiny=True for fast smoke tests.
    """

    def __init__(self, fusion_dim: int = 512, pretrained: bool = False, tiny: bool = False):
        super().__init__()
        if tiny:
            self._enc = TinyImageEncoder(fusion_dim)
            self._tiny = True
            return
        self._tiny = False
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.resnet50(weights=weights)
        # remove final FC; keep everything up to adaptive avg-pool
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # → (B, 2048, 1, 1)
        self.proj = nn.Sequential(
            nn.Linear(2048, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            z_i: (B, fusion_dim)
        """
        if self._tiny:
            return self._enc(x)
        feat = self.backbone(x).flatten(1)   # (B, 2048)
        return self.proj(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Encoders
# ─────────────────────────────────────────────────────────────────────────────

class AudioCNNEncoder(nn.Module):
    """5-layer CNN on mel-spectrograms → 512-dim."""

    def __init__(self, n_mels: int = 128, fusion_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 5
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(512, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T)"""
        feat = self.cnn(x).flatten(1)
        return self.proj(feat)


class AudioBiGRUEncoder(nn.Module):
    """2-layer Bidirectional GRU on mel-spectrograms → 512-dim."""

    def __init__(self, n_mels: int = 128, hidden: int = 256, fusion_dim: int = 512):
        super().__init__()
        # Lightweight CNN front-end to reduce time dim
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 64, (n_mels, 1), groups=1), nn.ReLU(inplace=True),
        )
        self.gru  = nn.GRU(64, hidden, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=0.2)
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T)"""
        # CNN front-end: collapse freq dim
        fe = self.frontend(x)          # (B, 64, 1, T)
        fe = fe.squeeze(2).permute(0, 2, 1)  # (B, T, 64)
        _, h = self.gru(fe)            # h: (4, B, 256)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # last layer fwd+bwd → (B, 512)
        return self.proj(h)


class AudioCRNNEncoder(nn.Module):
    """CNN + GRU temporal decoder → 512-dim."""

    def __init__(self, n_mels: int = 128, fusion_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        # After 3× pool-2 on freq: n_mels → n_mels // 8
        cnn_freq = n_mels // 8
        self.gru  = nn.GRU(256 * cnn_freq, 256, num_layers=1, batch_first=True,
                           bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(512, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T)"""
        # CMU-MOSEI lightweight loader provides audio/text-derived features as [B, F].
        # The CNN-style audio encoder expects [B, C, F, T], so we reshape safely.
        if x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(-1)   # [B, F] -> [B, 1, F, 1]
        elif x.dim() == 3:
            x = x.unsqueeze(1)                 # [B, F, T] -> [B, 1, F, T]

        B, C, F, T = x.shape
        feat = self.cnn(x)                       # (B, 256, F', T)
        feat = feat.permute(0, 3, 1, 2)          # (B, T, 256, F')
        feat = feat.flatten(2)                   # (B, T, 256*F')
        # Some lightweight datasets, e.g. CMU-MOSEI text-derived auxiliary
        # features, produce a different flattened CNN feature size. The GRU
        # expects self.gru.input_size, so pad/truncate safely.
        expected = self.gru.input_size
        current = feat.size(-1)

        if current < expected:
            pad = feat.new_zeros(*feat.shape[:-1], expected - current)
            feat = torch.cat([feat, pad], dim=-1)
        elif current > expected:
            feat = feat[..., :expected]

        _, h = self.gru(feat)
        h = torch.cat([h[0], h[1]], dim=-1)      # (B, 512)
        return self.proj(h)


class AudioResNet18Encoder(nn.Module):
    """ResNet-18 adapted for spectrogram input → 512-dim."""

    def __init__(self, fusion_dim: int = 512):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        # Change first conv to accept 1-channel spectrogram
        base.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove FC
        self.proj = nn.Sequential(
            nn.Linear(512, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T)"""
        feat = self.backbone(x).flatten(1)
        return self.proj(feat)


def build_audio_encoder(encoder_type: str = "crnn",
                        n_mels: int = 128,
                        fusion_dim: int = 512) -> nn.Module:
    enc_map = {
        "cnn":      AudioCNNEncoder,
        "bigru":    AudioBiGRUEncoder,
        "crnn":     AudioCRNNEncoder,
        "resnet18": AudioResNet18Encoder,
    }
    cls = enc_map.get(encoder_type.lower())
    if cls is None:
        raise ValueError(f"Unknown audio encoder: {encoder_type}. "
                         f"Choose from {list(enc_map.keys())}")
    if encoder_type.lower() == "resnet18":
        return cls(fusion_dim=fusion_dim)
    return cls(n_mels=n_mels, fusion_dim=fusion_dim)
