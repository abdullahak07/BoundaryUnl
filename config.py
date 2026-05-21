"""
Configuration for MAF: Multimodal Unlearning Framework
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MAFConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    fusion_dim: int = 512
    text_model_name: str = "bert-base-uncased"
    image_backbone: str = "resnet50"           # resnet50
    audio_encoder: str = "crnn"                # cnn | bigru | crnn | resnet18

    # ── Training ───────────────────────────────────────────────────────────
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 30
    seed: int = 123

    # ── Unlearning ─────────────────────────────────────────────────────────
    unlearn_epochs: int = 10
    unlearn_lr: float = 5e-5
    unlearn_batch_size: int = 64
    early_stop_factor: float = 1.2
    max_unlearn_steps: int = 15  # cap batches per unlearn epoch     # stop when forget_acc < factor × chance

    # Loss weights (λ1, λ2, λ3)
    lambda1: float = 0.1   # gradient decoupling
    lambda2: float = 0.01  # curvature regularization
    lambda3: float = 0.05  # shared-space decorrelation
    alpha: float = 0.5     # retained structure preservation in LS
    lambda_h: float = 1.0  # scale inside LH
    lambda_r: float = 5.0

    # Warmup epochs for λ3 schedule
    warmup_epochs: int = 5

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset: str = "memotion7k"    # memotion7k | crema-d | meld
    data_root: str = "./data"
    max_text_len: int = 128
    image_size: int = 224
    n_mels: int = 128              # mel spectrogram bins
    audio_max_len: int = 300       # time frames

    # Forget class / speaker configuration
    # Memotion7k: forget "positive" (label 0)
    # CREMA-D:    forget emotions ["happy","sad"] by index
    # MELD:       forget specific speaker IDs
    forget_classes: List[int] = field(default_factory=lambda: [0])
    # MELD-specific: forget 'fear' emotion (label 2) — rare and isolated
    # Switch to speaker-level by passing --forget_speakers Ross Rachel
    meld_forget_classes: List[int] = field(default_factory=lambda: [2])

    # ── Privacy / MIA ─────────────────────────────────────────────────────
    n_shadow_models: int = 3
    mia_top_k: int = 3

    # ── Misc ──────────────────────────────────────────────────────────────
    device: str = "cuda"
    num_workers: int = 4
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


# Default config instance
DEFAULT_CONFIG = MAFConfig()

# Module-level constants expected by main.py
RESULTS_DIR = DEFAULT_CONFIG.results_dir          # "./results"
CONDITIONS  = DEFAULT_CONFIG.forget_classes        # [0]
TOP_K       = DEFAULT_CONFIG.mia_top_k             # 3