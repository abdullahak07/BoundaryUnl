

# ---------------------------------------------------------------------
# Fast-MU / MIA numerical guard
# ---------------------------------------------------------------------
def _finite_np(x):
    """
    Convert arrays/tensors/lists to finite numpy arrays.

    This prevents sklearn MIA classifiers from crashing when a baseline
    produces NaN or Inf logits/features.
    """
    return np.nan_to_num(
        np.asarray(x),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

"""
Evaluation metrics for MAF.

• Forget Accuracy  (↓)
• Retain Accuracy  (↑)
• Trade-off Score  (↑)  – harmonic mean of (1-FA) and RA
• MIA Attack Success Rate (↓)  – black-box shadow-model protocol
• Shared-Space Leakage (↓)  – Frobenius norm of cross-modal covariance
• Distance-to-Retrain (↓)   – CLEAR-style logit distance proxy
• Speaker-level metrics (SID Acc, EER, Cross-Modal Linkage)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from losses import _cross_modal_covariance

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


@torch.no_grad()
def _collect_logits_labels(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        logits, _ = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


@torch.no_grad()
def _collect_embeddings(
    model: nn.Module, loader: DataLoader, device: torch.device,
    modalities: Tuple[str, ...]
) -> Dict[str, torch.Tensor]:
    model.eval()
    acc = {m: [] for m in modalities}
    for batch in loader:
        batch = _to_device(batch, device)
        embs  = model.get_embeddings(batch)
        for m in modalities:
            if m in embs:
                acc[m].append(embs[m].cpu())
    return {m: torch.cat(v) for m, v in acc.items() if v}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Forget / Retain Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def forget_retain_accuracy(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (forget_acc, retain_acc)."""
    def _acc(loader):
        logits, labels = _collect_logits_labels(model, loader, device)
        return (logits.argmax(1) == labels).float().mean().item()
    return _acc(forget_loader), _acc(retain_loader)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trade-off Score  (Eq. 15)
# ─────────────────────────────────────────────────────────────────────────────

def trade_off_score(forget_acc: float, retain_acc: float) -> float:
    """
    T = 2·(1-FA)·RA / ((1-FA)+RA)   (harmonic mean)
    """
    a = 1 - forget_acc
    b = retain_acc
    denom = a + b
    if denom < 1e-9:
        return 0.0
    return 2 * a * b / denom


# ─────────────────────────────────────────────────────────────────────────────
# 3. Membership Inference Attack (MIA) ASR
# ─────────────────────────────────────────────────────────────────────────────

def _mia_features(logits: torch.Tensor, top_k: int = 3) -> np.ndarray:
    """
    Feature vector for MIA attacker:
    [top-K confidence scores, entropy, max-2nd gap, label]
    """
    probs = torch.softmax(logits, dim=-1)
    top_vals, _ = probs.topk(min(top_k, probs.size(-1)), dim=-1)
    entropy = -(probs * (probs + 1e-9).log()).sum(-1, keepdim=True)
    gap     = (top_vals[:, 0:1] - top_vals[:, 1:2]) if top_vals.size(1) > 1 \
              else torch.zeros(logits.size(0), 1)
    return torch.cat([top_vals, entropy, gap], dim=-1).numpy()


def mia_attack_success_rate(
    target_model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: torch.device,
    top_k: int = 3,
    n_shadow: int = 3,
) -> float:
    """
    Black-box MIA following Shokri et al. (2017).

    Shadow models are approximated by train/val splits of the retain set.
    An ASR near 0.5 indicates random guessing → strong privacy protection.
    """
    if not _HAS_SKLEARN:
        logger.warning("sklearn not available – returning 0.5 as MIA ASR")
        return 0.5

    # Collect target model outputs
    forget_logits, _ = _collect_logits_labels(target_model, forget_loader, device)
    retain_logits, _ = _collect_logits_labels(target_model, retain_loader, device)

    # Feature extraction
    X_member    = _mia_features(retain_logits, top_k)
    X_nonmember = _mia_features(forget_logits, top_k)

    n = min(len(X_member), len(X_nonmember))
    if n < 10:
        return 0.5

    X = _finite_np(np.vstack([X_member[:n], X_nonmember[:n]]))
    y = np.array([1] * n + [0] * n)

    # 3-fold cross-val approximation of shadow model accuracy
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    from sklearn.model_selection import cross_val_score
    clf    = LogisticRegression(max_iter=500, C=1.0)
    scores = cross_val_score(clf, X_s, y, cv=max(2, min(3, n_shadow)), scoring="accuracy")
    asr    = float(scores.mean())
    return asr


# ─────────────────────────────────────────────────────────────────────────────
# 4. Shared-Space Leakage
# ─────────────────────────────────────────────────────────────────────────────

def shared_space_leakage(
    model: nn.Module,
    forget_loader: DataLoader,
    device: torch.device,
    modalities: Tuple[str, ...],
) -> float:
    """
    Frobenius norm of cross-modal covariance on forgotten samples.
    Lower is better (less residual multimodal leakage).
    """
    pairs = [(modalities[i], modalities[j])
             for i in range(len(modalities))
             for j in range(i + 1, len(modalities))]
    if not pairs:
        return 0.0

    embs = _collect_embeddings(model, forget_loader, device, modalities)
    total = 0.0
    for (p, q) in pairs:
        if p in embs and q in embs:
            cov = _cross_modal_covariance(embs[p].to(device), embs[q].to(device))
            total += cov.pow(2).sum().sqrt().item()   # Frobenius norm
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 5. Distance-to-Retrain  (CLEAR-style proxy)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def distance_to_retrain(
    unlearned_model: nn.Module,
    retrain_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    L2 distance between logit distributions of unlearned and retrained models.
    Closer to 0 → unlearned model approximates oracle retrain.
    """
    unlearned_model.eval()
    retrain_model.eval()
    dists = []
    for batch in test_loader:
        batch = _to_device(batch, device)
        l_u, _ = unlearned_model(batch)
        l_r, _ = retrain_model(batch)
        d = F.mse_loss(
            F.softmax(l_u, dim=-1),
            F.softmax(l_r, dim=-1),
        )
        dists.append(d.item())
    return float(np.mean(dists)) if dists else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Speaker-level metrics (SID Acc, EER, Cross-Modal Linkage)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def speaker_level_metrics(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: torch.device,
    num_speakers: int = 10,
) -> Dict[str, float]:
    """
    SID Acc : Speaker-ID classification accuracy on forget speakers
              (proxy: classification accuracy on forgotten set)
    EER     : Equal Error Rate (approximated via threshold sweep)
    Linkage : Cross-modal (text-audio) cosine similarity on forgotten data
    """
    results = {}

    # SID Acc ≈ forget accuracy (a low value means speaker identity removed)
    forget_logits, forget_labels = _collect_logits_labels(model, forget_loader, device)
    results["sid_acc"] = (forget_logits.argmax(1) == forget_labels).float().mean().item()

    # EER approximation: use max softmax score as confidence
    probs  = torch.softmax(forget_logits, dim=-1)
    scores = probs.max(dim=-1).values.numpy()
    labels = (forget_logits.argmax(1) == forget_labels).long().numpy()

    thresholds = np.linspace(0, 1, 200)
    far_list, frr_list = [], []
    for t in thresholds:
        accepted  = scores >= t
        far = (accepted & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        frr = (~accepted & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        far_list.append(far); frr_list.append(frr)
    far_arr = np.array(far_list); frr_arr = np.array(frr_list)
    eer_idx  = np.argmin(np.abs(far_arr - frr_arr))
    results["eer"] = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2)

    # Cross-modal linkage: cosine similarity between text and audio embeddings
    # Only meaningful for models with both modalities
    forget_embs = _collect_embeddings(model, forget_loader, device,
                                      ("text", "audio"))
    if "text" in forget_embs and "audio" in forget_embs:
        zt = F.normalize(forget_embs["text"], dim=-1)
        za = F.normalize(forget_embs["audio"], dim=-1)
        linkage = (zt * za).sum(-1).mean().item()
    else:
        linkage = 0.0
    results["linkage"] = linkage

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full Evaluation Suite
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: torch.device,
    modalities: Tuple[str, ...],
    retrain_model: Optional[nn.Module] = None,
    mia_n_shadow: int = 3,
    mia_top_k: int = 3,
) -> Dict[str, float]:
    """
    Run all MAF evaluation metrics and return a flat results dict.
    """
    logger.info("Evaluating forget/retain accuracy …")
    forget_acc, retain_acc = forget_retain_accuracy(
        model, forget_loader, retain_loader, device
    )
    t_score = trade_off_score(forget_acc, retain_acc)

    logger.info("Running MIA …")
    asr = mia_attack_success_rate(
        model, forget_loader, retain_loader, device, mia_top_k, mia_n_shadow
    )

    logger.info("Computing shared-space leakage …")
    leakage = shared_space_leakage(model, forget_loader, device, modalities)

    results = {
        "forget_acc":     forget_acc,
        "retain_acc":     retain_acc,
        "trade_off":      t_score,
        "mia_asr":        asr,
        "ss_leakage":     leakage,
    }

    if retrain_model is not None:
        logger.info("Computing distance-to-retrain …")
        d2r = distance_to_retrain(model, retrain_model, forget_loader, device)
        results["distance_to_retrain"] = d2r

    return results


def print_results_table(results: Dict[str, float], dataset: str = ""):
    """Pretty-print evaluation results."""
    header = f"\n{'─'*55}\n  MAF Evaluation Results  {dataset}\n{'─'*55}"
    print(header)
    for k, v in results.items():
        arrow = "↓" if k in ("forget_acc", "mia_asr", "ss_leakage", "distance_to_retrain") else "↑"
        print(f"  {k:<25s} {arrow}  {v:.4f}")
    print("─" * 55)
