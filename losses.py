"""
MAF Loss Functions
==================

LBU  – Boundary Unlearning Loss  (Eq. 3–4)
LG   – Gradient-Guided Forgetting (Eq. 5–6)
LH   – Hessian-Guided Stabilization (Eq. 7–8)
LS   – Shared-Space Unlearning (Eq. 9–12)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Boundary Unlearning  (LBU = LB + λ_r · LR)
# ─────────────────────────────────────────────────────────────────────────────

class BoundaryUnlearningLoss(nn.Module):
    """
    LB: Pushes predicted distribution of forgotten samples toward uniform,
        encouraging high-entropy / chance-level predictions.

    LBU = LB + lambda_r * LR
    """

    def __init__(self, num_classes: int, lambda_r: float = 1.0):
        super().__init__()
        self.C = num_classes
        self.lambda_r = lambda_r

    def forward(
        self,
        logits_f: torch.Tensor,        # (B_f, C)  – on forgotten samples
        logits_r: torch.Tensor,        # (B_r, C)  – on retained samples
        labels_r: torch.Tensor,        # (B_r,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            L_BU, L_B, L_R  (all scalars)
        """
        B_f = logits_f.size(0)
        # Uniform target: (B_f, C) filled with 1/C
        uniform = torch.full_like(logits_f, 1.0 / self.C)

        # LB = mean cross-entropy between predictions and uniform distribution
        log_probs = F.log_softmax(logits_f, dim=-1)
        L_B = -(uniform * log_probs).sum(dim=-1).mean()

        # LR = standard cross-entropy on retained data
        L_R = F.cross_entropy(logits_r, labels_r)

        L_BU = L_B + self.lambda_r * L_R
        return L_BU, L_B, L_R


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gradient-Guided Forgetting  (LG, g_perp)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_gradient_penalty(
    gf: torch.Tensor,
    gr: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    LG = cos²(gf, gr)  – penalises alignment without sign ambiguity.

    Both gf and gr are flattened parameter-gradient vectors.
    """
    gf_flat = torch.cat([g.reshape(-1) for g in gf])
    gr_flat = torch.cat([g.reshape(-1) for g in gr])
    cos = F.cosine_similarity(gf_flat.unsqueeze(0), gr_flat.unsqueeze(0))
    return cos.pow(2)


def orthogonalize_gradient(
    gf: Tuple[torch.Tensor, ...],
    gr: Tuple[torch.Tensor, ...],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, ...]:
    """
    Eq. 6: g_perp = gf - (gf·gr / ||gr||² + ε) * gr

    Removes the component of the forgetting gradient that aligns
    with the retained gradient.

    Args:
        gf, gr: tuples of per-parameter gradients (same structure)
    Returns:
        g_perp: tuple with same structure as gf
    """
    # Flatten
    gf_flat = torch.cat([g.reshape(-1) for g in gf])
    gr_flat = torch.cat([g.reshape(-1) for g in gr])

    projection = (gf_flat @ gr_flat) / (gr_flat @ gr_flat + eps)
    gf_perp_flat = gf_flat - projection * gr_flat

    # Reconstruct original shapes
    result, idx = [], 0
    for g in gf:
        n = g.numel()
        result.append(gf_perp_flat[idx:idx + n].view_as(g))
        idx += n
    return tuple(result)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hessian-Guided Stabilization  (LH)
# ─────────────────────────────────────────────────────────────────────────────

class HessianStabilizationLoss(nn.Module):
    """
    Eq. 7–8: Diagonal Fisher approximation of Hessian as curvature estimate.

    LH = (λ_h / 2) · Σ_i  F_i · (θ_i - θ0_i)²

    Penalises large changes to high-curvature (high-Fisher) parameters,
    analogous to Elastic Weight Consolidation (EWC).
    """

    def __init__(self, lambda_h: float = 1.0):
        super().__init__()
        self.lambda_h = lambda_h

    def compute_fisher(
        self,
        model: nn.Module,
        forget_loader,
        device: torch.device,
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Accumulate diagonal Fisher over forget set.

        Returns dict: param_name → Fisher diagonal (same shape as param).
        """
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}
        # MUST stay in train() mode — cuDNN RNN (GRU/LSTM) only supports
        # backward in training mode.
        was_training = model.training
        model.train()
        count = 0
        for batch in forget_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits, _ = model(batch)
            log_probs = F.log_softmax(logits, dim=-1)
            pred = log_probs.argmax(dim=-1)
            loss = F.nll_loss(log_probs, pred)
            model.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            count += 1
            if n_samples and count >= n_samples:
                break

        # Restore original mode and normalise
        model.train(was_training)
        for name in fisher:
            fisher[name] /= max(count, 1)
        return fisher

    def forward(
        self,
        model: nn.Module,
        theta0: Dict[str, torch.Tensor],
        fisher: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            model:  current model
            theta0: reference parameters (before unlearning)
            fisher: diagonal Fisher dict (from compute_fisher)
        Returns:
            L_H scalar
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in fisher and name in theta0:
                F_i   = fisher[name]
                diff  = param - theta0[name].to(param.device)
                loss += (F_i * diff.pow(2)).sum()
        return (self.lambda_h / 2) * loss


# ─────────────────────────────────────────────────────────────────────────────
# 4. Shared-Space Unlearning  (LS)
# ─────────────────────────────────────────────────────────────────────────────

def _cross_modal_covariance(
    zp: torch.Tensor, zq: torch.Tensor
) -> torch.Tensor:
    """
    Eq. 9: Cov(z_p, z_q) = 1/(B-1) Σ (z_p,b - ẑ_p)(z_q,b - ẑ_q)ᵀ

    Args:
        zp, zq: (B, D)
    Returns:
        cov: (D, D)
    """
    B = zp.size(0)
    zp_c = zp - zp.mean(0, keepdim=True)
    zq_c = zq - zq.mean(0, keepdim=True)
    return (zp_c.t() @ zq_c) / max(B - 1, 1)


class SharedSpaceLoss(nn.Module):
    """
    Eq. 10–12:

    L_S^(f) = Σ_{(p,q)} ||Cov(z_p^f, z_q^f)||_F²     (forget: minimise correlation)
    L_S^(r) = Σ_{(p,q)} ||Cov(z_p^r, z_q^r) - Cov0||_F  (retain: preserve structure)
    L_S     = L_S^(f) + α · L_S^(r)

    For unimodal datasets (e.g. CREMA-D audio-only), LS is disabled (returns 0).
    """

    def __init__(self, modalities: Tuple[str, ...], alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        # All unordered pairs
        mods = list(modalities)
        self.pairs = [(mods[i], mods[j])
                      for i in range(len(mods))
                      for j in range(i + 1, len(mods))]
        self.enabled = len(self.pairs) > 0

    def forward(
        self,
        forget_embs: Dict[str, torch.Tensor],
        retain_embs: Dict[str, torch.Tensor],
        ref_covs: Optional[Dict[Tuple[str, str], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            forget_embs: {mod: (B_f, D)}
            retain_embs: {mod: (B_r, D)}
            ref_covs:    precomputed reference covariances from θ0 (optional)
        Returns:
            L_S, L_S_f, L_S_r
        """
        device = next(iter(forget_embs.values())).device
        L_S_f = torch.zeros(1, device=device)
        L_S_r = torch.zeros(1, device=device)

        if not self.enabled:
            zero = torch.zeros(1, device=device)
            return zero, zero, zero

        for (p, q) in self.pairs:
            if p not in forget_embs or q not in forget_embs:
                continue

            # Forget: minimise cross-modal covariance  (Eq. 10)
            cov_f = _cross_modal_covariance(forget_embs[p], forget_embs[q])
            L_S_f += cov_f.pow(2).sum()   # Frobenius² = sum of squared entries

            # Retain: stabilise relative to reference  (Eq. 11)
            if p in retain_embs and q in retain_embs:
                cov_r = _cross_modal_covariance(retain_embs[p], retain_embs[q])
                if ref_covs and (p, q) in ref_covs:
                    diff = cov_r - ref_covs[(p, q)].to(device)
                    L_S_r += diff.pow(2).sum().sqrt()   # Frobenius norm
                else:
                    # If no reference, penalise absolute retained covariance lightly
                    L_S_r += cov_r.pow(2).sum() * 0.1

        L_S = L_S_f + self.alpha * L_S_r
        return L_S, L_S_f, L_S_r

    @torch.no_grad()
    def compute_reference_covs(
        self,
        model: nn.Module,
        retain_loader,
        device: torch.device,
        n_batches: int = 20,
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Pre-compute reference covariances from θ0 on retained data."""
        cov_accum = {pair: [] for pair in self.pairs}
        model.eval()
        for i, batch in enumerate(retain_loader):
            if i >= n_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            embs = model.get_embeddings(batch)
            for (p, q) in self.pairs:
                if p in embs and q in embs:
                    cov_accum[(p, q)].append(
                        _cross_modal_covariance(embs[p], embs[q]).cpu()
                    )
        ref = {}
        for pair, covs in cov_accum.items():
            if covs:
                ref[pair] = torch.stack(covs).mean(0)
        return ref


# ─────────────────────────────────────────────────────────────────────────────
# 5. Unified MAF Objective  (Eq. 13)
# ─────────────────────────────────────────────────────────────────────────────

class MAFLoss(nn.Module):
    """
    L_total = L_BU + λ1·L_G + λ2·L_H + λ3(e)·L_S

    with λ3(e) warm-up schedule (Eq. 14).
    """

    def __init__(
        self,
        num_classes: int,
        modalities: Tuple[str, ...],
        lambda1: float = 0.1,
        lambda2: float = 0.01,
        lambda3: float = 0.05,
        alpha: float = 0.5,
        lambda_h: float = 1.0,
        lambda_r: float = 1.0,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.warmup_epochs = warmup_epochs

        self.bul  = BoundaryUnlearningLoss(num_classes, lambda_r)
        self.ssl  = SharedSpaceLoss(modalities, alpha)
        self.hsl  = HessianStabilizationLoss(lambda_h)

    def lambda3_schedule(self, epoch: int) -> float:
        """Eq. 14: linear warm-up over first warmup_epochs."""
        return min(self.lambda3, (epoch / max(self.warmup_epochs, 1)) * self.lambda3)

    def forward(
        self,
        logits_f: torch.Tensor,
        logits_r: torch.Tensor,
        labels_r: torch.Tensor,
        forget_embs: Dict[str, torch.Tensor],
        retain_embs: Dict[str, torch.Tensor],
        model: nn.Module,
        theta0: Dict[str, torch.Tensor],
        fisher: Dict[str, torch.Tensor],
        ref_covs: Optional[Dict] = None,
        epoch: int = 0,
        grad_gf: Optional[Tuple] = None,
        grad_gr: Optional[Tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with all loss components for logging.
        """
        # 1. Boundary unlearning
        L_BU, L_B, L_R = self.bul(logits_f, logits_r, labels_r)

        # 2. Gradient penalty (computed externally; we just record it here)
        if grad_gf is not None and grad_gr is not None:
            L_G = cosine_gradient_penalty(grad_gf, grad_gr)
        else:
            L_G = torch.zeros(1, device=logits_f.device)

        # 3. Hessian stabilisation
        L_H = self.hsl(model, theta0, fisher)

        # 4. Shared-space
        L3_w = self.lambda3_schedule(epoch)
        L_S, L_S_f, L_S_r = self.ssl(forget_embs, retain_embs, ref_covs)

        # Total
        L_total = L_BU + self.lambda1 * L_G + self.lambda2 * L_H + L3_w * L_S

        return {
            "total":  L_total,
            "L_BU":   L_BU,
            "L_B":    L_B,
            "L_R":    L_R,
            "L_G":    L_G,
            "L_H":    L_H,
            "L_S":    L_S,
            "L_S_f":  L_S_f,
            "L_S_r":  L_S_r,
            "lambda3_eff": torch.tensor(L3_w),
        }
