"""
MAF Unlearner — implements Algorithm 1 from the paper.

Steps per epoch:
    1. Sample Br ⊂ Dr, Bf ⊂ Df
    2. Compute L_R, L_BU
    3. Compute gr = ∇θ L_R,  gf = ∇θ L_BU
    4. Orthogonalise: g_perp = gf − (gf·gr / ||gr||²) gr
    5. Compute L_G = cos²(gf, gr)  [diagnostic only]
    6. Compute Fisher diagonal Fi
    7. Compute L_H (Hessian stabilisation)
    8. Compute L_S with warm-up λ3(e)
    9. Total loss: L_BU + λ1·L_G + λ2·L_H + λ3(e)·L_S
   10. Replace ∇θ L_BU with g_perp in backward pass
   11. Adam update
"""

import copy
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from losses import (
    BoundaryUnlearningLoss,
    HessianStabilizationLoss,
    SharedSpaceLoss,
    cosine_gradient_penalty,
    orthogonalize_gradient,
    MAFLoss,
)
from models.maf_model import DATASET_CONFIGS

logger = logging.getLogger(__name__)


class MAFUnlearner:
    """
    Implements the MAF unlearning loop (Algorithm 1).

    Usage:
        unlearner = MAFUnlearner(model, config, device)
        unlearner.setup(forget_loader, retain_loader)
        results   = unlearner.unlearn()
    """

    def __init__(self, model: nn.Module, config, device: torch.device):
        self.model  = model
        self.config = config
        self.device = device
        self.model.to(device)

        dataset_cfg  = DATASET_CONFIGS[config.dataset]
        self.num_classes  = dataset_cfg["num_classes"]
        self.chance_level = dataset_cfg["chance"]
        self.modalities   = dataset_cfg["modalities"]

        # Freeze a copy of θ0 (pre-unlearning parameters)
        self.theta0: Dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }

        # Losses
        self.bul = BoundaryUnlearningLoss(self.num_classes, config.lambda_r)
        self.hsl = HessianStabilizationLoss(config.lambda_h)
        self.ssl = SharedSpaceLoss(self.modalities, config.alpha)

        self.optimizer = optim.Adam(model.parameters(), lr=config.unlearn_lr)
        self.dataset   = getattr(config, "dataset", "model")
        self.ckpt_dir  = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated in setup()
        self.forget_loader = None
        self.retain_loader = None
        self.fisher: Dict[str, torch.Tensor] = {}
        self.ref_covs = {}

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup(self, forget_loader: DataLoader, retain_loader: DataLoader):
        """
        Pre-compute Fisher diagonals and reference covariances before unlearning.
        """
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader

        logger.info("Computing diagonal Fisher on forget set …")
        self.fisher = self.hsl.compute_fisher(
            self.model, forget_loader, self.device, n_samples=50
        )

        if self.ssl.enabled:
            # Reference covariance precompute skipped — too slow on large
            # datasets (MELD: 13k utterances × BERT+audio = hours per batch).
            # SharedSpaceLoss uses absolute covariance fallback instead.
            self.ref_covs = {}
            logger.info("Shared-space loss active (fast mode, no ref-cov precompute)")
    # ── Main unlearning loop ──────────────────────────────────────────────────

    def unlearn(self) -> Dict[str, Any]:
        """
        Run Algorithm 1 for config.unlearn_epochs epochs.
        Returns a history dict with per-epoch metrics.
        """
        assert self.forget_loader and self.retain_loader, \
            "Call setup() before unlearn()"

        history = {k: [] for k in [
            "forget_acc", "retain_acc", "L_total",
            "L_B", "L_G", "L_H", "L_S",
        ]}

        forget_iter = iter(self.forget_loader)
        retain_iter = iter(self.retain_loader)

        for epoch in range(1, self.config.unlearn_epochs + 1):
            t0 = time.time()
            metrics = self._unlearn_epoch(epoch, forget_iter, retain_iter)

            # Reset iterators if exhausted
            for k, v in metrics.items():
                if k in history:
                    history[k].append(v)

            elapsed = time.time() - t0
            logger.info(
                f"[Unlearn] Epoch {epoch:02d}/{self.config.unlearn_epochs}  "
                f"forget_acc={metrics['forget_acc']:.4f}  "
                f"retain_acc={metrics['retain_acc']:.4f}  "
                f"L_total={metrics['L_total']:.4f}  ({elapsed:.1f}s)"
            )

            # Early stopping: forget acc < 1.5 × chance
            if metrics["forget_acc"] < self.config.early_stop_factor * self.chance_level:
                logger.info(
                    f"Early stop at epoch {epoch}: "
                    f"forget_acc {metrics['forget_acc']:.4f} < "
                    f"{self.config.early_stop_factor}×chance={self.config.early_stop_factor * self.chance_level:.4f}"
                )
                break

            # Refresh iterators each epoch
            forget_iter = iter(self.forget_loader)
            retain_iter = iter(self.retain_loader)

        self._save_checkpoint(f"{self.dataset}_unlearned_model.pt")
        return history

    def _unlearn_epoch(
        self,
        epoch: int,
        forget_iter,
        retain_iter,
    ) -> Dict[str, float]:
        self.model.train()

        total_loss_sum = 0.0
        lb_sum = lg_sum = lh_sum = ls_sum = 0.0
        n_steps = 0

        # Cap steps per epoch — prevents full-dataset sweep from
        # destroying retained knowledge on large datasets like MELD.
        max_steps = getattr(self.config, "max_unlearn_steps", 100)
        for step_idx, batch_f in enumerate(forget_iter):
            if step_idx >= max_steps:
                break
            # Get a retain batch (cycle)
            try:
                batch_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(self.retain_loader)
                batch_r = next(retain_iter)

            batch_f = self._to_device(batch_f)
            batch_r = self._to_device(batch_r)
            labels_r = batch_r["label"]

            # ── Step 1: Forward pass ─────────────────────────────────────────
            logits_f, _ = self.model(batch_f)
            logits_r, _ = self.model(batch_r)

            # ── Step 2: L_BU and base gradients ─────────────────────────────
            L_BU, L_B, L_R = self.bul(logits_f, logits_r, labels_r)

            # Compute gr = ∇θ L_R  (no graph for gr)
            self.model.zero_grad()
            L_R.backward(retain_graph=True)
            gr = tuple(
                p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p)
                for p in self.model.parameters()
            )

            # Compute gf = ∇θ L_BU
            self.model.zero_grad()
            L_BU.backward(retain_graph=True)
            gf = tuple(
                p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p)
                for p in self.model.parameters()
            )

            # ── Step 3: Gradient orthogonalisation (g_perp) ─────────────────
            g_perp = orthogonalize_gradient(gf, gr)

            # ── Step 4: Gradient penalty L_G (diagnostic) ───────────────────
            L_G = cosine_gradient_penalty(gf, gr)

            # ── Step 5: Hessian stabilisation L_H ───────────────────────────
            L_H = self.hsl(self.model, self.theta0, self.fisher)

            # ── Step 6: Shared-space loss L_S ────────────────────────────────
            forget_embs = self.model.get_embeddings(batch_f)
            retain_embs = self.model.get_embeddings(batch_r)
            L3_w = min(
                self.config.lambda3,
                (epoch / max(self.config.warmup_epochs, 1)) * self.config.lambda3,
            )
            L_S, _, _ = self.ssl(forget_embs, retain_embs, self.ref_covs)

            # ── Step 7: Total loss (Eq. 13) ──────────────────────────────────
            L_total = (
                L_BU
                + self.config.lambda1 * L_G
                + self.config.lambda2 * L_H
                + L3_w * L_S
            )

            # ── Step 8: Backward with g_perp replacing ∇L_BU ─────────────────
            self.optimizer.zero_grad()
            L_total.backward()

            # Override gradients with orthogonalised version
            for param, gp in zip(self.model.parameters(), g_perp):
                if param.grad is not None:
                    param.grad.data.copy_(gp)
                else:
                    param.grad = gp.clone()

            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss_sum += L_total.item()
            lb_sum += L_B.item()
            lg_sum += L_G.item()
            lh_sum += L_H.item()
            ls_sum += L_S.item()
            n_steps += 1

        n_steps = max(n_steps, 1)
        forget_acc = self._eval_acc(self.forget_loader)
        retain_acc = self._eval_acc(self.retain_loader)

        return {
            "forget_acc": forget_acc,
            "retain_acc": retain_acc,
            "L_total":    total_loss_sum / n_steps,
            "L_B":        lb_sum / n_steps,
            "L_G":        lg_sum / n_steps,
            "L_H":        lh_sum / n_steps,
            "L_S":        ls_sum / n_steps,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_acc(self, loader: DataLoader) -> float:
        # Stay in train() mode so cuDNN RNN backward remains valid.
        # Dropout noise is acceptable for a quick unlearning accuracy check.
        correct = total = 0
        for batch in loader:
            batch  = self._to_device(batch)
            labels = batch["label"]
            logits, _ = self.model(batch)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        return correct / max(total, 1)

    def _to_device(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _save_checkpoint(self, name: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "theta0":      self.theta0,
        }, self.ckpt_dir / name)
        logger.info(f"Unlearned model saved → {self.ckpt_dir / name}")

    def load_unlearned_model(self, name: str = None):
        if name is None: name = f"{self.dataset}_unlearned_model.pt"
        ckpt = torch.load(self.ckpt_dir / name, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(f"Unlearned model loaded from {self.ckpt_dir / name}")
