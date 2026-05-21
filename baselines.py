"""
Baseline unlearning methods compared in the MAF paper.

• Retrain from scratch (oracle)
• Fine-tuning on Dr (delete-and-refine)
• Random Relabeling
• Fisher Forgetting
• Amnesiac Unlearning
• Boundary Unlearning (single-module)
• Fast Yet Effective MU (Tarun et al., 2023)
"""

import copy
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Retrain from scratch  (Oracle)
# ─────────────────────────────────────────────────────────────────────────────

def retrain_oracle(
    model_factory,       # callable() → fresh MAFModel
    retain_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-4,
) -> nn.Module:
    """Full retraining on Dr only (oracle upper bound)."""
    model = model_factory().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    ce    = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for batch in retain_loader:
            batch = _to_device(batch, device)
            opt.zero_grad()
            logits, _ = model(batch)
            ce(logits, batch["label"]).backward()
            opt.step()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fine-tuning on Dr  (Delete-and-Refine)
# ─────────────────────────────────────────────────────────────────────────────

def delete_and_refine(
    model: nn.Module,
    retain_loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-4,
) -> nn.Module:
    """Brief fine-tuning on Dr after data removal."""
    model = copy.deepcopy(model).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    ce    = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch in retain_loader:
            batch = _to_device(batch, device)
            opt.zero_grad()
            logits, _ = model(batch)
            ce(logits, batch["label"]).backward()
            opt.step()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Relabeling
# ─────────────────────────────────────────────────────────────────────────────

def random_relabeling(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int = 5,
    lr: float = 1e-4,
) -> nn.Module:
    """Assign random labels to forget samples and continue training."""
    model = copy.deepcopy(model).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    ce    = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch_f, batch_r in zip(forget_loader, retain_loader):
            batch_f = _to_device(batch_f, device)
            batch_r = _to_device(batch_r, device)
            # Random labels for forget batch
            rand_labels = torch.randint(0, num_classes, batch_f["label"].shape).to(device)

            opt.zero_grad()
            lf, _ = model(batch_f)
            lr_,_ = model(batch_r)
            loss  = ce(lf, rand_labels) + ce(lr_, batch_r["label"])
            loss.backward()
            opt.step()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fisher Forgetting  (Golatkar et al., 2020)
# ─────────────────────────────────────────────────────────────────────────────

def fisher_forgetting(
    model: nn.Module,
    forget_loader: DataLoader,
    device: torch.device,
    noise_scale: float = 1e-3,
    n_batches: int = 20,
) -> nn.Module:
    """
    Scrub parameters by adding noise proportional to inverse Fisher.
    Approximates scrubbing the forget data's influence.
    """
    model = copy.deepcopy(model).to(device)
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
              if p.requires_grad}

    model.eval()
    model.train()  # cuDNN needs train mode
    count = 0
    for batch in forget_loader:
        if count >= n_batches:
            break
        batch = _to_device(batch, device)
        logits, _ = model(batch)
        log_p  = F.log_softmax(logits, dim=-1)
        pred   = logits.argmax(1)
        loss   = F.nll_loss(log_p, pred)
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)
        count += 1

    for name, param in model.named_parameters():
        if name in fisher:
            fi = fisher[name] / max(count, 1)
            noise = torch.randn_like(param) * noise_scale / (fi.sqrt() + 1e-6)
            param.data.add_(noise)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. Amnesiac Unlearning  (Graves et al., 2021)
# ─────────────────────────────────────────────────────────────────────────────

def amnesiac_unlearning(
    model: nn.Module,
    forget_loader: DataLoader,
    device: torch.device,
    gradient_scale: float = 1.0,
    n_batches: int = 20,
) -> nn.Module:
    """
    Gradient subtraction: reverse the update that would have been applied
    if forget data had been included in training.
    """
    model = copy.deepcopy(model).to(device)
    opt   = optim.SGD(model.parameters(), lr=gradient_scale * 1e-4)
    ce    = nn.CrossEntropyLoss()
    model.train()
    count = 0
    for batch in forget_loader:
        if count >= n_batches:
            break
        batch = _to_device(batch, device)
        # Compute gradient on forget data
        logits, _ = model(batch)
        loss = ce(logits, batch["label"])
        # Subtract gradient (gradient ascent on forget loss)
        opt.zero_grad()
        (-loss).backward()
        opt.step()
        count += 1
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. Boundary Unlearning (single-module, no gradient/Hessian/SS)
# ─────────────────────────────────────────────────────────────────────────────

def boundary_unlearning(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int = 10,
    lr: float = 1e-4,
    lambda_r: float = 1.0,
) -> nn.Module:
    """Decision-space boundary erasure (LBU only, no other components)."""
    from losses import BoundaryUnlearningLoss
    model = copy.deepcopy(model).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    bul   = BoundaryUnlearningLoss(num_classes, lambda_r)
    retain_iter = iter(retain_loader)
    model.train()
    for epoch in range(epochs):
        for batch_f in forget_loader:
            try:
                batch_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                batch_r = next(retain_iter)
            batch_f = _to_device(batch_f, device)
            batch_r = _to_device(batch_r, device)
            lf, _ = model(batch_f)
            lr_, _ = model(batch_r)
            L_BU, _, _ = bul(lf, lr_, batch_r["label"])
            opt.zero_grad(); L_BU.backward(); opt.step()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fast Yet Effective MU  (Tarun et al., 2023)
# ─────────────────────────────────────────────────────────────────────────────

def fast_yet_effective_mu(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    noise_std: float = 0.1,
    repair_epochs: int = 3,
    lr: float = 1e-4,
) -> nn.Module:
    """
    Impair–Repair mechanism:
    Impair : add error-maximizing noise to forget samples
    Repair : fine-tune on retain data to restore retained performance
    Adapted to multimodal: noise applied in shared embedding space.
    """
    model = copy.deepcopy(model).to(device)
    ce    = nn.CrossEntropyLoss()

    # ── Impair step: gradient ascent with noise on forget set ─────────────
    opt = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for batch_f in forget_loader:
        batch_f = _to_device(batch_f, device)
        logits, _ = model(batch_f)
        # Error-maximising: ascend on CE loss
        loss = -ce(logits, batch_f["label"])
        opt.zero_grad(); loss.backward(); opt.step()
        # Add parameter noise proportional to noise_std
        with torch.no_grad():
            for p in model.parameters():
                p.data.add_(torch.randn_like(p) * noise_std)

    # ── Repair step: fine-tune on retain set ─────────────────────────────
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(repair_epochs):
        for batch_r in retain_loader:
            batch_r = _to_device(batch_r, device)
            logits, _ = model(batch_r)
            loss = ce(logits, batch_r["label"])
            opt.zero_grad(); loss.backward(); opt.step()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Registry
# ─────────────────────────────────────────────────────────────────────────────

BASELINES = {
    "random_relabeling":    random_relabeling,
    "fisher_forgetting":    fisher_forgetting,
    "amnesiac":             amnesiac_unlearning,
    "boundary":             boundary_unlearning,
    "fast_mu":              fast_yet_effective_mu,
    "delete_and_refine":    delete_and_refine,
}
