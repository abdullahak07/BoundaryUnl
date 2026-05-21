"""
Standard Trainer for initial (pre-unlearning) model training.
"""

import os
import time
import copy
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """
    Plain cross-entropy trainer for initial model training on the full dataset.
    """

    def __init__(self, model: nn.Module, config, device: torch.device):
        self.model  = model
        self.config = config
        self.device = device

        self.model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        self.criterion = nn.CrossEntropyLoss()
        self.dataset   = getattr(config, "dataset", "model")
        self.ckpt_dir  = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.best_state   = None

    # ── Training loop ────────────────────────────────────────────────────────

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        history = {"train_loss": [], "val_acc": []}

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_acc    = self._eval(val_loader)
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{self.config.epochs}  "
                f"loss={train_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.1f}s)"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state   = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(f"{self.dataset}_best_model.pt")

        # Restore best
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            batch  = self._to_device(batch)
            labels = batch["label"]

            self.optimizer.zero_grad()
            logits, _ = self.model(batch)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _eval(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        for batch in loader:
            batch  = self._to_device(batch)
            labels = batch["label"]
            logits, _ = self.model(batch)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        return correct / max(total, 1)

    # ── Utilities ────────────────────────────────────────────────────────────

    def _to_device(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _save_checkpoint(self, name: str):
        path = self.ckpt_dir / name
        torch.save({
            "model_state": self.model.state_dict(),
            "best_val_acc": self.best_val_acc,
        }, path)
        logger.debug(f"Checkpoint saved → {path}")

    def load_checkpoint(self, name: str = None):
        if name is None: name = f"{self.dataset}_best_model.pt"
        path = self.ckpt_dir / name
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        logger.info(f"Loaded checkpoint from {path} (val_acc={self.best_val_acc:.4f})")
