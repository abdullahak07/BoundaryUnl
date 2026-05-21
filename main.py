"""
main.py – MAF Experiment Runner

Usage examples:
    # Full experiment on Memotion7k
    python main.py --dataset memotion7k --mode full

    # Unlearning only (load pretrained model)
    python main.py --dataset crema-d --mode unlearn --load_pretrained

    # Evaluate a saved unlearned model
    python main.py --dataset meld --mode eval

    # Run all baselines comparison
    python main.py --dataset memotion7k --mode baselines
"""

import argparse
import copy
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import MAFConfig
from data.datasets import MAFDataModule
from models.maf_model import build_model, DATASET_CONFIGS
from trainer import Trainer
from unlearner import MAFUnlearner
from evaluate import evaluate_all, print_results_table
from baselines import BASELINES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("maf.main")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(config: MAFConfig) -> torch.device:
    if config.device == "cuda" and torch.cuda.is_available():
        d = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device("cpu")
        logger.info("Using CPU")
    return d


def aggregate_seed_results(all_results):
    """Average results across multiple seeds."""
    keys = all_results[0].keys()
    agg  = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        agg[k] = {"mean": np.mean(vals), "std": np.std(vals)}
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Core workflow functions
# ─────────────────────────────────────────────────────────────────────────────

def run_training(config: MAFConfig, device: torch.device, loaders: dict) -> torch.nn.Module:
    """Train the base model on the full dataset."""
    logger.info("=" * 60)
    logger.info(f"PHASE 1: Initial Training  ({config.dataset})")
    logger.info("=" * 60)

    model   = build_model(config.dataset, config)
    trainer = Trainer(model, config, device)

    t0 = time.time()
    history = trainer.train(loaders["train"], loaders["val"])
    elapsed = time.time() - t0

    logger.info(f"Training complete in {elapsed:.1f}s  "
                f"| best val_acc={trainer.best_val_acc:.4f}")
    return model


def run_unlearning(
    config: MAFConfig,
    device: torch.device,
    model: torch.nn.Module,
    loaders: dict,
) -> torch.nn.Module:
    """Run MAF unlearning on a pretrained model."""
    logger.info("=" * 60)
    logger.info(f"PHASE 2: MAF Unlearning  ({config.dataset})")
    logger.info("=" * 60)

    unlearner = MAFUnlearner(model, config, device)
    unlearner.setup(loaders["train_forget"], loaders["train_retain"])

    t0 = time.time()
    history = unlearner.unlearn()
    elapsed = time.time() - t0

    logger.info(f"Unlearning complete in {elapsed:.1f}s")
    return model


def run_evaluation(
    config: MAFConfig,
    device: torch.device,
    model: torch.nn.Module,
    loaders: dict,
    label: str = "MAF",
    retrain_model=None,
) -> dict:
    """Evaluate a model with all MAF metrics."""
    logger.info(f"Evaluating: {label}")
    modalities = DATASET_CONFIGS[config.dataset]["modalities"]
    results = evaluate_all(
        model,
        forget_loader=loaders["test_forget"],
        retain_loader=loaders["test_retain"],
        device=device,
        modalities=modalities,
        retrain_model=retrain_model,
        mia_n_shadow=config.n_shadow_models,
        mia_top_k=config.mia_top_k,
    )
    print_results_table(results, dataset=f"[{config.dataset} | {label}]")
    return results


def run_baselines(
    config: MAFConfig,
    device: torch.device,
    pretrained_model: torch.nn.Module,
    loaders: dict,
) -> dict:
    """Run all baseline methods and compare."""
    logger.info("=" * 60)
    logger.info("PHASE: Baselines Comparison")
    logger.info("=" * 60)

    num_classes = DATASET_CONFIGS[config.dataset]["num_classes"]
    modalities  = DATASET_CONFIGS[config.dataset]["modalities"]
    all_results = {}

    for name, fn in BASELINES.items():
        logger.info(f"Running baseline: {name}")
        try:
            # Each baseline function has slightly different signatures
            if name == "random_relabeling":
                m = fn(copy.deepcopy(pretrained_model),
                       loaders["train_forget"], loaders["train_retain"],
                       device, num_classes)
            elif name in ("fisher_forgetting", "amnesiac"):
                m = fn(copy.deepcopy(pretrained_model),
                       loaders["train_forget"], device)
            elif name in ("boundary", "fast_mu"):
                m = fn(copy.deepcopy(pretrained_model),
                       loaders["train_forget"], loaders["train_retain"],
                       device, num_classes)
            elif name == "delete_and_refine":
                m = fn(copy.deepcopy(pretrained_model),
                       loaders["train_retain"], device)
            else:
                continue

            results = evaluate_all(
                m, loaders["test_forget"], loaders["test_retain"],
                device, modalities
            )
            all_results[name] = results
            print_results_table(results, dataset=f"[{name}]")

        except Exception as e:
            logger.warning(f"Baseline {name} failed: {e}")

    return all_results


def run_multi_seed(config: MAFConfig, device: torch.device, mode: str = "full") -> dict:
    """Run full experiment across multiple seeds and report mean±std."""
    all_maf_results = []

    for seed in config.random_seeds:
        logger.info(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        set_seed(seed)

        # Reload data and model each seed
        data_module = MAFDataModule(config)
        loaders     = data_module.get_loaders()

        if mode == "full":
            model = run_training(config, device, loaders)
            model = run_unlearning(config, device, model, loaders)
        else:
            model = build_model(config.dataset, config)
            ckpt  = Path(config.checkpoint_dir) / f"{config.dataset}_unlearned_model.pt"
            if ckpt.exists():
                ckpt_data = torch.load(ckpt, map_location=device)
                model.load_state_dict(ckpt_data["model_state"])
                model.to(device)

        results = run_evaluation(config, device, model, loaders, label=f"MAF (seed={seed})")
        all_maf_results.append(results)

    # Aggregate
    agg = aggregate_seed_results(all_maf_results)
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATED RESULTS (mean ± std)")
    logger.info("=" * 60)
    for k, v in agg.items():
        arrow = "↓" if k in ("forget_acc", "mia_asr", "ss_leakage") else "↑"
        logger.info(f"  {k:<25s} {arrow}  {v['mean']:.4f} ± {v['std']:.4f}")

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MAF Multimodal Unlearning Framework")
    p.add_argument("--dataset", default="memotion7k",
                   choices=["memotion7k", "crema-d", "meld"])
    p.add_argument("--mode", default="full",
                   choices=["full", "train", "unlearn", "eval", "baselines", "multi_seed"],
                   help="Execution mode")
    p.add_argument("--data_root",   default="./data")
    p.add_argument("--ckpt_dir",    default="./checkpoints")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--unlearn_epochs", type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--audio_encoder", default="crnn",
                   choices=["cnn", "bigru", "crnn", "resnet18"])
    p.add_argument("--forget_classes", nargs="+", type=int, default=[0])
    return p.parse_args()


def main():
    args = parse_args()

    # Build config from CLI args
    config = MAFConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        checkpoint_dir=args.ckpt_dir,
        results_dir=args.results_dir,
        device=args.device,
        epochs=args.epochs,
        unlearn_epochs=args.unlearn_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        audio_encoder=args.audio_encoder,
        forget_classes=args.forget_classes,
    )

    set_seed(config.seed)
    device      = get_device(config)
    data_module = MAFDataModule(config)
    loaders     = data_module.get_loaders()

    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "multi_seed":
        results = run_multi_seed(config, device)
    elif args.mode == "full":
        model = run_training(config, device, loaders)
        model = run_unlearning(config, device, model, loaders)
        results = run_evaluation(config, device, model, loaders)
    elif args.mode == "train":
        run_training(config, device, loaders)
        return
    elif args.mode == "unlearn":
        model = build_model(config.dataset, config).to(device)
        ckpt  = Path(config.checkpoint_dir) / f"{config.dataset}_best_model.pt"
        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location=device)["model_state"]
            )
        model = run_unlearning(config, device, model, loaders)
        results = run_evaluation(config, device, model, loaders)
    elif args.mode == "eval":
        model = build_model(config.dataset, config).to(device)
        ckpt  = Path(config.checkpoint_dir) / "unlearned_model.pt"
        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location=device)["model_state"]
            )
        results = run_evaluation(config, device, model, loaders)
    elif args.mode == "baselines":
        model = build_model(config.dataset, config).to(device)
        ckpt  = Path(config.checkpoint_dir) / "best_model.pt"
        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location=device)["model_state"]
            )
        results = run_baselines(config, device, model, loaders)

    # Save results
    out_path = Path(config.results_dir) / f"{config.dataset}_{args.mode}_results.json"
    with open(out_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (float, int)) else str(v)
                   for k, v in results.items()}, f, indent=2)
    logger.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
