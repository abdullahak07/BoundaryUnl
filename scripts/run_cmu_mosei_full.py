"""
run_cmu_mosei_full.py
Full MAF experiment on CMU-MOSEI: train + unlearn + evaluate + baselines.

Prerequisites (already done):
    py setup_cmu_mosei.py       ✓ data in data/cmu-mosei/
    py patch_for_cmu_mosei.py   ✓ datasets.py + maf_model.py patched

Fix syntax error in datasets.py first:
    python -c "p=open('data/datasets.py',encoding='utf-8').read(); open('data/datasets.py','w',encoding='utf-8').write(p.replace('elif self.dataset_name == \"meld\"', 'elif self.dataset_name == \"meld\":'))"

Then:
    py run_cmu_mosei_full.py
"""

import multiprocessing
multiprocessing.freeze_support()

import json, logging, random, shutil, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("cmu_mosei")
sys.path.insert(0, ".")

DATASET = "cmu-mosei"


def set_seed(seed=42):
    import torch, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def verify_datasets_syntax():
    """Check datasets.py has the colon after meld elif — auto-fix if not."""
    p = Path("data/datasets.py")
    src = p.read_text(encoding="utf-8")
    broken = 'elif self.dataset_name == "meld"\n'
    if broken in src:
        p.write_text(src.replace(broken, 'elif self.dataset_name == "meld":\n'),
                     encoding="utf-8")
        logger.info("  Auto-fixed: missing colon in data/datasets.py")


def main():
    import torch

    # Auto-fix common patch issue
    verify_datasets_syntax()

    from config import MAFConfig
    from data.datasets import MAFDataModule
    from models.maf_model import build_model, DATASET_CONFIGS
    from trainer import Trainer
    from unlearner import MAFUnlearner
    from evaluate import evaluate_all

    if not Path(f"data/{DATASET}/train.json").exists():
        logger.error(f"Data not found at data/{DATASET}/. Run: py setup_cmu_mosei.py")
        return

    train_count = len(json.load(open(f"data/{DATASET}/train.json")))
    logger.info(f"CMU-MOSEI: {train_count} train samples")
    logger.info("Forget class: 0 (negative sentiment)")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = MAFConfig(
        dataset=DATASET,
        device=str(device),
        seed=42,
        num_workers=0,
        forget_classes=[0],
        epochs=30,
        lambda_r=5.0,
    )
    Path(cfg.checkpoint_dir).mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    dm      = MAFDataModule(cfg)
    loaders = dm.get_loaders()

    # Sanity check
    s = next(iter(loaders["train"]))
    logger.info(f"Batch: audio={s['audio'].shape} text={s['input_ids'].shape} "
                f"labels={s['label'][:5].tolist()}")
    logger.info(f"Forget set: {len(loaders['train_forget'].dataset)} "
                f"Retain set: {len(loaders['train_retain'].dataset)}")

    # ── Phase 1: Training ─────────────────────────────────────────────────
    logger.info("\n" + "="*55)
    logger.info("PHASE 1: Training CMU-MOSEI base model")
    logger.info("="*55)
    model   = build_model(DATASET, cfg).to(device)
    trainer = Trainer(model, cfg, device)
    history = trainer.train(loaders["train"], loaders["val"])
    best_acc = max(history["val_acc"]) if history["val_acc"] else 0
    logger.info(f"Training done | best val_acc={best_acc:.4f}")

    auto  = Path(cfg.checkpoint_dir) / f"{DATASET}_best_model.pt"
    saved = Path(cfg.checkpoint_dir) / f"{DATASET}_seed42_best_model.pt"
    if auto.exists():
        shutil.copy2(auto, saved)

    # ── Phase 2: Unlearning ───────────────────────────────────────────────
    logger.info("\n" + "="*55)
    logger.info("PHASE 2: MAF Unlearning (forget negative sentiment)")
    logger.info("="*55)
    if saved.exists():
        model.load_state_dict(
            torch.load(saved, map_location=device)["model_state"]
        )
    unlearner = MAFUnlearner(model, cfg, device)
    unlearner.setup(loaders["train_forget"], loaders["train_retain"])
    unlearner.unlearn()

    # ── Evaluate MAF ─────────────────────────────────────────────────────
    logger.info("\nEvaluating MAF ...")
    mods = DATASET_CONFIGS[DATASET]["modalities"]
    maf  = evaluate_all(
        unlearner.model,
        forget_loader=loaders["test_forget"],
        retain_loader=loaders["test_retain"],
        device=device,
        modalities=mods,
        mia_n_shadow=1,
    )
    logger.info(
        f"MAF Results:\n"
        f"  forget_acc  = {maf['forget_acc']:.4f}\n"
        f"  retain_acc  = {maf['retain_acc']:.4f}\n"
        f"  trade_off   = {maf['trade_off']:.4f}\n"
        f"  mia_asr     = {maf['mia_asr']:.4f}\n"
        f"  ss_leakage  = {maf['ss_leakage']:.4f}"
    )
    json.dump(maf, open(f"results/{DATASET}_full_results.json","w"), indent=2)

    # ── Baselines ─────────────────────────────────────────────────────────
    logger.info("\n" + "="*55)
    logger.info(f"Baselines on {DATASET.upper()}")
    logger.info("="*55)

    try:
        from baselines import BASELINES
        nc = DATASET_CONFIGS[DATASET]["num_classes"]

        def fresh():
            m = build_model(DATASET, cfg).to(device)
            m.load_state_dict(torch.load(saved, map_location=device)["model_state"])
            return m

        baseline_results = {}
        for name, fn in BASELINES.items():
            logger.info(f"  {name} ...")
            try:
                m = fresh()
                if name == "random_relabeling":
                    m = fn(m, loaders["train_forget"], loaders["train_retain"], device, nc)
                elif name in ("fisher_forgetting", "amnesiac"):
                    m = fn(m, loaders["train_forget"], device)
                elif name in ("boundary", "fast_mu"):
                    m = fn(m, loaders["train_forget"], loaders["train_retain"], device, nc)
                elif name == "delete_and_refine":
                    m = fn(m, loaders["train_retain"], device)
                elif name == "finetune_dr":
                    m = fn(m, loaders["train_retain"])
                else:
                    continue
                r = evaluate_all(m, loaders["test_forget"], loaders["test_retain"],
                                 device, mods, mia_n_shadow=1)
                baseline_results[name] = r
                logger.info(f"    forget={r['forget_acc']:.4f}  "
                            f"retain={r['retain_acc']:.4f}  "
                            f"trade={r['trade_off']:.4f}")
            except Exception as e:
                logger.warning(f"    {name} FAILED: {e}")

        json.dump(baseline_results,
                  open(f"results/{DATASET}_baselines.json","w"), indent=2)
        logger.info(f"Baselines saved → results/{DATASET}_baselines.json")

    except ImportError:
        logger.warning("baselines.py not found — skipping baselines")

    logger.info(f"\nDone. Results → results/{DATASET}_full_results.json")


if __name__ == "__main__":
    main()