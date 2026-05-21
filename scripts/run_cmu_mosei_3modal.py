"""
run_cmu_mosei_3modal.py
Full MAF experiment on CMU-MOSEI 3-modal (text + audio + video).
This directly addresses Reviewer 1's request for text+video+audio evaluation.

Prerequisites:
    py setup_cmu_mosei_3modal.py
    copy cmu_mosei_3modal_dataset.py data\\cmu_mosei_3modal_dataset.py
    py patch_cmu_mosei_3modal.py

Usage:
    py run_cmu_mosei_3modal.py
"""

import multiprocessing
multiprocessing.freeze_support()

import json, logging, random, shutil, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("mosei_3m")
sys.path.insert(0, ".")

DATASET = "cmu-mosei-3modal"


def set_seed(seed=42):
    import torch, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    import torch
    from config import MAFConfig
    from data.datasets import MAFDataModule
    from models.maf_model import build_model, DATASET_CONFIGS
    from trainer import Trainer
    from unlearner import MAFUnlearner
    from evaluate import evaluate_all

    if not Path(f"data/{DATASET}/train.json").exists():
        logger.error("Data not found. Run: py setup_cmu_mosei_3modal.py")
        return

    train_n = len(json.load(open(f"data/{DATASET}/train.json")))
    logger.info(f"CMU-MOSEI 3-modal: {train_n} train samples")
    logger.info("Modalities: text (BERT) + audio (COVAREP proxy) + video (FACET42 facial)")
    logger.info("Forget class: 0 (negative sentiment)")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = MAFConfig(
        dataset=DATASET, device=str(device),
        seed=42, num_workers=0, forget_classes=[0], lambda_r=5.0,
    )
    Path(cfg.checkpoint_dir).mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    dm      = MAFDataModule(cfg)
    loaders = dm.get_loaders()

    s = next(iter(loaders["train"]))
    logger.info(f"Batch: text={s['input_ids'].shape} "
                f"audio={s['audio'].shape} video={s['video'].shape} "
                f"labels={s['label'][:5].tolist()}")
    logger.info(f"Forget set: {len(loaders['train_forget'].dataset)}  "
                f"Retain set: {len(loaders['train_retain'].dataset)}")

    # ── Phase 1: Training ─────────────────────────────────────────────
    logger.info("\n" + "="*55)
    logger.info("PHASE 1: Training (text + audio + video)")
    logger.info("="*55)
    model   = build_model(DATASET, cfg).to(device)
    trainer = Trainer(model, cfg, device)
    history = trainer.train(loaders["train"], loaders["val"])
    logger.info(f"Done | best val_acc={max(history['val_acc']):.4f}")

    auto  = Path(cfg.checkpoint_dir) / f"{DATASET}_best_model.pt"
    saved = Path(cfg.checkpoint_dir) / f"{DATASET}_seed42_best.pt"
    if auto.exists(): shutil.copy2(auto, saved)

    # ── Phase 2: Unlearning ───────────────────────────────────────────
    logger.info("\n" + "="*55)
    logger.info("PHASE 2: MAF Unlearning (forget negative sentiment)")
    logger.info("="*55)
    if saved.exists():
        model.load_state_dict(torch.load(saved, map_location=device)["model_state"])
    unlearner = MAFUnlearner(model, cfg, device)
    unlearner.setup(loaders["train_forget"], loaders["train_retain"])
    unlearner.unlearn()

    # ── Evaluate ─────────────────────────────────────────────────────
    logger.info("\nEvaluating MAF ...")
    mods = DATASET_CONFIGS[DATASET]["modalities"]
    maf = evaluate_all(
        unlearner.model,
        forget_loader=loaders["test_forget"],
        retain_loader=loaders["test_retain"],
        device=device, modalities=mods, mia_n_shadow=1,
    )
    logger.info(
        f"MAF 3-modal Results:\n"
        f"  forget_acc  = {maf['forget_acc']:.4f}\n"
        f"  retain_acc  = {maf['retain_acc']:.4f}\n"
        f"  trade_off   = {maf['trade_off']:.4f}\n"
        f"  mia_asr     = {maf['mia_asr']:.4f}\n"
        f"  ss_leakage  = {maf['ss_leakage']:.4f}"
    )
    json.dump(maf, open(f"results/{DATASET}_full_results.json","w"), indent=2)

    # ── Baselines ─────────────────────────────────────────────────────
    logger.info("\n" + "="*55)
    logger.info("Baselines on CMU-MOSEI 3-modal")
    logger.info("="*55)
    try:
        from baselines import BASELINES
        nc = DATASET_CONFIGS[DATASET]["num_classes"]

        def fresh():
            m = build_model(DATASET, cfg).to(device)
            m.load_state_dict(torch.load(saved, map_location=device)["model_state"])
            return m

        bl = {}
        for name, fn in BASELINES.items():
            logger.info(f"  {name} ...")
            try:
                m = fresh()
                if name == "random_relabeling":
                    m = fn(m, loaders["train_forget"], loaders["train_retain"], device, nc)
                elif name in ("fisher_forgetting","amnesiac"):
                    m = fn(m, loaders["train_forget"], device)
                elif name in ("boundary","fast_mu"):
                    m = fn(m, loaders["train_forget"], loaders["train_retain"], device, nc)
                elif name == "delete_and_refine":
                    m = fn(m, loaders["train_retain"], device)
                elif name == "finetune_dr":
                    m = fn(m, loaders["train_retain"])
                else: continue
                r = evaluate_all(m, loaders["test_forget"], loaders["test_retain"],
                                 device, mods, mia_n_shadow=1)
                bl[name] = r
                logger.info(f"    forget={r['forget_acc']:.4f}  "
                            f"retain={r['retain_acc']:.4f}  trade={r['trade_off']:.4f}")
            except Exception as e:
                logger.warning(f"    {name} FAILED: {e}")

        json.dump(bl, open(f"results/{DATASET}_baselines.json","w"), indent=2)
        logger.info(f"Baselines saved → results/{DATASET}_baselines.json")
    except ImportError:
        logger.warning("baselines.py not found")

    logger.info(f"\nDONE. Add to rebuttal: R1.4 now answered with text+audio+video.")
    logger.info(f"Results → results/{DATASET}_full_results.json")


if __name__ == "__main__":
    main()
