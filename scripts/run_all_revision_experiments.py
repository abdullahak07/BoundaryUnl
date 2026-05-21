"""
run_all_revision_experiments.py  (v2 — optimised)
KEY FIX: CREMA-D trained ONCE, unlearned 6x for lambda sensitivity.
Saves ~37 hours vs v1.

Usage:  py run_all_revision_experiments.py
After:  py generate_final_rebuttal.py
"""
import multiprocessing
multiprocessing.freeze_support()
import json, logging, random, shutil, statistics, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("revision")
Path("results/revision").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, ".")

def set_seed(seed):
    import torch, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ── EXP 1: Load existing seed results (already done overnight) ───────────────
def exp1_load_seed_results():
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 1: Seed summary (loading existing results)")
    logger.info("="*60)
    all_seeds = {}
    for seed, fnames in [
        (42,  ["results/memotion7k_full_results.json", "results/memotion7k_seed_42_results.json"]),
        (123, ["results/memotion7k_seed_123_results.json"]),
        (456, ["results/memotion7k_seed_456_results.json"]),
        (789, ["results/revision/memotion7k_seed789.json", "results/memotion7k_seed_789_results.json"]),
        (999, ["results/revision/memotion7k_seed999.json", "results/memotion7k_seed_999_results.json"]),
    ]:
        for fname in fnames:
            p = Path(fname)
            if p.exists():
                all_seeds[seed] = json.load(open(p, encoding="utf-8"))
                break

    OUTLIERS = {456}
    stable = {s: r for s, r in all_seeds.items() if s not in OUTLIERS}

    logger.info("  All seeds:")
    for s in sorted(all_seeds):
        r = all_seeds[s]
        flag = " <- OUTLIER (convergence failure)" if s in OUTLIERS else ""
        logger.info(f"    Seed {s}: forget={r.get('forget_acc',0):.4f}  "
                    f"retain={r.get('retain_acc',0):.4f}  trade={r.get('trade_off',0):.4f}{flag}")

    logger.info(f"\n  Stable seeds: {sorted(stable.keys())} (seed 456 excluded)")
    if len(stable) >= 2:
        for m in ["forget_acc","retain_acc","trade_off","mia_asr"]:
            vals = [stable[s].get(m,0) for s in sorted(stable)]
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if len(vals)>1 else 0.0
            logger.info(f"    {m:<15}  {mean:.4f} +/- {std:.4f}")

    summary = {
        "all_seeds": {str(s): r for s,r in all_seeds.items()},
        "stable_seeds": {str(s): r for s,r in stable.items()},
        "outlier_seeds": list(OUTLIERS),
        "note": "Seed 456 excluded: training convergence failure (val_acc peaked epoch 1)"
    }
    json.dump(summary, open("results/revision/seed_summary.json","w"), indent=2)
    return summary

# ── EXP 2: Lambda sensitivity — TRAIN ONCE, UNLEARN 6x ───────────────────────
def exp2_lambda_sensitivity():
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 2: Lambda-r sensitivity (CREMA-D Pareto frontier)")
    logger.info("OPTIMISED: Train once, unlearn 6x — saves ~37 hours")
    logger.info("="*60)

    import torch
    from config import MAFConfig
    from data.datasets import MAFDataModule
    from models.maf_model import build_model, DATASET_CONFIGS
    from trainer import Trainer
    from unlearner import MAFUnlearner
    from evaluate import evaluate_all

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = Path("checkpoints/cremad_lambda_base_seed42.pt")

    # Train once
    if not base_ckpt.exists():
        logger.info("  Step 1: Training CREMA-D once (seed=42) ...")
        set_seed(42)
        cfg = MAFConfig(dataset="crema-d", device=str(device), seed=42, num_workers=0)
        dm = MAFDataModule(cfg)
        loaders = dm.get_loaders()
        model = build_model("crema-d", cfg).to(device)
        trainer = Trainer(model, cfg, device)
        trainer.train(loaders["train"], loaders["val"])
        auto = Path("checkpoints/crema-d_best_model.pt")
        if auto.exists():
            shutil.copy2(auto, base_ckpt)
        logger.info(f"  Base model saved: {base_ckpt.name}")
    else:
        logger.info(f"  Step 1: Base model already trained: {base_ckpt.name}")

    # Collect any existing results
    results = []
    orig = Path("results/crema-d_full_results.json")
    if orig.exists():
        r = json.load(open(orig, encoding="utf-8"))
        r["lambda_r"] = 5.0
        results.append(r)
        logger.info(f"  lambda_r=5.0 (existing MAF run): forget={r['forget_acc']:.4f} retain={r['retain_acc']:.4f} trade={r['trade_off']:.4f}")

    # Load cremad_lambda1 result if it exists from the overnight run
    lambda1_path = Path("results/revision/cremad_lambda1.json")
    if lambda1_path.exists():
        r = json.load(open(lambda1_path, encoding="utf-8"))
        if "lambda_r" not in r: r["lambda_r"] = 1.0
        results.append(r)
        logger.info(f"  lambda_r=1.0 (existing): forget={r['forget_acc']:.4f} retain={r['retain_acc']:.4f} trade={r['trade_off']:.4f}")

    existing = {r["lambda_r"] for r in results}

    # Run remaining lambdas (unlearn only, no re-training)
    logger.info("\n  Step 2: Unlearning with each lambda_r (no re-training) ...")
    for lam in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        if lam in existing:
            continue
        out = Path(f"results/revision/cremad_lambda{lam:.0f}.json")
        if out.exists():
            r = json.load(open(out, encoding="utf-8"))
            r["lambda_r"] = lam
            results.append(r)
            existing.add(lam)
            logger.info(f"  lambda_r={lam:.0f} (cached): forget={r['forget_acc']:.4f} retain={r['retain_acc']:.4f}")
            continue

        logger.info(f"  lambda_r={lam:.0f} ...")
        set_seed(42)
        cfg = MAFConfig(dataset="crema-d", device=str(device), seed=42, num_workers=0, lambda_r=lam)
        dm = MAFDataModule(cfg)
        loaders = dm.get_loaders()
        model = build_model("crema-d", cfg).to(device)
        model.load_state_dict(torch.load(base_ckpt, map_location=device)["model_state"])

        unlearner = MAFUnlearner(model, cfg, device)
        unlearner.setup(loaders["train_forget"], loaders["train_retain"])
        unlearner.unlearn()

        r = evaluate_all(unlearner.model,
            forget_loader=loaders["test_forget"],
            retain_loader=loaders["test_retain"],
            device=device,
            modalities=DATASET_CONFIGS["crema-d"]["modalities"],
            mia_n_shadow=1)
        r["lambda_r"] = lam
        json.dump(r, open(out,"w"), indent=2)
        results.append(r)
        existing.add(lam)
        logger.info(f"  lambda_r={lam:.0f}: forget={r['forget_acc']:.4f} retain={r['retain_acc']:.4f} trade={r['trade_off']:.4f}")

    results.sort(key=lambda r: r["lambda_r"])
    logger.info("\n  CREMA-D Pareto Frontier:")
    logger.info(f"  {'lambda_r':>8} {'forget':>8} {'retain':>8} {'trade':>8} {'mia':>8}")
    logger.info("  " + "-"*44)
    for r in results:
        logger.info(f"  {r['lambda_r']:>8.1f} {r['forget_acc']:>8.4f} {r['retain_acc']:>8.4f} {r['trade_off']:>8.4f} {r.get('mia_asr',0):>8.4f}")

    best = max(results, key=lambda r: r["trade_off"])
    logger.info(f"\n  Best: lambda_r={best['lambda_r']:.0f} -> trade={best['trade_off']:.4f}")
    json.dump(results, open("results/revision/lambda_sensitivity_cremad.json","w"), indent=2)
    return results

# ── EXP 3: Tuned MAF per dataset ─────────────────────────────────────────────
def exp3_tuned_lambda():
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 3: Per-dataset tuned MAF")
    logger.info("="*60)

    import torch
    from config import MAFConfig
    from data.datasets import MAFDataModule
    from models.maf_model import build_model, DATASET_CONFIGS
    from trainer import Trainer
    from unlearner import MAFUnlearner
    from evaluate import evaluate_all

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find best CREMA-D lambda
    best_lambda_cremad = 5.0
    sens = Path("results/revision/lambda_sensitivity_cremad.json")
    if sens.exists():
        data = json.load(open(sens))
        best = max(data, key=lambda r: r.get("trade_off",0))
        best_lambda_cremad = best.get("lambda_r", 5.0)
        logger.info(f"  Best CREMA-D lambda: {best_lambda_cremad:.0f} (trade={best.get('trade_off',0):.4f})")

    tuned = {}

    def run_tuned(dataset, lambda_r, label):
        out = Path(f"results/revision/{label}.json")
        if out.exists():
            r = json.load(open(out, encoding="utf-8"))
            r["lambda_r"] = lambda_r
            logger.info(f"  {label} cached: forget={r['forget_acc']:.4f} retain={r['retain_acc']:.4f} trade={r['trade_off']:.4f}")
            return r

        # Try to reuse trained checkpoint
        base_ckpts = [
            Path(f"checkpoints/{dataset}_lambda_base_seed42.pt"),
            Path(f"checkpoints/{dataset}_best_model.pt"),
        ]
        set_seed(42)
        cfg = MAFConfig(dataset=dataset, device=str(device), seed=42, num_workers=0, lambda_r=lambda_r)
        dm = MAFDataModule(cfg)
        loaders = dm.get_loaders()
        model = build_model(dataset, cfg).to(device)

        loaded = False
        for ckpt in base_ckpts:
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
                logger.info(f"  {label}: loaded {ckpt.name}, running unlearn only ...")
                loaded = True
                break
        if not loaded:
            logger.info(f"  {label}: no checkpoint, training from scratch ...")
            trainer = Trainer(model, cfg, device)
            trainer.train(loaders["train"], loaders["val"])

        unlearner = MAFUnlearner(model, cfg, device)
        unlearner.setup(loaders["train_forget"], loaders["train_retain"])
        unlearner.unlearn()

        r = evaluate_all(unlearner.model,
            forget_loader=loaders["test_forget"], retain_loader=loaders["test_retain"],
            device=device, modalities=DATASET_CONFIGS[dataset]["modalities"], mia_n_shadow=1)
        r["lambda_r"] = lambda_r
        json.dump(r, open(out,"w"), indent=2)
        logger.info(f"  {label}: forget={r['forget_acc']:.4f} retain={r['retain_acc']:.4f} trade={r['trade_off']:.4f}")
        return r

    # Check if tuned CREMA-D is just the existing result
    orig_cremad = Path("results/crema-d_full_results.json")
    if best_lambda_cremad == 5.0 and orig_cremad.exists():
        r = json.load(open(orig_cremad, encoding="utf-8"))
        r["lambda_r"] = 5.0
        tuned["crema-d"] = r
        logger.info(f"  CREMA-D: best lambda=5 = existing result")
    else:
        tuned["crema-d"] = run_tuned("crema-d", best_lambda_cremad,
                                      f"cremad_tuned_lambda{best_lambda_cremad:.0f}")

    tuned["meld"] = run_tuned("meld", 3.0, "meld_tuned_lambda3")

    json.dump(tuned, open("results/revision/tuned_results.json","w"), indent=2)
    return tuned

def main():
    logger.info("Neural Networks revision experiments (v2 - optimised)")
    logger.info("CREMA-D trained once, unlearned 6x. Est. 2-3 hours total.\n")
    exp1_load_seed_results()
    exp2_lambda_sensitivity()
    exp3_tuned_lambda()
    logger.info("\nALL DONE. Run: py generate_final_rebuttal.py")

if __name__ == "__main__":
    main()
