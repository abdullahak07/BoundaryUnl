"""
run_cmu_mosei_mia_all_baselines.py

Runs full MIA for every CMU-MOSEI baseline.

Fix in this version:
- Auto-detects the audio encoder required by the saved checkpoint.
- Your checkpoint contains audio_encoder.cnn / audio_encoder.gru / audio_encoder.proj,
  so the script must build the CRNN-style audio encoder, not audio_encoder.net.
- Does NOT use strict=False, because that would make the results invalid.

Usage:
    py run_cmu_mosei_mia_all_baselines.py

Output:
    results/cmu_mosei_mia_all_baselines.csv
    results/cmu_mosei_mia_all_baselines.json
"""

import multiprocessing
multiprocessing.freeze_support()

import csv
import json
import logging
import random
import sys
import warnings
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mosei_mia")

sys.path.insert(0, ".")

DATASET = "cmu-mosei"
CKPT_BASE = Path("checkpoints") / "cmu-mosei_seed42_best_model.pt"
MAF_CKPT = Path("checkpoints") / "cmu-mosei_unlearned_model.pt"
OUT_DIR = Path("results")

WORKING_AUDIO_ENCODER = None


def set_seed(seed=42):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_checkpoint_state(path: Path, device):
    import torch

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run the CMU-MOSEI experiment first, or check the checkpoint path."
        )

    obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]

    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]

    if isinstance(obj, dict):
        return obj

    raise RuntimeError(f"Unsupported checkpoint format: {type(obj)}")


def infer_audio_encoder_from_state(state_dict):
    """
    Infer likely audio encoder type from checkpoint keys.
    Your failing log shows:
        audio_encoder.cnn.*
        audio_encoder.gru.*
        audio_encoder.proj.*
    That is CRNN-style, not audio_encoder.net.
    """
    keys = list(state_dict.keys())
    key_blob = "\n".join(keys)

    if "audio_encoder.cnn." in key_blob and "audio_encoder.gru." in key_blob:
        return "crnn"

    if "audio_encoder.gru." in key_blob:
        return "gru"

    if "audio_encoder.cnn." in key_blob:
        return "cnn"

    if "audio_encoder.net." in key_blob:
        # This is usually feature/proxy MLP/COVAREP-style.
        return "covarep"

    return None


def make_cfg(audio_encoder=None):
    from config import MAFConfig

    cfg = MAFConfig(
        dataset=DATASET,
        device="cuda",
        seed=42,
        num_workers=0,
        forget_classes=[0],
        lambda_r=5.0,
    )

    if audio_encoder is not None:
        # Some config classes accept dynamic attributes, some already have it.
        setattr(cfg, "audio_encoder", audio_encoder)

    return cfg


def patch_dataset_config(audio_encoder):
    """
    Some versions of build_model() read from DATASET_CONFIGS and some read from cfg.
    Patch both to be safe.
    """
    try:
        from models.maf_model import DATASET_CONFIGS

        if DATASET in DATASET_CONFIGS:
            DATASET_CONFIGS[DATASET]["audio_encoder"] = audio_encoder
            DATASET_CONFIGS[DATASET]["audio_encoder_type"] = audio_encoder

        if "cmu-mosi" in DATASET_CONFIGS:
            # CMU-MOSEI is often aliased from CMU-MOSI in this project.
            DATASET_CONFIGS["cmu-mosi"]["audio_encoder"] = audio_encoder
            DATASET_CONFIGS["cmu-mosi"]["audio_encoder_type"] = audio_encoder

    except Exception as e:
        logger.warning(f"Could not patch DATASET_CONFIGS audio encoder: {e}")


def build_model_with_encoder(cfg, device, audio_encoder):
    from models.maf_model import build_model

    setattr(cfg, "audio_encoder", audio_encoder)
    patch_dataset_config(audio_encoder)

    model = build_model(DATASET, cfg).to(device)
    return model


def choose_working_audio_encoder(device):
    """
    Tries to build a model with the checkpoint-compatible audio encoder.
    Uses strict=True loading. If strict loading fails, that encoder is invalid.
    """
    global WORKING_AUDIO_ENCODER

    if WORKING_AUDIO_ENCODER is not None:
        return WORKING_AUDIO_ENCODER

    state = load_checkpoint_state(CKPT_BASE, device)
    inferred = infer_audio_encoder_from_state(state)

    candidates = []
    if inferred is not None:
        candidates.append(inferred)

    # Add common aliases/candidates used in different versions of this project.
    for c in ["crnn", "gru", "cnn", "covarep", "mlp", "feature", "features"]:
        if c not in candidates:
            candidates.append(c)

    logger.info(f"Checkpoint audio encoder inferred as: {inferred}")
    logger.info(f"Trying audio encoder candidates: {candidates}")

    last_errors = []

    for enc in candidates:
        try:
            cfg = make_cfg(audio_encoder=enc)
            cfg.device = str(device)
            model = build_model_with_encoder(cfg, device, enc)

            # STRICT load. Do not change this to strict=False.
            model.load_state_dict(state, strict=True)

            WORKING_AUDIO_ENCODER = enc
            logger.info(f"Selected audio_encoder='{enc}' because checkpoint loaded strictly.")
            return enc

        except Exception as e:
            msg = str(e).split("\n")[0]
            last_errors.append((enc, msg))
            logger.info(f"audio_encoder='{enc}' did not match checkpoint: {msg}")

    logger.error("Could not find a checkpoint-compatible audio encoder.")
    logger.error("Tried:")
    for enc, err in last_errors:
        logger.error(f"  {enc}: {err}")

    raise RuntimeError(
        "Failed to build a model compatible with the checkpoint. "
        "Do not use strict=False. Check models/maf_model.py audio encoder names."
    )


def load_base_model(cfg, device):
    """
    Load the trained CMU-MOSEI base model from checkpoint using the correct
    audio encoder architecture.
    """
    from models.maf_model import build_model

    state = load_checkpoint_state(CKPT_BASE, device)
    enc = choose_working_audio_encoder(device)

    setattr(cfg, "audio_encoder", enc)
    cfg.device = str(device)
    patch_dataset_config(enc)

    model = build_model(DATASET, cfg).to(device)
    model.load_state_dict(state, strict=True)

    logger.info(f"  Loaded base checkpoint: {CKPT_BASE}")
    logger.info(f"  Using audio_encoder='{enc}'")
    return model


def sanitize_model_parameters(model):
    import torch

    with torch.no_grad():
        for p in model.parameters():
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
    return model


def run_mia_for_model(model, loaders, device, modalities):
    """
    Run MIA using existing evaluate_all() in evaluate.py.
    """
    from evaluate import evaluate_all

    model.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = evaluate_all(
            model,
            forget_loader=loaders["test_forget"],
            retain_loader=loaders["test_retain"],
            device=device,
            modalities=modalities,
            mia_n_shadow=1,
        )

    return results


def apply_baseline(name, base_model_fn, loaders, device, num_classes):
    """
    Apply a baseline and return the unlearned model.
    Returns None on failure.
    """
    from baselines import BASELINES

    if name not in BASELINES:
        logger.warning(f"  {name}: not found in BASELINES dict — skipping")
        return None

    fn = BASELINES[name]

    try:
        model = base_model_fn()

        if name == "random_relabeling":
            model = fn(
                model,
                loaders["train_forget"],
                loaders["train_retain"],
                device,
                num_classes,
            )

        elif name in ("fisher_forgetting", "amnesiac"):
            model = fn(
                model,
                loaders["train_forget"],
                device,
            )

        elif name in ("boundary", "fast_mu"):
            model = fn(
                model,
                loaders["train_forget"],
                loaders["train_retain"],
                device,
                num_classes,
            )

        elif name == "delete_and_refine":
            model = fn(
                model,
                loaders["train_retain"],
                device,
            )

        elif name in ("finetune_dr", "fine_tuning", "finetune"):
            model = fn(
                model,
                loaders["train_retain"],
                device,
            )

        else:
            logger.warning(f"  {name}: unknown calling convention — skipping")
            return None

        return sanitize_model_parameters(model)

    except Exception as e:
        logger.warning(f"  {name}: FAILED — {e}")
        return None


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def print_metric(r, key):
    val = safe_float(r.get(key))
    if val is None:
        return "FAILED"
    return f"{val:.4f}"


def main():
    import torch
    from data.datasets import MAFDataModule
    from models.maf_model import DATASET_CONFIGS
    from unlearner import MAFUnlearner

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Base checkpoint: {CKPT_BASE}")

    OUT_DIR.mkdir(exist_ok=True)

    # Choose encoder first so DataModule/model config stays consistent.
    enc = choose_working_audio_encoder(device)

    cfg = make_cfg(audio_encoder=enc)
    cfg.device = str(device)
    patch_dataset_config(enc)

    dm = MAFDataModule(cfg)
    loaders = dm.get_loaders()

    num_classes = DATASET_CONFIGS[DATASET]["num_classes"]
    modalities = DATASET_CONFIGS[DATASET]["modalities"]

    logger.info(
        f"Forget set: {len(loaders['test_forget'].dataset)}  "
        f"Retain set: {len(loaders['test_retain'].dataset)}"
    )
    logger.info(f"Modalities: {modalities}")
    logger.info(f"Audio encoder: {enc}")

    def fresh_model():
        return load_base_model(cfg, device)

    methods = [
        "random_relabeling",
        "fisher_forgetting",
        "amnesiac",
        "boundary",
        "fast_mu",
        "delete_and_refine",
    ]

    all_results = {}

    for name in methods:
        logger.info(f"\n{'=' * 45}")
        logger.info(f"Method: {name}")
        logger.info(f"{'=' * 45}")

        model = apply_baseline(name, fresh_model, loaders, device, num_classes)

        if model is None:
            all_results[name] = {
                "forget_acc": None,
                "retain_acc": None,
                "trade_off": None,
                "mia_asr": None,
                "status": "FAILED",
            }
            continue

        try:
            r = run_mia_for_model(model, loaders, device, modalities)
            r["status"] = "OK"
            all_results[name] = r

            logger.info(
                f"  forget={print_metric(r, 'forget_acc')}  "
                f"retain={print_metric(r, 'retain_acc')}  "
                f"trade={print_metric(r, 'trade_off')}  "
                f"MIA={print_metric(r, 'mia_asr')}"
            )

        except Exception as e:
            logger.warning(f"  {name}: MIA FAILED — {e}")
            all_results[name] = {
                "forget_acc": None,
                "retain_acc": None,
                "trade_off": None,
                "mia_asr": None,
                "status": f"MIA_FAILED: {e}",
            }

    # MAF
    logger.info(f"\n{'=' * 45}")
    logger.info("Method: MAF (ours)")
    logger.info(f"{'=' * 45}")

    try:
        if MAF_CKPT.exists():
            maf_state = load_checkpoint_state(MAF_CKPT, device)
            maf_model = fresh_model()
            maf_model.load_state_dict(maf_state, strict=True)
            logger.info(f"  Loaded MAF unlearned checkpoint: {MAF_CKPT}")
        else:
            logger.info("  No saved MAF checkpoint — re-running MAF unlearning ...")
            maf_model = fresh_model()
            unlearner = MAFUnlearner(maf_model, cfg, device)
            unlearner.setup(loaders["train_forget"], loaders["train_retain"])
            unlearner.unlearn()
            maf_model = unlearner.model

        maf_model = sanitize_model_parameters(maf_model)
        r = run_mia_for_model(maf_model, loaders, device, modalities)
        r["status"] = "OK"
        all_results["MAF"] = r

        logger.info(
            f"  forget={print_metric(r, 'forget_acc')}  "
            f"retain={print_metric(r, 'retain_acc')}  "
            f"trade={print_metric(r, 'trade_off')}  "
            f"MIA={print_metric(r, 'mia_asr')}"
        )

    except Exception as e:
        logger.warning(f"  MAF: FAILED — {e}")
        all_results["MAF"] = {
            "forget_acc": None,
            "retain_acc": None,
            "trade_off": None,
            "mia_asr": None,
            "status": f"FAILED: {e}",
        }

    # Print final table
    logger.info(f"\n{'=' * 70}")
    logger.info("FULL CMU-MOSEI MIA TABLE")
    logger.info(f"{'=' * 70}")
    logger.info(
        f"  {'Method':<22} {'Forget':>8} {'Retain':>8} "
        f"{'Trade':>8} {'MIA':>8}  {'Status'}"
    )
    logger.info("  " + "-" * 68)

    maf_mia = safe_float(all_results.get("MAF", {}).get("mia_asr"))
    if maf_mia is None:
        maf_mia = 1.0

    for method, r in all_results.items():
        status = r.get("status", "UNKNOWN")

        if status != "OK":
            logger.info(f"  {method:<22} {'FAILED':>8} {'FAILED':>8} {'FAILED':>8} {'FAILED':>8}  {status}")
            continue

        mia_val = safe_float(r.get("mia_asr"))
        mia_flag = ""

        if method != "MAF" and mia_val is not None:
            if mia_val > maf_mia + 0.02:
                mia_flag = " ← MAF better privacy"
            elif mia_val < maf_mia - 0.02:
                mia_flag = " ← baseline better"

        logger.info(
            f"  {method:<22} "
            f"{print_metric(r, 'forget_acc'):>8} "
            f"{print_metric(r, 'retain_acc'):>8} "
            f"{print_metric(r, 'trade_off'):>8} "
            f"{print_metric(r, 'mia_asr'):>8}  "
            f"{status}{mia_flag}"
        )

    # Decision guidance
    logger.info(f"\n{'=' * 70}")
    logger.info("DECISION GUIDANCE")
    logger.info(f"{'=' * 70}")

    valid = {
        k: v
        for k, v in all_results.items()
        if k != "MAF" and v.get("status") == "OK" and safe_float(v.get("mia_asr")) is not None
    }

    maf_mia_val = safe_float(all_results.get("MAF", {}).get("mia_asr"))

    if maf_mia_val is None:
        logger.info("  MAF MIA not available. Do not add full MIA table.")
    elif not valid:
        logger.info("  No valid baseline MIA results. Do not add full MIA table.")
    else:
        better_count = sum(1 for v in valid.values() if safe_float(v["mia_asr"]) > maf_mia_val + 0.02)
        worse_count = sum(1 for v in valid.values() if safe_float(v["mia_asr"]) < maf_mia_val - 0.02)
        similar_count = len(valid) - better_count - worse_count

        logger.info(f"  MAF MIA = {maf_mia_val:.4f}")
        logger.info(f"  Methods with HIGHER MIA than MAF (MAF better): {better_count}/{len(valid)}")
        logger.info(f"  Methods with SIMILAR MIA: {similar_count}/{len(valid)}")
        logger.info(f"  Methods with LOWER MIA than MAF (baseline better): {worse_count}/{len(valid)}")

        if better_count >= len(valid) // 2 + 1:
            logger.info("\n  RECOMMENDATION: ADD full MIA column to paper.")
            logger.info("  MAF has clearly better privacy than the majority of baselines.")
        elif worse_count >= len(valid) // 2 + 1:
            logger.info("\n  RECOMMENDATION: DO NOT ADD. Keep diagnostic wording.")
            logger.info("  Baselines have better or equal MIA, so adding would weaken the paper.")
        else:
            logger.info("\n  RECOMMENDATION: BORDERLINE. Keep current diagnostic wording.")
            logger.info("  MIA results are mixed, so adding could invite scrutiny.")

    # Save results
    json_out = OUT_DIR / "cmu_mosei_mia_all_baselines.json"
    csv_out = OUT_DIR / "cmu_mosei_mia_all_baselines.csv"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "forget_acc", "retain_acc", "trade_off", "mia_asr", "status"])

        for method, r in all_results.items():
            writer.writerow([
                method,
                print_metric(r, "forget_acc"),
                print_metric(r, "retain_acc"),
                print_metric(r, "trade_off"),
                print_metric(r, "mia_asr"),
                r.get("status", ""),
            ])

    logger.info(f"\nSaved → {json_out}")
    logger.info(f"Saved → {csv_out}")
    logger.info("Now check the MIA column and apply the decision rule.")


if __name__ == "__main__":
    main()