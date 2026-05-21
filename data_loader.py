"""
data_loader.py — Load MLLMU-Bench for two distinct experiment types.

EXPERIMENT 1 — Fictitious entities (forget_10 config)
  Forget: 30 fictitious profiles (no LLaVA parametric memory)
  Retain: 100 real celebrities (Retain_Set)
  Expected: index deletion alone is sufficient → FA drops to 0

EXPERIMENT 2 — Celebrity entities (split from Retain_Set)
  Forget: first 30 real celebrities from Retain_Set
  Retain: next 100 real celebrities from Retain_Set
  Expected: index deletion insufficient → parametric memory leaks
  This is the key experiment proving dual-channel necessity.

Together the two experiments characterise WHEN each channel matters.
"""

import os, pickle, logging, json
from typing import List, Dict, Tuple, Optional
from PIL import Image

from config import DATA_DIR, MAX_FORGET, MAX_RETAIN

logger = logging.getLogger(__name__)

HF_DATASET_ID  = "MLLMMU/MLLMU-Bench"
FORGET_CONFIG  = "forget_10"   # 50 fictitious profiles
RETAIN_CONFIG  = "Retain_Set"  # 153 real celebrities


class Sample(dict):
    """Keys: entity_id, image (PIL), caption, question, answer, split"""
    pass


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def load_mllmu_bench(
    cache_path: Optional[str] = None,
    max_forget: int = MAX_FORGET,
    max_retain: int = MAX_RETAIN,
) -> Tuple[List[Sample], List[Sample]]:
    """Experiment 1: fictitious forget + celebrity retain."""
    cache_file = cache_path or os.path.join(DATA_DIR, "mllmu_exp1.pkl")
    return _load_cached(
        cache_file,
        lambda: _download_exp1(max_forget, max_retain),
        max_forget, max_retain,
    )


def load_celebrity_experiment(
    cache_path: Optional[str] = None,
    max_forget: int = 30,
    max_retain: int = 100,
) -> Tuple[List[Sample], List[Sample]]:
    """
    Experiment 2: celebrity forget + celebrity retain.
    Split Retain_Set (153) into:
      - forget: first 30  (entities LLaVA genuinely knows)
      - retain: next 100  (indices 30–129)
    """
    cache_file = cache_path or os.path.join(DATA_DIR, "mllmu_exp2.pkl")
    return _load_cached(
        cache_file,
        lambda: _download_exp2(max_forget, max_retain),
        max_forget, max_retain,
    )


# ─────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────

def _load_cached(cache_file, downloader, max_forget, max_retain):
    if os.path.exists(cache_file):
        logger.info(f"Loading cached samples from {cache_file}")
        with open(cache_file, "rb") as f:
            forget, retain = pickle.load(f)
        return forget[:max_forget], retain[:max_retain]
    forget, retain = downloader()
    with open(cache_file, "wb") as f:
        pickle.dump((forget, retain), f)
    logger.info(f"Cached {len(forget)} forget + {len(retain)} retain → {cache_file}")
    return forget[:max_forget], retain[:max_retain]


def _download_exp1(max_forget, max_retain):
    """Fictitious forget (forget_10) + celebrity retain (Retain_Set)."""
    logger.info(f"Downloading {HF_DATASET_ID} [Exp 1: fictitious forget]...")
    try:
        from datasets import load_dataset
        forget_ds = load_dataset(HF_DATASET_ID, FORGET_CONFIG, split="train")
        retain_ds = load_dataset(HF_DATASET_ID, RETAIN_CONFIG,  split="train")
        logger.info(f"Forget '{FORGET_CONFIG}': {len(forget_ds)} | Retain '{RETAIN_CONFIG}': {len(retain_ds)}")
        return (
            _process_rows(forget_ds, "forget", max_forget),
            _process_rows(retain_ds, "retain", max_retain),
        )
    except Exception as e:
        logger.warning(f"HF load failed: {e}. Falling back to synthetic.")
        return _build_synthetic_data(max_forget, max_retain)


def _download_exp2(max_forget, max_retain):
    """Celebrity forget + celebrity retain (both from Retain_Set)."""
    logger.info(f"Downloading {HF_DATASET_ID} [Exp 2: celebrity forget]...")
    try:
        from datasets import load_dataset
        all_celebrities = load_dataset(HF_DATASET_ID, RETAIN_CONFIG, split="train")
        logger.info(f"Retain_Set total: {len(all_celebrities)} celebrities")
        all_rows = list(all_celebrities)
        # Hard split: first 30 = forget, next 100 = retain
        forget_rows = all_rows[:max_forget]
        retain_rows = all_rows[max_forget: max_forget + max_retain]
        return (
            _process_rows(forget_rows, "forget", max_forget),
            _process_rows(retain_rows, "retain", max_retain),
        )
    except Exception as e:
        logger.warning(f"HF load failed: {e}. Falling back to synthetic.")
        return _build_synthetic_data(max_forget, max_retain)


# ─────────────────────────────────────────────────────────────────
# Row processing — MLLMU-Bench specific schema
# ─────────────────────────────────────────────────────────────────

def _process_rows(rows, split: str, max_samples: int) -> List[Sample]:
    samples = []
    for i, row in enumerate(rows):
        if i >= max_samples:
            break
        s = _row_to_sample(row, split)
        if s is not None:
            samples.append(s)
    return samples


def _row_to_sample(row: Dict, split: str) -> Optional[Sample]:
    image = _extract_image(row)
    if image is None:
        return None

    bio = row.get("biography", {})
    if isinstance(bio, str):
        try: bio = json.loads(bio)
        except: bio = {}

    name      = bio.get("Name") or bio.get("name") or str(row.get("ID", "unknown"))
    entity_id = str(name).replace(" ", "_")
    caption   = bio.get("Description") or bio.get("description") or ""

    if not caption and isinstance(bio, dict):
        parts = [f"{k}: {v}" for k in ["Name","Employment","Residence","Educated at"]
                 if (v := bio.get(k))]
        caption = ". ".join(parts)

    question = str(row.get("question") or
                   "Tell me about this person including their name and background.")
    answer   = str(row.get("answer") or caption or "")

    if not answer:
        gen = row.get("Generation_Task", [])
        if gen and isinstance(gen, list):
            first = gen[0]
            if isinstance(first, dict):
                answer   = first.get("Ground_Truth", "")
                question = first.get("Question", question)

    if not answer:
        return None

    return Sample(entity_id=entity_id, image=image, caption=caption,
                  question=question, answer=answer, split=split)


def _extract_image(row: Dict) -> Optional[Image.Image]:
    val = row.get("image")
    if val is None: return None
    if isinstance(val, Image.Image): return val.convert("RGB")
    if isinstance(val, dict):
        raw = val.get("bytes")
        if raw:
            from io import BytesIO
            try: return Image.open(BytesIO(raw)).convert("RGB")
            except: pass
        path = val.get("path")
        if path and os.path.exists(path):
            try: return Image.open(path).convert("RGB")
            except: pass
    return None


# ─────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────

def _build_synthetic_data(max_forget: int, max_retain: int):
    import numpy as np
    def _img(seed):
        rng = np.random.RandomState(seed)
        return Image.fromarray(rng.randint(0,255,(224,224,3),dtype=np.uint8))

    ENTITIES = [
        ("Alice Johnson",  "an environmental scientist from Copenhagen."),
        ("Bob Chen",       "a celebrated chef known for fusion cuisine."),
        ("Clara Müller",   "a German novelist famous for dystopian fiction."),
        ("David Osei",     "an Olympic sprinter from Ghana."),
        ("Elena Rossi",    "an Italian soprano who debuted at La Scala."),
    ]
    forget, retain = [], []
    for i in range(max_forget):
        name, desc = ENTITIES[i % len(ENTITIES)]
        forget.append(Sample(entity_id=f"{name.replace(' ','_')}_{i}",
            image=_img(i), caption=f"{name} is {desc}",
            question=f"Who is {name} and what are they known for?",
            answer=f"{name} is {desc}", split="forget"))
    for i in range(max_retain):
        retain.append(Sample(entity_id=f"retain_entity_{i}",
            image=_img(1000+i),
            caption=f"A person known for contributions to field {i%10}.",
            question=f"Describe the contributions of person {i}.",
            answer=f"Person {i} contributed significantly to field {i%10}.",
            split="retain"))
    logger.info(f"[Synthetic] {len(forget)} forget + {len(retain)} retain.")
    return forget, retain
