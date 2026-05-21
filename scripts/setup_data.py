"""
setup_data.py  –  Download & prepare all three MAF datasets.

Run with:
    py setup_data.py                        # all datasets
    py setup_data.py --dataset memotion7k   # single dataset
    py setup_data.py --dataset crema-d
    py setup_data.py --dataset meld

PRE-REQUISITES  (install once):
    py -m pip install kaggle datasets gdown torchaudio tqdm requests pillow

CREDENTIALS (one-time setup):
    ┌─ Kaggle (for Memotion7k) ────────────────────────────────────────────┐
    │  1. Go to kaggle.com → Your Profile → Settings → API → Create Token │
    │  2. Move the downloaded kaggle.json to:                               │
    │       Windows : C:\\Users\\<you>\\.kaggle\\kaggle.json                 │
    │       macOS/Linux: ~/.kaggle/kaggle.json                              │
    └──────────────────────────────────────────────────────────────────────┘
    No credentials needed for CREMA-D or MELD.
"""

import argparse
import csv
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

# ── Try optional imports; they are checked per-dataset ──────────────────────
try:
    import requests
    from tqdm import tqdm
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import gdown
    _HAS_GDOWN = True
except ImportError:
    _HAS_GDOWN = False

try:
    from datasets import load_dataset as hf_load
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

try:
    import torchaudio
    import torch
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT = Path("./data")
SPLITS    = ("train", "val", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_split_json(samples: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(samples):>5} samples → {path}")


def stratified_split(samples: list, label_key: str = "label"):
    """Split into train/val/test preserving class balance."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for s in samples:
        buckets[s[label_key]].append(s)

    train, val, test = [], [], []
    for lbl, items in buckets.items():
        random.shuffle(items)
        n  = len(items)
        n1 = int(n * SPLIT_RATIOS[0])
        n2 = int(n * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))
        train += items[:n1]
        val   += items[n1:n2]
        test  += items[n2:]

    random.shuffle(train)
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar. Returns True on success."""
    if not _HAS_REQUESTS:
        print("  [SKIP] requests not installed — py -m pip install requests tqdm")
        return False
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [FAIL] {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def run(cmd: list) -> bool:
    """Run a shell command, return True on success."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] {result.stderr[:300]}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ── MEMOTION7K ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

MEMOTION_DIR = DATA_ROOT / "memotion7k"

MEMOTION_LABEL_MAP = {
    "positive": 0, "very positive": 0,
    "neutral":  1,
    "negative": 2, "very negative": 2,
    "0": 0, "1": 1, "2": 2,
}


def _memotion_from_csv(csv_path: Path, img_dir: Path) -> list:
    """Parse a memotion CSV → list of sample dicts."""
    samples = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Column names vary across versions of the dataset
            img   = (row.get("image_name") or row.get("Image_name") or "").strip()
            text  = (row.get("text_ocr")   or row.get("Text_Corrected") or
                     row.get("sentence")   or "").strip()
            label_raw = (row.get("overall_sentiment") or
                         row.get("Sentiment") or
                         row.get("label")     or "").strip().lower()
            label = MEMOTION_LABEL_MAP.get(label_raw)
            if img and label is not None:
                samples.append({"img": img, "text": text, "label": label})
    return samples


def _memotion_hf_fallback(out_dir: Path):
    """Try to pull Memotion7k from HuggingFace."""
    if not _HAS_HF:
        return False
    HF_IDS = [
        "wangjing0128/memotion_dataset_7k",
        "esFullName/memotion_7k",
        "Peilin/memotion7k",
    ]
    for hf_id in HF_IDS:
        try:
            print(f"  Trying HuggingFace: {hf_id} …")
            ds = hf_load(hf_id)
            all_samples = []
            img_dir = out_dir / "images"
            img_dir.mkdir(exist_ok=True)
            for split_name, split_ds in ds.items():
                for row in split_ds:
                    label_raw = str(row.get("label", row.get("sentiment", "1"))).lower()
                    label     = MEMOTION_LABEL_MAP.get(label_raw, 1)
                    img_name  = row.get("image_name", f"img_{len(all_samples):05d}.jpg")
                    text      = row.get("text_ocr", row.get("caption", ""))
                    # Save image bytes if present
                    if "image" in row and hasattr(row["image"], "save"):
                        row["image"].save(img_dir / img_name)
                    all_samples.append({"img": img_name, "text": text, "label": label})
            return all_samples
        except Exception as e:
            print(f"  [SKIP] {e}")
    return False


def setup_memotion7k():
    print("\n" + "═" * 60)
    print("  MEMOTION7K  (text + image, 3-class sentiment)")
    print("═" * 60)

    out_dir = MEMOTION_DIR
    ensure_dir(out_dir / "images")

    # ── Check if already prepared ──────────────────────────────────────────
    if (out_dir / "train.json").exists():
        print("  Already prepared. Delete data/memotion7k/ to re-download.")
        return

    # ── Method 1: Kaggle CLI ───────────────────────────────────────────────
    print("\n  [Method 1] Kaggle download …")
    kaggle_zip = out_dir / "memotion7k.zip"
    if not kaggle_zip.exists():
        ok = run([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "williamscott701/memotion-dataset-7k",
            "-p", str(out_dir),
        ])
    else:
        ok = True

    if ok and kaggle_zip.exists():
        print("  Extracting …")
        with zipfile.ZipFile(kaggle_zip) as z:
            z.extractall(out_dir)
        # Find the CSV and image folder
        csv_files = list(out_dir.rglob("*.csv"))
        print(f"  Found CSVs: {[f.name for f in csv_files]}")
        img_dirs  = [d for d in out_dir.rglob("*") if d.is_dir()
                     and any(d.iterdir()) and d.name not in ("images",)]
        samples = []
        for csv_f in csv_files:
            samples += _memotion_from_csv(csv_f, out_dir / "images")
        if samples:
            # Move images to canonical location
            for img_dir in img_dirs:
                for img_f in img_dir.glob("*.jpg"):
                    dst = out_dir / "images" / img_f.name
                    if not dst.exists():
                        shutil.copy2(img_f, dst)
            _write_memotion_splits(samples, out_dir)
            return

    # ── Method 2: HuggingFace ─────────────────────────────────────────────
    print("\n  [Method 2] HuggingFace …")
    result = _memotion_hf_fallback(out_dir)
    if result:
        _write_memotion_splits(result, out_dir)
        return

    # ── Method 3: Manual instructions ─────────────────────────────────────
    print("""
  [MANUAL DOWNLOAD REQUIRED]
  ──────────────────────────────────────────────────────────────
  Memotion7k is hosted on Kaggle and requires a free account.

  Option A – Kaggle CLI (recommended):
    1. Create free account at kaggle.com
    2. Go to Settings → API → Create New Token
    3. Place kaggle.json in ~/.kaggle/  (or C:\\Users\\<you>\\.kaggle\\)
    4. Re-run:  py setup_data.py --dataset memotion7k

  Option B – Manual:
    1. Download from: https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k
    2. Extract so that you have:
         data/memotion7k/labels.csv   (or memotion_dataset_7k.csv)
         data/memotion7k/images/*.jpg
    3. Re-run:  py setup_data.py --dataset memotion7k
  ──────────────────────────────────────────────────────────────
  Generating SYNTHETIC PLACEHOLDER data so the codebase runs.
  Replace with real data for valid results.
""")
    _write_memotion_synthetic(out_dir)


def _write_memotion_splits(samples: list, out_dir: Path):
    if not samples:
        print("  No samples found — falling back to synthetic")
        _write_memotion_synthetic(out_dir)
        return
    train, val, test = stratified_split(samples)
    save_split_json(train, out_dir / "train.json")
    save_split_json(val,   out_dir / "val.json")
    save_split_json(test,  out_dir / "test.json")
    print(f"  Total: {len(samples)} samples  ({len(set(s['label'] for s in samples))} classes)")


def _write_memotion_synthetic(out_dir: Path):
    """Generate synthetic placeholder data so the pipeline doesn't crash."""
    TEXTS = [
        "This is hilarious 😂", "I can has cheezburger?",
        "Just another day", "Feeling neutral about this meme",
        "This doesn't even make sense", "Why is this funny?",
        "So relatable omg", "Mood.", "Big oof", "Not funny, didn't laugh",
    ]
    samples = []
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    try:
        from PIL import Image
        import numpy as np
        for i in range(600):
            name = f"img_{i:04d}.jpg"
            img  = Image.fromarray(
                (np.random.rand(64, 64, 3) * 255).astype("uint8")
            )
            img.save(img_dir / name)
            samples.append({
                "img":   name,
                "text":  TEXTS[i % len(TEXTS)],
                "label": i % 3,
            })
    except ImportError:
        for i in range(600):
            samples.append({
                "img":   f"img_{i:04d}.jpg",
                "text":  TEXTS[i % len(TEXTS)],
                "label": i % 3,
            })
    _write_memotion_splits(samples, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ── CREMA-D ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

CREMA_DIR = DATA_ROOT / "crema-d"

# Emotion code → integer label
CREMA_EMO_MAP = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}

# Direct download URL for the CREMA-D summary CSV (no auth needed)
CREMA_CSV_URL = (
    "https://raw.githubusercontent.com/CheyneyComputerScience/"
    "CREMA-D/master/processedResults/summaryTable.csv"
)

# The audio archive is large (~500MB) — we download per-file from GitHub LFS
# as a fallback when the archive isn't available.
CREMA_AUDIO_BASE = (
    "https://media.githubusercontent.com/media/"
    "CheyneyComputerScience/CREMA-D/master/AudioWAV/"
)


def _parse_cremad_filename(fname: str):
    """
    CREMA-D filename pattern: <speakerID>_<sentence>_<emotion>_<intensity>.wav
    e.g. 1001_DFA_ANG_XX.wav
    Returns (speaker_id, emotion_label) or None.
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    speaker = int(parts[0])
    emo_code = parts[2].upper()
    label = CREMA_EMO_MAP.get(emo_code)
    if label is None:
        return None
    return speaker, label


def _cremad_from_summary_csv(csv_path: Path, audio_dir: Path) -> list:
    samples = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("FileName") or row.get("fileName") or "").strip()
            if not fname.endswith(".wav"):
                fname += ".wav"
            parsed = _parse_cremad_filename(fname)
            if parsed is None:
                continue
            speaker, label = parsed
            samples.append({
                "file":    fname,
                "emotion": label,
                "speaker": speaker,
            })
    return samples


def _cremad_scan_audio_dir(audio_dir: Path) -> list:
    """Build sample list from local audio files."""
    samples = []
    for wav in audio_dir.glob("*.wav"):
        parsed = _parse_cremad_filename(wav.name)
        if parsed:
            speaker, label = parsed
            samples.append({"file": wav.name, "emotion": label, "speaker": speaker})
    return samples


def _cremad_hf_fallback(out_dir: Path):
    if not _HAS_HF:
        return False
    HF_IDS = ["jlr3/cremad", "msclm/CREMA-D", "alfredcs/CREMA-D"]
    for hf_id in HF_IDS:
        try:
            print(f"  Trying HuggingFace: {hf_id} …")
            ds     = hf_load(hf_id)
            audio_dir = out_dir / "AudioWAV"
            audio_dir.mkdir(exist_ok=True)
            samples = []
            for split_ds in ds.values():
                for row in split_ds:
                    # Save audio
                    audio_data = row.get("audio") or row.get("speech")
                    fname      = row.get("file", f"utt_{len(samples):05d}.wav")
                    if audio_data and _HAS_AUDIO:
                        arr  = torch.tensor(audio_data["array"]).unsqueeze(0).float()
                        sr   = audio_data.get("sampling_rate", 16000)
                        _save_wav(audio_dir / fname, arr, sr)
                    lbl = CREMA_EMO_MAP.get(str(row.get("label", "")).upper(), 0)
                    samples.append({
                        "file":    fname,
                        "emotion": lbl,
                        "speaker": row.get("speaker", 0),
                    })
            return samples
        except Exception as e:
            print(f"  [SKIP] {e}")
    return False


def setup_cremad():
    print("\n" + "═" * 60)
    print("  CREMA-D  (audio, 6-class emotion)")
    print("═" * 60)

    out_dir   = CREMA_DIR
    audio_dir = out_dir / "AudioWAV"
    ensure_dir(audio_dir)

    if (out_dir / "train.json").exists():
        print("  Already prepared. Delete data/crema-d/ to re-download.")
        return

    # ── Method 1: Scan existing audio files ───────────────────────────────
    existing = list(audio_dir.glob("*.wav"))
    if len(existing) > 100:
        print(f"  Found {len(existing)} existing WAV files in {audio_dir}")
        samples = _cremad_scan_audio_dir(audio_dir)
        if samples:
            _write_cremad_splits(samples, out_dir)
            return

    # ── Method 2: torchaudio (if available) ───────────────────────────────
    if _HAS_AUDIO:
        print("\n  [Method 2] torchaudio download …")
        try:
            print("  Downloading CREMA-D via torchaudio …")
            ds = torchaudio.datasets.CREMA_D(root=str(out_dir), download=True)
            samples = []
            for i in range(len(ds)):
                waveform, sr, label, speaker, *_ = ds[i]
                fname = f"utt_{i:05d}.wav"
                _save_wav(audio_dir / fname, waveform, sr)
                samples.append({
                    "file":    fname,
                    "emotion": CREMA_EMO_MAP.get(label.upper(), 0),
                    "speaker": int(speaker),
                })
            if samples:
                _write_cremad_splits(samples, out_dir)
                return
        except Exception as e:
            print(f"  [SKIP] torchaudio: {e}")

    # ── Method 3: Download summary CSV + audio from GitHub ────────────────
    print("\n  [Method 3] GitHub direct download …")
    summary_csv = out_dir / "summaryTable.csv"
    if not summary_csv.exists():
        ok = download_file(CREMA_CSV_URL, summary_csv, "CREMA-D summary CSV")
    else:
        ok = True

    if ok and summary_csv.exists():
        samples = _cremad_from_summary_csv(summary_csv, audio_dir)
        print(f"  Got {len(samples)} entries from summary CSV")
        if samples:
            # Download audio files from GitHub LFS (first 500 for speed;
            # remove the slice to download all 7,442)
            print(f"  Downloading audio WAV files from GitHub (first 500) …")
            print("  Tip: remove the [:500] slice in setup_data.py to get all 7,442 files.")
            downloaded = 0
            for s in tqdm(samples[:500], desc="  WAV files"):
                wav_path = audio_dir / s["file"]
                if not wav_path.exists():
                    url = CREMA_AUDIO_BASE + s["file"]
                    if download_file(url, wav_path, ""):
                        downloaded += 1
                else:
                    downloaded += 1
            print(f"  Downloaded/found {downloaded} audio files")
            _write_cremad_splits(samples[:500], out_dir)
            return

    # ── Method 4: HuggingFace ─────────────────────────────────────────────
    print("\n  [Method 4] HuggingFace …")
    result = _cremad_hf_fallback(out_dir)
    if result:
        _write_cremad_splits(result, out_dir)
        return

    # ── Fallback: Manual instructions + synthetic ─────────────────────────
    print("""
  [MANUAL DOWNLOAD OPTION]
  ──────────────────────────────────────────────────────────────
  CREMA-D is freely available. Best options:

  Option A – Git LFS (gets all 7,442 audio files):
    git lfs install
    git clone https://github.com/CheyneyComputerScience/CREMA-D.git
    cp -r CREMA-D/AudioWAV  data/crema-d/AudioWAV
    py setup_data.py --dataset crema-d

  Option B – Hugging Face (easier):
    py -m pip install datasets
    Then re-run this script (HuggingFace fallback will activate).
  ──────────────────────────────────────────────────────────────
  Generating SYNTHETIC PLACEHOLDER data.
""")
    _write_cremad_synthetic(out_dir)


def _write_cremad_splits(samples: list, out_dir: Path):
    if not samples:
        print("  No samples — falling back to synthetic")
        _write_cremad_synthetic(out_dir)
        return
    train, val, test = stratified_split(samples, label_key="emotion")
    save_split_json(train, out_dir / "train.json")
    save_split_json(val,   out_dir / "val.json")
    save_split_json(test,  out_dir / "test.json")
    print(f"  Total: {len(samples)} utterances  "
          f"({len(set(s['speaker'] for s in samples))} speakers)")


def _write_cremad_synthetic(out_dir: Path):
    audio_dir = out_dir / "AudioWAV"
    audio_dir.mkdir(exist_ok=True)
    speakers  = list(range(1001, 1092))  # 91 actors
    emos      = list(CREMA_EMO_MAP.keys())
    sentences = ["DFA", "IEO", "IOM", "ITH", "ITS", "IWL", "IWW",
                 "JSI", "LES", "MTI", "NEI", "OOM", "TIE", "TSI", "WSI"]
    intensities = ["HI", "LO", "MD", "XX"]
    samples, idx = [], 0

    if _HAS_AUDIO:
        print("  Generating synthetic WAV files …")
        for sp in speakers:
            for emo in emos:
                sent  = sentences[idx % len(sentences)]
                intns = intensities[idx % len(intensities)]
                fname = f"{sp}_{sent}_{emo}_{intns}.wav"
                wav_path = audio_dir / fname
                if not wav_path.exists():
                    noise = torch.randn(1, 16000)
                    _save_wav(wav_path, noise, 16000)
                samples.append({
                    "file":    fname,
                    "emotion": CREMA_EMO_MAP[emo],
                    "speaker": sp,
                })
                idx += 1
    else:
        for sp in speakers:
            for emo in emos:
                fname = f"{sp}_{sentences[idx%len(sentences)]}_{emo}_XX.wav"
                samples.append({"file": fname, "emotion": CREMA_EMO_MAP[emo], "speaker": sp})
                idx += 1
    _write_cremad_splits(samples, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ── MELD ──────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

MELD_DIR = DATA_ROOT / "meld"

MELD_EMO_MAP = {
    "neutral":  0, "surprise": 1, "fear": 2, "sadness": 3,
    "joy":      4, "disgust":  5, "anger": 6,
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
}

# Direct GitHub raw URLs for the MELD CSV files
MELD_CSV_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD.Raw/train_sent_emo.csv",
    "val":   "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD.Raw/dev_sent_emo.csv",
    "test":  "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD.Raw/test_sent_emo.csv",
}

# Audio tarballs on Google Drive (from official MELD repo)
MELD_AUDIO_GDRIVE = {
    "train": "1TqP4uFPzm4mGXOXBXUxNsXnBuaxaqCLT",  # train.tar.gz
    "val":   "1mKqiHneTdntKPkYnKJt1qbzreRv8nU9v",   # dev.tar.gz
    "test":  "1XH-7O8lqFiDuHtL2kIEjPVBsUqJTVlA9",   # test.tar.gz
}


def _save_wav(path: Path, tensor: "torch.Tensor", sr: int = 16000):
    """Save a 1D/2D float tensor as WAV, with fallback if torchcodec missing."""
    try:
        torchaudio.save(str(path), tensor, sr, format="wav")
        return
    except Exception:
        pass
    # Fallback: use Python's built-in wave module
    import wave, struct
    arr = (tensor.squeeze().numpy() * 32767).astype("int16")
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(arr)}h", *arr))


def _parse_meld_csv(csv_path: Path) -> list:
    samples = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text     = (row.get("Utterance") or row.get("utterance") or "").strip()
            emo_raw  = (row.get("Emotion")   or row.get("emotion")   or "neutral").strip().lower()
            speaker  = (row.get("Speaker")   or row.get("speaker")   or "Unknown").strip()
            dia_id   = row.get("Dialogue_ID", row.get("dialogue_id", "0"))
            utt_id   = row.get("Utterance_ID", row.get("utterance_id", "0"))
            emotion  = MELD_EMO_MAP.get(emo_raw, 0)
            audio_f  = f"dia{dia_id}_utt{utt_id}.wav"
            samples.append({
                "text":    text,
                "audio":   audio_f,
                "emotion": emotion,
                "speaker": speaker,
                "label":   emotion,
            })
    return samples


def _meld_hf_fallback(out_dir: Path):
    if not _HAS_HF:
        return {}
    HF_IDS = [
        "declare-lab/MELD",
        "joankusuma/meld",
        "Zahra99/meld_dataset",
    ]
    for hf_id in HF_IDS:
        try:
            print(f"  Trying HuggingFace: {hf_id} …")
            ds = hf_load(hf_id)
            audio_dir = out_dir / "audio"
            audio_dir.mkdir(exist_ok=True)
            result = {"train": [], "val": [], "test": []}
            split_map = {
                "train": "train", "validation": "val", "dev": "val", "test": "test"
            }
            for hf_split, rows in ds.items():
                dest_split = split_map.get(hf_split, "train")
                for i, row in enumerate(rows):
                    text    = row.get("Utterance", row.get("text", ""))
                    emo_raw = str(row.get("Emotion", row.get("label", "0"))).lower()
                    speaker = row.get("Speaker", "Unknown")
                    emotion = MELD_EMO_MAP.get(emo_raw, 0)
                    fname   = f"dia0_utt{i:05d}.wav"
                    if "audio" in row and _HAS_AUDIO:
                        arr = torch.tensor(row["audio"]["array"]).unsqueeze(0).float()
                        sr  = row["audio"].get("sampling_rate", 16000)
                        _save_wav(audio_dir / fname, arr, sr)
                    result[dest_split].append({
                        "text": text, "audio": fname,
                        "emotion": emotion, "speaker": speaker, "label": emotion,
                    })
            return result
        except Exception as e:
            print(f"  [SKIP] {e}")
    return {}


def setup_meld():
    print("\n" + "═" * 60)
    print("  MELD  (text + audio, 7-class emotion, speaker-level forgetting)")
    print("═" * 60)

    out_dir   = MELD_DIR
    audio_dir = out_dir / "audio"
    ensure_dir(audio_dir)

    if (out_dir / "train.json").exists():
        print("  Already prepared. Delete data/meld/ to re-download.")
        return

    split_samples = {}

    # ── Method 1: Download CSVs from GitHub ───────────────────────────────
    print("\n  [Method 1] Downloading MELD CSVs from GitHub …")
    csv_ok = True
    for split_name, url in MELD_CSV_URLS.items():
        csv_path = out_dir / f"{split_name}_sent_emo.csv"
        if not csv_path.exists():
            ok = download_file(url, csv_path, f"MELD {split_name}.csv")
            csv_ok = csv_ok and ok

    if csv_ok:
        for split_name in ("train", "val", "test"):
            csv_path = out_dir / f"{split_name}_sent_emo.csv"
            if csv_path.exists():
                split_samples[split_name] = _parse_meld_csv(csv_path)
                print(f"  Parsed {len(split_samples[split_name])} {split_name} utterances")

        # ── Method 1b: Download audio tarballs via gdown ──────────────────
        if _HAS_GDOWN and split_samples:
            print("\n  Downloading MELD audio from Google Drive (via gdown) …")
            for split_name, gdrive_id in MELD_AUDIO_GDRIVE.items():
                tar_path = out_dir / f"{split_name}_audio.tar.gz"
                if not tar_path.exists():
                    try:
                        url = f"https://drive.google.com/uc?id={gdrive_id}"
                        gdown.download(url, str(tar_path), quiet=False)
                    except Exception as e:
                        print(f"  [SKIP] gdown {split_name}: {e}")
                if tar_path.exists():
                    print(f"  Extracting {tar_path.name} …")
                    with tarfile.open(tar_path) as t:
                        t.extractall(out_dir)
                    # Move WAV files to audio/
                    for wav in out_dir.rglob("*.wav"):
                        dst = audio_dir / wav.name
                        if not dst.exists():
                            shutil.move(str(wav), str(dst))

    # ── Method 2: HuggingFace ─────────────────────────────────────────────
    if not split_samples:
        print("\n  [Method 2] HuggingFace …")
        split_samples = _meld_hf_fallback(out_dir)

    if split_samples and any(v for v in split_samples.values()):
        _write_meld_splits(split_samples, out_dir)
        return

    # ── Manual instructions + synthetic ───────────────────────────────────
    print("""
  [MANUAL DOWNLOAD OPTION]
  ──────────────────────────────────────────────────────────────
  MELD is freely available from the official GitHub repo.

  Option A – Download script from repo:
    git clone https://github.com/declare-lab/MELD
    cd MELD && py download_meld.py
    cp -r data/MELD.Raw/train_sent_emo.csv data/meld/
    cp -r data/MELD.Raw/dev_sent_emo.csv   data/meld/val_sent_emo.csv
    cp -r data/MELD.Raw/test_sent_emo.csv  data/meld/
    py setup_data.py --dataset meld

  Option B – gdown (for audio):
    py -m pip install gdown
    Then re-run this script.
  ──────────────────────────────────────────────────────────────
  Generating SYNTHETIC PLACEHOLDER data.
""")
    _write_meld_synthetic(out_dir)


def _write_meld_splits(split_samples: dict, out_dir: Path):
    total = 0
    for split_name in ("train", "val", "test"):
        samples = split_samples.get(split_name, [])
        if not samples:
            continue
        # Ensure "label" key exists
        for s in samples:
            if "label" not in s:
                s["label"] = s.get("emotion", 0)
        save_split_json(samples, out_dir / f"{split_name}.json")
        total += len(samples)
    print(f"  Total: {total} utterances")


def _write_meld_synthetic(out_dir: Path):
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    SPEAKERS = ["Ross", "Rachel", "Monica", "Chandler", "Joey", "Phoebe",
                "Mike", "Gunther", "Janice", "Richard"]
    TEXTS    = [
        "I was on a break!", "How you doin'?", "Could this BE any more obvious?",
        "Oh. My. God.", "He's her lobster.", "Smelly cat, smelly cat.",
        "We were on a break!", "I'm so not okay.", "Seven! Seven! Seven!",
        "Gum would be perfection.", "Joey doesn't share food!",
    ]
    split_samples = {"train": [], "val": [], "test": []}
    counts = {"train": 500, "val": 100, "test": 150}
    idx = 0
    for split_name, n in counts.items():
        for i in range(n):
            sp    = SPEAKERS[idx % len(SPEAKERS)]
            fname = f"dia{idx//10}_utt{idx%10}.wav"
            if _HAS_AUDIO:
                wav_path = audio_dir / fname
                if not wav_path.exists():
                    noise = torch.randn(1, 16000)
                    _save_wav(wav_path, noise, 16000)
            emo = idx % 7
            split_samples[split_name].append({
                "text":    TEXTS[idx % len(TEXTS)],
                "audio":   fname,
                "emotion": emo,
                "label":   emo,
                "speaker": sp,
            })
            idx += 1
    _write_meld_splits(split_samples, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ── Verify ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def verify_all():
    print("\n" + "═" * 60)
    print("  VERIFICATION")
    print("═" * 60)
    all_ok = True
    checks = {
        "memotion7k": [
            DATA_ROOT / "memotion7k" / "train.json",
            DATA_ROOT / "memotion7k" / "val.json",
            DATA_ROOT / "memotion7k" / "test.json",
        ],
        "crema-d": [
            DATA_ROOT / "crema-d" / "train.json",
            DATA_ROOT / "crema-d" / "val.json",
            DATA_ROOT / "crema-d" / "test.json",
        ],
        "meld": [
            DATA_ROOT / "meld" / "train.json",
            DATA_ROOT / "meld" / "val.json",
            DATA_ROOT / "meld" / "test.json",
        ],
    }
    for dataset, files in checks.items():
        ok = all(f.exists() for f in files)
        status = "✓" if ok else "✗"
        print(f"  {status}  {dataset}")
        if ok:
            for f in files:
                with open(f) as fp:
                    n = len(json.load(fp))
                print(f"       {f.name:12s}  {n:5d} samples")
        else:
            all_ok = False
            for f in files:
                print(f"       {'MISSING' if not f.exists() else 'OK':8s}  {f}")
    print()
    if all_ok:
        print("  All datasets ready.  Run your experiment with:")
        print("    py main.py --dataset memotion7k --mode full")
        print("    py main.py --dataset crema-d    --mode full")
        print("    py main.py --dataset meld        --mode full")
    else:
        print("  Some datasets missing — see above for manual download instructions.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ── Entry point ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download and prepare MAF datasets")
    parser.add_argument(
        "--dataset",
        choices=["memotion7k", "crema-d", "meld", "all"],
        default="all",
    )
    parser.add_argument(
        "--data_root", default="./data",
        help="Root directory for datasets (default: ./data)"
    )
    args = parser.parse_args()

    global DATA_ROOT, MEMOTION_DIR, CREMA_DIR, MELD_DIR
    DATA_ROOT    = Path(args.data_root)
    MEMOTION_DIR = DATA_ROOT / "memotion7k"
    CREMA_DIR    = DATA_ROOT / "crema-d"
    MELD_DIR     = DATA_ROOT / "meld"

    print(f"\nData root: {DATA_ROOT.resolve()}")
    random.seed(42)

    if args.dataset in ("memotion7k", "all"):
        setup_memotion7k()
    if args.dataset in ("crema-d", "all"):
        setup_cremad()
    if args.dataset in ("meld", "all"):
        setup_meld()

    verify_all()


if __name__ == "__main__":
    main()
