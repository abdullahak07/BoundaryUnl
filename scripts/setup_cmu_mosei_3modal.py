"""
setup_cmu_mosei_3modal.py

Extends CMU-MOSEI to 3-modality:
  text + audio proxy + video FACET42.

Downloads:
  CMU-MOSEI/labels/CMU_MOSEI_Labels.csd
  CMU-MOSEI/languages/CMU_MOSEI_TimestampedWords.csd
  CMU-MOSEI/visuals/CMU_MOSEI_VisualFacet42.csd

Important:
  This script uses real text and real video FACET42 features.
  Audio is a lightweight text-derived proxy unless you separately download COVAREP.
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np


DATASET = "samarwarsi/cmu-mosei"
RAW_DIR = Path("data/cmu-mosei/raw")
OUT_DIR = Path("data/cmu-mosei-3modal")

# IMPORTANT:
# Kaggle file paths must include the top-level CMU-MOSEI/ folder.
NEEDED = [
    ("CMU-MOSEI/labels/CMU_MOSEI_Labels.csd", "CMU_MOSEI_Labels.csd"),
    ("CMU-MOSEI/languages/CMU_MOSEI_TimestampedWords.csd", "CMU_MOSEI_TimestampedWords.csd"),
    ("CMU-MOSEI/visuals/CMU_MOSEI_VisualFacet42.csd", "CMU_MOSEI_VisualFacet42.csd"),
]

FACET_DIM = 35
AUDIO_DIM = 74


def kaggle_cmd():
    """
    Return a working Kaggle command on Windows/Python 3.13.

    Avoids broken:
        python -m kaggle

    Uses:
        kaggle.exe
    or:
        python -m kaggle.cli
    """
    exe = shutil.which("kaggle")
    if exe:
        return [exe]

    scripts_dir = Path(sys.executable).parent / "Scripts"
    kaggle_exe = scripts_dir / "kaggle.exe"
    if kaggle_exe.exists():
        return [str(kaggle_exe)]

    return [sys.executable, "-m", "kaggle.cli"]


def run_kaggle_download(rel_path, dest_name):
    """
    Download one file from Kaggle dataset.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dest = RAW_DIR / dest_name

    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  SKIP (exists): {dest_name}  ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    cmd = kaggle_cmd() + [
        "datasets",
        "download",
        DATASET,
        "--file",
        rel_path,
        "--path",
        str(RAW_DIR),
    ]

    print(f"  Downloading: {dest_name} ...")
    print("  Kaggle file:", rel_path)

    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        print("\n  ERROR: Kaggle download failed.")
        print("  Command:")
        print("  " + " ".join(cmd))

        print("\n  STDOUT:")
        print(r.stdout if r.stdout else "  <empty>")

        print("\n  STDERR:")
        print(r.stderr if r.stderr else "  <empty>")

        print("\n  Most likely causes:")
        print("  1. Wrong Kaggle file path")
        print("  2. Missing Kaggle API token")
        print("  3. Dataset access/rules not accepted")
        print("  4. Network interruption")
        sys.exit(1)

    # Kaggle usually downloads a zip.
    zip_files = list(RAW_DIR.glob("*.zip"))

    for zf in zip_files:
        print(f"  Extracting: {zf.name}")
        with zipfile.ZipFile(zf) as z:
            z.extractall(RAW_DIR)
        zf.unlink()

    # Find the downloaded file anywhere under RAW_DIR.
    matches = list(RAW_DIR.rglob(dest_name))

    if not matches:
        print(f"\n  ERROR: Download finished but {dest_name} was not found.")
        print("  Files currently under RAW_DIR:")
        for p in RAW_DIR.rglob("*"):
            if p.is_file():
                print("   ", p)
        sys.exit(1)

    found = matches[0]

    if found.resolve() != dest.resolve():
        shutil.move(str(found), str(dest))

    # Clean empty CMU-MOSEI folders if extraction created them.
    extracted_top = RAW_DIR / "CMU-MOSEI"
    if extracted_top.exists():
        try:
            shutil.rmtree(extracted_top)
        except Exception:
            pass

    print(f"  OK: {dest_name}  ({dest.stat().st_size / 1e6:.1f} MB)")


def download():
    print("\nStep 1: Downloading files ...")

    for rel_path, dest_name in NEEDED:
        run_kaggle_download(rel_path, dest_name)


def read_csd(path):
    """
    Robust CMU-MOSEI .csd reader.

    Handles structures like:
      /data/<video_id>/features
      /All Labels/<video_id>/features
      /CMU_MOSEI_.../<video_id>/features

    It recursively searches for groups that contain both:
      features
      intervals
    """
    p = Path(path)

    if not p.exists():
        print(f"  MISSING: {p.name}")
        return {}

    print(f"  Reading {p.name}  ({p.stat().st_size / 1e6:.1f} MB) ...")

    try:
        import h5py

        out = {}

        def clean_key(x):
            if isinstance(x, bytes):
                return x.decode("utf-8", "ignore")
            return str(x)

        def visit_group(name, obj):
            if not isinstance(obj, h5py.Group):
                return

            if "features" in obj and "intervals" in obj:
                try:
                    features = obj["features"][()]
                    intervals = obj["intervals"][()]

                    # Use final path component as video id.
                    vid = clean_key(name.split("/")[-1])

                    out[vid] = {
                        "features": features,
                        "intervals": intervals,
                    }
                except Exception:
                    pass

        with h5py.File(p, "r") as f:
            print(f"  Top-level groups: {list(f.keys())[:10]}")
            f.visititems(visit_group)

        print(f"  Loaded {len(out)} sequences from {p.name}")

        if len(out) == 0:
            print("  WARNING: no groups with both 'features' and 'intervals' were found.")
            print("  Run the debug command below to inspect structure.")

        return out

    except Exception as e:
        print(f"  HDF5 read failed for {p.name}: {e}")

    print(f"  ERROR: could not read {p.name}")
    return {}

def text_to_audio_proxy(text, dim=74):
    """
    Lightweight 74-dim audio proxy.

    This is NOT real COVAREP audio. It is a proxy feature used when the 11.6GB
    COVAREP file is not downloaded.
    """
    vec = np.zeros(dim, dtype=np.float32)
    words = text.split()

    vec[0] = min(len(words), 100) / 100.0
    vec[1] = min(len(text), 500) / 500.0
    vec[2] = (np.mean([len(w) for w in words]) / 10.0) if words else 0.0

    for w in words:
        idx = 3 + (abs(hash(w.lower())) % (dim - 3))
        vec[idx] = min(vec[idx] + 0.1, 1.0)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.astype(np.float32).tolist()


def get_from_csd_dict(d, vid):
    """
    Handle string keys and bytes keys.
    """
    if not isinstance(d, dict) or not d:
        return {}

    if vid in d:
        return d[vid]

    bvid = vid.encode("utf-8")
    if bvid in d:
        return d[bvid]

    return {}


def extract_words_for_segment(t_ints, t_feats, s0, s1):
    words = []

    for ws, wf in zip(t_ints, t_feats):
        try:
            w0 = float(ws[0])
            w1 = float(ws[1])

            if w0 >= s0 and w1 <= s1 + 0.1:
                if hasattr(wf, "__len__"):
                    w = wf[0]
                else:
                    w = wf

                if isinstance(w, bytes):
                    w = w.decode("utf-8", "ignore")

                w = str(w).strip()

                if w and w not in ("sp", "sil", "PUNCUATION", "PUNCTUATION"):
                    words.append(w)

        except Exception:
            continue

    text = " ".join(words).strip()

    if not text:
        text = "multimodal sentiment utterance"

    return text


def extract_video_for_segment(f_ints, f_feats, s0, s1):
    """
    Average FACET42 visual vectors over the segment.
    """
    video_vecs = []

    for fs, ff in zip(f_ints, f_feats):
        try:
            f0 = float(fs[0])
            f1 = float(fs[1])

            if f0 >= s0 and f1 <= s1 + 0.1:
                v = np.array(ff, dtype=np.float32).flatten()

                if len(v) >= FACET_DIM:
                    v = v[:FACET_DIM]

                    if not np.any(np.isnan(v)):
                        video_vecs.append(v)

        except Exception:
            continue

    if video_vecs:
        return np.mean(video_vecs, axis=0).astype(np.float32).tolist(), True

    return [0.0] * FACET_DIM, False


def convert():
    print("\nStep 2: Converting to MAF 3-modal JSON ...")
    print("\nReading CSD files ...")

    labels_d = read_csd(RAW_DIR / "CMU_MOSEI_Labels.csd")
    text_d = read_csd(RAW_DIR / "CMU_MOSEI_TimestampedWords.csd")
    facet_d = read_csd(RAW_DIR / "CMU_MOSEI_VisualFacet42.csd")

    if not labels_d:
        print("ERROR: labels file empty.")
        sys.exit(1)

    print(
        f"  labels: {len(labels_d)} videos | "
        f"text: {len(text_d)} | video: {len(facet_d)}"
    )

    all_vids = sorted(str(v.decode("utf-8", "ignore") if isinstance(v, bytes) else v) for v in labels_d.keys())

    n = len(all_vids)

    train_v = set(all_vids[: int(n * 0.70)])
    val_v = set(all_vids[int(n * 0.70): int(n * 0.80)])
    test_v = set(all_vids[int(n * 0.80):])

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    skipped_videos = 0
    skipped_segments = 0

    for vid in all_vids:
        if vid in train_v:
            split = "train"
        elif vid in val_v:
            split = "val"
        else:
            split = "test"

        ld = get_from_csd_dict(labels_d, vid)

        if not isinstance(ld, dict):
            skipped_videos += 1
            continue

        l_ints = ld.get("intervals", [])
        l_feats = ld.get("features", [])

        td = get_from_csd_dict(text_d, vid)
        t_ints = td.get("intervals", []) if isinstance(td, dict) else []
        t_feats = td.get("features", []) if isinstance(td, dict) else []

        fd = get_from_csd_dict(facet_d, vid)
        f_ints = fd.get("intervals", []) if isinstance(fd, dict) else []
        f_feats = fd.get("features", []) if isinstance(fd, dict) else []

        for seg_idx, (seg, label_feat) in enumerate(zip(l_ints, l_feats)):
            try:
                sentiment = float(np.array(label_feat).flatten()[0])

                if sentiment < -0.5:
                    label = 0
                elif sentiment > 0.5:
                    label = 2
                else:
                    label = 1

                s0 = float(seg[0])
                s1 = float(seg[1])

            except Exception:
                skipped_segments += 1
                continue

            text = extract_words_for_segment(t_ints, t_feats, s0, s1)
            audio = text_to_audio_proxy(text, AUDIO_DIM)
            video, has_video = extract_video_for_segment(f_ints, f_feats, s0, s1)

            splits[split].append(
                {
                    "id": f"{vid}_{seg_idx}",
                    "text": text,
                    "audio_features": audio,
                    "video_features": video,
                    "has_video": has_video,
                    "label": label,
                    "speaker": str(vid),
                    "raw_score": sentiment,
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print()

    for split, data in splits.items():
        out_path = OUT_DIR / f"{split}.json"

        labels = [d["label"] for d in data]
        has_video_count = sum(1 for d in data if d["has_video"])

        print(
            f"  {split:6s}: {len(data):6d} samples  "
            f"neg={labels.count(0)} neu={labels.count(1)} pos={labels.count(2)}  "
            f"with_video={has_video_count}/{len(data)}"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    print(f"\n  Skipped videos: {skipped_videos}")
    print(f"  Skipped segments: {skipped_segments}")
    print(f"  Saved to: {OUT_DIR}")


def main():
    print("=" * 60)
    print("  CMU-MOSEI 3-Modal Setup")
    print("  text + audio proxy + video FACET42")
    print("  New download: VisualFacet42 ~1.66 GB")
    print("=" * 60)

    if (OUT_DIR / "train.json").exists():
        with open(OUT_DIR / "train.json", "r", encoding="utf-8") as f:
            n_train = len(json.load(f))

        print(f"Already set up ({n_train} train samples).")
        print("Delete data/cmu-mosei-3modal if you want to rebuild.")
        print("Run: py run_cmu_mosei_3modal.py")
        return

    download()
    convert()

    print("\n" + "=" * 60)
    print("  DONE. CMU-MOSEI 3-modal JSON ready.")
    print("  Run: py patch_cmu_mosei_3modal.py")
    print("  Run: py run_cmu_mosei_3modal.py")
    print("=" * 60)


if __name__ == "__main__":
    main()