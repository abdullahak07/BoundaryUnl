"""
setup_cmu_mosei.py

Lightweight CMU-MOSEI setup for BoundaryUnl / MAF-style JSON.

Downloads only:
  - CMU_MOSEI_Labels
  - CMU_MOSEI_TimestampedWords

Skips:
  - COVAREP acoustic file (~11.59 GB)
  - visual files

Output:
  data/cmu-mosei/train.json
  data/cmu-mosei/val.json
  data/cmu-mosei/test.json

Run:
  py -m pip install --upgrade kaggle h5py numpy
  py setup_cmu_mosei.py
"""

import json
import random
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np


DATASET = "samarwarsi/cmu-mosei"
OUT_DIR = Path("data/cmu-mosei")
RAW_DIR = OUT_DIR / "raw"


def check_kaggle_key():
    key_path = Path.home() / ".kaggle" / "kaggle.json"
    if not key_path.exists():
        raise FileNotFoundError(
            f"Kaggle key not found: {key_path}\n\n"
            "Fix:\n"
            "1. Go to Kaggle > Account > Create New API Token\n"
            "2. Put kaggle.json here:\n"
            f"   {key_path}"
        )
    print(f"  Kaggle key found: {key_path}")


def get_kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def list_dataset_files(api):
    print("\nListing Kaggle dataset files...")
    files = api.dataset_list_files(DATASET).files
    names = [f.name for f in files]

    print("\nAvailable files:")
    for name in names:
        print(f"  - {name}")

    return names


def choose_file(names, kind):
    """
    Select exact Kaggle file path from dataset file list.
    Kaggle paths may look like:
      CMU-MOSEI/labels/CMU_MOSEI_Labels.csd
      CMU-MOSEI/languages/CMU_MOSEI_TimestampedWords.csd
    or may display without extension in UI.
    """
    lowered = [(n, n.lower()) for n in names]

    if kind == "labels":
        candidates = [
            n for n, low in lowered
            if "label" in low and "cmu_mosei" in low
        ]
    elif kind == "words":
        candidates = [
            n for n, low in lowered
            if (
                "language" in low
                and "timestamped" in low
                and "word" in low
                and "vector" not in low
                and "glove" not in low
                and "phoneme" not in low
            )
        ]

        # fallback if folder name is not included
        if not candidates:
            candidates = [
                n for n, low in lowered
                if (
                    "timestamped" in low
                    and "word" in low
                    and "vector" not in low
                    and "glove" not in low
                    and "phoneme" not in low
                )
            ]
    else:
        raise ValueError(f"Unknown file kind: {kind}")

    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-detect {kind} file.\n"
            "Copy the printed file list and send it to me."
        )

    # Prefer .csd if available.
    candidates = sorted(candidates, key=lambda x: (not x.lower().endswith(".csd"), len(x)))

    selected = candidates[0]
    print(f"\nSelected {kind} file:")
    print(f"  {selected}")
    return selected


def find_local_file_by_name(target_name):
    matches = list(RAW_DIR.rglob(target_name))
    if matches:
        return matches[0]

    # Kaggle UI may hide .csd extension, so search by stem too.
    target_stem = Path(target_name).stem.lower()
    for p in RAW_DIR.rglob("*"):
        if p.is_file() and p.stem.lower() == target_stem:
            return p

    return None


def download_one(api, kaggle_file_path):
    """
    Download one exact file from Kaggle dataset.
    Works even when Kaggle stores it inside CMU-MOSEI/... subfolders.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    target_name = Path(kaggle_file_path).name
    local_existing = find_local_file_by_name(target_name)
    if local_existing and local_existing.stat().st_size > 0:
        print(f"  Already exists: {local_existing}")
        return local_existing

    print(f"\nDownloading:")
    print(f"  {kaggle_file_path}")

    api.dataset_download_file(
        DATASET,
        file_name=kaggle_file_path,
        path=str(RAW_DIR),
        force=False,
        quiet=False,
    )

    # Kaggle usually downloads a zip for single file.
    zips = list(RAW_DIR.glob("*.zip"))
    for z in zips:
        print(f"  Extracting {z.name} ...")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(RAW_DIR)
        z.unlink()

    found = find_local_file_by_name(target_name)
    if found is None:
        print("\nFiles currently in raw folder:")
        for p in RAW_DIR.rglob("*"):
            if p.is_file():
                print(f"  - {p}")
        raise FileNotFoundError(f"Download finished but could not find: {target_name}")

    final_path = RAW_DIR / target_name
    if found.resolve() != final_path.resolve():
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, final_path)

    print(f"  Saved as: {final_path}")
    return final_path


def read_csd(csd_path):
    """
    Read CMU SDK .csd HDF5 file.

    Handles both structures:
      /data/<video_id>/features
      /data/<video_id>/intervals

    and Kaggle MOSEI structure:
      /All Labels/data/<video_id>/features
      /CMU_MOSEI_TimestampedWords/data/<video_id>/features
      or similar nested groups.
    """
    data = {}

    print(f"\nReading CSD: {csd_path}")

    def find_data_group(h5file):
        # Case 1: direct /data group
        if "data" in h5file:
            return h5file["data"]

        # Case 2: nested /<sequence_name>/data group
        for key in h5file.keys():
            obj = h5file[key]
            if hasattr(obj, "keys") and "data" in obj:
                print(f"  Found nested data group under: {key}")
                return obj["data"]

        # Case 3: recursive search
        found = []

        def visitor(name, obj):
            if name.endswith("/data") and hasattr(obj, "keys"):
                found.append(obj)

        h5file.visititems(visitor)

        if found:
            print("  Found recursive data group.")
            return found[0]

        print("Top-level keys:", list(h5file.keys()))
        raise KeyError(f"No 'data' group found in {csd_path}")

    with h5py.File(csd_path, "r") as f:
        data_group = find_data_group(f)

        for vid in data_group.keys():
            group = data_group[vid]

            if "features" not in group or "intervals" not in group:
                continue

            data[vid] = {
                "features": group["features"][()],
                "intervals": group["intervals"][()],
            }

    print(f"  Loaded videos: {len(data)}")
    return data


def decode_word(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "ignore")

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        return decode_word(x.flatten()[0])

    if isinstance(x, (list, tuple)):
        if not x:
            return ""
        return decode_word(x[0])

    return str(x)


def score_to_label(score):
    if score < -0.5:
        return 0
    if score > 0.5:
        return 2
    return 1


def text_to_features(text, dim=74):
    """
    Lightweight second modality so old code expecting audio_features works.

    NOTE:
    This is not real COVAREP acoustic data.
    It is a text-derived numeric vector because we intentionally skip
    the 11.59 GB COVAREP file.
    """
    vec = np.zeros(dim, dtype=np.float32)

    words = text.split()
    vec[0] = len(words)
    vec[1] = len(text)
    vec[2] = np.mean([len(w) for w in words]) if words else 0.0
    vec[3] = text.count("!")
    vec[4] = text.count("?")
    vec[5] = sum(1 for c in text if c.isupper())
    vec[6] = sum(1 for c in text if c.isdigit())

    for w in words:
        idx = 7 + (abs(hash(w.lower())) % (dim - 7))
        vec[idx] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.astype(float).tolist()


def build_video_splits(video_ids, seed=42):
    vids = sorted(video_ids)
    random.Random(seed).shuffle(vids)

    n = len(vids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = set(vids[:n_train])
    val = set(vids[n_train:n_train + n_val])
    test = set(vids[n_train + n_val:])

    return train, val, test


def convert(labels_path, words_path):
    print("\nConverting CMU-MOSEI to JSON...")

    labels_d = read_csd(labels_path)
    words_d = read_csd(words_path)

    common_vids = sorted(set(labels_d.keys()) & set(words_d.keys()))
    if not common_vids:
        raise RuntimeError("No overlapping video IDs between labels and words.")

    print(f"\nCommon videos with labels + words: {len(common_vids)}")

    train_vids, val_vids, test_vids = build_video_splits(common_vids)

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    total = 0
    skipped = 0

    for vid in common_vids:
        if vid in train_vids:
            split = "train"
        elif vid in val_vids:
            split = "val"
        else:
            split = "test"

        label_feats = labels_d[vid]["features"]
        label_ints = labels_d[vid]["intervals"]

        word_feats = words_d[vid]["features"]
        word_ints = words_d[vid]["intervals"]

        for i, (seg, feat) in enumerate(zip(label_ints, label_feats)):
            total += 1

            try:
                s0 = float(seg[0])
                s1 = float(seg[1])
                score = float(np.asarray(feat).flatten()[0])
            except Exception:
                skipped += 1
                continue

            words = []

            for wint, wfeat in zip(word_ints, word_feats):
                try:
                    w0 = float(wint[0])
                    w1 = float(wint[1])
                except Exception:
                    continue

                if w0 >= s0 and w1 <= s1 + 0.05:
                    word = decode_word(wfeat).strip()
                    if word and word.lower() not in {
                        "sp",
                        "sil",
                        "punctuation",
                        "puncuation",
                    }:
                        words.append(word)

            text = " ".join(words).strip()
            if not text:
                text = "sentiment utterance"

            item = {
                "id": f"{vid}_{i}",
                "text": text,
                "audio_features": text_to_features(text, dim=74),
                "label": int(score_to_label(score)),
                "speaker": vid,
                "raw_score": float(score),
                "start": s0,
                "end": s1,
                "note": "audio_features are text-derived; COVAREP skipped",
            }

            splits[split].append(item)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nConverted segments: {total - skipped}")
    print(f"Skipped segments: {skipped}")

    for split, rows in splits.items():
        labels = [r["label"] for r in rows]
        print(
            f"  {split}: {len(rows)} samples | "
            f"neg={labels.count(0)} neu={labels.count(1)} pos={labels.count(2)}"
        )

        out_file = OUT_DIR / f"{split}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'train.json'}")
    print(f"  {OUT_DIR / 'val.json'}")
    print(f"  {OUT_DIR / 'test.json'}")


def main():
    print("=======================================================")
    print("  CMU-MOSEI Lightweight Setup")
    print("  Downloads labels + timestamped words only")
    print("  Skips COVAREP and visuals")
    print("=======================================================")

    if (
        (OUT_DIR / "train.json").exists()
        and (OUT_DIR / "val.json").exists()
        and (OUT_DIR / "test.json").exists()
    ):
        print("\nCMU-MOSEI already converted.")
        print("Delete data/cmu-mosei if you want to rebuild.")
        return

    print("\nStep 1: Checking Kaggle...")
    check_kaggle_key()

    api = get_kaggle_api()

    names = list_dataset_files(api)
    labels_file = choose_file(names, "labels")
    words_file = choose_file(names, "words")

    print("\nStep 2: Downloading selected files...")
    labels_path = download_one(api, labels_file)
    words_path = download_one(api, words_file)

    print("\nStep 3: Convert...")
    convert(labels_path, words_path)

    print("\nDONE.")
    print("Run:")
    print("  py main.py --dataset cmu-mosei --mode full")


if __name__ == "__main__":
    main()