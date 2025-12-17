#!/usr/bin/env python
"""
Build Ekman-style 7-class IEMOCAP CSV files from Tae Kim's
EmoBERTa raw-text splits.

Source:
  https://github.com/tae898/multimodal-datasets/tree/main/IEMOCAP/raw-texts

The raw-texts folder already contains the official train/val/test split
used by EmoBERTa, encoded as:
  IEMOCAP/raw-texts/train/*.json
  IEMOCAP/raw-texts/val/*.json
  IEMOCAP/raw-texts/test/*.json

Each JSON file corresponds to a single utterance and contains at least:
  - "Utterance": text string
  - "Emotion": original IEMOCAP-style label
  - "Speaker": "Female" / "Male"
  - "SessionID": e.g. "Ses05"

Goal:
  1. Read those JSON files.
  2. Keep the EmoBERTa splits (train/val/test).
  3. Map the original 11 IEMOCAP emotions into a 7-class Ekman-like
     label space compatible with a GoEmotions-style head:

        {anger, joy, sadness, neutral, surprise, fear, disgust}

  4. Drop samples with emotions that cannot be mapped:
        {undecided, other}

Output:
  - iemocap_ekman7_emoberta_all.csv
  - iemocap_ekman7_emoberta_train.csv
  - iemocap_ekman7_emoberta_val.csv
  - iemocap_ekman7_emoberta_test.csv

Columns:
  Split, Dialogue_ID, Utterance_ID, Speaker, Utterance, Emotion, Original_Emotion
"""

import os
import json
import argparse
from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------
# Emotion mapping
# ---------------------------------------------------------------------
# The raw JSONs contain 11 IEMOCAP emotion labels:
#   {undecided, frustration, neutral, anger, sadness,
#    excited, happiness, surprise, fear, other, disgust}
#
# We map these into a 7-class Ekman-style space:
#   {anger, joy, sadness, neutral, surprise, fear, disgust}
#
# Rules:
#   - anger, frustration        -> anger
#   - sadness                   -> sadness
#   - neutral                   -> neutral
#   - happiness, excited        -> joy
#   - surprise                  -> surprise
#   - fear                      -> fear
#   - disgust                   -> disgust
#
# We drop:
#   - undecided                 -> no clear target class
#   - other                     -> heterogeneous / noisy
#
MAP_11_TO_7: Dict[str, str] = {
    "anger": "anger",
    "frustration": "anger",

    "sadness": "sadness",

    "neutral": "neutral",

    "happiness": "joy",
    "excited": "joy",

    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust",
    # "undecided" and "other" are intentionally not mapped (dropped)
}


def load_split(split_dir: str, split_name: str) -> List[Dict]:
    """
    Load all JSON utterances from a split directory (train/val/test).

    Each JSON file corresponds to one utterance and is named:
      <Utterance_ID>.json

    Example:
      Ses05F_impro01_F000.json

    We construct:
      - Utterance_ID: filename without extension
      - Dialogue_ID: all parts before the last underscore
                     e.g. "Ses05F_impro01_F000" -> "Ses05F_impro01"
                     e.g. "Ses05M_script01_1_F000" -> "Ses05M_script01_1"
    """
    rows: List[Dict] = []

    if not os.path.isdir(split_dir):
        print(f"[WARN] Split directory not found: {split_dir}")
        return rows

    print(f"[INFO] Processing split='{split_name}' in {split_dir}")

    for fname in os.listdir(split_dir):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(split_dir, fname)

        # Utterance_ID = file name without extension
        utt_id = os.path.splitext(fname)[0]

        # Dialogue_ID = all components before the last underscore
        parts = utt_id.split("_")
        if len(parts) < 2:
            # Unexpected name, skip
            print(f"[WARN] Unexpected utt_id format: {utt_id}")
            continue
        dialogue_id = "_".join(parts[:-1])

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = str(data.get("Utterance", "")).strip()
        orig_emotion = str(data.get("Emotion", "")).strip().lower()
        speaker_raw = str(data.get("Speaker", "")).strip()

        # Discard emotions that we do not map into the 7-class space
        if orig_emotion not in MAP_11_TO_7:
            # Most often: "undecided" or "other"
            continue

        ekman_emotion = MAP_11_TO_7[orig_emotion]

        # Map "Female"/"Male" -> "F"/"M"
        s_lower = speaker_raw.lower()
        if s_lower.startswith("f"):
            speaker = "F"
        elif s_lower.startswith("m"):
            speaker = "M"
        else:
            # Fallback: keep whatever is there
            speaker = speaker_raw

        rows.append(
            {
                "Split": split_name,       # train / val / test
                "Dialogue_ID": dialogue_id,
                "Utterance_ID": utt_id,
                "Speaker": speaker,
                "Utterance": text,
                "Emotion": ekman_emotion,          # final 7-class label
                "Original_Emotion": orig_emotion,  # for reference/debugging
            }
        )

    return rows


def build_iemocap_ekman7(raw_text_root: str, out_dir: str) -> None:
    """
    Build Ekman-7 CSVs from Tae Kim's IEMOCAP raw-text splits.

    Parameters
    ----------
    raw_text_root : str
        Path to IEMOCAP/raw-texts, which must contain the folders
        'train', 'val', and 'test'.
    out_dir : str
        Output directory where CSV files will be stored.
    """
    splits = ["train", "val", "test"]
    all_rows: List[Dict] = []

    for split in splits:
        split_dir = os.path.join(raw_text_root, split)
        rows = load_split(split_dir, split)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(
            f"No utterances were loaded. Check RAW_TEXT_ROOT: {raw_text_root}"
        )

    df = pd.DataFrame(all_rows)

    # Basic statistics
    print("\n[INFO] Total utterances after 7-class mapping:", len(df))
    print("\n[INFO] Label distribution (7-class):")
    print(df["Emotion"].value_counts())

    print("\n[INFO] Split sizes (rows per split):")
    print(df["Split"].value_counts())

    print("\n[INFO] Split x Emotion:")
    print(df.groupby(["Split", "Emotion"]).size())

    os.makedirs(out_dir, exist_ok=True)

    # Save full CSV
    all_path = os.path.join(out_dir, "iemocap_ekman7_emoberta_all.csv")
    df.to_csv(all_path, index=False, encoding="utf-8")
    print(f"\n[INFO] Saved full dataset -> {all_path}")

    # Save per-split CSVs
    for split in splits:
        split_df = df[df["Split"] == split].copy()
        out_path = os.path.join(
            out_dir, f"iemocap_ekman7_emoberta_{split}.csv"
        )
        split_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[INFO] Saved {split} split -> {out_path} ({len(split_df)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Ekman-style 7-class IEMOCAP CSVs "
            "from EmoBERTa raw-text splits."
        )
    )
    parser.add_argument(
        "--raw_text_root",
        type=str,
        required=True,
        help=(
            "Path to IEMOCAP/raw-texts directory from the "
            "multimodal-datasets repo. "
            "This folder must contain 'train', 'val', 'test'."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory for CSV files (default: current directory).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_iemocap_ekman7(args.raw_text_root, args.out_dir)
