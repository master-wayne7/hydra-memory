"""
Download LongMemEval's s_cleaned split (500 instances, ~40 sessions / ~115k tokens each,
the exact scale Track 03 names) and filter it down to a small, reproducible validation subset:
the first 4 `knowledge-update` instances and first 4 abstention (`_abs`-suffixed question_id)
instances found in file order. These two categories are the direct match for what this project's
graph mechanism is built to demonstrate (supersession, correct abstention).

Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (MIT licensed).
"""

import json
import os

import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_PATH = os.path.join(DATA_DIR, "longmemeval_s_cleaned.json")
SUBSET_PATH = os.path.join(DATA_DIR, "subset.json")

SOURCE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

TARGET_PER_CATEGORY = 4


def download_raw() -> None:
    if os.path.exists(RAW_PATH):
        print(f"already have {RAW_PATH}")
        return
    print(f"downloading {SOURCE_URL} -> {RAW_PATH} (277MB, this takes a few minutes)...")
    with requests.get(SOURCE_URL, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        downloaded = 0
        with open(RAW_PATH, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
                downloaded += len(chunk)
                print(f"\r  {downloaded / 1024 / 1024:.0f} MB", end="", flush=True)
    print("\ndownload complete")


def build_subset() -> None:
    print(f"loading {RAW_PATH}...")
    with open(RAW_PATH, encoding="utf-8") as fh:
        instances = json.load(fh)
    print(f"  {len(instances)} total instances")

    knowledge_update = [i for i in instances if i["question_type"] == "knowledge-update"][:TARGET_PER_CATEGORY]
    abstention = [i for i in instances if i["question_id"].endswith("_abs")][:TARGET_PER_CATEGORY]

    subset = knowledge_update + abstention
    print(f"  selected {len(knowledge_update)} knowledge-update + {len(abstention)} abstention = {len(subset)}")

    with open(SUBSET_PATH, "w", encoding="utf-8") as fh:
        json.dump(subset, fh, indent=2)
    print(f"wrote {SUBSET_PATH}")


if __name__ == "__main__":
    download_raw()
    build_subset()
