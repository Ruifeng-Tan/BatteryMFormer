#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.scripts.loao_common import (
    SPLIT_DIR,
    VAL_RATIO,
    VAL_SPLIT_SEED,
    condition_groups,
    ensure_dirs,
    loao_tasks,
    split_json_path,
    write_manifest,
)


def build_loao_split(dataset: str, held_out_condition: int, val_seed: int, val_ratio: float):
    groups = condition_groups(dataset)
    if held_out_condition not in groups:
        raise KeyError(f"Held-out condition {held_out_condition} not found in {dataset}")

    rng = random.Random(val_seed + held_out_condition)
    train_files = []
    val_files = []
    test_files = list(groups[held_out_condition])

    for condition_id, file_names in groups.items():
        if condition_id == held_out_condition:
            continue

        candidates = list(file_names)
        rng.shuffle(candidates)

        if len(candidates) <= 1:
            train_files.extend(candidates)
            continue

        val_count = max(1, int(math.floor(len(candidates) * val_ratio)))
        val_count = min(val_count, len(candidates) - 1)
        val_files.extend(sorted(candidates[:val_count]))
        train_files.extend(sorted(candidates[val_count:]))

    split = {
        "dataset": dataset,
        "protocol": "leave_one_aging_condition_out",
        "held_out_condition_id": held_out_condition,
        "val_seed": val_seed,
        "val_ratio": val_ratio,
        "train": sorted(train_files),
        "val": sorted(val_files),
        "test": sorted(test_files),
    }
    return split


def generate_for_dataset(dataset: str, val_seed: int, val_ratio: float):
    generated = []
    for condition_id in condition_groups(dataset).keys():
        split = build_loao_split(dataset, condition_id, val_seed=val_seed, val_ratio=val_ratio)
        out_path = split_json_path(dataset, condition_id, val_seed=val_seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(split, f, indent=2)
        generated.append((condition_id, out_path, len(split["train"]), len(split["val"]), len(split["test"])))
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate LOAO splits for CALB / NA-ion")
    parser.add_argument("--dataset", choices=["CALB", "NA-ion", "all"], default="all")
    parser.add_argument("--val_seed", type=int, default=VAL_SPLIT_SEED)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args()

    ensure_dirs()

    datasets = ["CALB", "NA-ion"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        generated = generate_for_dataset(dataset, val_seed=args.val_seed, val_ratio=args.val_ratio)
        print(f"\n{dataset}: generated {len(generated)} LOAO splits")
        for condition_id, out_path, train_count, val_count, test_count in generated:
            print(
                f"  cond{condition_id}: train={train_count} val={val_count} test={test_count} -> {out_path}"
            )

    manifest_path = write_manifest(loao_tasks())
    print(f"\nManifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
