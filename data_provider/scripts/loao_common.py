from __future__ import annotations

import csv
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = REPO_ROOT / "data_provider" / "split_json" / "loao"
CACHE_ROOT = REPO_ROOT / ".cache_loao"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "loao_20260404"
RESULT_ROOT = REPO_ROOT / "results" / "loao_20260404"
LOG_ROOT = REPO_ROOT / "logs" / "loao_20260404"

RAW_ROOTS = {
    "CALB": Path("/path/to/your/dataset"),
    "NA-ion": Path("/path/to/your/dataset"),
}

PROCESSED_SOH_ROOTS = {
    "CALB": Path("/path/to/your/processed_SOH"),
    "NA-ion": Path("/path/to/your/processed_SOH"),
}

BASE_ARGS_PATHS = {
    "CALB": Path("/path/to/your/base_checkpoint/CALB/args.json"),
    "NA-ion": Path("/path/to/your/base_checkpoint/NA-ion/args.json"),
}

# Use the strongest settings we have already validated on the current codebase as a starting point.
BASE_ARG_OVERRIDES = {
    "CALB": {
        "model": "BatteryMFormer",
        "learning_rate": 1e-4,
    },
    "NA-ion": {
        "model": "BatteryMFormer",
        "dropout": 0.1,
    },
}

VAL_SPLIT_SEED = 2024
VAL_RATIO = 0.25
WAND_PROJECT = "BattMFormer_paper"


def ensure_dirs() -> None:
    for path in [SPLIT_DIR, CACHE_ROOT, CHECKPOINT_ROOT, RESULT_ROOT, LOG_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def load_name2aging() -> Dict[str, int]:
    with open(REPO_ROOT / "name2agingConditionID.json", "r") as f:
        return json.load(f)


def dataset_files(dataset: str) -> List[str]:
    raw_dir = RAW_ROOTS[dataset] / dataset
    soh_dir = PROCESSED_SOH_ROOTS[dataset] / dataset
    raw_files = {path.name for path in raw_dir.glob("*.pkl")}
    soh_files = {path.name for path in soh_dir.glob("*.pkl")}
    mapped_files = {
        file_name for file_name, cond_id in load_name2aging().items()
        if file_name.startswith(dataset)
    }
    return sorted(raw_files & soh_files & mapped_files)


def condition_groups(dataset: str) -> Dict[int, List[str]]:
    name2aging = load_name2aging()
    groups = defaultdict(list)
    for file_name in dataset_files(dataset):
        groups[int(name2aging[file_name])].append(file_name)
    return {cond_id: sorted(file_names) for cond_id, file_names in sorted(groups.items())}


def split_json_path(dataset: str, condition_id: int, val_seed: int = VAL_SPLIT_SEED) -> Path:
    return SPLIT_DIR / f"{dataset}_loao_cond{condition_id}_seed{val_seed}.json"


def split_tag(dataset: str, condition_id: int, val_seed: int = VAL_SPLIT_SEED) -> str:
    return f"{dataset}_loao_cond{condition_id}_seed{val_seed}"


def checkpoint_dir(dataset: str, condition_id: int) -> Path:
    return CHECKPOINT_ROOT / dataset / f"cond{condition_id}"


def load_base_args(dataset: str) -> Dict:
    with open(BASE_ARGS_PATHS[dataset], "r") as f:
        args = json.load(f)
    merged = deepcopy(args)
    merged.update(BASE_ARG_OVERRIDES.get(dataset, {}))
    merged["model"] = "BatteryMFormer"
    merged["root_path"] = str(RAW_ROOTS[dataset])
    merged["processed_SOH_path"] = str(PROCESSED_SOH_ROOTS[dataset])
    merged["cache_root"] = str(CACHE_ROOT)
    merged["dataset"] = dataset
    merged["input_mode"] = "current_voltage"
    merged["use_capacity_resample"] = True
    merged["resume_existing"] = False
    merged["force_reload"] = False
    merged["use_multi_gpu"] = False
    merged["gpu"] = 0
    merged["num_workers"] = 0
    merged["wandb_project"] = WAND_PROJECT
    merged["eol_threshold"] = 0.9 if dataset == "CALB" else 0.8
    return merged


def loao_tasks() -> List[Dict]:
    tasks = []
    for dataset in ["CALB", "NA-ion"]:
        for condition_id, file_names in condition_groups(dataset).items():
            tasks.append({
                "dataset": dataset,
                "condition_id": condition_id,
                "test_count": len(file_names),
                "split_json": str(split_json_path(dataset, condition_id)),
                "checkpoint_dir": str(checkpoint_dir(dataset, condition_id)),
            })
    return tasks


def manifest_csv_path() -> Path:
    return RESULT_ROOT / "loao_manifest.csv"


def write_manifest(tasks: List[Dict]) -> Path:
    ensure_dirs()
    path = manifest_csv_path()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "condition_id", "test_count", "split_json", "checkpoint_dir"],
        )
        writer.writeheader()
        writer.writerows(tasks)
    return path
