#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.scripts.loao_common import CHECKPOINT_ROOT, CACHE_ROOT, split_json_path, split_tag


RUN_SCRIPT = REPO_ROOT / "run_main.py"
EVAL_SCRIPT = REPO_ROOT / "evaluate_model.py"
MANIFEST_PATH = REPO_ROOT / "data_provider" / "scripts" / "organized_loao_params" / "manifest.json"
RAW_ROOTS = {
    "CALB": "/path/to/your/dataset",
    "NA-ion": "/path/to/your/dataset",
}
PROCESSED_ROOTS = {
    "CALB": "/path/to/your/processed_SOH",
    "NA-ion": "/path/to/your/processed_SOH",
}


def load_manifest():
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def load_trial_args(model: str, dataset: str, condition_id: int) -> dict:
    manifest = load_manifest()
    entry = manifest["models"][model][dataset]
    if not entry["available"]:
        raise SystemExit(f"No parameter source available for {model} / {dataset}")

    with open(entry["copied_args"], "r") as f:
        args = json.load(f)

    ckpt_dir = CHECKPOINT_ROOT / model / dataset / f"cond{condition_id}"
    args["model"] = model
    args["dataset"] = dataset
    args["root_path"] = RAW_ROOTS[dataset]
    args["processed_SOH_path"] = PROCESSED_ROOTS[dataset]
    args["cache_root"] = str(CACHE_ROOT)
    args["checkpoints"] = str(ckpt_dir)
    args["split_json_path"] = str(split_json_path(dataset, condition_id))
    args["split_tag"] = split_tag(dataset, condition_id)
    args["eol_threshold"] = 0.9 if dataset == "CALB" else 0.8
    args["use_multi_gpu"] = False
    args["resume_existing"] = False
    args["force_reload"] = False
    args["num_workers"] = 0
    args["wandb_project"] = "BattMFormer_paper"
    args["wandb_group"] = f"loao_20260404_{model}_{dataset}"
    args["wandb_name"] = f"{model}_{dataset}_cond{condition_id}"
    args["wandb_tags"] = ",".join(["loao_20260404", model, dataset, f"cond{condition_id}"])
    return args


def load_trial_args_from_source(model: str, dataset: str, condition_id: int, source_args_path: str, checkpoint_root: str, tag: str) -> dict:
    with open(source_args_path, "r") as f:
        args = json.load(f)

    ckpt_dir = Path(checkpoint_root) / model / dataset / tag / f"cond{condition_id}"
    args["model"] = model
    args["dataset"] = dataset
    args["root_path"] = RAW_ROOTS[dataset]
    args["processed_SOH_path"] = PROCESSED_ROOTS[dataset]
    args["cache_root"] = str(CACHE_ROOT)
    args["checkpoints"] = str(ckpt_dir)
    args["split_json_path"] = str(split_json_path(dataset, condition_id))
    args["split_tag"] = split_tag(dataset, condition_id)
    args["eol_threshold"] = 0.9 if dataset == "CALB" else 0.8
    args["use_multi_gpu"] = False
    args["resume_existing"] = False
    args["force_reload"] = False
    args["num_workers"] = 0
    args["wandb_project"] = "BattMFormer_paper"
    args["wandb_group"] = f"loao_tune_20260405_{model}_{dataset}_{tag}"
    args["wandb_name"] = f"{model}_{dataset}_{tag}_cond{condition_id}"
    args["wandb_tags"] = ",".join(["loao_tune_20260405", model, dataset, tag, f"cond{condition_id}"])
    return args


def format_run_args(args: dict) -> list[str]:
    bool_flags = {
        "use_amp",
        "use_capacity_resample",
        "use_grad_clip",
        "use_cycle_encode",
        "channel_mixing",
        "no_decomposition",
        "use_multi_gpu",
        "resume_existing",
        "force_reload",
    }
    ordered_keys = [
        "model", "dataset", "root_path", "processed_SOH_path", "checkpoints",
        "input_mode", "split_json_path", "split_tag", "prompt_embeddings_path", "structured_metadata_path", "structured_embed_dim", "batch_size", "train_epochs",
        "learning_rate", "lradj", "warmup_epochs", "weight_decay", "patience",
        "d_model", "n_heads", "e_layers", "e_layers2", "d_layers", "d_ff",
        "dropout", "activation", "factor", "seq_len", "pred_len", "eol_threshold",
        "truncate_start_cycle", "early_cycle_threshold", "charge_discharge_length",
        "task_name", "d_llm", "cache_root", "gpu", "seed", "use_amp",
        "use_capacity_resample", "num_query", "accumulation_steps", "lambda_recovery",
        "num_slots", "temperature", "top_k", "lambda_mem", "num_segments",
        "lambda_life_loss", "k_dim", "enc_in", "kernel_size", "cnn_channels", "stride",
        "d_ffs", "moving_avg", "patch_len", "patch_ks", "patch_sd", "dw_ks",
        "padding_patch", "re_param", "re_param_kernel", "enable_res_param",
        "head_dropout", "revin", "affine", "subtract_last", "deformable",
        "context", "horizon", "num_workers", "wandb_project", "wandb_group",
        "wandb_name", "wandb_tags",
    ]

    cli = []
    for key in ordered_keys:
        if key not in args:
            continue
        value = args[key]
        if key in bool_flags:
            if value:
                cli.append(f"--{key}")
            continue
        if value is None or value == "":
            continue
        cli.extend([f"--{key}", str(value)])
    return cli


def outputs_complete(ckpt_dir: Path) -> bool:
    return (
        (ckpt_dir / "training_history.pkl").exists()
        and (ckpt_dir / "seen_unseen_results.json").exists()
    )


def run_command(cmd: list[str], dry_run: bool) -> int:
    print("Command:")
    print(" ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main():
    parser = argparse.ArgumentParser(description="Run one LOAO fold for any supported model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["CALB", "NA-ion"], required=True)
    parser.add_argument("--condition_id", type=int, required=True)
    parser.add_argument("--source_args", type=str, default="", help="Explicit args.json path for tuning runs")
    parser.add_argument("--checkpoint_root", type=str, default="", help="Explicit checkpoint root for tuning runs")
    parser.add_argument("--tag", type=str, default="", help="Experiment tag for tuning runs")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.source_args:
        if not args.checkpoint_root or not args.tag:
            raise SystemExit("--source_args requires --checkpoint_root and --tag")
        trial_args = load_trial_args_from_source(
            args.model, args.dataset, args.condition_id, args.source_args, args.checkpoint_root, args.tag
        )
    else:
        trial_args = load_trial_args(args.model, args.dataset, args.condition_id)
    ckpt_dir = Path(trial_args["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "loao_spec.json", "w") as f:
        json.dump(trial_args, f, indent=2)

    if args.skip_existing and outputs_complete(ckpt_dir):
        print(f"Skipping completed fold: {args.model} {args.dataset} cond{args.condition_id}")
        return 0

    train_cmd = [sys.executable, str(RUN_SCRIPT)] + format_run_args(trial_args)
    eval_cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--args_path",
        str(ckpt_dir),
        "--target_dataset",
        args.dataset,
    ]

    status = run_command(train_cmd, args.dry_run)
    if status != 0:
        return status
    status = run_command(eval_cmd, args.dry_run)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
