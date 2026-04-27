#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch
from safetensors.torch import load_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_factory import data_provider_soh
from run_main import get_model, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Measure single-sample inference cost for one trained checkpoint")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory containing args.json and model.safetensors")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--sample_limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def load_namespace(ckpt_dir: Path) -> SimpleNamespace:
    args_path = ckpt_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.json in {ckpt_dir}")
    args_dict = json.load(open(args_path))
    args_dict["batch_size"] = 1
    args_dict["num_workers"] = 0
    args_dict["resume_existing"] = False
    args_dict["force_reload"] = False
    args_dict["use_multi_gpu"] = False
    return SimpleNamespace(**args_dict)


def build_model(args: SimpleNamespace, ckpt_dir: Path, device: torch.device) -> torch.nn.Module:
    model = get_model(args)
    state_path = ckpt_dir / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors in {ckpt_dir}")
    state_dict = load_file(str(state_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: missing keys={len(missing)}, unexpected keys={len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def prepare_loader(args: SimpleNamespace, split: str):
    train_set, _ = data_provider_soh(args, "train", args.input_mode)
    label_scaler = getattr(train_set, "label_scaler", None)
    _, loader = data_provider_soh(args, split, args.input_mode, label_scaler)
    return loader


def get_cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def get_gpu_name(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    try:
        return torch.cuda.get_device_name(device)
    except Exception:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={device.index or 0}",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            return out
        except Exception:
            return "unknown"


def move_item(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x


def run_model(args: SimpleNamespace, model: torch.nn.Module, batch, device: torch.device):
    cycle_curve_data, curve_attn_mask, soh_input, soh_trajectory, trajectory_mask, aging_condition_embedding, soc_curves, cycle_level_features, life_labels, file_names = batch
    cycle_curve_data = move_item(cycle_curve_data, device)
    curve_attn_mask = move_item(curve_attn_mask, device)
    soh_input = move_item(soh_input, device)
    soh_trajectory = move_item(soh_trajectory, device)
    trajectory_mask = move_item(trajectory_mask, device)
    aging_condition_embedding = move_item(aging_condition_embedding, device)
    soc_curves = move_item(soc_curves, device)
    cycle_level_features = move_item(cycle_level_features, device)
    life_labels = move_item(life_labels, device)

    if args.input_mode == "current_voltage":
        outputs = model(
            cycle_curve_data=cycle_curve_data,
            curve_attn_mask=curve_attn_mask,
            aging_condition_embedding=aging_condition_embedding,
            soh_trajectory=soh_trajectory,
            trajectory_mask=trajectory_mask,
            soc_input=soc_curves,
            soh_input=soh_input,
            cycle_level_features=cycle_level_features,
            life_labels=life_labels,
        )
    elif args.input_mode == "capacity_increment":
        _, outputs, _ = model(capacity_increment=cycle_curve_data, capacity_mask=curve_attn_mask)
    else:
        outputs = model(soh_input=soh_input)

    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs


def benchmark(args: SimpleNamespace, model: torch.nn.Module, batches, device: torch.device, warmup: int):
    process = psutil.Process()
    rss_before = process.memory_info().rss / (1024 ** 2)
    peak_rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    gpu_alloc_after_model_mb = 0.0
    gpu_reserved_after_model_mb = 0.0
    gpu_peak_total_including_model_mb = 0.0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        gpu_alloc_after_model_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        gpu_reserved_after_model_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
        gpu_peak_total_including_model_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    warmup_batches = batches[:warmup]
    timed_batches = batches[warmup:]

    with torch.inference_mode():
        for batch in warmup_batches:
            _ = run_model(args, model, batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        sample_timings_ms = []
        start = time.perf_counter()
        for batch in timed_batches:
            sample_start = time.perf_counter()
            _ = run_model(args, model, batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sample_timings_ms.append((time.perf_counter() - sample_start) * 1000.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()

    timed_count = max(len(timed_batches), 1)
    avg_ms = (end - start) * 1000.0 / timed_count

    rss_after = process.memory_info().rss / (1024 ** 2)
    peak_rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    peak_gpu_mb = 0.0
    peak_gpu_extra_mb = 0.0
    if device.type == "cuda":
        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_gpu_extra_mb = max(0.0, peak_gpu_mb - gpu_alloc_after_model_mb)
        gpu_peak_total_including_model_mb = max(
            gpu_peak_total_including_model_mb,
            gpu_alloc_after_model_mb + peak_gpu_extra_mb,
        )

    return {
        "avg_latency_ms_per_sample": avg_ms,
        "single_sample_total_ms_mean": statistics.mean(sample_timings_ms) if sample_timings_ms else avg_ms,
        "single_sample_total_ms_median": statistics.median(sample_timings_ms) if sample_timings_ms else avg_ms,
        "single_sample_total_ms_p95": (
            sorted(sample_timings_ms)[min(len(sample_timings_ms) - 1, max(0, int(len(sample_timings_ms) * 0.95) - 1))]
            if sample_timings_ms
            else avg_ms
        ),
        "timed_samples": timed_count,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "peak_rss_mb": peak_rss_after,
        "peak_rss_delta_mb": peak_rss_after - peak_rss_before,
        "gpu_alloc_after_model_load_mb": gpu_alloc_after_model_mb,
        "gpu_reserved_after_model_load_mb": gpu_reserved_after_model_mb,
        "peak_gpu_allocated_mb": peak_gpu_mb,
        "peak_gpu_extra_allocated_mb": peak_gpu_extra_mb,
        "peak_gpu_total_including_model_mb": gpu_peak_total_including_model_mb,
    }


def main():
    cli = parse_args()
    ckpt_dir = Path(cli.checkpoint_dir).resolve()
    args = load_namespace(ckpt_dir)
    set_seed(getattr(args, "seed", 42))

    if cli.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        device = torch.device(f"cuda:{cli.gpu_id}")
    else:
        device = torch.device("cpu")

    model = build_model(args, ckpt_dir, device)
    loader = prepare_loader(args, cli.split)
    batches = []
    needed = cli.warmup + cli.sample_limit
    for idx, batch in enumerate(loader):
        batches.append(batch)
        if len(batches) >= needed:
            break
    if len(batches) <= cli.warmup:
        raise SystemExit(f"Not enough batches collected for warmup={cli.warmup}, got {len(batches)}")

    metrics = benchmark(args, model, batches, device, cli.warmup)
    result = {
        "checkpoint_dir": str(ckpt_dir),
        "model": args.model,
        "dataset": args.dataset,
        "input_mode": args.input_mode,
        "device": str(device),
        "cpu_name": get_cpu_name(),
        "gpu_name": get_gpu_name(device),
        "split": cli.split,
        "sample_limit": cli.sample_limit,
        "warmup": cli.warmup,
        **metrics,
    }

    print(json.dumps(result, indent=2))
    if cli.output_json:
        out_path = Path(cli.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
