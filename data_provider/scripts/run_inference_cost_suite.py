#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MEASURE_SCRIPT = SCRIPT_DIR / "measure_inference_cost.py"
TABLE_DIR = REPO_ROOT / "results" / "table"


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference-cost benchmarks for completed result-table checkpoints")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sample_limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--datasets", type=str, default="CALB,NA-ion")
    return parser.parse_args()


def load_rows(csv_path: Path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    out_dir = Path(args.output_dir) if args.output_dir else (TABLE_DIR / "inference_cost")
    out_dir = out_dir / args.device
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if "CALB" in datasets:
        rows.extend(load_rows(TABLE_DIR / "calb_main_ranking_best_available.csv"))
    if "NA-ion" in datasets:
        rows.extend(load_rows(TABLE_DIR / "naion_main_ranking_best_available.csv"))

    summary_rows = []
    for row in rows:
        dataset = row["dataset"]
        model = row["model"]
        ckpt = row["example_checkpoint"]
        out_json = out_dir / f"{dataset}_{model}.json"
        cmd = [
            sys.executable,
            str(MEASURE_SCRIPT),
            "--checkpoint_dir",
            ckpt,
            "--device",
            args.device,
            "--sample_limit",
            str(args.sample_limit),
            "--warmup",
            str(args.warmup),
            "--output_json",
            str(out_json),
        ]
        if args.device == "cuda":
            cmd.extend(["--gpu_id", str(args.gpu_id)])

        print("Running:", " ".join(cmd))
        status = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if status != 0:
            print(f"FAILED: {dataset} {model}", file=sys.stderr)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "device": args.device,
                    "status": "failed",
                    "checkpoint_dir": ckpt,
                }
            )
            continue

        result = json.load(open(out_json))
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "device": args.device,
                "status": "ok",
                "cpu_name": result.get("cpu_name", ""),
                "gpu_name": result.get("gpu_name", ""),
                "avg_latency_ms_per_sample": f"{result['avg_latency_ms_per_sample']:.6f}",
                "single_sample_total_ms_mean": f"{result.get('single_sample_total_ms_mean', result['avg_latency_ms_per_sample']):.6f}",
                "single_sample_total_ms_median": f"{result.get('single_sample_total_ms_median', result['avg_latency_ms_per_sample']):.6f}",
                "single_sample_total_ms_p95": f"{result.get('single_sample_total_ms_p95', result['avg_latency_ms_per_sample']):.6f}",
                "timed_samples": result["timed_samples"],
                "gpu_alloc_after_model_load_mb": f"{result.get('gpu_alloc_after_model_load_mb', 0.0):.6f}",
                "gpu_reserved_after_model_load_mb": f"{result.get('gpu_reserved_after_model_load_mb', 0.0):.6f}",
                "peak_gpu_allocated_mb": f"{result['peak_gpu_allocated_mb']:.6f}",
                "peak_gpu_extra_allocated_mb": f"{result.get('peak_gpu_extra_allocated_mb', 0.0):.6f}",
                "peak_gpu_total_including_model_mb": f"{result.get('peak_gpu_total_including_model_mb', result['peak_gpu_allocated_mb']):.6f}",
                "peak_rss_mb": f"{result['peak_rss_mb']:.6f}",
                "peak_rss_delta_mb": f"{result['peak_rss_delta_mb']:.6f}",
                "checkpoint_dir": ckpt,
            }
        )

    summary_csv = out_dir / f"summary_{args.device}.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"Saved summary to {summary_csv}")


if __name__ == "__main__":
    main()
