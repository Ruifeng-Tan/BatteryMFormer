#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

# This script only needs text embedding. On this machine, torchvision is installed
# but mismatched with the active PyTorch in `batterylife`, which makes modern
# transformers fail while importing optional vision helpers. Hide torchvision from
# transformers' package-availability probe so the text-only Qwen embedding model
# can be loaded without touching the training environment.
_orig_find_spec = importlib.util.find_spec


def _patched_find_spec(name, *args, **kwargs):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _orig_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _patched_find_spec

from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Prompts.Mapping_helper import Mapping_helper


RAW_ROOTS = {
    "CALB": Path("/path/to/your/dataset/CALB"),
    "NA-ion": Path("/path/to/your/dataset/NA-ion"),
}

DEFAULT_LLM_PATH = "/path/to/your/llm/Qwen3-Embedding-0.6B"
DEFAULT_BASE_EMBED = REPO_ROOT / "data_provider" / "prompt_embeddings" / "Qwen3_total.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prompt-embedding variants for template robustness")
    parser.add_argument("--variant", choices=["condition_first", "field_shuffle", "mild_shuffle"], required=True)
    parser.add_argument("--datasets", default="CALB,NA-ion")
    parser.add_argument("--llm_path", default=DEFAULT_LLM_PATH)
    parser.add_argument("--base_embeddings", default=str(DEFAULT_BASE_EMBED))
    parser.add_argument("--output_pickle", required=True)
    parser.add_argument("--output_prompts_json", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:2")
    return parser.parse_args()


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def split_sentences(section_text: str):
    parts = [x.strip() for x in re.split(r'(?<=\.)\s+', section_text.strip()) if x.strip()]
    return parts


def reorder_sentences(sentences):
    if len(sentences) <= 2:
        return list(reversed(sentences))
    odds = sentences[1::2]
    evens = sentences[::2]
    return odds + evens


def apply_variant(protocol_prompt: str, variant: str) -> str:
    text = protocol_prompt.strip()
    if "Battery specifications:" not in text or "Operating condition:" not in text:
        return text
    _, after_spec = text.split("Battery specifications:", 1)
    spec_body, op_body = after_spec.split("Operating condition:", 1)
    spec_body = spec_body.strip()
    op_body = op_body.strip()

    spec_sentences = split_sentences(spec_body)
    op_sentences = split_sentences(op_body)

    if variant == "condition_first":
        return f"Operating condition: {' '.join(op_sentences)} Battery specifications: {' '.join(spec_sentences)}"
    if variant == "field_shuffle":
        spec_variant = reorder_sentences(spec_sentences)
        op_variant = reorder_sentences(op_sentences)
        return f"Battery specifications: {' '.join(spec_variant)} Operating condition: {' '.join(op_variant)}"
    if variant == "mild_shuffle":
        # Reviewer/meeting-aligned mild order shuffle:
        # 1. swap the two major prompt blocks;
        # 2. swap a few sentence orders within each block, but keep semantics intact.
        if len(spec_sentences) >= 6:
            spec_variant = [
                spec_sentences[0],  # keep dataset/format statement first inside specs
                spec_sentences[2],  # negative electrode before positive electrode
                spec_sentences[1],
                spec_sentences[3],  # electrolyte unchanged
                spec_sentences[5],  # nominal capacity before manufacturer
                spec_sentences[4],
                *spec_sentences[6:],
            ]
        elif len(spec_sentences) >= 2:
            spec_variant = [spec_sentences[0], *reversed(spec_sentences[1:])]
        else:
            spec_variant = spec_sentences

        if len(op_sentences) >= 3:
            op_variant = [op_sentences[1], op_sentences[0], *op_sentences[2:]]
        elif len(op_sentences) >= 2:
            op_variant = [op_sentences[1], op_sentences[0]]
        else:
            op_variant = op_sentences

        return f"Operating condition: {' '.join(op_variant)} Battery specifications: {' '.join(spec_variant)}"
    raise ValueError(variant)


def build_task_description(cell_name: str) -> str:
    if "CALB" in cell_name:
        return (
            "Task description: "
            "The end of life of a battery is the number of charge-discharge cycles until the battery's discharge capacity reaches 90% of its nominal capacity. "
            "The discharge capacity is calculated under the described operating condition. "
            "The state of the health (SOH) is computed by the ratio of the remaining capacity to the initial capacity. "
            "The target is the SOH degradation trajecotry until the end of life of the battery. "
            "Please directly output the target of the battery based on the provided data. "
        )
    return (
        "Task description: "
        "The end of life of a battery is the number of charge-discharge cycles until the battery's discharge capacity reaches 80% of its nominal capacity. "
        "The discharge capacity is calculated under the described operating condition. "
        "The state of the health (SOH) is computed by the ratio of the remaining capacity to the nominal capacity. "
        "The target is the SOH degradation trajecotry until the end of life of the battery. "
        "Please directly output the target of the battery based on the provided data. "
    )


def collect_cells(datasets):
    cells = []
    for dataset in datasets:
        cells.extend(sorted(p.name for p in RAW_ROOTS[dataset].glob("*.pkl")))
    return cells


def build_prompts(cell_names, variant):
    prompts = {}
    for file_name in cell_names:
        cell_name = file_name[:-4] if file_name.endswith(".pkl") else file_name
        base_protocol = Mapping_helper(prompt_type="PROTOCOL", cell_name=cell_name).do_mapping()
        protocol_variant = apply_variant(base_protocol, variant)
        full_prompt = build_task_description(cell_name) + protocol_variant
        prompts[file_name] = full_prompt
    return prompts


def encode_prompts(prompts, llm_path: str, batch_size: int, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        llm_path,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    cell_names = list(prompts.keys())
    out = {}
    with torch.inference_mode():
        for i in range(0, len(cell_names), batch_size):
            batch_names = cell_names[i : i + batch_size]
            batch_prompts = [get_detailed_instruct("classification", prompts[name]) for name in batch_names]
            tokenized = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            outputs = model(**tokenized)
            embeddings = last_token_pool(outputs.last_hidden_state, tokenized["attention_mask"])
            features = embeddings.detach().cpu().numpy()
            for name, feat in zip(batch_names, features):
                out[name] = feat.reshape(1, -1)
    return out


def main():
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    base_embeddings = pickle.load(open(args.base_embeddings, "rb"))
    cell_names = collect_cells(datasets)
    prompts = build_prompts(cell_names, args.variant)
    encoded = encode_prompts(prompts, args.llm_path, args.batch_size, args.device)

    merged = dict(base_embeddings)
    merged.update(encoded)

    out_pickle = Path(args.output_pickle)
    out_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pickle, "wb") as f:
        pickle.dump(merged, f)

    out_json = Path(args.output_prompts_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(prompts, f, indent=2)

    summary = {
        "variant": args.variant,
        "datasets": datasets,
        "cell_count": len(cell_names),
        "output_pickle": str(out_pickle),
        "output_prompts_json": str(out_json),
        "llm_path": args.llm_path,
        "device": args.device,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
