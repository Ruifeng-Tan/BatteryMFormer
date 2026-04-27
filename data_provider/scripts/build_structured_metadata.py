#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Prompts.Mapping_helper import Mapping_helper


RAW_ROOTS = {
    "CALB": Path("/path/to/your/dataset/CALB"),
    "NA-ion": Path("/path/to/your/dataset/NA-ion"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build structured metadata json from prompt text")
    parser.add_argument("--datasets", default="CALB,NA-ion")
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def extract_field(pattern: str, text: str, default: str = "unknown") -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else default


def prompt_to_metadata(prompt: str):
    return {
        "format": extract_field(r"format of (.*?) battery\\.", prompt),
        "cathode": extract_field(r"Its positive electrode is (.*?)\\.", prompt),
        "anode": extract_field(r"Its negative electrode is (.*?)\\.", prompt),
        "manufacturer": extract_field(r"The battery manufacturer is (.*?)\\.", prompt),
        "temperature": extract_field(r"The working ambient temperature of this battery is (.*?) degrees Celsius\\.", prompt),
    }


def main():
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    files = []
    for dataset in datasets:
        files.extend(sorted(p.name for p in RAW_ROOTS[dataset].glob("*.pkl")))

    field_values = {
        "format": [],
        "cathode": [],
        "anode": [],
        "manufacturer": [],
        "temperature": [],
    }
    raw_records = {}

    for file_name in files:
        cell_name = file_name[:-4]
        prompt = Mapping_helper("PROTOCOL", cell_name).do_mapping()
        meta = prompt_to_metadata(prompt)
        raw_records[file_name] = meta
        for field, value in meta.items():
            if value not in field_values[field]:
                field_values[field].append(value)

    mappings = {
        field: {value: idx for idx, value in enumerate(values)}
        for field, values in field_values.items()
    }
    records = {}
    for file_name, meta in raw_records.items():
        records[file_name] = {
            field: mappings[field][value]
            for field, value in meta.items()
        }

    payload = {
        "fields": ["format", "cathode", "anode", "manufacturer", "temperature"],
        "cardinalities": {field: len(mappings[field]) for field in mappings},
        "value_mappings": mappings,
        "records": records,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(out)


if __name__ == "__main__":
    main()
