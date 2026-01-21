#!/usr/bin/env python3
"""
process_stage_chunk.py

Run a single stage chunk and emit a results CSV keyed by compound_key.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from dock_vina import process_batch as dock_batch
from pdbqt_to_sdf import process_batch as convert_batch
from prepare_all_ligands import process_batch as prep_batch


def load_config(config_path: Path) -> dict:
    """Load workflow config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def process_preparation(items: list[dict], config: dict) -> list[dict]:
    results = prep_batch(items, config)
    output = []
    for item, result in zip(items, results):
        success = result.get("success", False)
        skipped = result.get("skipped", False)
        output.append({
            "compound_key": item["compound_key"],
            "preparation_status": success or skipped,
            "error": result.get("error"),
        })
    return output


def process_docking(items: list[dict], config: dict) -> list[dict]:
    results = dock_batch(items, config)
    output = []
    for item, result in zip(items, results):
        output.append({
            "compound_key": item["compound_key"],
            "docking_status": result.get("success", False),
            "vina_score": result.get("score"),
            "error": result.get("error"),
        })
    return output


def process_conversion(items: list[dict], config: dict) -> list[dict]:
    results = convert_batch(items, config)
    output = []
    for item, result in zip(items, results):
        output.append({
            "compound_key": item["compound_key"],
            "conversion_status": result.get("success", False),
            "error": result.get("error"),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a stage chunk")
    parser.add_argument("--stage", choices=["preparation", "docking", "conversion"], required=True)
    parser.add_argument("--chunk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))

    args = parser.parse_args()
    config = load_config(args.config)

    df = pd.read_csv(args.chunk)
    if df.empty:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        return

    items = df.to_dict(orient="records")

    if args.stage == "preparation":
        results = process_preparation(items, config)
    elif args.stage == "docking":
        results = process_docking(items, config)
    else:
        results = process_conversion(items, config)

    output_df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
