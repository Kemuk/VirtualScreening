#!/usr/bin/env python3
"""
backfill_vina_chunk.py

Process a single chunk for vina_backfill stage.
Extracts Vina scores from docked PDBQT files using multithreading.
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import polars as pl


def extract_vina_score(pdbqt_path: str) -> Optional[float]:
    """
    Extract Vina score from docked PDBQT file (line 2, first number).

    Args:
        pdbqt_path: Path to docked PDBQT file

    Returns:
        Vina score (kcal/mol) or None if extraction failed
    """
    path = Path(pdbqt_path)
    if not path.exists():
        return None

    try:
        with open(path) as f:
            next(f)  # Skip MODEL line
            line2 = next(f)
            match = re.search(r'-?\d+\.\d+', line2)
            if match:
                return float(match.group())
    except Exception:
        pass

    return None


def process_chunk(chunk_path: Path, output_path: Path, workers: int = 8) -> None:
    """
    Process a chunk CSV and extract Vina scores.

    Args:
        chunk_path: Input chunk CSV with compound_key, docked_pdbqt_path
        output_path: Output CSV with compound_key, vina_score
        workers: Number of threads for parallel extraction
    """
    df = pl.read_csv(chunk_path)

    if df.is_empty():
        # Write empty results file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            'compound_key': [],
            'vina_score': [],
        }).write_csv(output_path)
        print(f"Empty chunk, wrote empty results to {output_path}")
        return

    paths = df['docked_pdbqt_path'].to_list()
    compound_keys = df['compound_key'].to_list()

    print(f"Processing {len(paths)} ligands with {workers} threads...")

    # Extract scores in parallel using threads (I/O bound)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        scores = list(executor.map(extract_vina_score, paths))

    # Count successes
    successful = sum(1 for s in scores if s is not None)
    print(f"Extracted {successful} / {len(scores)} scores")

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pl.DataFrame({
        'compound_key': compound_keys,
        'vina_score': scores,
    })
    results.write_csv(output_path)
    print(f"Wrote results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Vina scores from docked PDBQT files"
    )
    parser.add_argument(
        "--chunk",
        type=Path,
        required=True,
        help="Input chunk CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output results CSV"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of threads (default: 8)"
    )

    args = parser.parse_args()
    process_chunk(args.chunk, args.output, args.workers)


if __name__ == "__main__":
    main()
