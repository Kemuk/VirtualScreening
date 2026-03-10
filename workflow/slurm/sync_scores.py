#!/usr/bin/env python3
"""
sync_scores.py

Scan for existing docked files and extract Vina scores into manifest.
Updates docking_status and vina_score without re-docking.

Usage:
    python -m workflow.slurm.sync_scores
    python -m workflow.slurm.sync_scores --dry-run  # preview changes
"""

import argparse
import fcntl
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from tqdm import tqdm


VINA_LOG_PATTERN = re.compile(r"^\s*1\s+(-?\d+\.?\d*)\s+", re.MULTILINE)
VINA_PDBQT_PATTERN = re.compile(r"REMARK\s+VINA\s+RESULT:\s+(-?\d+\.?\d*)", re.IGNORECASE)


def extract_score_from_log(log_path: Path, max_bytes: int = 32768) -> float | None:
    """Extract best Vina score from log file (reads up to max_bytes)."""
    if not log_path.exists():
        return None
    try:
        with open(log_path, "r", errors="ignore") as f:
            content = f.read(max_bytes)
        match = VINA_LOG_PATTERN.search(content)
        if match:
            return float(match.group(1))
    except Exception:
        return None
    return None


def extract_score_from_pdbqt(pdbqt_path: Path) -> float | None:
    """Extract best Vina score from docked PDBQT file header."""
    if not pdbqt_path.exists():
        return None
    try:
        with open(pdbqt_path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("REMARK"):
                    match = VINA_PDBQT_PATTERN.search(line)
                    if match:
                        return float(match.group(1))
                elif line.startswith("ATOM") or line.startswith("HETATM"):
                    break
    except Exception:
        return None
    return None


def check_item_min(compound_key: str, docked_pdbqt_path: str | None) -> dict:
    if not docked_pdbqt_path:
        return {"compound_key": compound_key, "exists": False, "score": None}

    docked_path = Path(docked_pdbqt_path)
    if not docked_path.exists():
        return {"compound_key": compound_key, "exists": False, "score": None}

    log_path = docked_path.parent / "log" / f"{docked_path.stem.replace('_docked', '')}.log"
    score = extract_score_from_log(log_path)
    if score is None:
        score = extract_score_from_pdbqt(docked_path)

    return {"compound_key": compound_key, "exists": True, "score": score}


def sync_scores(
    manifest_path: Path,
    dry_run: bool = False,
    max_workers: int = 8,
) -> tuple[int, int]:
    """
    Sync docking_status and vina_score from existing files.

    Fast native update strategy:
      - Build minimal update DataFrames keyed by compound_key
      - Use Polars DataFrame.update(...) (no joins, no helper columns, no Python per-row mapping)
    """
    df = pl.read_parquet(manifest_path)
    print(f"Loaded manifest: {df.height:,} rows")

    required = {"compound_key", "docking_status", "docked_pdbqt_path", "vina_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    needs_check = (
        df.filter(pl.col("docking_status") == False)
          .select(["compound_key", "docked_pdbqt_path"])
    )
    print(f"Items with docking_status=False: {needs_check.height:,}")

    if needs_check.height == 0:
        print("Nothing to sync - all items already have docking_status=True")
        return 0, 0

    compound_keys = needs_check.get_column("compound_key").to_list()
    docked_paths = needs_check.get_column("docked_pdbqt_path").to_list()

    print("\nScanning for existing docked files...")
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(check_item_min, ck, dp)
            for ck, dp in zip(compound_keys, docked_paths, strict=False)
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            try:
                results.append(fut.result())
            except Exception:
                pass

    found = [r for r in results if r.get("exists")]
    with_scores = [r for r in found if r.get("score") is not None]

    print("\nResults:")
    print(f"  Found existing docked files: {len(found):,}")
    print(f"  With extractable scores: {len(with_scores):,}")

    if not found:
        print("\nNo existing docked files found that need syncing.")
        return 0, 0
    # Replace your current "Sample of found files" block with this:

    print("\nSample of found files (highest scores):")

    # Sort only those with a score, descending (highest first)
    top_scored = sorted(
        (r for r in found if r.get("score") is not None),
        key=lambda r: r["score"],
        reverse=True,
    )

    # Show up to 5 highest-scoring; if none have scores, fall back to first 5 found
    sample = top_scored[:5] if top_scored else found[:5]

    for r in sample:
        score = r.get("score")
        score_str = f"{score:.2f}" if score is not None else "N/A"
        print(f"  {r['compound_key']}: score={score_str}")
        
    if dry_run:
        print(f"\n[DRY RUN] Would update {len(found):,} rows")
        return len(found), len(with_scores)

    # Build minimal update frames (only existing columns; no helper columns)
    status_updates = pl.DataFrame(
        {
            "compound_key": [r["compound_key"] for r in found],
            "docking_status": [True] * len(found),
        }
    )

    score_updates = pl.DataFrame(
        {
            "compound_key": [r["compound_key"] for r in with_scores],
            "vina_score": [r["score"] for r in with_scores],
        }
    )

    lock_path = manifest_path.with_suffix(".lock")
    print("\nUpdating manifest...")

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            df = pl.read_parquet(manifest_path)

            # Native in-place-style updates keyed by compound_key (no joins / no extra columns)
            # Note: requires Polars DataFrame.update(..., on=...)
            df2 = df.update(status_updates, on="compound_key")
            if score_updates.height:
                df2 = df2.update(score_updates, on="compound_key")

            temp_path = manifest_path.with_suffix(".tmp")
            df2.write_parquet(temp_path)
            temp_path.rename(manifest_path)

            print(f"Updated manifest: {manifest_path}")

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return len(found), len(with_scores)


def main():
    parser = argparse.ArgumentParser(
        description="Sync docking_status and vina_score from existing files"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/master/manifest.parquet"),
        help="Path to manifest (default: data/master/manifest.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )

    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    num_status, num_scores = sync_scores(
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        max_workers=args.workers,
    )

    print(f"\nDone. Updated {num_status:,} status flags, {num_scores:,} scores.")
    sys.exit(0)


if __name__ == "__main__":
    main()