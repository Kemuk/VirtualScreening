#!/usr/bin/env python3
"""
prepare_aev_plig_shard.py

Generate an AEV-PLIG shard CSV from a rescoring chunk.

Input chunk columns (from shard_stage.py rescoring stage):
    - compound_key
    - vina_score
    - docked_sdf_path
    - receptor_pdb_path
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def vina_score_to_pK(vina_score: float, temperature: float = 298.0) -> float:
    """
    Convert Vina docking score to pK using thermodynamic relationship.

    pK = -ΔG / (2.303 * R * T)
    """
    R = 0.001987  # kcal/(mol·K)
    return -vina_score / (2.303 * R * temperature)


def resolve_path(project_root: Path, path_value: str) -> Path:
    if not path_value or pd.isna(path_value):
        return Path()
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def prepare_shard(chunk_path: Path, output_path: Path, project_root: Path) -> int:
    chunk = pd.read_csv(chunk_path)
    if chunk.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["unique_id", "pK", "sdf_file", "pdb_file"]).to_csv(
            output_path,
            index=False,
        )
        return 0

    required_cols = {"compound_key", "vina_score", "docked_sdf_path", "receptor_pdb_path"}
    missing = required_cols - set(chunk.columns)
    if missing:
        print(f"ERROR: Missing columns in chunk {chunk_path}: {sorted(missing)}", file=sys.stderr)
        return 0

    chunk["vina_score"] = pd.to_numeric(chunk["vina_score"], errors="coerce")
    chunk = chunk[chunk["vina_score"].notna()].copy()

    def make_row(row: pd.Series) -> dict | None:
        sdf_path = resolve_path(project_root, row["docked_sdf_path"])
        pdb_path = resolve_path(project_root, row["receptor_pdb_path"])
        if not sdf_path.exists() or not pdb_path.exists():
            return None
        return {
            "unique_id": row["compound_key"],
            "pK": vina_score_to_pK(row["vina_score"]),
            "sdf_file": str(sdf_path.resolve()),
            "pdb_file": str(pdb_path.resolve()),
        }

    rows = [make_row(row) for _, row in chunk.iterrows()]
    rows = [row for row in rows if row is not None]
    output_df = pd.DataFrame(rows, columns=["unique_id", "pK", "sdf_file", "pdb_file"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return len(output_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AEV-PLIG shard CSV from rescoring chunk")
    parser.add_argument("--chunk", type=Path, required=True, help="Input rescoring chunk CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output AEV-PLIG shard CSV")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root for resolving relative paths",
    )

    args = parser.parse_args()

    if not args.chunk.exists():
        print(f"ERROR: Chunk file not found: {args.chunk}", file=sys.stderr)
        sys.exit(1)

    count = prepare_shard(args.chunk, args.output, args.project_root)
    if count == 0:
        print(f"WARNING: No rows written for {args.output}", file=sys.stderr)

    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
