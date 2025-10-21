#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import RDLogger
from src.preprocess import preprocess_target
from src.conformers import generate_conformers_for_target
from src.descriptors import compute_descr_for_target
from src.index_search import rank_by_usrcat
from src.eval import evaluate_rankings
from src.io import read_vina_boxes, list_targets
import yaml

def disable_rdkit_logs():
    RDLogger.DisableLog('rdApp.*')

def load_cfg(path="config.yaml"):
    return yaml.safe_load(open(path))

def run_target(target, cfg, vina, data_root, processed_root, results_root):
    # Each target run (parallel across targets or serial)

    print(f"==> START target {target}")
    manifest = vina.get(target, {})
    preprocess_target(data_root, processed_root, target, cfg)
    generate_conformers_for_target(processed_root, target, cfg)
    compute_descr_for_target(processed_root, target, cfg)
    ranked_csv = rank_by_usrcat(processed_root, results_root, target, cfg, manifest)
    metrics = evaluate_rankings(Path(results_root) / target / ranked_csv,
                                Path(processed_root) / target / "labels.csv")
    print(f"  {target} metrics: {metrics}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", help="Single target; if omitted process all")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = p.parse_args()

    disable_rdkit_logs()

    if args.no_progress:
        os.environ["TQDM_DISABLE"] = "1"

    cfg = load_cfg(args.config)
    data_root = Path(cfg["data_root"])
    processed_root = Path(cfg["processed_root"])
    results_root = Path(cfg["results_root"])
    processed_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)


    vina = read_vina_boxes(data_root / "vina_boxes.csv")
    available = list_targets(data_root)

    if args.target:
        targets_to_run = [args.target] if args.target in available else []
    else:
        targets_to_run = available[:]

    if not targets_to_run:
        raise SystemExit("No valid targets to run")

    # Determine cores per target
    total_cores = os.cpu_count() or 1
    n = len(targets_to_run)
    # divide cores per target
    per_target_workers = max(1, total_cores // n)
    max_target_workers = n

    # run targets in parallel
    with ProcessPoolExecutor(max_workers=max_target_workers) as exe:
        futures = {exe.submit(run_target, t, cfg, vina, data_root, processed_root, results_root): t
                   for t in targets_to_run}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"ERROR for target {t}: {e}")

if __name__ == "__main__":
    main()
