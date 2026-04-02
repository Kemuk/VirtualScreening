"""
ligand_based.smk

Optional Snakemake rule for ligand-based virtual screening.

Activated by setting `ligand_based.enabled: true` in config/config.yaml.
Runs from SMILES alone — no dependency on docking or rescoring stages.

Outputs a copy of the manifest with three additional columns:
  ligand_based_score   float32  similarity score (higher = more similar to template)
  ligand_based_status  bool     True when scoring succeeded
  ligand_based_method  str      method used (e.g. "usrcat")
"""

_LB_CFG = config.get("ligand_based", {})


rule ligand_based_screen:
    """
    Score all ligands by shape similarity to a per-target template molecule.

    Wraps workflow/slurm/workers/ligand_based.py, which in turn wraps the
    existing ligand_based/src/ pipeline (conformers → USR/USRCAT → similarity).

    The output manifest is written to data/master/manifest_ligand_based.parquet
    and is independent of the main manifest — the main pipeline is unaffected.
    """
    input:
        manifest = MANIFEST_PATH,

    output:
        manifest_lb = "data/master/manifest_ligand_based.parquet",
        checkpoint  = touch("data/logs/ligand_based/ligand_based_checkpoint.done"),

    log:
        "data/logs/ligand_based/ligand_based_screen.log"

    params:
        method   = _LB_CFG.get("method", "usrcat"),
        work_dir = "data/ligand_based",

    shell:
        """
        python -m workflow.slurm.workers.ligand_based \
            --manifest  {input.manifest} \
            --output    {output.manifest_lb} \
            --work-dir  {params.work_dir} \
            --method    {params.method} \
            --config    config/config.yaml \
            2>&1 | tee {log}
        """
