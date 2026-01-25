"""
preparation.smk

Snakemake rules for receptor and ligand preparation.
"""

MODE = config.get('mode', 'production')
PREP_CHUNKS = get_chunk_count("cpu")
PREP_MAX_ITEMS = get_chunk_max_items()
PREP_CHUNK_IDS = list(range(PREP_CHUNKS))


# Receptor Preparation
rule prepare_receptor:
    """Convert receptor MOL2 → PDBQT + PDB."""
    input:
        mol2 = lambda wildcards: get_target_config(wildcards.target)['receptor_mol2']

    output:
        pdbqt = "{dataset}/{target}/{target}_protein.pdbqt",
        pdb = "{dataset}/{target}/{target}_protein.pdb"

    log:
        "data/logs/preparation/{dataset}_{target}_receptor.log"

    conda:
        "../envs/vscreen.yaml"

    params:
        ph = lambda wildcards: config.get('preparation', {}).get('ph', 7.4),
        partial_charge = lambda wildcards: config.get('preparation', {}).get('partial_charge', 'gasteiger'),

    shell:
        """
        python workflow/scripts/mol2_to_pdbqt.py \
            --input {input.mol2} \
            --pdbqt {output.pdbqt} \
            --pdb {output.pdb} \
            --ph {params.ph} \
            --partial-charge {params.partial_charge} \
            --overwrite \
            2>&1 | tee {log}
        """


rule prepare_all_receptors:
    """Prepare all receptors defined in targets.yaml."""
    input:
        expand(
            "{dataset}/{target}/{target}_protein.pdbqt",
            dataset=config['dataset'],
            target=get_targets()
        ),
        expand(
            "{dataset}/{target}/{target}_protein.pdb",
            dataset=config['dataset'],
            target=get_targets()
        )

    message:
        "All receptors prepared!"


# Ligand Preparation (Batch Processing)
rule shard_preparation:
    """Shard ligands needing preparation into chunk CSVs."""
    input:
        manifest = MANIFEST_PATH

    output:
        expand("data/chunks/preparation/chunk_{chunk}.csv", chunk=PREP_CHUNK_IDS)

    log:
        "data/logs/preparation/shard_preparation.log"

    params:
        num_chunks = PREP_CHUNKS,
        max_items_flag = lambda wildcards: f"--max-items {PREP_MAX_ITEMS}" if PREP_MAX_ITEMS else "",

    conda:
        "../envs/vscreen.yaml"

    run:
        with notify(rule):
            shell(
                "python workflow/scripts/shard_stage.py "
                "--stage preparation "
                "--manifest {input.manifest} "
                "--outdir data/chunks/preparation "
                "--num-chunks {params.num_chunks} "
                "{params.max_items_flag} "
                "2>&1 | tee {log}"
            )


rule prepare_array:
    """Submit a SLURM array to prepare all chunks."""
    input:
        expand("data/chunks/preparation/chunk_{chunk}.csv", chunk=PREP_CHUNK_IDS)

    output:
        touch("data/logs/preparation/preparation_array.done")

    log:
        "data/logs/preparation/preparation_array.log"

    conda:
        "../envs/vscreen.yaml"

    params:
        mode = config.get("mode", "production"),

    shell:
        """
        bash workflow/scripts/submit_preparation_array.sh \
            --chunks-dir data/chunks/preparation \
            --results-dir data/results/preparation \
            --log-dir data/logs/preparation \
            --slurm-log-dir data/logs/slurm \
            --config config/config.yaml \
            --mode {params.mode} \
            2>&1 | tee {log}
        """


rule merge_preparation_results:
    """Merge preparation chunk results into the manifest."""
    input:
        manifest = MANIFEST_PATH,
        array_done = "data/logs/preparation/preparation_array.done",

    output:
        touch("data/logs/preparation/ligands_checkpoint.done")

    log:
        "data/logs/preparation/merge_preparation_results.log"

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/merge_stage_results.py \
            --stage preparation \
            --manifest {input.manifest} \
            --results-dir data/results/preparation \
            2>&1 | tee {log}
        """


# Combined Preparation Rule
rule prepare_all:
    """Prepare all receptors and ligands."""
    input:
        rules.prepare_all_receptors.input,
        "data/logs/preparation/ligands_checkpoint.done"

    message:
        "Preparation stage complete!"
