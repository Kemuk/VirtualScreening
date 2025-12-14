# Virtual Screening Pipeline (Snakemake)

A modular Snakemake workflow for high-throughput virtual screening using AutoDock Vina (GPU/CPU) with AEV-PLIG rescoring.

## Overview

This pipeline performs structure-based virtual screening on the LIT-PCBA dataset (or custom targets) using:

- **Receptor preparation**: MOL2 → PDBQT/PDB conversion
- **Ligand preparation**: SMILES → PDBQT conversion with 3D coordinate generation
- **Molecular docking**: GPU-accelerated (Vina-GPU) or CPU-based (Vina)
- **Post-processing**: PDBQT → SDF conversion for downstream analysis
- **Rescoring**: AEV-PLIG machine learning-based rescoring
- **Ligand-based methods**: Fingerprint, shape, and pharmacophore similarity (optional)

## Project Structure

```
VirtualScreening/
├── workflow/
│   ├── Snakefile                 # Main workflow entry point
│   ├── rules/                    # Modular Snakemake rules
│   │   ├── common.smk            # Common functions and utilities
│   │   ├── preparation.smk       # Receptor/ligand preparation
│   │   ├── docking.smk           # GPU/CPU docking
│   │   ├── conversion.smk        # PDBQT → SDF conversion
│   │   ├── rescoring.smk         # AEV-PLIG rescoring
│   │   └── ligand_based.smk      # Ligand-based methods
│   ├── scripts/                  # Python scripts called by rules
│   ├── envs/                     # Conda environment definitions
│   └── profiles/                 # Cluster execution profiles
│       └── arc/                  # ARC cluster profile
│
├── config/
│   ├── config.yaml               # Main workflow configuration
│   └── targets.yaml              # Target-level configuration
│
├── data/
│   ├── master/
│   │   ├── manifest.parquet      # Main pipeline state tracker
│   │   └── backups/              # Timestamped manifest backups
│   └── logs/                     # Snakemake execution logs
│
├── LIT_PCBA/                     # Dataset directory
│   └── [TARGET]/
│       ├── [TARGET]_protein.mol2 # Receptor structure
│       ├── actives.smi           # Active ligands (SMILES)
│       ├── inactives.smi         # Inactive ligands (SMILES)
│       ├── receptor.pdbqt        # Generated receptor (PDBQT)
│       ├── receptor.pdb          # Generated receptor (PDB)
│       ├── pdbqt/                # Generated ligand PDBQTs
│       │   ├── actives/
│       │   └── inactives/
│       ├── docked_vina/          # Docking outputs
│       │   ├── actives/
│       │   └── inactives/
│       └── docked_sdf/           # SDF conversions
│           ├── actives/
│           └── inactives/
│
├── results/                      # Analysis outputs
│   ├── metrics/                  # Performance metrics
│   ├── plots/                    # Visualizations
│   └── rescored/                 # AEV-PLIG results
│
├── AEV-PLIG/                     # AEV-PLIG rescoring tool
└── vina-gpu-dev/                 # Vina-GPU binaries
```

## Quick Start

### 1. Configure Targets

Edit `config/targets.yaml` to define your targets:

```yaml
targets:
  ADRB2:
    receptor_mol2: "LIT_PCBA/ADRB2/ADRB2_protein.mol2"
    actives_smi: "LIT_PCBA/ADRB2/actives.smi"
    inactives_smi: "LIT_PCBA/ADRB2/inactives.smi"
    box_center:
      x: 10.5
      y: 20.3
      z: 15.7
```

### 2. Adjust Workflow Settings

Edit `config/config.yaml` for global parameters (docking settings, resources, etc.).

### 3. Run the Workflow

#### Local execution (dry-run to check):
```bash
snakemake -n
```

#### Run with 8 cores:
```bash
snakemake --cores 8
```

#### Cluster execution (ARC):
```bash
snakemake --profile workflow/profiles/arc
```

## Configuration

### Main Configuration (`config/config.yaml`)

Key settings:
- **Docking parameters**: exhaustiveness, num_modes, energy_range
- **GPU/CPU settings**: thread counts, device IDs
- **Resource allocation**: memory, CPU, GPU requirements per rule
- **Tool paths**: Vina-GPU, Vina-CPU, OpenBabel

### Target Configuration (`config/targets.yaml`)

Per-target settings:
- **receptor_mol2**: Path to receptor MOL2 file
- **actives_smi/inactives_smi**: Paths to SMILES files
- **box_center**: Docking box center coordinates (x, y, z)
- **box_size** (optional): Override default box size

## Workflow Stages

### Stage 1: Preparation
- Convert receptor MOL2 → PDBQT + PDB
- Convert ligand SMILES → PDBQT with 3D coordinates
- Update manifest with preparation status

### Stage 2: Docking
- GPU (Vina-GPU) or CPU (Vina) docking
- Parallel execution across ligands
- Store binding affinities in manifest

### Stage 3: Post-processing
- Convert docked PDBQT → SDF for visualization/analysis
- Extract specific binding modes (default: best scoring)

### Stage 4: Rescoring
- AEV-PLIG machine learning-based rescoring
- Parallel sharded execution for large datasets
- Integrate scores into manifest

### Stage 5: Ligand-Based (Optional)
- Fingerprint similarity
- Shape-based similarity
- Pharmacophore-based screening

## Manifest System

The pipeline uses a **Parquet manifest** (`data/master/manifest.parquet`) to track:

- Ligand identity (ligand_id, protein_id, compound_key)
- SMILES (input and canonical)
- File paths (inputs, intermediates, outputs)
- Status flags (preparation, docking, rescoring)
- Scores (Vina binding affinity, AEV-PLIG score)
- Metadata (timestamps)

Benefits:
- **Incremental execution**: Only process incomplete work
- **Fault tolerance**: Resume from interruptions
- **Efficient lookups**: Fast filtering by target/status
- **Timestamped backups**: Automatic versioning

## Resource Requirements

### GPU Docking (per job)
- 20 GB RAM
- 2 CPUs
- 1 GPU (CUDA)
- ~12 hours (depends on ligand count)

### CPU Docking (per job)
- 64 GB RAM
- 32 CPUs
- ~12 hours

### Preparation
- 16 GB RAM
- 16 CPUs
- ~2 hours

## Migration from SLURM Scripts

This Snakemake workflow replaces the previous SLURM-based pipeline:

| Old SLURM Script         | New Snakemake Rule      |
|--------------------------|-------------------------|
| `mol2_to_pdbqt.slurm`    | `rule mol2_to_pdbqt`    |
| `smi2pdbqt_array.slurm`  | `rule smi2pdbqt`        |
| `submit_gpu.slurm`       | `rule dock_gpu`         |
| `submit_cpu.slurm`       | `rule dock_cpu`         |
| `sdf_conversion.slurm`   | `rule pdbqt_to_sdf`     |
| `plig_manifest.slurm`    | `rule rescore_aev_plig` |

Benefits of Snakemake:
- Automatic dependency tracking
- Parallel execution
- Cluster resource management
- Reproducible configuration
- Resumable workflows

## Dependencies

- **Python 3.8+**
- **Snakemake 7.0+**
- **RDKit** (ligand preparation)
- **OpenBabel** (format conversions)
- **AutoDock Vina** or **Vina-GPU**
- **AEV-PLIG** (rescoring)
- **NumPy, Pandas, PyArrow** (manifest management)

Install via conda:
```bash
conda env create -f workflow/envs/vscreen.yaml
conda activate vscreen
```

### Vina-GPU Additional Requirements

If using GPU-accelerated docking, ensure required modules are loaded:
```bash
module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0
```

**Note**: GPU version of Vina does not use `--exhaustiveness` or `--energy_range` parameters (these are CPU-only).

## Contributing

When extending this workflow:
1. Add new rules to appropriate `workflow/rules/*.smk` files
2. Place scripts in `workflow/scripts/`
3. Update manifest schema if adding new columns
4. Document resource requirements in `config/config.yaml`

## License

[Specify your license here]

## Citation

If you use this workflow, please cite:
- AutoDock Vina: [DOI]
- AEV-PLIG: [DOI]
- LIT-PCBA dataset: [DOI]