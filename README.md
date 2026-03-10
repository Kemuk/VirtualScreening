# Virtual Screening Pipeline (Snakemake)

A modular Snakemake workflow for high-throughput virtual screening using AutoDock Vina (GPU/CPU) with AEV-PLIG rescoring.

> **🎉 Recently Consolidated (Feb 2026):** The workflow has been significantly simplified with 70% fewer rules and 83% fewer SLURM scripts. See [`MIGRATION.md`](MIGRATION.md) for details and migration instructions.

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
│       ├── slurm/                # SLURM cluster profile (production)
│       └── slurm_devel/          # SLURM cluster profile (development)
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

#### Production Mode (Full Pipeline):
```bash
# Validate configuration
snakemake validate_config

# Create manifest
snakemake create_manifest --cores 1

# Run all stages
snakemake prepare_all --cores 1    # Prepare receptors & ligands
snakemake dock_all --cores 1       # Dock all ligands
snakemake convert_all --cores 1    # Convert to SDF
snakemake rescore_all --cores 1    # Rescore with AEV-PLIG
snakemake results_all --cores 4    # Compute metrics & plots
```

#### Development Mode (Testing with smaller chunks):
```bash
snakemake prepare_all --config mode=devel --cores 1
snakemake dock_all --config mode=devel --cores 1
```

#### Local execution (dry-run to check):
```bash
snakemake -n
```

#### Testing single files:
```bash
snakemake --cores 8
```

#### Cluster execution (SLURM):
```bash
snakemake --profile workflow/profiles/slurm
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

> **Note:** The workflow has been significantly simplified. See `MIGRATION.md` for details about recent consolidation (34 rules → 10 rules, 70% reduction).

### Stage 1: Preparation (`prepare_all`)
- Convert receptor MOL2 → PDBQT + PDB
- Convert ligand SMILES → PDBQT with 3D coordinates
- Parallel SLURM array jobs for large-scale processing
- Update manifest with preparation status

**Command:**
```bash
snakemake prepare_all --cores 1
```

### Stage 2: Docking (`dock_all`)
- GPU (Vina-GPU) or CPU (Vina) docking
- Parallel SLURM array jobs across ligands
- Store binding affinities in manifest

**Command:**
```bash
snakemake dock_all --cores 1
```

### Stage 3: Conversion (`convert_all`)
- Convert docked PDBQT → SDF for visualization/analysis
- Extract specific binding modes (default: best scoring)
- Parallel SLURM array jobs

**Command:**
```bash
snakemake convert_all --cores 1
```

### Stage 4: Rescoring (`rescore_all`)
- AEV-PLIG machine learning-based rescoring
- GPU-accelerated neural network predictions
- Parallel SLURM array jobs for large datasets
- Integrate scores into manifest

**Command:**
```bash
snakemake rescore_all --cores 1
```

### Stage 5: Results (`results_all`)
- Compute virtual screening metrics (ROC-AUC, BEDROC, EF, NEF)
- Generate visualization plots
- Bootstrap confidence intervals

**Command:**
```bash
snakemake results_all --cores 4
```

### All Stages Available Rules

**Production (batch processing):**
- `prepare_all` - Prepare all receptors and ligands
- `dock_all` - Dock all ligands
- `convert_all` - Convert all to SDF
- `rescore_all` - Rescore all with AEV-PLIG
- `compute_results` - Compute metrics
- `make_plots` - Generate plots
- `results_all` - Complete results stage

**Development (single-file testing):**
- `prepare_receptor` - Test single receptor preparation
- `dock_ligand` - Test single ligand docking
- `convert_to_sdf` - Test single conversion

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

## Simplified Workflow Architecture

The workflow now uses a **unified SLURM infrastructure** that consolidates all batch processing:

### Before (Old Approach)
- 34 Snakemake rules across 5 stages
- 12 separate SLURM submission scripts
- Complex multi-step chunking and merging
- Different patterns for each stage

### After (New Consolidated Approach)
- **10 Snakemake rules** (70% reduction)
- **2 unified SLURM scripts** (83% reduction)
- Single submission system: `submit_stage.py`
- Single worker template: `stage_worker.slurm`

### New Workflow Pattern

| Stage | Production Rule | Single-File Testing |
|-------|----------------|---------------------|
| **Preparation** | `prepare_all` | `prepare_receptor` |
| **Docking** | `dock_all` | `dock_ligand` |
| **Conversion** | `convert_all` | `convert_to_sdf` |
| **Rescoring** | `rescore_all` | *(use mode=devel)* |
| **Results** | `results_all` | `compute_results`, `make_plots` |

### Benefits of the New System

- **Simpler**: One command per stage instead of shard → submit → merge
- **Consistent**: All stages use the same submission logic
- **Maintainable**: Change SLURM settings in one place
- **Reliable**: Single unified codebase reduces bugs
- **Automatic**: Handles chunking, submission, and merging internally

See [`MIGRATION.md`](MIGRATION.md) for detailed migration instructions and troubleshooting.

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
