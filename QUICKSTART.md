# Quick Start Guide

Fast-track guide to running the virtual screening pipeline.

## Prerequisites

1. **Install Conda/Mamba**
   ```bash
   # If not already installed
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
   bash Mambaforge-Linux-x86_64.sh
   ```

2. **Install Snakemake**
   ```bash
   conda install -c conda-forge snakemake
   # or
   mamba install -c conda-forge snakemake
   ```

3. **Verify Installation**
   ```bash
   snakemake --version  # Should show v7.0+
   ```

---

## Setup (First Time Only)

### 1. Configure Your Targets

Edit `config/targets.yaml` with your target information:

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

### 2. Adjust Workflow Settings (Optional)

Edit `config/config.yaml` to customize:
- Docking parameters (exhaustiveness, num_modes)
- Resource allocations (CPUs, memory)
- Tool paths (Vina-GPU, OpenBabel)

### 3. Validate Configuration

```bash
snakemake validate_config
```

Expected: `✓ Configuration validation passed!`

---

## Running the Pipeline

### Complete Pipeline (Automatic)

```bash
# Dry-run to preview (doesn't execute)
snakemake --dry-run -n

# Run complete pipeline
snakemake --cores 16 --use-conda

# For cluster (with profile)
snakemake --profile workflow/profiles/arc
```

### Stage-by-Stage Execution

**Stage 1: Create Manifest**
```bash
snakemake create_manifest --cores 1
snakemake show_manifest_stats  # View statistics
```

**Stage 2: Preparation**
```bash
# Prepare all receptors and ligands
snakemake prepare_all --cores 16

# Or separately:
snakemake prepare_all_receptors --cores 4
snakemake prepare_all_ligands --cores 16
```

**Stage 3: Docking**
```bash
# GPU docking (if Vina-GPU is available)
snakemake dock_all_gpu --cores 100

# CPU docking
snakemake dock_all_cpu --cores 32

# Generic (uses config default)
snakemake dock_all --cores 32
```

**Stage 4: SDF Conversion**
```bash
snakemake convert_all_to_sdf --cores 16
```

**Stage 5: AEV-PLIG Data Prep**
```bash
snakemake prepare_all_aev_plig --cores 8
```

---

## Monitoring Progress

### Check Pipeline Status

```bash
# View manifest statistics
snakemake show_manifest_stats

# Check what will run next
snakemake --dry-run | head -50

# View DAG (requires graphviz)
snakemake --dag | dot -Tpng > workflow_dag.png
```

### Monitor Logs

```bash
# Watch specific stage logs
tail -f data/logs/preparation/*.log
tail -f data/logs/docking/*.log

# Check for errors
grep -i error data/logs/**/*.log
grep -i warning data/logs/**/*.log
```

---

## Common Commands

### Single Target Operations

```bash
# Prepare specific target
snakemake rescore_target --config target=ADRB2

# Convert specific target to SDF
snakemake convert_target_to_sdf --config target=ADRB2
```

### Specific Files

```bash
# Prepare specific receptor
snakemake LIT_PCBA/ADRB2/receptor.pdbqt

# Dock specific ligand
snakemake LIT_PCBA/ADRB2/docked_vina/actives/compound_001_docked.pdbqt

# Convert specific ligand to SDF
snakemake LIT_PCBA/ADRB2/docked_sdf/actives/compound_001.sdf
```

### Maintenance

```bash
# Backup manifest
snakemake backup_manifest

# Clean logs
snakemake clean_logs

# Force rerun (ignoring existing files)
snakemake --forceall
```

---

## Typical Workflows

### Full Pipeline (Start to Finish)

```bash
# 1. Validate and create manifest
snakemake validate_config
snakemake create_manifest --cores 1

# 2. Run entire pipeline
snakemake --cores 32 --use-conda

# 3. Monitor progress
watch -n 30 snakemake show_manifest_stats
```

### Incremental Execution (Resume After Failure)

```bash
# Pipeline stopped/failed? Just run again:
snakemake --cores 32

# Snakemake automatically:
# - Skips completed work
# - Resumes from where it stopped
# - Only processes what's needed
```

### Testing with Small Subset

```bash
# Edit targets.yaml to include only 1-2 targets
# Then run:
snakemake --cores 8

# Or limit number of jobs:
snakemake --cores 8 --jobs 50
```

---

## Resource Optimization

### Local Machine

```bash
# Conservative (safe for laptops)
snakemake --cores 4 --resources mem_mb=16000

# Moderate (desktop/workstation)
snakemake --cores 16 --resources mem_mb=64000

# Aggressive (high-end server)
snakemake --cores 64 --resources mem_mb=256000
```

### HPC Cluster

```bash
# Using cluster profile
snakemake --profile workflow/profiles/arc

# Custom SLURM submission
snakemake --cluster "sbatch --partition=htc --mem={resources.mem_mb} --cpus-per-task={threads}" \
          --cores 100 \
          --jobs 100
```

---

## Troubleshooting Quick Fixes

### Pipeline Won't Start

```bash
# Check configuration
snakemake validate_config

# Verify manifest exists
ls -lh data/master/manifest.parquet

# Recreate if needed
snakemake create_manifest --cores 1
```

### "Nothing to be done"

```bash
# Force checkpoint re-evaluation
rm -rf .snakemake/

# Or force complete rerun
snakemake --forceall
```

### Conda Environment Issues

```bash
# Create environment manually
conda env create -f workflow/envs/vscreen.yaml
conda activate vscreen

# Then run without --use-conda
snakemake --cores 16
```

### Out of Memory

```bash
# Reduce parallelism
snakemake --cores 8 --resources mem_mb=32000

# Or adjust in config.yaml
```

---

## Output Locations

After running the pipeline, find outputs in:

```
LIT_PCBA/{target}/
├── receptor.pdbqt              # Prepared receptor
├── receptor.pdb                # For visualization
├── pdbqt/                      # Prepared ligands
│   ├── actives/
│   └── inactives/
├── docked_vina/                # Docking results
│   ├── actives/
│   │   ├── *.pdbqt            # Docked poses
│   │   └── log/*.log          # Binding affinities
│   └── inactives/
├── docked_sdf/                 # SDF format
│   ├── actives/
│   └── inactives/
└── rescoring/                  # AEV-PLIG data
    └── datasets/
        └── aev_plig_{target}.csv
```

---

## Performance Expectations

Typical throughput on a modern server:

| Stage | Time per Ligand | Throughput (32 cores) |
|-------|----------------|----------------------|
| Preparation | ~1 second | 1000-2000/min |
| Docking (CPU) | ~30-60 seconds | 30-60/min |
| Docking (GPU) | ~5-10 seconds | 200-400/min |
| SDF Conversion | ~0.5 seconds | 2000-4000/min |
| AEV-PLIG Prep | ~0.1 seconds | 5000+/min |

For 10,000 ligands on one target:
- Preparation: ~10 minutes
- Docking (GPU): ~30-60 minutes
- Docking (CPU): ~5-8 hours
- Post-processing: ~5 minutes

---

## Next Steps

1. **Small Test Run**
   - Start with 1 target, 100 ligands
   - Verify all stages work correctly

2. **Scale Up**
   - Add more targets to targets.yaml
   - Increase --cores as needed

3. **Production Deployment**
   - Set up cluster profile for your HPC
   - Configure monitoring and backups
   - See TESTING.md for detailed validation

4. **Analysis**
   - Load AEV-PLIG CSV files for analysis
   - Use scripts/make_plots.py for visualization
   - Calculate metrics with scripts/make_metrics.py

---

## Getting Help

- Check TESTING.md for detailed testing procedures
- Check README.md for full documentation
- View rule documentation: `snakemake --help`
- Snakemake docs: https://snakemake.readthedocs.io

## Common Snakemake Options

```bash
--dry-run, -n          # Preview without executing
--cores N, -c N        # Use N CPU cores
--use-conda            # Use conda environments
--printshellcmds, -p   # Show shell commands
--forceall, -F         # Force rerun everything
--dag                  # Show DAG graph
--summary              # Show file status summary
--quiet, -q            # Suppress output
--keep-going           # Continue on errors
--rerun-incomplete     # Rerun incomplete jobs
```
