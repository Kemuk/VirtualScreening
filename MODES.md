# Execution Modes Guide

The pipeline supports two mutually exclusive execution modes: **test** and **slurm**.

## Test Mode (Default)

Run interactively on a small subset of ligands for quick testing and validation.

### Configuration

Test mode is enabled by default in `config/config.yaml`:

```yaml
mode: "test"  # Default mode

test:
  ligands_per_protein: 50        # Total ligands per protein
  actives_per_protein: 25        # Number of actives (rest will be inactives)
```

### Usage

```bash
# Run with test mode (default)
snakemake --cores 8

# Or explicitly specify test mode
snakemake --cores 8 --config mode=test

# Customize test size
snakemake --cores 8 --config mode=test test.ligands_per_protein=100
```

### What It Does

- **Filters manifest** to 50 ligands per protein (25 actives + 25 inactives by default)
- **Runs locally** in your current session
- **Uses local cores** (no SLURM submission)
- **Fast turnaround** for development and debugging

### When to Use

- ✅ Initial testing and validation
- ✅ Debugging workflow issues
- ✅ Parameter tuning
- ✅ Quick sanity checks
- ✅ Interactive development

### Example Workflow

```bash
# 1. Set test mode (already default)
# Edit config/config.yaml if needed

# 2. Create manifest (filtered to 50 ligands/protein)
snakemake create_manifest --cores 1

# 3. Check what was selected
snakemake show_manifest_stats

# Output:
# Total entries: 150  (3 targets × 50 ligands)
# Unique proteins: 3
# ...

# 4. Run full pipeline interactively
snakemake --cores 16
```

---

## SLURM Mode

Run the full dataset on the ARC cluster with proper resource management.

### Configuration

Set SLURM mode in `config/config.yaml`:

```yaml
mode: "slurm"  # Full dataset, cluster submission
```

### Usage

```bash
# Set SLURM mode
snakemake --config mode=slurm --profile workflow/profiles/arc

# Or edit config.yaml to set mode: "slurm" permanently
# Then just run:
snakemake --profile workflow/profiles/arc
```

### What It Does

- **Uses full dataset** (all ligands from SMILES files)
- **Submits jobs to SLURM** using cluster profile
- **Manages resources** according to `config.yaml` specifications
- **Parallel execution** across cluster nodes

### Resource Specifications

The SLURM profile uses resource specifications from `config.yaml`:

```yaml
resources:
  docking_gpu:
    mem_mb: 20000
    cpus: 2
    gpus: 1
    time_min: 720        # 12 hours
    partition: "htc"
```

These are automatically converted to SLURM flags:
- `--mem=20000M`
- `--cpus-per-task=2`
- `--gres=gpu:1`
- `--time=12:00:00`
- `--partition=htc`

### Cluster Profile Settings

Located in `workflow/profiles/arc/config.yaml`:

```yaml
jobs: 100                    # Max simultaneous jobs
restart-times: 3             # Retry failed jobs
max-jobs-per-second: 10     # Submission rate limit
latency-wait: 60            # Filesystem delay tolerance
```

### When to Use

- ✅ Production runs
- ✅ Full dataset processing
- ✅ Large-scale screening (1000s of ligands)
- ✅ Long-running jobs (>1 hour)
- ✅ GPU-intensive docking

### Example Workflow

```bash
# 1. Set SLURM mode
# Edit config/config.yaml:
#   mode: "slurm"

# 2. Create manifest (full dataset)
snakemake create_manifest --cores 1

# 3. Check what will be processed
snakemake show_manifest_stats

# Output:
# Total entries: 15000  (3 targets × ~5000 ligands each)
# Unique proteins: 3
# ...

# 4. Submit to cluster
snakemake --profile workflow/profiles/arc

# 5. Monitor jobs
squeue -u $USER
sacct -u $USER

# 6. Check logs
tail -f data/logs/slurm/dock_ligand_gpu-*.out
```

---

## Mode Comparison

| Feature | Test Mode | SLURM Mode |
|---------|-----------|------------|
| **Dataset Size** | 50 ligands/protein | Full dataset |
| **Execution** | Interactive/local | Cluster submission |
| **Cores** | Local (e.g., 8-16) | Cluster (100s) |
| **Resources** | Local machine | SLURM-managed |
| **Runtime** | Minutes-hours | Hours-days |
| **Use Case** | Development/testing | Production |
| **Cost** | Free (local) | Compute credits |

---

## Switching Between Modes

### Method 1: Config File (Persistent)

Edit `config/config.yaml`:

```yaml
# For testing
mode: "test"

# For production
mode: "slurm"
```

Then run normally:
```bash
snakemake --cores 8                       # Test mode
snakemake --profile workflow/profiles/arc # SLURM mode
```

### Method 2: Command Line (Temporary)

Override config from command line:

```bash
# Force test mode
snakemake --cores 8 --config mode=test

# Force SLURM mode
snakemake --profile workflow/profiles/arc --config mode=slurm
```

---

## Important Notes

### 1. Manifest Reflects Mode

The manifest is generated **according to the current mode**:

- **Test mode**: Manifest contains only 50 ligands/protein
- **SLURM mode**: Manifest contains all ligands

**To switch modes completely:**
```bash
# Switch from test to SLURM
# 1. Change mode in config.yaml
# 2. Regenerate manifest
rm data/master/manifest.parquet
snakemake create_manifest --cores 1
# 3. Run with new mode
snakemake --profile workflow/profiles/arc
```

### 2. Mode Is Mutually Exclusive

You cannot mix modes:
- ❌ Don't use `--profile` with test mode
- ❌ Don't use local `--cores` with SLURM mode

### 3. Test Before Production

**Best practice workflow:**

```bash
# 1. Test with small subset
echo 'mode: "test"' >> config/config.yaml
snakemake create_manifest --cores 1
snakemake --cores 8

# 2. Validate results
snakemake show_manifest_stats
ls -lh LIT_PCBA/ADRB2/docked_vina/actives/

# 3. Switch to production
echo 'mode: "slurm"' >> config/config.yaml
rm data/master/manifest.parquet
snakemake create_manifest --cores 1
snakemake --profile workflow/profiles/arc
```

---

## Troubleshooting

### "Nothing to be done"

You changed mode but Snakemake sees no work:

**Solution**: Regenerate manifest
```bash
rm data/master/manifest.parquet
snakemake create_manifest --cores 1
```

### SLURM job failures

Check SLURM logs:
```bash
ls -lh data/logs/slurm/
tail data/logs/slurm/dock_ligand_gpu-*.err
```

Common issues:
- Insufficient memory → Increase `mem_mb` in config.yaml
- Timeout → Increase `time_min` in config.yaml
- GPU unavailable → Check `--gres` allocation

### Test mode too slow

Reduce test size:
```bash
snakemake --config mode=test test.ligands_per_protein=10 --cores 8
```

---

## Performance Expectations

### Test Mode (50 ligands/protein, 3 proteins = 150 total)

On 16-core workstation:
- Preparation: ~5 minutes
- Docking (CPU): ~2-4 hours
- Docking (GPU): ~15-30 minutes
- Post-processing: ~2 minutes

### SLURM Mode (5000 ligands/protein, 3 proteins = 15000 total)

On ARC cluster (100 parallel jobs):
- Preparation: ~30 minutes
- Docking (GPU): ~4-8 hours
- Docking (CPU): ~24-48 hours
- Post-processing: ~20 minutes

---

## Summary

- **Test mode**: Quick, local, 50 ligands/protein - for development
- **SLURM mode**: Full, cluster, all ligands - for production
- **Switch by changing** `mode` in `config/config.yaml`
- **Regenerate manifest** when switching modes
- **Test first**, then scale to production
