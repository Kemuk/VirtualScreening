# Testing Guide for Virtual Screening Pipeline

A step-by-step guide to test and validate the Snakemake workflow.

## Prerequisites

Before testing, ensure you have:
- Conda/Mamba installed
- Snakemake installed (`conda install -c conda-forge snakemake`)
- Sample data in LIT_PCBA directory

## Testing Strategy

Test in order from simple to complex:
1. Configuration validation
2. Manifest generation
3. Individual stages (preparation → docking → conversion → rescoring)
4. Full pipeline

---

## Stage 0: Setup and Configuration

### 1. Validate Configuration Files

```bash
# Check YAML syntax and required fields
snakemake validate_config
```

**Expected output:**
```
✓ Configuration validation passed!
  Found 3 targets
```

**Common errors:**
- Missing required keys → Add them to config.yaml
- Invalid YAML syntax → Check indentation
- Missing box_center coordinates → Add to targets.yaml

### 2. Check Directory Structure

```bash
# Verify all required directories exist
ls -la workflow/{rules,scripts,envs}
ls -la config/
ls -la data/master/
```

**Expected:**
- `workflow/rules/` contains .smk files
- `workflow/scripts/` contains .py files (executable)
- `config/` contains config.yaml and targets.yaml

---

## Stage 1: Manifest Generation

### 1. Dry-Run Test

```bash
# See what Snakemake will do (doesn't execute)
snakemake create_manifest --dry-run -n
```

**Expected output:**
```
Job stats:
job                count
---------------  -------
create_manifest        1
total                  1
```

### 2. Create Test Manifest

```bash
# Actually create the manifest
snakemake create_manifest --cores 1
```

**Expected output:**
```
Generating manifest entries...
Processing targets: 100%|████████| 3/3
Saved manifest: data/master/manifest.parquet
  Total entries: 1500
  Prepared: 0
  Docked: 0
  Rescored: 0
✓ Manifest generation complete!
```

**Verify:**
```bash
# Check manifest was created
ls -lh data/master/manifest.parquet

# View manifest statistics
snakemake show_manifest_stats
```

**Expected stats output:**
```
============================================================
MANIFEST STATISTICS
============================================================
Total entries: 1500
Unique proteins: 3
Unique ligands: 500

By protein:
ADRB2      500
ALDH1      500
ESR1_ant   500

Pipeline progress:
  Prepared:      0 / 1500 (0.0%)
  Docked:        0 / 1500 (0.0%)
  Rescored:      0 / 1500 (0.0%)

By class:
  Actives:     750
  Inactives:   750
============================================================
```

**Common errors:**
- `FileNotFoundError: SMILES file not found` → Check paths in targets.yaml
- `No entries generated` → Verify SMILES files exist and are readable

---

## Stage 2: Preparation

### 1. Test Receptor Preparation (Single Target)

```bash
# Test receptor conversion for one target
snakemake LIT_PCBA/ADRB2/receptor.pdbqt --cores 1 -p
```

**Expected output:**
```
python workflow/scripts/mol2_to_pdbqt.py \
    --input LIT_PCBA/ADRB2/ADRB2_protein.mol2 \
    --pdbqt LIT_PCBA/ADRB2/receptor.pdbqt \
    --pdb LIT_PCBA/ADRB2/receptor.pdb \
    --ph 7.4 \
    --partial-charge gasteiger

Converting receptor: LIT_PCBA/ADRB2/ADRB2_protein.mol2
  pH: 7.4
  Partial charge method: gasteiger
✓ Created PDBQT: LIT_PCBA/ADRB2/receptor.pdbqt
✓ Created PDB: LIT_PCBA/ADRB2/receptor.pdb
✓ Receptor preparation complete!
```

**Verify output:**
```bash
# Check files were created
ls -lh LIT_PCBA/ADRB2/receptor.{pdbqt,pdb}

# Inspect PDBQT (should have ATOM records and charges)
head -20 LIT_PCBA/ADRB2/receptor.pdbqt
```

**Common errors:**
- `obabel not found` → Install OpenBabel in conda env
- `Invalid MOL2 file` → Check MOL2 file integrity

### 2. Test Ligand Preparation (Single Ligand)

```bash
# First, you need to know a ligand ID from your SMILES files
# Look at one:
head -5 LIT_PCBA/ADRB2/actives.smi

# Test converting one ligand (replace LIGAND_ID with actual ID)
snakemake LIT_PCBA/ADRB2/pdbqt/actives/LIGAND_ID.pdbqt --cores 1 -p
```

**Expected output:**
```
Processing: LIGAND_ID
  SMILES: CC(C)Cc1ccc(cc1)C(C)C(O)=O
✓ Created: LIT_PCBA/ADRB2/pdbqt/actives/LIGAND_ID.pdbqt
```

**Verify:**
```bash
# Check PDBQT was created
ls -lh LIT_PCBA/ADRB2/pdbqt/actives/LIGAND_ID.pdbqt

# Should contain ATOM/HETATM records
head -20 LIT_PCBA/ADRB2/pdbqt/actives/LIGAND_ID.pdbqt
```

### 3. Test Batch Preparation (Small Subset)

```bash
# Dry-run to see what will be prepared
snakemake prepare_all --dry-run | head -50

# Prepare all receptors (fast)
snakemake prepare_all_receptors --cores 3

# Prepare just a few ligands to test
# (checkpoint will trigger automatically)
snakemake prepare_all --cores 8 --resources mem_mb=32000
```

**Monitor progress:**
```bash
# Watch the checkpoint output
tail -f data/logs/preparation/ligands_checkpoint.log

# Check manifest to see progress
snakemake show_manifest_stats
```

**Common errors:**
- `Invalid SMILES` → Check SMILES file format
- `3D embedding failed` → Some SMILES may be invalid, expected for small %
- Out of memory → Reduce cores or increase memory

---

## Stage 3: Docking

### 1. Test Single Ligand Docking (CPU)

```bash
# Test docking one prepared ligand (replace LIGAND_ID)
snakemake LIT_PCBA/ADRB2/docked_vina/actives/LIGAND_ID_docked.pdbqt \
  --cores 1 \
  --config mode=cpu
```

**Expected output:**
```
Docking: LIGAND_ID.pdbqt
  Receptor: LIT_PCBA/ADRB2/receptor.pdbqt
  Box center: (10.5, 20.3, 15.7)
  Box size: (25.0, 25.0, 25.0)
  Mode: cpu
  Exhaustiveness: 8
  Num modes: 9
✓ Docking complete: LIT_PCBA/ADRB2/docked_vina/actives/LIGAND_ID_docked.pdbqt
  Log: LIT_PCBA/ADRB2/docked_vina/actives/log/LIGAND_ID.log
  Best score: -8.5 kcal/mol
```

**Verify:**
```bash
# Check docked file
ls -lh LIT_PCBA/ADRB2/docked_vina/actives/LIGAND_ID_docked.pdbqt

# Check log for scores
cat LIT_PCBA/ADRB2/docked_vina/actives/log/LIGAND_ID.log

# Should see binding modes like:
#    mode |   affinity | dist from best mode
#         | (kcal/mol) | rmsd l.b.| rmsd u.b.
# -----+------------+----------+----------
#    1       -8.5          0.000      0.000
#    2       -8.3          1.234      2.456
```

### 2. Test GPU Docking (if available)

```bash
# Check if GPU is available
nvidia-smi

# Test GPU docking (if you have Vina-GPU installed)
snakemake LIT_PCBA/ADRB2/docked_vina/actives/LIGAND_ID_docked.pdbqt \
  --cores 1 \
  --use-conda \
  --config mode=gpu
```

### 3. Test Batch Docking (Small Subset)

```bash
# Dry-run to estimate job count
snakemake dock_all --dry-run | grep "job counts"

# Run docking checkpoint (shows what needs docking)
snakemake docking_checkpoint

# Dock small batch (limit with --until or time)
snakemake dock_all --cores 4 --until dock_all
```

**Common errors:**
- `vina not found` → Check Vina installation path in config.yaml
- `Receptor/ligand not prepared` → Run preparation stage first
- GPU errors → Check CUDA installation and device availability

---

## Stage 4: SDF Conversion

### 1. Test Single Conversion

```bash
# Convert one docked ligand to SDF (replace LIGAND_ID)
snakemake LIT_PCBA/ADRB2/docked_sdf/actives/LIGAND_ID.sdf --cores 1 -p
```

**Expected output:**
```
Converting: LIGAND_ID_docked
  Input: LIT_PCBA/ADRB2/docked_vina/actives/LIGAND_ID_docked.pdbqt
  Output: LIT_PCBA/ADRB2/docked_sdf/actives/LIGAND_ID.sdf
  Model: 0
✓ Created SDF: LIT_PCBA/ADRB2/docked_sdf/actives/LIGAND_ID.sdf
  Extracted model: 0
```

**Verify:**
```bash
# Check SDF file
head -50 LIT_PCBA/ADRB2/docked_sdf/actives/LIGAND_ID.sdf

# Should contain molecular structure with atoms
```

### 2. Test Batch Conversion

```bash
# Convert all docked ligands
snakemake convert_all_to_sdf --cores 8
```

---

## Stage 5: AEV-PLIG Data Preparation

### 1. Test Single Target

```bash
# Prepare AEV-PLIG CSV for one target
snakemake LIT_PCBA/ADRB2/rescoring/datasets/aev_plig_ADRB2.csv --cores 1 -p
```

**Expected output:**
```
Processing 500 docked ligands for ADRB2
✓ Created AEV-PLIG CSV: LIT_PCBA/ADRB2/rescoring/datasets/aev_plig_ADRB2.csv
  Ligands: 485
  Actives: 242
  Inactives: 243
✓ AEV-PLIG data preparation complete!
```

**Verify CSV:**
```bash
# Check CSV structure
head -3 LIT_PCBA/ADRB2/rescoring/datasets/aev_plig_ADRB2.csv

# Should show:
# unique_id,Protein_ID,sdf_file,protein_pdb,MW,LogP,HBD,HBA,DockingScore,pK,is_active
# ADRB2_ligand1,ADRB2,/path/to/ligand1.sdf,/path/to/receptor.pdb,342.4,2.3,2,4,-8.5,6.2,1

# Count entries
wc -l LIT_PCBA/ADRB2/rescoring/datasets/aev_plig_ADRB2.csv
```

### 2. Test All Targets

```bash
snakemake prepare_all_aev_plig --cores 4
```

---

## Full Pipeline Test

### 1. Dry-Run Full Pipeline

```bash
# See complete DAG without executing
snakemake --dry-run -n

# Visualize DAG (requires graphviz)
snakemake --dag | dot -Tpdf > workflow_dag.pdf
```

### 2. Run Full Pipeline (Small Test Set)

**Option A: Limit to one target**
```bash
# Edit targets.yaml to include only one target
# Then run:
snakemake --cores 8 --use-conda
```

**Option B: Use Snakemake's job limiting**
```bash
# Run but limit to 10 jobs total
snakemake --cores 8 --jobs 10 --use-conda
```

### 3. Monitor Progress

```bash
# Check manifest stats periodically
watch -n 10 snakemake show_manifest_stats

# Check log files
tail -f data/logs/preparation/*.log
tail -f data/logs/docking/*.log
tail -f data/logs/conversion/*.log
tail -f data/logs/rescoring/*.log
```

---

## Troubleshooting

### Common Issues and Solutions

**1. Conda environment issues**
```bash
# Create environment manually
conda env create -f workflow/envs/vscreen.yaml
conda activate vscreen

# Test RDKit
python -c "from rdkit import Chem; print('RDKit OK')"

# Test OpenBabel
which obabel
obabel --version
```

**2. Manifest corruption**
```bash
# Recreate manifest from scratch
rm data/master/manifest.parquet
snakemake create_manifest --cores 1

# Restore from backup
cp data/master/backups/manifest_TIMESTAMP.parquet data/master/manifest.parquet
```

**3. Checkpoint issues**
```bash
# Clear checkpoint cache
rm -rf .snakemake/

# Force checkpoint rerun
snakemake --forceall create_manifest
```

**4. File permission errors**
```bash
# Make scripts executable
chmod +x workflow/scripts/*.py

# Fix directory permissions
chmod -R u+w data/
```

**5. Path issues**
```bash
# Check working directory
snakemake --printshellcmds

# Use absolute paths in config.yaml if needed
```

---

## Performance Testing

### Benchmark Individual Stages

```bash
# Time each stage
time snakemake prepare_all_receptors --cores 3
time snakemake prepare_all --cores 16
time snakemake dock_all --cores 32

# Use Snakemake's benchmark
snakemake --benchmark benchmarks.tsv
```

### Resource Monitoring

```bash
# Monitor resource usage
htop  # CPU/memory
nvidia-smi -l 1  # GPU (if applicable)

# Snakemake resource usage
snakemake --profile workflow/profiles/arc --cluster-status slurm_status.py
```

---

## Validation Checklist

After each stage, verify:

- [ ] **Configuration**
  - [ ] `validate_config` passes
  - [ ] All targets in targets.yaml are valid

- [ ] **Manifest**
  - [ ] Manifest created without errors
  - [ ] Entry count matches expected (ligands × targets)
  - [ ] All SMILES files found

- [ ] **Preparation**
  - [ ] All receptor PDBQT/PDB files created
  - [ ] Ligand PDBQT files created (expect ~95%+ success)
  - [ ] preparation_status updated in manifest

- [ ] **Docking**
  - [ ] Docked PDBQT files created
  - [ ] Log files contain binding affinities
  - [ ] docking_status updated in manifest

- [ ] **Conversion**
  - [ ] SDF files created for all docked ligands
  - [ ] SDF files contain valid structures

- [ ] **Rescoring**
  - [ ] AEV-PLIG CSV files created per target
  - [ ] CSV contains all required columns
  - [ ] Molecular descriptors are reasonable

---

## Testing on HPC/Cluster

If using ARC or similar cluster:

```bash
# Test cluster profile
snakemake --profile workflow/profiles/arc --dry-run

# Submit small test job
snakemake prepare_all_receptors --profile workflow/profiles/arc

# Monitor SLURM queue
squeue -u $USER
sacct -u $USER

# Check cluster logs
cat data/logs/preparation/*.log
```

---

## Quick Smoke Test

Run this to quickly validate the entire pipeline is working:

```bash
#!/bin/bash
# smoke_test.sh - Quick validation of pipeline

echo "=== Configuration Test ==="
snakemake validate_config || exit 1

echo "=== Manifest Test ==="
snakemake create_manifest --cores 1 || exit 1
snakemake show_manifest_stats

echo "=== Preparation Test (1 receptor) ==="
snakemake LIT_PCBA/ADRB2/receptor.pdbqt --cores 1 || exit 1

echo "=== All tests passed! ==="
```

Run with: `bash smoke_test.sh`

---

## Next Steps

After testing:

1. Document any configuration changes needed
2. Adjust resource requirements in config.yaml based on testing
3. Create cluster profile for your HPC system
4. Set up monitoring/logging for production runs
5. Create backup scripts for manifest files

For production deployment, consider:
- Setting up automatic backups
- Monitoring disk space
- Log rotation
- Error notifications
- Progress tracking dashboard
