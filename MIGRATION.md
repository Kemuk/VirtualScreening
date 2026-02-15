# Migration Guide: Workflow Consolidation

**Last Updated:** 2026-02-15
**Branch:** `claude/consolidate-workflow-scripts-C45Qg`

## Overview

The virtual screening pipeline has been significantly simplified and consolidated. This migration guide explains the changes and how to adapt your existing workflows.

## What Changed?

### Summary

- **34 Snakemake rules → 10 rules** (70% reduction)
- **12 SLURM scripts → 2 unified scripts** (83% reduction)
- **16 redundant files deleted** (1,751 lines removed)
- **Net reduction: -1,830 lines of code** (69% reduction)

### Key Improvements

1. **Unified SLURM Infrastructure**: All stages now use the same submission logic
2. **Simplified Rules**: Removed redundant chunking/merging rules
3. **Consistent Interface**: All production rules follow the same pattern
4. **Easier Maintenance**: Single source of truth for SLURM configuration

---

## Rule Changes

### Before vs. After

| Stage | Old Rules | New Rules | Change |
|-------|-----------|-----------|--------|
| **Preparation** | 6 rules | 2 rules | -67% |
| **Docking** | 7 rules | 2 rules | -71% |
| **Conversion** | 5 rules | 2 rules | -60% |
| **Rescoring** | 11 rules | 1 rule | -91% |
| **Results** | 5 rules | 3 rules | -40% |
| **TOTAL** | **34 rules** | **10 rules** | **-70%** |

---

## Migration Instructions

### 1. Preparation Stage

**BEFORE:**
```bash
# Old fragmented approach
snakemake shard_preparation
snakemake prepare_array
snakemake merge_preparation_results
```

**AFTER:**
```bash
# New unified approach
snakemake prepare_all
```

**What changed:**
- Removed: `shard_preparation`, `prepare_receptors_array`, `prepare_ligands_array`, `merge_preparation_results`
- Kept: `prepare_receptor` (single-file testing), `prepare_all` (production)

### 2. Docking Stage

**BEFORE:**
```bash
# Old fragmented approach
snakemake shard_docking
snakemake docking_array
snakemake merge_docking_results
```

**AFTER:**
```bash
# New unified approach
snakemake dock_all
```

**What changed:**
- Removed: `shard_docking`, `docking_array`, `merge_docking_results`, `dock_target`, `dock_target_gpu`
- Kept: `dock_ligand` (single-file testing), `dock_all` (production)

### 3. Conversion Stage

**BEFORE:**
```bash
# Old fragmented approach
snakemake shard_conversion
snakemake convert_array
snakemake merge_conversion_results
```

**AFTER:**
```bash
# New unified approach
snakemake convert_all
```

**What changed:**
- Removed: `shard_conversion`, `convert_array`, `merge_conversion_results`, `convert_target_to_sdf`
- Kept: `convert_to_sdf` (single-file testing), `convert_all` (production)

### 4. Rescoring Stage

**BEFORE:**
```bash
# Old complex multi-step approach
snakemake shard_rescoring
snakemake prepare_aev_plig_shard
snakemake prepare_aev_plig_array
snakemake aev_plig_array
snakemake merge_aev_plig_predictions
snakemake update_manifest_aev_plig
```

**AFTER:**
```bash
# New unified approach
snakemake rescore_all
```

**What changed:**
- Removed: All 10 intermediate rules
- Kept: `rescore_all` (production only)
- Note: Single-shard testing removed (use `mode=devel` instead)

### 5. Results Stage

**BEFORE:**
```bash
# Old approach with extra convenience rules
snakemake results_metrics_only
snakemake results_plots_only
```

**AFTER:**
```bash
# New simplified approach
snakemake compute_results  # For metrics only
snakemake make_plots       # For plots only
snakemake results_all      # For both
```

**What changed:**
- Removed: `results_metrics_only`, `results_plots_only` (redundant)
- Kept: `compute_results`, `make_plots`, `results_all`

---

## New Workflow Pattern

### Development Workflow (Testing Single Files)

```bash
# Test single receptor
snakemake LIT_PCBA/ADRB2/receptor.pdbqt --cores 1

# Test single ligand
snakemake LIT_PCBA/ADRB2/docked_vina/actives/LIGAND_ID_docked.pdbqt --cores 1

# Test single conversion
snakemake LIT_PCBA/ADRB2/docked_sdf/actives/LIGAND_ID.sdf --cores 1
```

### Production Workflow (Batch Processing)

```bash
# Run all stages sequentially
snakemake prepare_all --cores 1
snakemake dock_all --cores 1
snakemake convert_all --cores 1
snakemake rescore_all --cores 1
snakemake results_all --cores 4

# Or run the full pipeline
snakemake --cores 1
```

### Development Mode (Smaller Chunks)

```bash
# Use development mode for faster testing with smaller chunks
snakemake prepare_all --config mode=devel
snakemake dock_all --config mode=devel
```

---

## Script Changes

### Deleted Files (16 total)

**SLURM submission scripts (10 files):**
- `workflow/scripts/submit_preparation_array.sh`
- `workflow/scripts/submit_prepare_receptors_array.sh`
- `workflow/scripts/submit_prepare_ligands_array.sh`
- `workflow/scripts/submit_docking_array.sh`
- `workflow/scripts/submit_conversion_array.sh`
- `workflow/scripts/submit_prepare_aev_plig_array.sh`
- `workflow/scripts/submit_aev_plig_array.sh`
- And 3 other redundant scripts

**Python scripts (3 files):**
- `workflow/scripts/shard_stage.py` → replaced by `prepare_stage.py`
- `workflow/scripts/merge_stage_results.py` → replaced by `update_manifest.py`
- `workflow/scripts/write_stage_chunk.py` → integrated into `prepare_stage.py`

**Template files (2 files):**
- `workflow/slurm/worker_template.sh` → replaced by `stage_worker.slurm`
- Old manifest merger

### New Unified Files (2 files)

**`workflow/slurm/submit_stage.py`** (497 lines)
- Handles all SLURM job submissions
- Supports all stages: preparation, docking, conversion, aev_infer
- Single source of truth for chunking, resource allocation, job submission

**`workflow/slurm/stage_worker.slurm`** (124 lines)
- Unified SLURM worker template
- Works for all stages via stage-specific dispatching
- Handles resource allocation, conda activation, error handling

### Enhanced Files

**`workflow/slurm/stage_config.py`** (+124 lines)
- Added comprehensive stage configuration
- Defines resources, scripts, conda environments per stage
- Centralized configuration for all stages

---

## Configuration Changes

### No Breaking Changes

All existing configuration in `config/config.yaml` and `config/targets.yaml` remains compatible.

### New Optional Settings

```yaml
# config/config.yaml

# Mode selection (unchanged)
mode: production  # or 'devel'

# Chunking configuration (unchanged)
chunking:
  production:
    chunks: 200      # CPU-based stages
    gpu_chunks: 100  # GPU-based stages
  devel:
    chunks: 5
    gpu_chunks: 2
```

---

## Benefits

### 1. **Simpler Mental Model**

**Before:**
- Multiple steps per stage (shard → array → merge)
- Different submission scripts per stage
- Complex rule dependencies

**After:**
- One rule per stage (`prepare_all`, `dock_all`, etc.)
- Single unified submission system
- Clear, linear workflow

### 2. **Easier Maintenance**

**Before:**
- Change SLURM settings in 12 different files
- Update chunking logic in multiple scripts
- Fix bugs in parallel implementations

**After:**
- Change SLURM settings in `submit_stage.py` once
- Update chunking logic in `stage_config.py` once
- Fix bugs in a single worker template

### 3. **Consistent Behavior**

All stages now:
- Use the same chunking strategy
- Have the same error handling
- Generate logs in the same format
- Update the manifest in the same way

### 4. **Better Testing**

**Before:**
- Test each stage's submission script separately
- Validate each merger separately
- Debug different SLURM templates

**After:**
- Test `submit_stage.py` once for all stages
- Validate `update_manifest.py` once for all stages
- Debug `stage_worker.slurm` once for all stages

---

## Troubleshooting

### If you have running jobs from the old workflow

```bash
# Cancel old jobs
scancel -u $USER

# Clean up old chunk directories
rm -rf data/chunks/*/chunk_*.csv

# Re-run with new workflow
snakemake prepare_all --cores 1
```

### If you encounter "rule not found" errors

**Error:**
```
RuleException: No rule to produce shard_preparation
```

**Solution:**
The old rules have been removed. Use the new unified rules:
- `shard_preparation` → `prepare_all`
- `shard_docking` → `dock_all`
- `shard_conversion` → `convert_all`
- `shard_rescoring` → `rescore_all`

### If chunk files are missing

The new workflow handles chunking internally. You don't need to manually create chunks anymore.

```bash
# Just run the production rule
snakemake prepare_all
```

---

## Testing the Migration

### Quick Validation

```bash
# 1. Check configuration
snakemake validate_config

# 2. Dry-run to see new rules
snakemake --dry-run -n

# 3. Test single-file rules (should work as before)
snakemake LIT_PCBA/ADRB2/receptor.pdbqt --cores 1

# 4. Test new unified rules
snakemake prepare_all --config mode=devel --dry-run
```

### Full Testing

See `TESTING.md` for comprehensive testing procedures.

---

## Questions?

If you encounter issues or have questions about the migration:

1. Check this migration guide
2. Review `TESTING.md` for testing procedures
3. Check the git history: `git log --oneline --graph`
4. View the PR/commit messages for detailed explanations

---

## Rollback Instructions

If you need to rollback to the old workflow:

```bash
# Switch to the previous branch
git checkout main

# Or view specific old files
git show main:workflow/rules/preparation.smk
```

Note: The old workflow is deprecated and will not receive updates. Please report any issues with the new workflow for fixes.
