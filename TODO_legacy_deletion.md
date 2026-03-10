# Legacy Code Deletion Checklist

This document tracks legacy code that can be safely deleted in future commits.

## Batch 1: Safe to Delete (High Priority) - COMPLETED ✓

These files have been deleted (commit 311fa10):

### Duplicate Files
- [x] `pymol_box_tmp.py` - Exact duplicate of `pymol_box.py`

### Test/Development Scripts
- [x] `hello_mpi.py` - Simple MPI hello world test script (7 lines)

### Root-level Legacy SLURM Scripts
- [x] `test_submit_cpu.slurm` - Devel test for CPU docking
- [x] `test_submit_gpu.slurm` - Devel test for GPU docking
- [x] `test_sdf.slurm` - Test SDF conversion
- [x] `vina_smoke.slurm` - Smoke test for GPU docking
- [x] `submit_cpu.slurm` - CPU docking submission
- [x] `submit_gpu.slurm` - GPU docking submission
- [x] `vina_gpu_array.slurm` - GPU array job variant
- [x] `aev_plig_array.slurm` - AEV-PLIG rescoring (root level)
- [x] `chunks_and_csv_array.slurm` - Ligand preparation
- [x] `generate_sdfs_array.slurm` - SDF conversion
- [x] `smi2pdbqt_array.slurm` - SMILES to PDBQT conversion
- [x] `mol2_to_pdbqt.slurm` - MOL2 to PDBQT conversion
- [x] `sdf_conversion.slurm` - PDBQT to SDF conversion
- [x] `plig_manifest.slurm` - Manifest generation

## Batch 2: Verify Before Deleting (Medium Priority)

Verify these are not referenced in documentation or external tools before deleting.

### Superseded Python Scripts
- [ ] `run_vina_gpu.py` - Non-MPI version, replaced by `run_vina_gpu_mpi.py` (183 lines)

### Alternative Workflows
- [ ] `workflow/Snakefile_slurm` - Alternative SLURM workflow, may be obsolete

### Old CLI Tools
These appear to be old standalone tools that may have been replaced by the unified workflow:

- [ ] `make_boxes.py` (180 lines) - Box generation
- [ ] `make_sdf_files.py` (575 lines) - SDF file creation
- [ ] `make_aev_plig.py` (222 lines) - AEV-PLIG processing
- [ ] `make_aev_plig_best.py` (174 lines) - AEV-PLIG best selection
- [ ] `make_metrics.py` (201 lines) - Metrics calculation
- [ ] `dock.py` (321 lines) - Docking CLI
- [ ] `analyse_datasets.py` (172 lines) - Dataset analysis

## Batch 3: Consider Consolidating (Low Priority)

### PyMOL Box Scripts
- [ ] Consolidate `pymol_box.py` and `pymol_box_from_lig.py` into a single script
  - `pymol_box.py`: 23 lines, basic implementation
  - `pymol_box_from_lig.py`: 19 lines, safer approach with fallback

## Notes

- **Total files identified**: 26
- **Batch 1 deleted**: 16 files (~1,100 lines)
- **Remaining**: 10 files (~1,400 lines)
- Always run tests after deletion to ensure nothing breaks
- Check git history for any documentation referencing these files
