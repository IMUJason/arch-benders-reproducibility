# C&OR Reproducibility Map

This map identifies the exact result files, scripts, and manuscript components used by the active `paper/cor_submission/` package.

## Manuscript-level evidence map

- Pilot architecture effect:
  - Manuscript location: `paper/cor_submission/sections/results_generated.tex`
  - Source files: `results/per_run_metrics.csv`, `results/summary_metrics.csv`
  - Primary script lineage: `legacy/reproducibility_package/run_pipeline.py`

- Public benchmark single-pass screening:
  - Manuscript location: `paper/cor_submission/sections/arch_rp_single_pass_generated.tex`
  - Source files: `results/arch_rp_stage8_compare_runs.csv`, `results/arch_rp_stage8_compare_overall.csv`
  - Primary script lineage: `legacy/reproducibility_package/run_arch_rp_stage8_compare.py`

- Public benchmark repeated timing audit:
  - Manuscript location: `paper/cor_submission/sections/arch_rp_repeat_generated.tex`
  - Source files: `results/arch_rp_stage8_repeat_timing_runs.csv`, `results/arch_rp_stage8_repeat_timing_paired.csv`, `results/arch_rp_stage8_repeat_timing_summary.csv`, `results/arch_rp_stage8_repeat_timing_overall.csv`
  - Conservative inference layer: `results/arch_rp_stage8_repeat_timing_clustered_summary.csv`, `results/arch_rp_stage8_repeat_timing_clustered_overall.csv`
  - Primary script lineage: `legacy/reproducibility_package/run_arch_rp_stage8_repeat_timing.py`, `legacy/reproducibility_package/run_arch_rp_reporting.py`

- Stage VIII ranking ablation:
  - Manuscript location: `paper/cor_submission/sections/arch_rp_ranking_ablation_generated.tex`
  - Source files: `results/arch_rp_stage8_ranking_ablation_runs.csv`, `results/arch_rp_stage8_ranking_ablation_overall.csv`, `results/arch_rp_stage8_ranking_ablation_direct.csv`
  - Hard-family repeat layer: `results/arch_rp_stage8_ranking_ablation_hard_repeat_runs.csv`, `results/arch_rp_stage8_ranking_ablation_hard_repeat_overall.csv`, `results/arch_rp_stage8_ranking_ablation_hard_repeat_clustered_overall.csv`
  - Primary script lineage: `legacy/reproducibility_package/run_arch_rp_stage8_ranking_ablation.py`, `legacy/reproducibility_package/run_arch_rp_stage8_ranking_ablation_hard_repeat.py`, `legacy/reproducibility_package/run_arch_rp_stage8_ranking_ablation_reporting.py`

- Extensive-form contextualization:
  - Manuscript location: `paper/cor_submission/sections/extensive_form_generated.tex`
  - Source files: `results/extensive_form_grid_context_summary.csv`

- Large-scale Stage IX diagnostics:
  - Manuscript location: `paper/cor_submission/sections/stage9_large_generated.tex`, `paper/cor_submission/sections/stage9_cut_productivity_generated.tex`, `paper/cor_submission/sections/stage9_sensitivity_generated.tex`
  - Source files: `results/arch_rp_stage9_sensitivity_runs.csv`, `results/arch_rp_stage9_cut_productivity_runs.csv`, related generated summaries in `results/`

- Long-running Stage X benchmark:
  - Manuscript location: `paper/cor_submission/sections/stage10_long_generated.tex`
  - Source files: `results/arch_rp_stage10_long_runs.csv`, `results/arch_rp_stage10_long_summary.csv`

## Artifact package for submission

- Code supplement root: `legacy/reproducibility_package/`
- Results supplement root: `results/`
- Manuscript package: `paper/cor_submission/`

## Supersession note

- The active C&OR manuscript is based on the `ARCH-RP` / Stage VIII-IX-X result family.
- Earlier `round2_*` files remain archived for historical traceability but are not the evidentiary basis for the active C&OR manuscript.
- The historical `paper/opre_submission/` package is retained unchanged and should not be mixed with the C&OR upload bundle.
- See `results/manifests/cor_submission_supersession_note.md` for a compact file-family precedence rule.
