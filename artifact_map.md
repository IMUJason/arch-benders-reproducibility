# Artifact Map: Paper Tables to Data Files

Each table and figure in the manuscript is computed directly from the
aggregate CSV files in `results_table_data/` (relative paths below). All
aggregate files were produced from the archived per-run summaries and traces;
no number in the paper was transcribed by hand.

| Manuscript table | Source file(s) |
|---|---|
| Pilot summary (Tab. 1) | `summary_metrics.csv`, `per_run_metrics.csv` |
| Pilot robustness sweep (30 seeds, Sec. 5) | `pilot_robustness_30inst.csv` |
| Extensive-form contextualization (Tab. 2) | `extensive_form_grid_context_summary.csv`, `extensive_form_grid_context_runs.csv` |
| Public benchmark single-pass (Tab. 3) | `arch_rp_stage8_compare_overall.csv` |
| Public benchmark seven-repeat audit (Tab. 4) | `arch_rp_stage8_repeat_timing_overall.csv`, `arch_rp_stage8_repeat_timing_clustered_summary.csv` |
| Stage IX large-scale activation (Tab. 5) | `arch_rp_stage9_large_scale_summary.csv`, `arch_rp_stage9_large_scale_paired.csv` |
| Stage IX cut-productivity audit | `arch_rp_stage9_cut_productivity_summary.csv`, `arch_rp_stage9_cut_productivity_runs.csv` |
| Stage IX budget decomposition (Fig.) | `arch_rp_stage9_budget_audit.csv` |
| Stage IX sensitivity sweep (Tab. 6) | `arch_rp_stage9_sensitivity_summary.csv` |
| Ranking score correlation (Tab. 7) | `arch_rp_correlation_analysis.csv` |
| Stage X long-running benchmark (Tab. 8) | `arch_rp_stage10_long_summary.csv` |
| Hold-out generalization check (Tab. 9) | `arch_rp_stage9_expanded_8inst.csv`, `arch_rp_stage10_expanded_8inst.csv` |

Result-family precedence (which file family is authoritative when similarly
named families coexist) is recorded in
`results_table_data/manifests/cor_submission_supersession_note.md`.

## Scope note

The package contains the solver engine, the pilot entry point, the unit-test
suite, and the aggregate data behind the paper's tables. The batch harness
scripts that drove the stage-specific benchmark sweeps and the full per-run
JSON trace archive (≈750 MB) are part of the journal supplement deposit and
are available from the corresponding author on request.
