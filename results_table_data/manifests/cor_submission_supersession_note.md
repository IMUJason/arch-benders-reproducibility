# C&OR Submission Supersession Note

Updated: 2026-03-31

## Active package

- Active manuscript package: `paper/cor_submission/`
- Historical package retained for record only: `paper/opre_submission/`

## Active result families

- Pilot audit: `results/per_run_metrics.csv`, `results/summary_metrics.csv`
- Public benchmark: `results/arch_rp_stage8_compare_*.csv`, `results/arch_rp_stage8_repeat_timing_*.csv`, `results/arch_rp_stage8_repeat_timing_clustered_*.csv`
- Extensive-form context: `results/extensive_form_grid_context_summary.csv`
- Large-scale diagnostics: `results/arch_rp_stage9_*.csv`, `results/arch_rp_stage10_long_*.csv`

## Historical but non-claim-bearing files

- `results/round2_*`
- `results/arch_rp_repeat_timing_*`
- `results/arch_rp_adaptive_*`
- `results/hard_benchmark_*`

These historical files remain archived for traceability, but they are not the authoritative sources for the current C&OR manuscript claims.

## Use rule

- If a numerical statement appears in the C&OR manuscript, trace it first through `results/cor_reproducibility_map.md`.
- If similarly named result families coexist, prefer the Stage VIII, IX, and X files listed above unless the manuscript explicitly cites an earlier developmental run.
