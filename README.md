# ARCH-Benders: Exact Hybrid Benders with Ranked Probing
## Reproducibility Package

This package accompanies the manuscript

> **ARCH-Benders: Exact Hybrid Benders with Ranked Probing for Stochastic Network Design** (under review, *Computational Management Science*)

and contains the core implementation, the experiment entry point, and the aggregate result tables that support every table and figure in the paper.

## What this study shows, honestly

ARCH-Benders keeps the exact restricted master as the sole source of certified
lower bounds and adds a budgeted auxiliary layer that proposes candidate
first-stage designs and ranks scenarios for exact evaluation. The paper is a
**diagnostic study**: the pilot suite and extensive-form contextualization show
where the hybrid architecture helps, and the large-scale families show where it
does not. The main empirical finding is that **cut conversion, not activation
or overhead, is the bottleneck**: activated heuristic cuts lift the next-master
lower bound (24.7 objective units on average on the primary 32×48×320 family)
but none remains supporting at later iterations, and on eight fresh hold-out
seeds per family neither the hybrid advantage nor the activation rule
transfers. The value of the package is that every number in the paper is
traceable to the archives in `results_table_data/`.

## Quick start

Python 3.10+ with NumPy, SciPy (≥1.15), pandas, and matplotlib is sufficient.
The solver is HiGHS, called through `scipy.optimize.milp` (master MIP) and
`scipy.optimize.linprog` (recourse LPs); no commercial solver license is
required.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python experiments/run_pilot_study.py          # end-to-end pilot run
python -m pytest tests/ -q                     # 30 tests
```

## Package structure

```
├── src/                          # Core implementation (9 modules)
│   ├── solver.py                 # Main Benders solver with exactness guardrail
│   ├── master_problem.py         # Restricted master (scipy.optimize.milp / HiGHS)
│   ├── subproblem.py             # Scenario recourse LPs (scipy linprog / HiGHS)
│   ├── candidate_search.py       # Budgeted auxiliary candidate generation
│   ├── cut_management.py         # Cut storage, retention, productivity audit
│   ├── models.py                 # Instance and run data structures
│   ├── instance_generation.py    # Fixed-seed stochastic instance generator
│   └── provenance.py             # Run-summary and trace archiving
├── experiments/
│   └── run_pilot_study.py        # Pilot study entry point (21 runs)
├── data/sample_instances/        # Two sample hard-family instances
├── tests/test_solver.py          # 30 unit tests (solver, cuts, provenance)
├── results_table_data/           # Aggregate tables behind every paper table
│   └── manifests/                # Result-family precedence (supersession note)
└── artifact_map.md               # Paper table → data file map
```

## Result archives and provenance

`results_table_data/` contains the aggregate CSV files from which the paper's
tables are computed, including the fresh-seed hold-out families
(`arch_rp_stage9_expanded_8inst.csv`, `arch_rp_stage10_expanded_8inst.csv`) and
the 30-seed pilot robustness sweep (`pilot_robustness_30inst.csv`).
`artifact_map.md` maps every paper table to its source file.

The full per-run JSON traces of the large-scale families
(≈750 MB, one file per run, including cut pools and iteration logs) are
deposited with the journal supplement and are available from the corresponding
author on request.

## Citation

```bibtex
@article{archbenders2026,
  title={ARCH-Benders: Exact Hybrid Benders with Ranked Probing for Stochastic Network Design},
  author={Cao, Jin Xin},
  journal={Computational Management Science},
  year={2026},
  note={Under review}
}
```

## Acknowledgements

Supported in part by the National Natural Science Foundation of China (Grants
72461028 and 71961024) and the Key Technology Research Plan of Inner Mongolia
Autonomous Region (Grant 2019GG287).

## License

MIT License. See `LICENSE`.

Corresponding author: Jin Xin Cao (imucjx@163.com)
