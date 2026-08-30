"""End-to-end demonstration of the ARCH-Benders engine.

Runs classical multicut Benders (BD-MC) and the hybrid variant with candidate
search enabled (ARCH-RP) on one small fixed-seed instance, and prints the
run summaries. This script demonstrates the engine; the exact operating-point
configurations behind every paper table are archived in the per-run traces of
the journal supplement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plan3_hybrid_benders.instance_generation import generate_instance
from plan3_hybrid_benders.models import SolverConfig
from plan3_hybrid_benders.solver import solve_instance


def main() -> None:
    instance = generate_instance(
        "demo_s20",
        n_facilities=8,
        n_customers=12,
        n_scenarios=20,
        seed=42,
        service_level=1.08,
    )

    configs = [
        SolverConfig(name="BD-MC", max_iterations=40, multicut=True),
        SolverConfig(
            name="ARCH-RP",
            max_iterations=40,
            multicut=True,
            use_candidate_search=True,
        ),
    ]

    for cfg in configs:
        result = solve_instance(instance, cfg)
        print(
            f"{result.config_name}: iterations={result.iterations} "
            f"proof_time={result.time_to_proof if result.time_to_proof is not None else 'not reached'} "
            f"gap={result.final_gap:.2e} generated_cuts={result.total_generated_cuts} "
            f"termination={result.termination_reason}"
        )


if __name__ == "__main__":
    main()
