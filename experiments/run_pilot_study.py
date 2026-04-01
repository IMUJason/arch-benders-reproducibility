"""
Pilot Study: ARCH-Benders vs Classical Multicut Benders

This script runs a simple comparison between classical multicut Benders
and the ARCH-Benders hybrid variants on a small set of instances.

Usage:
    python experiments/run_pilot_study.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.instance_generation import generate_instance, save_instance
from src.solver import solve_instance
from src.models import SolverConfig


def run_classical_benders(instance):
    """Run classical multicut Benders (baseline)."""
    config = SolverConfig(
        solver_type="multicut",
        time_limit_seconds=300,
        gap_tolerance=1e-4,
    )
    return solve_instance(instance, config)


def run_arch_benders(instance, probe_budget=3, gap_weight=0.6, sensitivity_weight=0.4):
    """Run ARCH-Benders with ranked probing."""
    config = SolverConfig(
        solver_type="hybrid",
        time_limit_seconds=300,
        gap_tolerance=1e-4,
        hybrid_candidate_budget=10,
        hybrid_probe_budget=probe_budget,
        probe_ranking_mode="weighted_sum",
        probe_gap_weight=gap_weight,
        probe_sensitivity_weight=sensitivity_weight,
        candidate_targeting_mode="ranked",
    )
    return solve_instance(instance, config)


def main():
    """Run pilot study on small instances."""
    print("=" * 70)
    print("ARCH-Benders Pilot Study")
    print("=" * 70)

    # Test configurations
    test_configs = [
        {"n_facilities": 24, "n_customers": 36, "n_scenarios": 20, "seed": 42},
        {"n_facilities": 24, "n_customers": 36, "n_scenarios": 30, "seed": 43},
        {"n_facilities": 28, "n_customers": 42, "n_scenarios": 40, "seed": 44},
    ]

    results = []

    for i, config in enumerate(test_configs, 1):
        print(f"\n--- Test {i}/{len(test_configs)} ---")
        print(f"Configuration: {config['n_facilities']} facilities, "
              f"{config['n_customers']} customers, "
              f"{config['n_scenarios']} scenarios")

        # Generate instance
        print("Generating instance...")
        instance = generate_instance(
            instance_id=f"pilot_{i}",
            n_facilities=config["n_facilities"],
            n_customers=config["n_customers"],
            n_scenarios=config["n_scenarios"],
            seed=config["seed"],
        )

        # Save instance
        instance_path = Path(__file__).parent.parent / "data" / "sample_instances" / f"pilot_{i}.json"
        save_instance(instance, instance_path)
        print(f"Instance saved to: {instance_path}")

        # Run classical Benders
        print("\nRunning classical multicut Benders...")
        classical_result = run_classical_benders(instance)
        print(f"  Major iterations: {classical_result.major_iterations}")
        print(f"  Total time: {classical_result.total_time:.2f}s")
        print(f"  Final gap: {classical_result.relative_gap:.2%}")

        # Run ARCH-Benders
        print("\nRunning ARCH-Benders...")
        arch_result = run_arch_benders(instance)
        print(f"  Major iterations: {arch_result.major_iterations}")
        print(f"  Total time: {arch_result.total_time:.2f}s")
        print(f"  Final gap: {arch_result.relative_gap:.2%}")

        # Compare
        iteration_reduction = (
            (classical_result.major_iterations - arch_result.major_iterations) /
            max(1, classical_result.major_iterations) * 100
        )
        print(f"\n  Iteration reduction: {iteration_reduction:.1f}%")

        results.append({
            "test": i,
            "n_scenarios": config["n_scenarios"],
            "classical_iters": classical_result.major_iterations,
            "arch_iters": arch_result.major_iterations,
            "iteration_reduction_pct": iteration_reduction,
            "classical_time": classical_result.total_time,
            "arch_time": arch_result.total_time,
        })

    # Summary
    print("\n" + "=" * 70)
    print("PILOT STUDY SUMMARY")
    print("=" * 70)
    print(f"{'Test':<6} {'Scenarios':<12} {'Classical':<12} {'ARCH-Benders':<14} {'Reduction':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['test']:<6} {r['n_scenarios']:<12} {r['classical_iters']:<12} "
              f"{r['arch_iters']:<14} {r['iteration_reduction_pct']:<11.1f}%")

    mean_reduction = np.mean([r["iteration_reduction_pct"] for r in results])
    print("-" * 70)
    print(f"{'MEAN':<6} {'':<12} {'':<12} {'':<14} {mean_reduction:<11.1f}%")
    print("=" * 70)

    print("\nNote: This pilot study demonstrates the reduction in major iterations")
    print("achieved by ARCH-Benders through ranked scenario probing.")


if __name__ == "__main__":
    main()
