from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plan3_hybrid_benders.instance_generation import generate_instance
from plan3_hybrid_benders.candidate_search import (
    _compute_geometry_pair_proxy,
    generate_candidates,
)
from plan3_hybrid_benders.models import BendersCut, SolverConfig
from plan3_hybrid_benders.solver import (
    _build_cuts_for_design,
    _effective_anchor_gap_density_threshold,
    _effective_heuristic_eval_limit,
    _effective_heuristic_cut_limit,
    _passes_search_activation_concentration_gate,
    _probe_effective_support_ratio,
    _probe_top_mass_share,
    _passes_activation_support_gate,
    _effective_scenario_probe_limit,
    _progressive_heuristic_evaluation,
    _score_probe_scenarios,
    _select_nonredundant_probe_schedule,
    solve_instance,
)
from plan3_hybrid_benders.subproblem import solve_scenario_recourse


def test_subproblem_cut_matches_design_value() -> None:
    instance = generate_instance(
        "unit_test_instance",
        n_facilities=3,
        n_customers=4,
        n_scenarios=3,
        seed=5,
    )
    design = [1, 0, 1]
    evaluation = solve_scenario_recourse(instance, 0, design)
    cut_value = evaluation.intercept + sum(
        coefficient * variable
        for coefficient, variable in zip(evaluation.coefficients, design, strict=True)
    )
    assert abs(cut_value - evaluation.objective_value) < 1e-5
    assert evaluation.solve_time_seconds >= 0.0


def test_hybrid_solver_keeps_valid_bounds() -> None:
    instance = generate_instance(
        "unit_test_hybrid",
        n_facilities=4,
        n_customers=5,
        n_scenarios=4,
        seed=13,
    )
    config = SolverConfig(
        name="TEST-HEB",
        multicut=True,
        use_candidate_search=True,
        candidate_budget=4,
        annealing_steps=20,
        exact_correction_frequency=2,
        use_filtering=True,
        use_strong_cuts=True,
        max_iterations=10,
        random_seed=7,
    )
    result = solve_instance(instance, config)
    assert result.success
    assert result.incumbent_objective is not None
    assert result.certified_lower_bound is not None
    assert result.certified_lower_bound <= result.incumbent_objective + 1e-6
    assert result.cut_pool_size > 0


def test_progressive_heuristic_screening_prunes_dominated_candidate() -> None:
    instance = generate_instance(
        "unit_test_progressive_screening",
        n_facilities=4,
        n_customers=5,
        n_scenarios=6,
        seed=17,
    )
    config = SolverConfig(
        name="TEST-HEB-SCREEN",
        multicut=True,
        use_candidate_search=True,
        max_iterations=5,
        random_seed=19,
    )
    design = [1, 0, 1, 0]
    expected_recourse, cuts, _ = _build_cuts_for_design(
        instance=instance,
        config=config,
        design=design,
        iteration=1,
        source="exact_rmp",
        alternative_point=[0.5] * instance.n_facilities,
    )
    incumbent_objective = sum(
        cost * variable for cost, variable in zip(instance.opening_costs, design, strict=True)
    ) + expected_recourse

    (
        screened_recourse,
        screened_cuts,
        details,
        screened_out,
        screening_bound,
        partial_only,
        probe_economy_metadata,
    ) = (
        _progressive_heuristic_evaluation(
            instance=instance,
            config=config,
            design=design,
            iteration=2,
            source="annealed",
            alternative_point=[0.5] * instance.n_facilities,
            cut_pool=cuts,
            incumbent_objective=incumbent_objective,
        )
    )

    assert screened_out
    assert screened_recourse is None
    assert screening_bound is not None
    assert screening_bound >= incumbent_objective - config.tolerance
    assert 1 <= len(details) < instance.n_scenarios
    assert len(screened_cuts) == len(details)
    assert not partial_only
    assert probe_economy_metadata["probe_economy_early_stop"] is False


def test_progressive_heuristic_partial_probe_harvests_valid_cuts() -> None:
    instance = generate_instance(
        "unit_test_partial_probe",
        n_facilities=5,
        n_customers=6,
        n_scenarios=8,
        seed=23,
    )
    config = SolverConfig(
        name="TEST-HEB-PROBE",
        multicut=True,
        use_candidate_search=True,
        heuristic_scenario_probe_limit=3,
        max_iterations=5,
        random_seed=29,
    )
    anchor_design = [1, 0, 1, 0, 1]
    _, base_cuts, _ = _build_cuts_for_design(
        instance=instance,
        config=config,
        design=anchor_design,
        iteration=1,
        source="exact_rmp",
        alternative_point=[0.5] * instance.n_facilities,
    )

    (
        partial_recourse,
        partial_cuts,
        details,
        screened_out,
        screening_bound,
        partial_only,
        probe_economy_metadata,
    ) = _progressive_heuristic_evaluation(
        instance=instance,
        config=config,
        design=[1, 1, 1, 0, 0],
        iteration=2,
        source="greedy_refine",
        alternative_point=[0.5] * instance.n_facilities,
        cut_pool=base_cuts,
        incumbent_objective=None,
        scenario_probe_limit=3,
    )

    assert partial_recourse is None
    assert not screened_out
    assert screening_bound is None
    assert partial_only
    assert len(details) == 3
    assert len(partial_cuts) == 3
    assert probe_economy_metadata["probe_economy_early_stop"] is False


def test_progressive_probe_economy_stops_weak_second_probe() -> None:
    instance = generate_instance(
        "unit_test_probe_economy_stop",
        n_facilities=5,
        n_customers=6,
        n_scenarios=4,
        seed=25,
    )
    config = SolverConfig(
        name="TEST-PROBE-ECONOMY-STOP",
        multicut=True,
        use_candidate_search=True,
        heuristic_scenario_probe_limit=3,
        partial_probe_economy_enabled=True,
        partial_probe_min_steps=1,
        partial_probe_gain_target_ratio=0.90,
        partial_probe_optimism_factor=0.05,
        max_iterations=5,
        random_seed=31,
    )
    anchor_design = [1, 0, 1, 0, 1]
    _, base_cuts, _ = _build_cuts_for_design(
        instance=instance,
        config=config,
        design=anchor_design,
        iteration=1,
        source="exact_rmp",
        alternative_point=[0.5] * instance.n_facilities,
    )

    (
        partial_recourse,
        partial_cuts,
        details,
        screened_out,
        screening_bound,
        partial_only,
        probe_economy_metadata,
    ) = _progressive_heuristic_evaluation(
        instance=instance,
        config=config,
        design=[1, 1, 1, 0, 0],
        iteration=2,
        source="greedy_refine",
        alternative_point=[0.5] * instance.n_facilities,
        cut_pool=base_cuts,
        incumbent_objective=None,
        scenario_order=[0, 1, 2],
        scenario_probe_limit=3,
        probe_gap_mass_by_scenario={
            instance.scenarios[0].scenario_id: 80.0,
            instance.scenarios[1].scenario_id: 60.0,
            instance.scenarios[2].scenario_id: 60.0,
        },
    )

    assert partial_recourse is None
    assert not screened_out
    assert screening_bound is None
    assert partial_only
    assert len(details) == 1
    assert len(partial_cuts) == 1
    assert probe_economy_metadata["probe_economy_early_stop"] is True
    assert probe_economy_metadata["probe_economy_stop_reason"] == "target_miss"


def test_effective_density_threshold_scales_with_scenario_count() -> None:
    instance = generate_instance(
        "unit_test_density_scale",
        n_facilities=3,
        n_customers=4,
        n_scenarios=60,
        seed=31,
    )
    config = SolverConfig(
        name="TEST-DENSITY-SCALE",
        heuristic_anchor_gap_density_threshold=320.0,
        heuristic_anchor_gap_density_reference_scenarios=40,
        heuristic_anchor_gap_density_scenario_exponent=0.5,
    )
    threshold = _effective_anchor_gap_density_threshold(instance, config)
    expected = 320.0 * (60 / 40) ** 0.5
    assert abs(threshold - expected) < 1e-9


def test_effective_density_threshold_stays_dimensionless_under_anchor_share_metric() -> None:
    instance = generate_instance(
        "unit_test_density_scale_dimensionless",
        n_facilities=3,
        n_customers=4,
        n_scenarios=60,
        seed=131,
    )
    config = SolverConfig(
        name="TEST-DENSITY-SCALE-DIMENSIONLESS",
        heuristic_anchor_gap_density_threshold=0.08,
        heuristic_anchor_gap_density_reference_scenarios=40,
        heuristic_anchor_gap_density_scenario_exponent=0.5,
        heuristic_anchor_gap_density_metric="anchor_recourse_share",
    )
    threshold = _effective_anchor_gap_density_threshold(instance, config)
    assert abs(threshold - 0.08) < 1e-12


def test_share_rank_probe_scoring_prefers_gap_sensitivity_overlap() -> None:
    instance = generate_instance(
        "unit_test_share_rank_probe",
        n_facilities=2,
        n_customers=3,
        n_scenarios=3,
        seed=33,
    )
    config = SolverConfig(
        name="TEST-SHARE-RANK",
        probe_ranking_mode="share_rank",
        probe_gap_weight=1.0,
        probe_sensitivity_weight=1.5,
        probe_rank_weight=0.2,
        probe_overlap_weight=2.0,
    )
    anchor_design = [0, 0]
    proposal_design = [1, 0]
    exact_anchor_cuts = [
        BendersCut(
            scenario_id=instance.scenarios[0].scenario_id,
            intercept=0.0,
            coefficients=[0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=anchor_design,
            recourse_value=90.0,
            efficacy=1.0,
            strengthened=False,
        ),
        BendersCut(
            scenario_id=instance.scenarios[1].scenario_id,
            intercept=0.0,
            coefficients=[60.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=anchor_design,
            recourse_value=50.0,
            efficacy=1.0,
            strengthened=False,
        ),
        BendersCut(
            scenario_id=instance.scenarios[2].scenario_id,
            intercept=0.0,
            coefficients=[180.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=anchor_design,
            recourse_value=5.0,
            efficacy=1.0,
            strengthened=False,
        ),
    ]
    anchor_gap_by_scenario = {
        instance.scenarios[0].scenario_id: 90.0,
        instance.scenarios[1].scenario_id: 50.0,
        instance.scenarios[2].scenario_id: 5.0,
    }

    scored = _score_probe_scenarios(
        instance=instance,
        config=config,
        design=proposal_design,
        anchor_design=anchor_design,
        exact_anchor_cuts=exact_anchor_cuts,
        anchor_gap_by_scenario=anchor_gap_by_scenario,
    )

    assert scored[0][0] == 1


def test_search_activation_concentration_gate_blocks_diffuse_anchor_profile() -> None:
    instance = generate_instance(
        "unit_test_anchor_concentration_block",
        n_facilities=4,
        n_customers=5,
        n_scenarios=4,
        seed=35,
    )
    config = SolverConfig(
        name="TEST-ANCHOR-CONCENTRATION-BLOCK",
        search_activation_top_mass_share_floor=0.40,
        search_activation_effective_support_ratio_ceiling=0.60,
    )
    anchor_gap_by_scenario = {
        scenario.scenario_id: 10.0 for scenario in instance.scenarios
    }

    passed, policy, top_mass_share, effective_support_ratio = (
        _passes_search_activation_concentration_gate(
            instance=instance,
            config=config,
            anchor_gap_by_scenario=anchor_gap_by_scenario,
        )
    )

    assert not passed
    assert policy == "anchor_low_top_mass_blocked"
    assert top_mass_share < 0.40
    assert effective_support_ratio > 0.0


def test_search_activation_concentration_gate_passes_concentrated_anchor_profile() -> None:
    instance = generate_instance(
        "unit_test_anchor_concentration_pass",
        n_facilities=4,
        n_customers=5,
        n_scenarios=4,
        seed=37,
    )
    config = SolverConfig(
        name="TEST-ANCHOR-CONCENTRATION-PASS",
        search_activation_top_mass_share_floor=0.35,
        search_activation_effective_support_ratio_ceiling=0.75,
    )
    anchor_gap_by_scenario = {
        instance.scenarios[0].scenario_id: 80.0,
        instance.scenarios[1].scenario_id: 10.0,
        instance.scenarios[2].scenario_id: 5.0,
        instance.scenarios[3].scenario_id: 0.0,
    }

    passed, policy, top_mass_share, effective_support_ratio = (
        _passes_search_activation_concentration_gate(
            instance=instance,
            config=config,
            anchor_gap_by_scenario=anchor_gap_by_scenario,
        )
    )

    assert passed
    assert policy == "anchor_concentrated_pass"
    assert top_mass_share > 0.35
    assert effective_support_ratio < 0.75


def test_search_activation_uniform_ratio_metric_recovers_portability() -> None:
    instance = generate_instance(
        "unit_test_anchor_concentration_uniform_ratio",
        n_facilities=4,
        n_customers=5,
        n_scenarios=40,
        seed=137,
    )
    config = SolverConfig(
        name="TEST-ANCHOR-CONCENTRATION-UNIFORM-RATIO",
        search_activation_top_mass_share_floor=2.0,
        search_activation_top_mass_metric="uniform_ratio",
        search_activation_effective_support_ratio_ceiling=0.90,
    )
    anchor_gap_by_scenario = {
        instance.scenarios[0].scenario_id: 80.0,
        instance.scenarios[1].scenario_id: 10.0,
        instance.scenarios[2].scenario_id: 5.0,
        instance.scenarios[3].scenario_id: 5.0,
    }
    for scenario in instance.scenarios[4:]:
        anchor_gap_by_scenario[scenario.scenario_id] = 0.0

    passed, policy, top_mass_metric, effective_support_ratio = (
        _passes_search_activation_concentration_gate(
            instance=instance,
            config=config,
            anchor_gap_by_scenario=anchor_gap_by_scenario,
        )
    )

    assert passed
    assert policy == "anchor_concentrated_pass"
    assert top_mass_metric > 2.0
    assert effective_support_ratio < 0.90


def test_dynamic_probe_limit_upgrades_low_scenario_high_density_case() -> None:
    instance = generate_instance(
        "unit_test_dynamic_probe_high",
        n_facilities=4,
        n_customers=5,
        n_scenarios=20,
        seed=101,
    )
    config = SolverConfig(
        name="TEST-DYNAMIC-PROBE-HIGH",
        heuristic_scenario_probe_limit=1,
        heuristic_anchor_gap_activation_threshold=10000.0,
        dynamic_probe_limit_enabled=True,
        dynamic_probe_limit_base_value=1,
        dynamic_probe_limit_high_value=2,
        dynamic_probe_limit_low_scenario_cap=40,
        dynamic_probe_limit_high_density_ratio=1.5,
        dynamic_probe_limit_high_underestimation_ratio=1.2,
        dynamic_probe_limit_support_ratio_threshold=0.6,
    )
    limit, policy = _effective_scenario_probe_limit(
        instance=instance,
        config=config,
        should_search=True,
        expected_anchor_underestimation=13000.0,
        anchor_gap_density=500.0,
        effective_density_threshold=320.0,
        probe_support_ratio=0.75,
    )
    assert limit == 2
    assert policy == "dynamic_high"


def test_dynamic_probe_limit_stays_base_on_large_scenario_regime() -> None:
    instance = generate_instance(
        "unit_test_dynamic_probe_base",
        n_facilities=4,
        n_customers=5,
        n_scenarios=120,
        seed=103,
    )
    config = SolverConfig(
        name="TEST-DYNAMIC-PROBE-BASE",
        heuristic_scenario_probe_limit=1,
        heuristic_anchor_gap_activation_threshold=10000.0,
        dynamic_probe_limit_enabled=True,
        dynamic_probe_limit_base_value=1,
        dynamic_probe_limit_high_value=2,
        dynamic_probe_limit_low_scenario_cap=60,
        dynamic_probe_limit_high_density_ratio=1.2,
        dynamic_probe_limit_high_underestimation_ratio=1.1,
        dynamic_probe_limit_support_ratio_threshold=0.5,
    )
    limit, policy = _effective_scenario_probe_limit(
        instance=instance,
        config=config,
        should_search=True,
        expected_anchor_underestimation=20000.0,
        anchor_gap_density=800.0,
        effective_density_threshold=320.0,
        probe_support_ratio=0.9,
    )
    assert limit == 1
    assert policy == "dynamic_base"


def test_effective_heuristic_cut_limit_tracks_probe_budget() -> None:
    config = SolverConfig(
        name="TEST-DYNAMIC-CUT-LIMIT",
        max_heuristic_cuts_per_candidate=2,
    )
    assert _effective_heuristic_cut_limit(config, 2) == 2
    assert _effective_heuristic_cut_limit(config, 1) == 1
    assert _effective_heuristic_cut_limit(config, None) == 2


def test_dynamic_candidate_eval_upgrades_second_candidate_when_gate_passes() -> None:
    instance = generate_instance(
        "unit_test_dynamic_candidate_eval_high",
        n_facilities=4,
        n_customers=5,
        n_scenarios=40,
        seed=211,
    )
    config = SolverConfig(
        name="TEST-DYNAMIC-CANDIDATE-EVAL-HIGH",
        heuristic_eval_limit=1,
        dynamic_candidate_eval_enabled=True,
        dynamic_candidate_eval_base_value=1,
        dynamic_candidate_eval_high_value=2,
        dynamic_candidate_eval_low_scenario_cap=60,
        dynamic_candidate_eval_target_gap_ratio_threshold=0.03,
        dynamic_candidate_eval_density_ratio_threshold=1.5,
        dynamic_candidate_eval_harvest_advantage_threshold=0.0,
        dynamic_candidate_eval_geometry_advantage_threshold=0.0,
    )
    eligible_candidates = [
        generate_candidates(
            instance=instance,
            cuts=[],
            config=config,
            anchor_design=[0, 0, 0, 0],
            incumbent_design=None,
            rng=np.random.default_rng(7),
        )[0],
        generate_candidates(
            instance=instance,
            cuts=[],
            config=config,
            anchor_design=[0, 0, 0, 0],
            incumbent_design=None,
            rng=np.random.default_rng(11),
        )[0],
    ]
    eligible_candidates[0].surrogate_score = 100.0
    eligible_candidates[0].metadata["targeting_score"] = 100.0
    eligible_candidates[0].metadata["harvest_proxy"] = 1.0
    eligible_candidates[0].metadata["geometry_pair_proxy"] = 5.0
    eligible_candidates[1].surrogate_score = 102.0
    eligible_candidates[1].metadata["targeting_score"] = 102.0
    eligible_candidates[1].metadata["harvest_proxy"] = 1.4
    eligible_candidates[1].metadata["geometry_pair_proxy"] = 5.8

    limit, policy, metadata = _effective_heuristic_eval_limit(
        instance=instance,
        config=config,
        should_search=True,
        eligible_candidates=eligible_candidates,
        anchor_gap_density=480.0,
        effective_density_threshold=300.0,
    )

    assert limit == 2
    assert policy == "candidate_high"
    assert abs(float(metadata["candidate_eval_density_ratio"]) - 1.6) < 1e-9
    assert abs(float(metadata["second_candidate_target_gap_ratio"]) - 0.02) < 1e-9
    assert abs(float(metadata["second_candidate_harvest_advantage"]) - 0.4) < 1e-9
    assert abs(float(metadata["second_candidate_geometry_advantage"]) - 0.8) < 1e-9


def test_dynamic_candidate_eval_stays_base_when_gate_fails() -> None:
    instance = generate_instance(
        "unit_test_dynamic_candidate_eval_base",
        n_facilities=4,
        n_customers=5,
        n_scenarios=40,
        seed=223,
    )
    config = SolverConfig(
        name="TEST-DYNAMIC-CANDIDATE-EVAL-BASE",
        heuristic_eval_limit=1,
        dynamic_candidate_eval_enabled=True,
        dynamic_candidate_eval_base_value=1,
        dynamic_candidate_eval_high_value=2,
        dynamic_candidate_eval_low_scenario_cap=60,
        dynamic_candidate_eval_target_gap_ratio_threshold=0.03,
        dynamic_candidate_eval_density_ratio_threshold=1.5,
        dynamic_candidate_eval_harvest_advantage_threshold=0.0,
        dynamic_candidate_eval_geometry_advantage_threshold=0.0,
    )
    eligible_candidates = [
        generate_candidates(
            instance=instance,
            cuts=[],
            config=config,
            anchor_design=[0, 0, 0, 0],
            incumbent_design=None,
            rng=np.random.default_rng(13),
        )[0],
        generate_candidates(
            instance=instance,
            cuts=[],
            config=config,
            anchor_design=[0, 0, 0, 0],
            incumbent_design=None,
            rng=np.random.default_rng(17),
        )[0],
    ]
    eligible_candidates[0].surrogate_score = 100.0
    eligible_candidates[0].metadata["targeting_score"] = 100.0
    eligible_candidates[0].metadata["harvest_proxy"] = 1.0
    eligible_candidates[0].metadata["geometry_pair_proxy"] = 5.0
    eligible_candidates[1].surrogate_score = 105.0
    eligible_candidates[1].metadata["targeting_score"] = 105.0
    eligible_candidates[1].metadata["harvest_proxy"] = 0.8
    eligible_candidates[1].metadata["geometry_pair_proxy"] = 4.8

    limit, policy, metadata = _effective_heuristic_eval_limit(
        instance=instance,
        config=config,
        should_search=True,
        eligible_candidates=eligible_candidates,
        anchor_gap_density=420.0,
        effective_density_threshold=300.0,
    )

    assert limit == 1
    assert policy == "candidate_base"
    assert abs(float(metadata["candidate_eval_density_ratio"]) - 1.4) < 1e-9
    assert abs(float(metadata["second_candidate_target_gap_ratio"]) - 0.05) < 1e-9
    assert abs(float(metadata["second_candidate_harvest_advantage"]) + 0.2) < 1e-9
    assert abs(float(metadata["second_candidate_geometry_advantage"]) + 0.2) < 1e-9


def test_activation_support_gate_blocks_flat_ranked_low_scenario_candidate() -> None:
    instance = generate_instance(
        "unit_test_activation_support_gate_block",
        n_facilities=4,
        n_customers=5,
        n_scenarios=30,
        seed=227,
    )
    config = SolverConfig(
        name="TEST-ACTIVATION-SUPPORT-GATE-BLOCK",
        activation_support_ratio_ceiling=0.75,
        activation_support_ratio_low_scenario_cap=60,
    )
    passed, policy = _passes_activation_support_gate(
        instance=instance,
        config=config,
        probe_support_ratio=0.84,
    )
    assert not passed
    assert policy == "flat_rank_blocked"


def test_activation_support_gate_blocks_low_top_mass_candidate() -> None:
    instance = generate_instance(
        "unit_test_activation_top_mass_block",
        n_facilities=4,
        n_customers=5,
        n_scenarios=40,
        seed=228,
    )
    config = SolverConfig(
        name="TEST-ACTIVATION-TOP-MASS-BLOCK",
        activation_top_mass_share_floor=0.45,
        activation_top_mass_share_topk=1,
    )
    passed, policy = _passes_activation_support_gate(
        instance=instance,
        config=config,
        probe_support_ratio=0.25,
        probe_top_mass_share=0.31,
        probe_effective_support_ratio=0.20,
    )
    assert not passed
    assert policy == "low_top_mass_blocked"


def test_activation_support_gate_blocks_diffuse_effective_support_candidate() -> None:
    instance = generate_instance(
        "unit_test_activation_effective_support_block",
        n_facilities=4,
        n_customers=5,
        n_scenarios=40,
        seed=230,
    )
    config = SolverConfig(
        name="TEST-ACTIVATION-EFFECTIVE-SUPPORT-BLOCK",
        activation_effective_support_ratio_ceiling=0.35,
    )
    passed, policy = _passes_activation_support_gate(
        instance=instance,
        config=config,
        probe_support_ratio=0.25,
        probe_top_mass_share=0.50,
        probe_effective_support_ratio=0.61,
    )
    assert not passed
    assert policy == "diffuse_support_blocked"


def test_activation_support_gate_passes_concentrated_ranked_candidate() -> None:
    instance = generate_instance(
        "unit_test_activation_support_gate_pass",
        n_facilities=4,
        n_customers=5,
        n_scenarios=40,
        seed=229,
    )
    config = SolverConfig(
        name="TEST-ACTIVATION-SUPPORT-GATE-PASS",
        activation_support_ratio_ceiling=0.75,
        activation_support_ratio_low_scenario_cap=60,
    )
    passed, policy = _passes_activation_support_gate(
        instance=instance,
        config=config,
        probe_support_ratio=0.18,
        probe_top_mass_share=0.70,
        probe_effective_support_ratio=0.20,
    )
    assert passed
    assert policy == "concentrated_pass"


def test_probe_concentration_metrics_capture_top_mass_and_support_ratio() -> None:
    scored = [(0, 10.0, 0.5), (1, 6.0, 0.3), (2, 1.0, 0.2)]
    top_mass_share = _probe_top_mass_share(scored, top_k=1)
    effective_support_ratio = _probe_effective_support_ratio(scored)

    assert abs(top_mass_share - (10.0 / 17.0)) < 1e-9
    assert 0.0 < effective_support_ratio < 1.0


def test_nonredundant_probe_schedule_skips_redundant_runner_up() -> None:
    instance = generate_instance(
        "unit_test_nonredundant_second_probe",
        n_facilities=4,
        n_customers=5,
        n_scenarios=4,
        seed=107,
    )
    config = SolverConfig(
        name="TEST-NONREDUNDANT-SECOND",
        second_probe_nonredundancy_enabled=True,
        second_probe_directional_distance_threshold=0.2,
        second_probe_support_topk=2,
        second_probe_search_depth=4,
    )
    exact_anchor_cuts = [
        BendersCut(
            scenario_id=instance.scenarios[0].scenario_id,
            intercept=0.0,
            coefficients=[1.0, 0.0, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=[0, 0, 0, 0],
            recourse_value=1.0,
            efficacy=1.0,
            strengthened=False,
        ),
        BendersCut(
            scenario_id=instance.scenarios[1].scenario_id,
            intercept=0.0,
            coefficients=[0.99, 0.01, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=[0, 0, 0, 0],
            recourse_value=1.0,
            efficacy=1.0,
            strengthened=False,
        ),
        BendersCut(
            scenario_id=instance.scenarios[2].scenario_id,
            intercept=0.0,
            coefficients=[0.0, 1.0, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=[0, 0, 0, 0],
            recourse_value=1.0,
            efficacy=1.0,
            strengthened=False,
        ),
    ]
    scored = [(0, 10.0, 0.4), (1, 9.0, 0.3), (2, 8.0, 0.2), (3, 7.0, 0.1)]
    scenario_priority, effective_probe_limit, probe_limit_policy, metadata = (
        _select_nonredundant_probe_schedule(
            instance=instance,
            config=config,
            scored_scenarios=scored,
            exact_anchor_cuts=exact_anchor_cuts,
            effective_probe_limit=2,
            probe_limit_policy="dynamic_high",
        )
    )

    assert effective_probe_limit == 2
    assert probe_limit_policy == "dynamic_high_nonredundant"
    assert scenario_priority[:2] == [0, 2]
    assert metadata["second_probe_gate_passed"] is True
    assert metadata["second_probe_selected_rank"] == 3
    assert metadata["second_probe_directional_distance"] is not None


def test_nonredundant_probe_schedule_blocks_when_all_candidates_are_collinear() -> None:
    instance = generate_instance(
        "unit_test_nonredundant_block",
        n_facilities=4,
        n_customers=5,
        n_scenarios=3,
        seed=109,
    )
    config = SolverConfig(
        name="TEST-NONREDUNDANT-BLOCK",
        second_probe_nonredundancy_enabled=True,
        second_probe_directional_distance_threshold=0.2,
        second_probe_support_topk=2,
        second_probe_search_depth=3,
    )
    exact_anchor_cuts = [
        BendersCut(
            scenario_id=instance.scenarios[0].scenario_id,
            intercept=0.0,
            coefficients=[1.0, 0.0, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=[0, 0, 0, 0],
            recourse_value=1.0,
            efficacy=1.0,
            strengthened=False,
        ),
        BendersCut(
            scenario_id=instance.scenarios[1].scenario_id,
            intercept=0.0,
            coefficients=[0.99, 0.01, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=[0, 0, 0, 0],
            recourse_value=1.0,
            efficacy=1.0,
            strengthened=False,
        ),
        BendersCut(
            scenario_id=instance.scenarios[2].scenario_id,
            intercept=0.0,
            coefficients=[0.98, 0.02, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=[0, 0, 0, 0],
            recourse_value=1.0,
            efficacy=1.0,
            strengthened=False,
        ),
    ]
    scored = [(0, 10.0, 0.5), (1, 9.0, 0.3), (2, 8.0, 0.2)]
    scenario_priority, effective_probe_limit, probe_limit_policy, metadata = (
        _select_nonredundant_probe_schedule(
            instance=instance,
            config=config,
            scored_scenarios=scored,
            exact_anchor_cuts=exact_anchor_cuts,
            effective_probe_limit=2,
            probe_limit_policy="dynamic_high",
        )
    )

    assert effective_probe_limit == 1
    assert probe_limit_policy == "dynamic_high_blocked"
    assert scenario_priority == [0]
    assert metadata["second_probe_gate_passed"] is False
    assert metadata["second_probe_selected_rank"] == 2


def test_geometry_pair_proxy_detects_nonredundant_second_probe() -> None:
    instance = generate_instance(
        "unit_test_geometry_pair_proxy",
        n_facilities=4,
        n_customers=5,
        n_scenarios=4,
        seed=113,
    )
    anchor = np.array([0, 0, 0, 0], dtype=int)
    design = np.array([1, 1, 0, 0], dtype=int)
    exact_anchor_cuts = [
        BendersCut(
            scenario_id=instance.scenarios[0].scenario_id,
            intercept=0.0,
            coefficients=[1.0, 0.0, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=anchor.tolist(),
            recourse_value=10.0,
            efficacy=1.0,
            strengthened=False,
        ),
        BendersCut(
            scenario_id=instance.scenarios[1].scenario_id,
            intercept=0.0,
            coefficients=[0.0, 1.0, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=anchor.tolist(),
            recourse_value=9.0,
            efficacy=1.0,
            strengthened=False,
        ),
        BendersCut(
            scenario_id=instance.scenarios[2].scenario_id,
            intercept=0.0,
            coefficients=[1.0, 0.01, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=anchor.tolist(),
            recourse_value=8.0,
            efficacy=1.0,
            strengthened=False,
        ),
    ]
    gap = {
        instance.scenarios[0].scenario_id: 10.0,
        instance.scenarios[1].scenario_id: 9.0,
        instance.scenarios[2].scenario_id: 8.0,
        instance.scenarios[3].scenario_id: 0.0,
    }
    config = SolverConfig(
        name="TEST-GEOMETRY-PROXY",
        candidate_targeting_mode="geometry",
        candidate_geometry_pair_weight=0.01,
        candidate_geometry_distance_threshold=0.2,
        candidate_geometry_pair_search_depth=4,
    )
    proxy, rank, distance, pair = _compute_geometry_pair_proxy(
        instance=instance,
        config=config,
        design=design,
        anchor_design=anchor,
        exact_anchor_cuts=exact_anchor_cuts,
        anchor_gap_by_scenario=gap,
    )
    assert proxy > 0.0
    assert rank == 2
    assert distance is not None and distance >= 0.2
    assert len(pair) == 2
    assert len(set(pair)) == 2


def test_generate_candidates_adds_geometry_pair_family() -> None:
    instance = generate_instance(
        "unit_test_geometry_pair_family",
        n_facilities=4,
        n_customers=5,
        n_scenarios=4,
        seed=127,
    )
    anchor = [0, 0, 0, 0]
    exact_anchor_cuts = [
        BendersCut(
            scenario_id=instance.scenarios[0].scenario_id,
            intercept=0.0,
            coefficients=[1.0, 0.0, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=anchor,
            recourse_value=10.0,
            efficacy=1.0,
            strengthened=False,
        ),
        BendersCut(
            scenario_id=instance.scenarios[1].scenario_id,
            intercept=0.0,
            coefficients=[0.0, 1.0, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=anchor,
            recourse_value=9.0,
            efficacy=1.0,
            strengthened=False,
        ),
        BendersCut(
            scenario_id=instance.scenarios[2].scenario_id,
            intercept=0.0,
            coefficients=[1.0, 0.01, 0.0, 0.0],
            source_iteration=1,
            source_candidate="exact_rmp",
            generating_design=anchor,
            recourse_value=8.0,
            efficacy=1.0,
            strengthened=False,
        ),
    ]
    gap = {
        instance.scenarios[0].scenario_id: 10.0,
        instance.scenarios[1].scenario_id: 9.0,
        instance.scenarios[2].scenario_id: 8.0,
        instance.scenarios[3].scenario_id: 0.0,
    }
    config = SolverConfig(
        name="TEST-GEOMETRY-FAMILY",
        candidate_targeting_mode="geometry",
        candidate_budget=12,
        annealing_steps=0,
        candidate_flip_pool_size=4,
        candidate_geometry_pair_budget=1,
        candidate_geometry_pair_weight=0.01,
        candidate_geometry_distance_threshold=0.2,
        candidate_geometry_pair_search_depth=4,
        candidate_top_gap_support=4,
        random_seed=131,
    )
    proposals = generate_candidates(
        instance=instance,
        cuts=exact_anchor_cuts,
        config=config,
        anchor_design=anchor,
        incumbent_design=None,
        rng=np.random.default_rng(137),
        exact_anchor_cuts=exact_anchor_cuts,
        anchor_gap_by_scenario=gap,
    )
    geometry_proposals = [
        proposal
        for proposal in proposals
        if proposal.source in {"targeted_scenario_pair_single", "targeted_scenario_pair_pair"}
    ]
    assert geometry_proposals
    assert any(
        proposal.metadata.get("scenario_pair_distance", 0.0) >= 0.2
        for proposal in geometry_proposals
    )
    assert all("geometry_pair_proxy" in proposal.metadata for proposal in geometry_proposals)


def test_structured_candidate_targeting_generates_local_branch_moves() -> None:
    instance = generate_instance(
        "unit_test_targeted_candidates",
        n_facilities=5,
        n_customers=6,
        n_scenarios=5,
        seed=37,
    )
    config = SolverConfig(
        name="TEST-TARGETED-CANDIDATES",
        multicut=True,
        use_candidate_search=True,
        candidate_budget=12,
        annealing_steps=10,
        candidate_targeting_mode="local_branch",
        local_branching_radius=1,
        candidate_flip_pool_size=4,
        random_seed=41,
    )
    anchor_design = [1, 0, 1, 0, 1]
    _, exact_anchor_cuts, _ = _build_cuts_for_design(
        instance=instance,
        config=config,
        design=anchor_design,
        iteration=1,
        source="exact_rmp",
        alternative_point=[0.5] * instance.n_facilities,
    )
    anchor_gap_by_scenario = {
        cut.scenario_id: cut.recourse_value
        for cut in exact_anchor_cuts
        if cut.scenario_id != "ALL"
    }

    proposals = generate_candidates(
        instance=instance,
        cuts=exact_anchor_cuts,
        config=config,
        anchor_design=anchor_design,
        incumbent_design=None,
        rng=np.random.default_rng(43),
        exact_anchor_cuts=exact_anchor_cuts,
        anchor_gap_by_scenario=anchor_gap_by_scenario,
    )

    targeted = [proposal for proposal in proposals if proposal.source == "targeted_local_branch"]
    assert targeted
    assert all(
        proposal.metadata.get("anchor_hamming_distance") == 1 for proposal in targeted
    )
    assert any(proposal.metadata.get("flip_priority_sum", 0.0) > 0.0 for proposal in targeted)
    assert all("harvest_proxy" in proposal.metadata for proposal in proposals)


def test_solver_logs_stage1_candidate_metadata() -> None:
    instance = generate_instance(
        "unit_test_stage1_logging",
        n_facilities=5,
        n_customers=6,
        n_scenarios=6,
        seed=47,
    )
    config = SolverConfig(
        name="TEST-STAGE1-LOGGING",
        multicut=True,
        use_candidate_search=True,
        candidate_budget=6,
        heuristic_eval_limit=1,
        annealing_steps=10,
        max_iterations=6,
        random_seed=53,
        candidate_targeting_mode="local_branch",
        candidate_flip_pool_size=4,
        local_branching_radius=1,
        candidate_harvest_proxy_weight=0.01,
    )
    result = solve_instance(instance, config)

    assert result.candidate_history
    for key in [
        "proposal_rank",
        "proposal_family",
        "anchor_hamming_distance",
        "flip_set",
        "flip_priority_sum",
        "harvest_proxy",
        "geometry_pair_proxy",
        "geometry_pair_rank",
        "geometry_pair_distance",
        "geometry_pair_scenarios",
        "targeting_score",
        "selected_for_exact_probe",
        "effective_candidate_eval_limit",
        "candidate_eval_policy",
        "candidate_eval_density_ratio",
        "second_candidate_target_gap_ratio",
        "second_candidate_harvest_advantage",
        "second_candidate_geometry_advantage",
        "activation_support_ratio",
        "activation_support_ratio_ceiling",
        "activation_support_ratio_policy",
        "activation_top_mass_share",
        "activation_top_mass_share_floor",
        "activation_effective_support_ratio",
        "activation_effective_support_ratio_ceiling",
        "effective_scenario_probe_limit",
        "probe_limit_policy",
        "probe_ranking_mode",
        "probe_economy_early_stop",
        "probe_economy_stop_reason",
        "probe_economy_realized_gain",
        "probe_economy_target_gain",
        "probe_economy_realized_gap_mass",
        "probe_economy_scheduled_gap_mass",
        "scenario_probe_support_ratio",
        "candidate_eval_wall_seconds",
        "scenario_solve_time_seconds",
        "search_activation_policy",
        "search_activation_top_mass",
        "search_activation_top_mass_raw_share",
        "search_activation_effective_support_ratio",
        "anchor_expected_recourse",
        "anchor_gap_density",
        "anchor_gap_density_raw",
        "anchor_gap_density_threshold",
        "anchor_gap_density_metric",
        "second_probe_gate_passed",
        "second_probe_selected_rank",
        "second_probe_directional_distance",
        "second_probe_support_jaccard_distance",
        "scenario_priority_head",
    ]:
        assert key in result.candidate_history[0]
    assert result.candidate_history[0]["candidate_eval_wall_seconds"] >= 0.0
    assert result.candidate_history[0]["scenario_solve_time_seconds"] >= 0.0

    assert result.iteration_logs
    for key in [
        "eligible_heuristic_candidates",
        "selected_heuristic_candidates",
        "iteration_wall_seconds",
        "master_solve_seconds",
        "candidate_generation_seconds",
        "exact_anchor_eval_seconds",
        "exact_anchor_scenario_solve_seconds",
        "heuristic_eval_seconds",
        "heuristic_scenario_solve_seconds",
        "search_activated",
        "search_activation_policy",
        "search_activation_top_mass",
        "search_activation_top_mass_raw_share",
        "search_activation_effective_support_ratio",
        "anchor_expected_recourse",
        "anchor_expected_underestimation",
        "anchor_gap_density",
        "anchor_gap_density_raw",
        "anchor_gap_density_threshold",
        "anchor_gap_density_metric",
    ]:
        assert key in result.iteration_logs[0]


def test_solver_logs_stage5_candidate_eval_metadata() -> None:
    instance = generate_instance(
        "unit_test_stage5_logging",
        n_facilities=5,
        n_customers=6,
        n_scenarios=20,
        seed=241,
    )
    config = SolverConfig(
        name="TEST-STAGE5-LOGGING",
        multicut=True,
        use_candidate_search=True,
        candidate_budget=8,
        heuristic_eval_limit=1,
        annealing_steps=10,
        max_iterations=6,
        random_seed=251,
        candidate_targeting_mode="geometry",
        candidate_flip_pool_size=6,
        candidate_top_gap_support=6,
        candidate_harvest_proxy_weight=0.02,
        candidate_geometry_pair_budget=2,
        candidate_geometry_pair_weight=0.05,
        dynamic_candidate_eval_enabled=True,
        dynamic_candidate_eval_base_value=1,
        dynamic_candidate_eval_high_value=2,
        dynamic_candidate_eval_low_scenario_cap=60,
        dynamic_candidate_eval_target_gap_ratio_threshold=0.05,
        dynamic_candidate_eval_density_ratio_threshold=0.0,
        dynamic_candidate_eval_harvest_advantage_threshold=-1.0,
        dynamic_candidate_eval_geometry_advantage_threshold=-1.0,
        activation_support_ratio_ceiling=0.95,
        activation_support_ratio_low_scenario_cap=60,
    )
    result = solve_instance(instance, config)

    assert result.candidate_history
    exact_entry = result.candidate_history[0]
    assert exact_entry["candidate_eval_policy"] == "exact_full"
    assert exact_entry["activation_support_ratio_policy"] == "exact_full"
    assert exact_entry["candidate_eval_wall_seconds"] >= 0.0
    assert exact_entry["scenario_solve_time_seconds"] >= 0.0
    heuristic_entries = [
        entry
        for entry in result.candidate_history
        if entry.get("source") not in {"exact_rmp", "carry_forward"}
    ]
    assert heuristic_entries
    for entry in heuristic_entries:
        assert entry["effective_candidate_eval_limit"] in {1, 2}
        assert entry["candidate_eval_policy"] in {"candidate_base", "candidate_high", "static"}
        assert "candidate_eval_density_ratio" in entry
        assert "second_candidate_target_gap_ratio" in entry
        assert "second_candidate_harvest_advantage" in entry
        assert "second_candidate_geometry_advantage" in entry
        assert "activation_top_mass_share" in entry
        assert "activation_effective_support_ratio" in entry
        assert "probe_ranking_mode" in entry
        assert "probe_economy_early_stop" in entry
        assert "candidate_eval_wall_seconds" in entry
        assert "scenario_solve_time_seconds" in entry
        assert entry["activation_support_ratio_policy"] in {
            "concentrated_pass",
            "inactive",
            "single_cut",
            "low_top_mass_blocked",
            "diffuse_support_blocked",
        }


def test_structured_candidate_targeting_generates_pair_moves() -> None:
    instance = generate_instance(
        "unit_test_targeted_pair_candidates",
        n_facilities=6,
        n_customers=6,
        n_scenarios=5,
        seed=59,
    )
    config = SolverConfig(
        name="TEST-TARGETED-PAIR-CANDIDATES",
        multicut=True,
        use_candidate_search=True,
        candidate_budget=16,
        annealing_steps=0,
        candidate_targeting_mode="local_branch",
        local_branching_radius=2,
        candidate_flip_pool_size=5,
        local_branching_pair_budget=4,
        random_seed=61,
    )
    anchor_design = [1, 0, 1, 0, 1, 0]
    _, exact_anchor_cuts, _ = _build_cuts_for_design(
        instance=instance,
        config=config,
        design=anchor_design,
        iteration=1,
        source="exact_rmp",
        alternative_point=[0.5] * instance.n_facilities,
    )
    anchor_gap_by_scenario = {
        cut.scenario_id: cut.recourse_value
        for cut in exact_anchor_cuts
        if cut.scenario_id != "ALL"
    }

    proposals = generate_candidates(
        instance=instance,
        cuts=exact_anchor_cuts,
        config=config,
        anchor_design=anchor_design,
        incumbent_design=None,
        rng=np.random.default_rng(67),
        exact_anchor_cuts=exact_anchor_cuts,
        anchor_gap_by_scenario=anchor_gap_by_scenario,
    )

    pair_moves = [
        proposal for proposal in proposals if proposal.source == "targeted_local_branch_pair"
    ]
    assert pair_moves
    assert all(
        proposal.metadata.get("anchor_hamming_distance") == 2 for proposal in pair_moves
    )
    assert all(len(proposal.metadata.get("flip_set", [])) == 2 for proposal in pair_moves)


def test_structured_candidate_targeting_generates_disagreement_moves() -> None:
    instance = generate_instance(
        "unit_test_targeted_disagreement_candidates",
        n_facilities=6,
        n_customers=7,
        n_scenarios=6,
        seed=71,
    )
    config = SolverConfig(
        name="TEST-TARGETED-DISAGREEMENT",
        multicut=True,
        use_candidate_search=True,
        candidate_budget=18,
        annealing_steps=0,
        candidate_targeting_mode="local_branch",
        local_branching_radius=1,
        candidate_flip_pool_size=6,
        local_branching_pair_budget=0,
        candidate_disagreement_budget=2,
        candidate_disagreement_max_flips=3,
        random_seed=73,
    )
    anchor_design = [1, 0, 1, 0, 1, 0]
    _, exact_anchor_cuts, _ = _build_cuts_for_design(
        instance=instance,
        config=config,
        design=anchor_design,
        iteration=1,
        source="exact_rmp",
        alternative_point=[0.5] * instance.n_facilities,
    )
    anchor_gap_by_scenario = {
        cut.scenario_id: cut.recourse_value
        for cut in exact_anchor_cuts
        if cut.scenario_id != "ALL"
    }

    proposals = generate_candidates(
        instance=instance,
        cuts=exact_anchor_cuts,
        config=config,
        anchor_design=anchor_design,
        incumbent_design=None,
        rng=np.random.default_rng(79),
        exact_anchor_cuts=exact_anchor_cuts,
        anchor_gap_by_scenario=anchor_gap_by_scenario,
    )

    disagreement_moves = [
        proposal for proposal in proposals if proposal.source == "targeted_disagreement"
    ]
    assert disagreement_moves
    assert all(
        proposal.metadata.get("anchor_hamming_distance", 0) >= 2
        for proposal in disagreement_moves
    )
    assert all("disagreement_signed_sum" in proposal.metadata for proposal in disagreement_moves)
