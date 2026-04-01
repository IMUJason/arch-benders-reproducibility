from __future__ import annotations

import time
from math import inf

import numpy as np

from .candidate_search import generate_candidates, surrogate_objective
from .cut_management import filter_new_cuts
from .master_problem import solve_master
from .models import (
    BendersCut,
    CandidateProposal,
    IterationLog,
    RunResult,
    SolverConfig,
    SupplyChainInstance,
)
from .subproblem import solve_scenario_recourse


def _relative_gap(upper_bound: float | None, lower_bound: float | None) -> float | None:
    if upper_bound is None or lower_bound is None:
        return None
    denominator = max(1.0, abs(upper_bound))
    return max(0.0, upper_bound - lower_bound) / denominator


def _rank_heuristic_cuts(
    instance: SupplyChainInstance,
    cuts: list[BendersCut],
    limit: int | None,
) -> list[BendersCut]:
    if limit is None or limit >= len(cuts):
        return cuts

    probability_by_scenario = {
        scenario.scenario_id: scenario.probability for scenario in instance.scenarios
    }
    return sorted(
        cuts,
        key=lambda cut: (
            probability_by_scenario.get(cut.scenario_id, 1.0) * cut.efficacy,
            cut.efficacy,
        ),
        reverse=True,
    )[:limit]


def _candidate_priority_key(
    config: SolverConfig,
    proposal: CandidateProposal,
) -> tuple[float, ...]:
    if config.candidate_targeting_mode == "baseline":
        return (proposal.surrogate_score,)
    return (
        proposal.metadata.get("targeting_score", proposal.surrogate_score),
        -proposal.metadata.get("geometry_pair_proxy", 0.0),
        -proposal.metadata.get("harvest_proxy", 0.0),
        proposal.surrogate_score,
    )


def _candidate_eval_density_ratio(
    anchor_gap_density: float,
    effective_density_threshold: float,
) -> float:
    if effective_density_threshold > 0.0:
        return anchor_gap_density / effective_density_threshold
    if anchor_gap_density > 0.0:
        return 1.0
    return 0.0


def _top_mass_metric(
    raw_top_mass_share: float,
    n_scenarios: int,
    topk: int,
    metric: str,
) -> float:
    if metric == "share":
        return raw_top_mass_share
    if metric == "uniform_ratio":
        return raw_top_mass_share * max(1, n_scenarios) / max(1, topk)
    raise ValueError(f"Unsupported top-mass metric: {metric}")


def _anchor_gap_density(
    instance: SupplyChainInstance,
    config: SolverConfig,
    expected_anchor_underestimation: float,
    anchor_expected_recourse: float | None,
) -> tuple[float, float]:
    raw_density = expected_anchor_underestimation / max(1, instance.n_scenarios)
    if config.heuristic_anchor_gap_density_metric == "raw_per_scenario":
        return raw_density, raw_density
    if config.heuristic_anchor_gap_density_metric == "anchor_recourse_share":
        denominator = max(
            config.heuristic_anchor_gap_density_scale_floor,
            anchor_expected_recourse or 0.0,
        )
        return expected_anchor_underestimation / denominator, raw_density
    raise ValueError(
        "Unsupported anchor-gap density metric: "
        f"{config.heuristic_anchor_gap_density_metric}"
    )


def _scenario_solve_seconds(scenario_details: list[dict[str, object]]) -> float:
    return float(
        sum(float(detail.get("solve_time_seconds", 0.0)) for detail in scenario_details)
    )


def _scenario_cut_lower_bound(
    cuts: list[BendersCut],
    scenario_id: str,
    design: list[int],
) -> float:
    values = [cut.value_at(design) for cut in cuts if cut.scenario_id == scenario_id]
    if not values:
        return 0.0
    return max(0.0, max(values))


def _compute_anchor_gap_profile(
    instance: SupplyChainInstance,
    base_cuts: list[BendersCut],
    exact_anchor_cuts: list[BendersCut],
    anchor_design: list[int],
) -> dict[str, float]:
    anchor_gap_by_scenario: dict[str, float] = {}
    for cut in exact_anchor_cuts:
        lower_bound = _scenario_cut_lower_bound(base_cuts, cut.scenario_id, anchor_design)
        anchor_gap_by_scenario[cut.scenario_id] = max(0.0, cut.recourse_value - lower_bound)
    for scenario in instance.scenarios:
        anchor_gap_by_scenario.setdefault(scenario.scenario_id, 0.0)
    return anchor_gap_by_scenario


def _score_probe_scenarios(
    instance: SupplyChainInstance,
    config: SolverConfig,
    design: list[int],
    anchor_design: list[int],
    exact_anchor_cuts: list[BendersCut],
    anchor_gap_by_scenario: dict[str, float],
) -> list[tuple[int, float, float]]:
    delta = np.array(design, dtype=float) - np.array(anchor_design, dtype=float)
    exact_cut_by_scenario = {
        cut.scenario_id: cut for cut in exact_anchor_cuts if cut.scenario_id != "ALL"
    }

    raw_rows: list[dict[str, float | int]] = []
    for scenario_index in range(instance.n_scenarios):
        scenario = instance.scenarios[scenario_index]
        reference_cut = exact_cut_by_scenario.get(scenario.scenario_id)
        sensitivity = 0.0
        if reference_cut is not None:
            sensitivity = abs(float(np.dot(np.array(reference_cut.coefficients, dtype=float), delta)))
        anchor_gap = anchor_gap_by_scenario.get(scenario.scenario_id, 0.0)
        gap_mass = scenario.probability * anchor_gap
        sensitivity_mass = scenario.probability * sensitivity
        raw_rows.append(
            {
                "scenario_index": scenario_index,
                "probability": float(scenario.probability),
                "anchor_gap": float(anchor_gap),
                "gap_mass": float(gap_mass),
                "sensitivity": float(sensitivity),
                "sensitivity_mass": float(sensitivity_mass),
            }
        )

    if config.probe_ranking_mode == "weighted_sum":
        return sorted(
            [
                (
                    int(row["scenario_index"]),
                    float(
                        config.probe_gap_weight * row["gap_mass"]
                        + config.probe_sensitivity_weight * row["sensitivity_mass"]
                    ),
                    float(row["probability"]),
                )
                for row in raw_rows
            ],
            key=lambda item: (item[1], item[2]),
            reverse=True,
        )

    total_gap_mass = sum(float(row["gap_mass"]) for row in raw_rows)
    total_sensitivity_mass = sum(float(row["sensitivity_mass"]) for row in raw_rows)
    gap_rank_order = sorted(
        raw_rows,
        key=lambda row: (float(row["gap_mass"]), float(row["probability"])),
        reverse=True,
    )
    sensitivity_rank_order = sorted(
        raw_rows,
        key=lambda row: (float(row["sensitivity_mass"]), float(row["probability"])),
        reverse=True,
    )
    gap_rank = {
        int(row["scenario_index"]): rank
        for rank, row in enumerate(gap_rank_order, start=1)
    }
    sensitivity_rank = {
        int(row["scenario_index"]): rank
        for rank, row in enumerate(sensitivity_rank_order, start=1)
    }

    scored_scenarios: list[tuple[int, float, float]] = []
    for row in raw_rows:
        scenario_index = int(row["scenario_index"])
        gap_share = (
            float(row["gap_mass"]) / total_gap_mass if total_gap_mass > 0.0 else 0.0
        )
        sensitivity_share = (
            float(row["sensitivity_mass"]) / total_sensitivity_mass
            if total_sensitivity_mass > 0.0
            else 0.0
        )
        rank_bonus = config.probe_rank_weight * (
            config.probe_gap_weight / gap_rank[scenario_index]
            + config.probe_sensitivity_weight / sensitivity_rank[scenario_index]
        )
        overlap_bonus = config.probe_overlap_weight * gap_share * sensitivity_share
        blended = (
            config.probe_gap_weight * gap_share
            + config.probe_sensitivity_weight * sensitivity_share
            + rank_bonus
            + overlap_bonus
        )
        scored_scenarios.append(
            (
                scenario_index,
                float(blended),
                float(row["probability"]),
            )
        )

    return sorted(
        scored_scenarios,
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )


def _rank_probe_scenarios(
    instance: SupplyChainInstance,
    config: SolverConfig,
    design: list[int],
    anchor_design: list[int],
    exact_anchor_cuts: list[BendersCut],
    anchor_gap_by_scenario: dict[str, float],
) -> list[int]:
    return [
        scenario_index
        for scenario_index, _, _ in _score_probe_scenarios(
            instance=instance,
            config=config,
            design=design,
            anchor_design=anchor_design,
            exact_anchor_cuts=exact_anchor_cuts,
            anchor_gap_by_scenario=anchor_gap_by_scenario,
        )
    ]


def _probe_support_ratio(scored_scenarios: list[tuple[int, float, float]]) -> float:
    if len(scored_scenarios) < 2:
        return 0.0
    top_score = scored_scenarios[0][1]
    if top_score <= 0.0:
        return 0.0
    return scored_scenarios[1][1] / top_score


def _probe_top_mass_share(
    scored_scenarios: list[tuple[int, float, float]],
    top_k: int,
) -> float:
    if top_k <= 0 or not scored_scenarios:
        return 0.0
    positive_scores = [max(0.0, score) for _, score, _ in scored_scenarios]
    total_score = sum(positive_scores)
    if total_score <= 0.0:
        return 0.0
    return sum(sorted(positive_scores, reverse=True)[:top_k]) / total_score


def _probe_effective_support_ratio(
    scored_scenarios: list[tuple[int, float, float]],
) -> float:
    if not scored_scenarios:
        return 0.0
    positive_scores = np.array(
        [max(0.0, score) for _, score, _ in scored_scenarios],
        dtype=float,
    )
    total_score = float(positive_scores.sum())
    if total_score <= 0.0:
        return 1.0
    squared_sum = float(np.dot(positive_scores, positive_scores))
    if squared_sum <= 0.0:
        return 1.0
    effective_support = (total_score * total_score) / squared_sum
    return float(effective_support / len(positive_scores))


def _anchor_gap_mass_scores(
    instance: SupplyChainInstance,
    anchor_gap_by_scenario: dict[str, float],
) -> list[tuple[int, float, float]]:
    return [
        (
            scenario_index,
            float(
                scenario.probability
                * anchor_gap_by_scenario.get(scenario.scenario_id, 0.0)
            ),
            float(scenario.probability),
        )
        for scenario_index, scenario in enumerate(instance.scenarios)
    ]


def _passes_search_activation_concentration_gate(
    instance: SupplyChainInstance,
    config: SolverConfig,
    anchor_gap_by_scenario: dict[str, float],
) -> tuple[bool, str, float, float]:
    top_mass_active = config.search_activation_top_mass_share_floor > 0.0
    effective_support_active = config.search_activation_effective_support_ratio_ceiling > 0.0
    if not (top_mass_active or effective_support_active):
        return True, "inactive", 0.0, 0.0

    scored = _anchor_gap_mass_scores(instance, anchor_gap_by_scenario)
    raw_top_mass_share = _probe_top_mass_share(
        scored,
        max(1, config.search_activation_top_mass_topk),
    )
    top_mass_share = _top_mass_metric(
        raw_top_mass_share=raw_top_mass_share,
        n_scenarios=instance.n_scenarios,
        topk=max(1, config.search_activation_top_mass_topk),
        metric=config.search_activation_top_mass_metric,
    )
    effective_support_ratio = _probe_effective_support_ratio(scored)
    if top_mass_active and top_mass_share < config.search_activation_top_mass_share_floor:
        return False, "anchor_low_top_mass_blocked", top_mass_share, effective_support_ratio
    if (
        effective_support_active
        and effective_support_ratio > config.search_activation_effective_support_ratio_ceiling
    ):
        return (
            False,
            "anchor_diffuse_support_blocked",
            top_mass_share,
            effective_support_ratio,
        )
    return True, "anchor_concentrated_pass", top_mass_share, effective_support_ratio


def _effective_anchor_gap_density_threshold(
    instance: SupplyChainInstance,
    config: SolverConfig,
) -> float:
    threshold = config.heuristic_anchor_gap_density_threshold
    if (
        threshold <= 0.0
        or config.heuristic_anchor_gap_density_metric != "raw_per_scenario"
    ):
        return threshold
    reference_scenarios = config.heuristic_anchor_gap_density_reference_scenarios
    exponent = config.heuristic_anchor_gap_density_scenario_exponent
    if (
        reference_scenarios is None
        or reference_scenarios <= 0
        or exponent == 0.0
    ):
        return threshold
    scale = (max(1, instance.n_scenarios) / reference_scenarios) ** exponent
    return threshold * scale


def _effective_scenario_probe_limit(
    instance: SupplyChainInstance,
    config: SolverConfig,
    should_search: bool,
    expected_anchor_underestimation: float,
    anchor_gap_density: float,
    effective_density_threshold: float,
    probe_support_ratio: float,
) -> tuple[int | None, str]:
    static_limit = config.heuristic_scenario_probe_limit
    if not config.dynamic_probe_limit_enabled:
        return static_limit, "static"

    if static_limit is None:
        base_limit = config.dynamic_probe_limit_base_value
    else:
        base_limit = (
            config.dynamic_probe_limit_base_value
            if config.dynamic_probe_limit_base_value is not None
            else static_limit
        )

    if not should_search:
        return max(0, config.dynamic_probe_limit_dormant_value), "dormant"

    if base_limit is None:
        base_limit = 1
    base_limit = max(0, base_limit)
    high_limit = max(base_limit, config.dynamic_probe_limit_high_value)
    if high_limit == base_limit:
        return base_limit, "dynamic_base"

    if (
        config.dynamic_probe_limit_low_scenario_cap is not None
        and instance.n_scenarios > config.dynamic_probe_limit_low_scenario_cap
    ):
        return base_limit, "dynamic_base"

    density_trigger = config.dynamic_probe_limit_high_density_threshold
    if effective_density_threshold > 0.0 and config.dynamic_probe_limit_high_density_ratio > 0.0:
        density_trigger = max(
            density_trigger,
            effective_density_threshold * config.dynamic_probe_limit_high_density_ratio,
        )
    underestimation_trigger = config.dynamic_probe_limit_high_underestimation_threshold
    if (
        config.heuristic_anchor_gap_activation_threshold > 0.0
        and config.dynamic_probe_limit_high_underestimation_ratio > 0.0
    ):
        underestimation_trigger = max(
            underestimation_trigger,
            config.heuristic_anchor_gap_activation_threshold
            * config.dynamic_probe_limit_high_underestimation_ratio,
        )

    high_density = density_trigger <= 0.0 or anchor_gap_density >= density_trigger
    high_underestimation = (
        underestimation_trigger <= 0.0
        or expected_anchor_underestimation >= underestimation_trigger
    )
    strong_support = (
        config.dynamic_probe_limit_support_ratio_threshold <= 0.0
        or probe_support_ratio >= config.dynamic_probe_limit_support_ratio_threshold
    )
    if high_density and high_underestimation and strong_support:
        return high_limit, "dynamic_high"
    return base_limit, "dynamic_base"


def _effective_heuristic_cut_limit(
    config: SolverConfig,
    effective_probe_limit: int | None,
) -> int | None:
    if config.max_heuristic_cuts_per_candidate is None:
        return effective_probe_limit
    if effective_probe_limit is None:
        return config.max_heuristic_cuts_per_candidate
    return min(config.max_heuristic_cuts_per_candidate, effective_probe_limit)


def _effective_heuristic_eval_limit(
    instance: SupplyChainInstance,
    config: SolverConfig,
    should_search: bool,
    eligible_candidates: list[CandidateProposal],
    anchor_gap_density: float,
    effective_density_threshold: float,
) -> tuple[int, str, dict[str, float | int | None]]:
    base_limit = (
        config.dynamic_candidate_eval_base_value
        if config.dynamic_candidate_eval_base_value is not None
        else config.heuristic_eval_limit
    )
    base_limit = max(0, base_limit)
    metadata: dict[str, float | int | None] = {
        "second_candidate_target_gap_ratio": None,
        "second_candidate_harvest_advantage": None,
        "second_candidate_geometry_advantage": None,
        "candidate_eval_density_ratio": _candidate_eval_density_ratio(
            anchor_gap_density,
            effective_density_threshold,
        ),
    }
    if not config.dynamic_candidate_eval_enabled:
        return base_limit, "static", metadata
    if not should_search:
        return base_limit, "candidate_search_off", metadata
    if len(eligible_candidates) <= base_limit:
        return base_limit, "candidate_base", metadata
    if (
        config.dynamic_candidate_eval_low_scenario_cap is not None
        and instance.n_scenarios > config.dynamic_candidate_eval_low_scenario_cap
    ):
        return base_limit, "candidate_base", metadata

    first_candidate = eligible_candidates[0]
    second_candidate = eligible_candidates[1]
    first_target = first_candidate.metadata.get(
        "targeting_score",
        first_candidate.surrogate_score,
    )
    second_target = second_candidate.metadata.get(
        "targeting_score",
        second_candidate.surrogate_score,
    )
    target_gap_ratio = max(0.0, second_target - first_target) / max(
        1.0, abs(first_target)
    )
    harvest_advantage = second_candidate.metadata.get("harvest_proxy", 0.0) - first_candidate.metadata.get(
        "harvest_proxy", 0.0
    )
    geometry_advantage = second_candidate.metadata.get(
        "geometry_pair_proxy", 0.0
    ) - first_candidate.metadata.get("geometry_pair_proxy", 0.0)
    density_ratio = metadata["candidate_eval_density_ratio"]
    metadata.update(
        {
            "second_candidate_target_gap_ratio": float(target_gap_ratio),
            "second_candidate_harvest_advantage": float(harvest_advantage),
            "second_candidate_geometry_advantage": float(geometry_advantage),
        }
    )
    if density_ratio < config.dynamic_candidate_eval_density_ratio_threshold:
        return base_limit, "candidate_base", metadata
    if target_gap_ratio > config.dynamic_candidate_eval_target_gap_ratio_threshold:
        return base_limit, "candidate_base", metadata
    if (
        harvest_advantage < config.dynamic_candidate_eval_harvest_advantage_threshold
        and geometry_advantage < config.dynamic_candidate_eval_geometry_advantage_threshold
    ):
        return base_limit, "candidate_base", metadata
    return max(base_limit, config.dynamic_candidate_eval_high_value), "candidate_high", metadata


def _passes_activation_support_gate(
    instance: SupplyChainInstance,
    config: SolverConfig,
    probe_support_ratio: float,
    probe_top_mass_share: float = 0.0,
    probe_effective_support_ratio: float = 0.0,
) -> tuple[bool, str]:
    support_gate_active = config.activation_support_ratio_ceiling > 0.0
    if (
        support_gate_active
        and config.activation_support_ratio_low_scenario_cap is not None
        and instance.n_scenarios > config.activation_support_ratio_low_scenario_cap
    ):
        support_gate_active = False
    top_mass_gate_active = config.activation_top_mass_share_floor > 0.0
    top_mass_metric = _top_mass_metric(
        raw_top_mass_share=probe_top_mass_share,
        n_scenarios=instance.n_scenarios,
        topk=max(1, config.activation_top_mass_share_topk),
        metric=config.activation_top_mass_metric,
    )
    effective_support_gate_active = config.activation_effective_support_ratio_ceiling > 0.0
    if not (support_gate_active or top_mass_gate_active or effective_support_gate_active):
        return True, "inactive"
    if support_gate_active and probe_support_ratio > config.activation_support_ratio_ceiling:
        return False, "flat_rank_blocked"
    if top_mass_gate_active and top_mass_metric < config.activation_top_mass_share_floor:
        return False, "low_top_mass_blocked"
    if (
        effective_support_gate_active
        and probe_effective_support_ratio > config.activation_effective_support_ratio_ceiling
    ):
        return False, "diffuse_support_blocked"
    return True, "concentrated_pass"


def _cut_directional_distance(
    first_cut: BendersCut,
    second_cut: BendersCut,
) -> float:
    first = np.array(first_cut.coefficients, dtype=float)
    second = np.array(second_cut.coefficients, dtype=float)
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return 0.5 * (1.0 - cosine)


def _cut_support_jaccard_distance(
    first_cut: BendersCut,
    second_cut: BendersCut,
    topk: int,
) -> float:
    support_width = max(1, topk)
    first = np.array(first_cut.coefficients, dtype=float)
    second = np.array(second_cut.coefficients, dtype=float)
    first_support = set(np.argsort(np.abs(first))[-support_width:].tolist())
    second_support = set(np.argsort(np.abs(second))[-support_width:].tolist())
    support_union = first_support | second_support
    if not support_union:
        return 0.0
    return 1.0 - len(first_support & second_support) / len(support_union)


def _select_nonredundant_probe_schedule(
    instance: SupplyChainInstance,
    config: SolverConfig,
    scored_scenarios: list[tuple[int, float, float]],
    exact_anchor_cuts: list[BendersCut],
    effective_probe_limit: int | None,
    probe_limit_policy: str,
) -> tuple[list[int], int | None, str, dict[str, float | int | bool | None]]:
    scenario_priority = [scenario_index for scenario_index, _, _ in scored_scenarios]
    metadata: dict[str, float | int | bool | None] = {
        "second_probe_gate_passed": False,
        "second_probe_selected_rank": None,
        "second_probe_directional_distance": None,
        "second_probe_support_jaccard_distance": None,
    }
    if (
        not config.second_probe_nonredundancy_enabled
        or effective_probe_limit is None
        or effective_probe_limit <= 1
        or len(scored_scenarios) <= 1
    ):
        return scenario_priority, effective_probe_limit, probe_limit_policy, metadata

    exact_cut_by_scenario = {
        cut.scenario_id: cut for cut in exact_anchor_cuts if cut.scenario_id != "ALL"
    }
    top_scenario_id = instance.scenarios[scored_scenarios[0][0]].scenario_id
    top_cut = exact_cut_by_scenario.get(top_scenario_id)
    if top_cut is None:
        return scenario_priority, effective_probe_limit, probe_limit_policy, metadata

    selected_priority = [scored_scenarios[0][0]]
    search_depth = min(len(scored_scenarios), max(2, config.second_probe_search_depth))
    best_candidate: tuple[int, float, float] | None = None
    best_rank: int | None = None
    best_support_distance: float | None = None
    for rank, (scenario_index, _, _) in enumerate(scored_scenarios[1:search_depth], start=2):
        scenario_id = instance.scenarios[scenario_index].scenario_id
        candidate_cut = exact_cut_by_scenario.get(scenario_id)
        if candidate_cut is None:
            continue
        directional_distance = _cut_directional_distance(top_cut, candidate_cut)
        support_distance = _cut_support_jaccard_distance(
            top_cut,
            candidate_cut,
            config.second_probe_support_topk,
        )
        if best_candidate is None:
            best_candidate = (scenario_index, directional_distance, support_distance)
            best_rank = rank
            best_support_distance = support_distance
        if directional_distance >= config.second_probe_directional_distance_threshold:
            selected_priority.append(scenario_index)
            metadata.update(
                {
                    "second_probe_gate_passed": True,
                    "second_probe_selected_rank": rank,
                    "second_probe_directional_distance": directional_distance,
                    "second_probe_support_jaccard_distance": support_distance,
                }
            )
            remaining_priority = [
                scenario
                for scenario in scenario_priority
                if scenario not in selected_priority
            ]
            return (
                [*selected_priority, *remaining_priority],
                effective_probe_limit,
                "dynamic_high_nonredundant",
                metadata,
            )

    if best_candidate is not None and best_rank is not None:
        metadata.update(
            {
                "second_probe_selected_rank": best_rank,
                "second_probe_directional_distance": best_candidate[1],
                "second_probe_support_jaccard_distance": best_support_distance,
            }
        )
    return selected_priority, 1, "dynamic_high_blocked", metadata


def _build_cuts_for_design(
    instance: SupplyChainInstance,
    config: SolverConfig,
    design: list[int],
    iteration: int,
    source: str,
    alternative_point: list[float],
) -> tuple[float, list[BendersCut], list[dict[str, object]]]:
    scenario_cuts: list[BendersCut] = []
    scenario_details: list[dict[str, object]] = []
    expected_recourse = 0.0
    intercept_sum = 0.0
    coefficient_sum = np.zeros(instance.n_facilities)
    strengthened_any = False

    for scenario_index, scenario in enumerate(instance.scenarios):
        evaluation = solve_scenario_recourse(
            instance,
            scenario_index,
            design,
            use_strong_cuts=config.use_strong_cuts,
            alternative_point=alternative_point,
        )
        expected_recourse += scenario.probability * evaluation.objective_value
        intercept_sum += scenario.probability * evaluation.intercept
        coefficient_sum += scenario.probability * np.array(evaluation.coefficients, dtype=float)
        strengthened_any = strengthened_any or evaluation.strengthened

        efficacy = abs(evaluation.objective_value) / (
            1.0 + np.linalg.norm(np.array(evaluation.coefficients, dtype=float), ord=1)
        )
        cut = BendersCut(
            scenario_id=scenario.scenario_id,
            intercept=evaluation.intercept,
            coefficients=evaluation.coefficients,
            source_iteration=iteration,
            source_candidate=source,
            generating_design=list(design),
            recourse_value=evaluation.objective_value,
            efficacy=float(efficacy),
            strengthened=evaluation.strengthened,
            metadata={
                "probability": scenario.probability,
                "alpha": evaluation.alpha,
                "beta": evaluation.beta,
            },
        )
        scenario_cuts.append(cut)
        scenario_details.append(
            {
                "scenario_id": scenario.scenario_id,
                "probability": float(scenario.probability),
                "objective_value": evaluation.objective_value,
                "strengthened": float(evaluation.strengthened),
                "solve_time_seconds": evaluation.solve_time_seconds,
            }
        )

    if config.multicut:
        return expected_recourse, scenario_cuts, scenario_details

    aggregate_cut = BendersCut(
        scenario_id="ALL",
        intercept=float(intercept_sum),
        coefficients=coefficient_sum.astype(float).tolist(),
        source_iteration=iteration,
        source_candidate=source,
        generating_design=list(design),
        recourse_value=float(expected_recourse),
        efficacy=float(
            abs(expected_recourse) / (1.0 + np.linalg.norm(coefficient_sum, ord=1))
        ),
        strengthened=bool(strengthened_any),
        metadata={"aggregation": "expected_value_single_cut"},
    )
    return expected_recourse, [aggregate_cut], scenario_details


def _progressive_heuristic_evaluation(
    instance: SupplyChainInstance,
    config: SolverConfig,
    design: list[int],
    iteration: int,
    source: str,
    alternative_point: list[float],
    cut_pool: list[BendersCut],
    incumbent_objective: float | None,
    scenario_order: list[int] | None = None,
    scenario_probe_limit: int | None = None,
    probe_gap_mass_by_scenario: dict[str, float] | None = None,
) -> tuple[
    float | None,
    list[BendersCut],
    list[dict[str, object]],
    bool,
    float | None,
    bool,
    dict[str, float | bool | int | str | None],
]:
    first_stage = float(
        np.dot(np.array(instance.opening_costs, dtype=float), np.array(design, dtype=float))
    )
    if scenario_order is None:
        scenario_order = sorted(
            range(instance.n_scenarios),
            key=lambda index: instance.scenarios[index].probability,
            reverse=True,
        )
    if scenario_probe_limit is not None:
        scenario_order = scenario_order[: max(0, scenario_probe_limit)]

    lower_bounds_by_scenario = {
        scenario.scenario_id: _scenario_cut_lower_bound(cut_pool, scenario.scenario_id, design)
        for scenario in instance.scenarios
    }
    remaining_lower_bound = sum(
        scenario.probability * lower_bounds_by_scenario[scenario.scenario_id]
        for scenario in instance.scenarios
    )

    exact_partial = 0.0
    cuts: list[BendersCut] = []
    details: list[dict[str, object]] = []
    partial_probe_only = len(scenario_order) < instance.n_scenarios
    probe_economy_metadata: dict[str, float | bool | int | str | None] = {
        "probe_economy_early_stop": False,
        "probe_economy_stop_reason": None,
        "probe_economy_realized_gain": 0.0,
        "probe_economy_target_gain": None,
        "probe_economy_realized_gap_mass": 0.0,
        "probe_economy_scheduled_gap_mass": 0.0,
    }
    realized_gain = 0.0
    realized_gap_mass = 0.0
    if probe_gap_mass_by_scenario is None:
        probe_gap_mass_by_scenario = {}
    scheduled_gap_mass = float(
        sum(
            probe_gap_mass_by_scenario.get(instance.scenarios[scenario_index].scenario_id, 0.0)
            for scenario_index in scenario_order
        )
    )
    target_gain = (
        config.partial_probe_gain_target_ratio * scheduled_gap_mass
        if config.partial_probe_economy_enabled and scheduled_gap_mass > 0.0
        else None
    )
    probe_economy_metadata["probe_economy_target_gain"] = target_gain
    probe_economy_metadata["probe_economy_scheduled_gap_mass"] = scheduled_gap_mass

    for scenario_index in scenario_order:
        scenario = instance.scenarios[scenario_index]
        remaining_lower_bound -= (
            scenario.probability * lower_bounds_by_scenario[scenario.scenario_id]
        )
        evaluation = solve_scenario_recourse(
            instance,
            scenario_index,
            design,
            use_strong_cuts=config.use_strong_cuts,
            alternative_point=alternative_point,
        )
        exact_partial += scenario.probability * evaluation.objective_value
        efficacy = abs(evaluation.objective_value) / (
            1.0 + np.linalg.norm(np.array(evaluation.coefficients, dtype=float), ord=1)
        )
        cuts.append(
            BendersCut(
                scenario_id=scenario.scenario_id,
                intercept=evaluation.intercept,
                coefficients=evaluation.coefficients,
                source_iteration=iteration,
                source_candidate=source,
                generating_design=list(design),
                recourse_value=evaluation.objective_value,
                efficacy=float(efficacy),
                strengthened=evaluation.strengthened,
                metadata={
                    "probability": scenario.probability,
                    "alpha": evaluation.alpha,
                    "beta": evaluation.beta,
                    "progressive_screening": True,
                },
            )
        )
        realized_gain += scenario.probability * max(
            0.0,
            evaluation.objective_value - lower_bounds_by_scenario[scenario.scenario_id],
        )
        realized_gap_mass += probe_gap_mass_by_scenario.get(scenario.scenario_id, 0.0)
        details.append(
            {
                "scenario_id": scenario.scenario_id,
                "probability": float(scenario.probability),
                "objective_value": evaluation.objective_value,
                "strengthened": float(evaluation.strengthened),
                "solve_time_seconds": evaluation.solve_time_seconds,
            }
        )

        if incumbent_objective is not None:
            lower_bound_total = first_stage + exact_partial + remaining_lower_bound
            if lower_bound_total >= incumbent_objective - config.tolerance:
                probe_economy_metadata["probe_economy_realized_gain"] = float(realized_gain)
                probe_economy_metadata["probe_economy_realized_gap_mass"] = float(
                    realized_gap_mass
                )
                return (
                    None,
                    cuts,
                    details,
                    True,
                    lower_bound_total,
                    False,
                    probe_economy_metadata,
                )

        if (
            config.partial_probe_economy_enabled
            and partial_probe_only
            and scenario_probe_limit is not None
            and len(details) >= max(1, config.partial_probe_min_steps)
            and len(details) < len(scenario_order)
            and target_gain is not None
        ):
            remaining_gap_mass = max(0.0, scheduled_gap_mass - realized_gap_mass)
            optimistic_gain = realized_gain + config.partial_probe_optimism_factor * remaining_gap_mass
            early_accept_ratio = config.partial_probe_early_accept_ratio
            if early_accept_ratio > 0.0 and realized_gain >= early_accept_ratio * scheduled_gap_mass:
                probe_economy_metadata.update(
                    {
                        "probe_economy_early_stop": True,
                        "probe_economy_stop_reason": "target_hit",
                        "probe_economy_realized_gain": float(realized_gain),
                        "probe_economy_realized_gap_mass": float(realized_gap_mass),
                    }
                )
                return None, cuts, details, False, None, True, probe_economy_metadata
            if optimistic_gain < target_gain:
                probe_economy_metadata.update(
                    {
                        "probe_economy_early_stop": True,
                        "probe_economy_stop_reason": "target_miss",
                        "probe_economy_realized_gain": float(realized_gain),
                        "probe_economy_realized_gap_mass": float(realized_gap_mass),
                    }
                )
                return None, cuts, details, False, None, True, probe_economy_metadata

    if partial_probe_only:
        probe_economy_metadata["probe_economy_realized_gain"] = float(realized_gain)
        probe_economy_metadata["probe_economy_realized_gap_mass"] = float(realized_gap_mass)
        return None, cuts, details, False, None, True, probe_economy_metadata

    probe_economy_metadata["probe_economy_realized_gain"] = float(realized_gain)
    probe_economy_metadata["probe_economy_realized_gap_mass"] = float(realized_gap_mass)
    return exact_partial, cuts, details, False, None, False, probe_economy_metadata


def solve_instance(
    instance: SupplyChainInstance,
    config: SolverConfig,
) -> RunResult:
    rng = np.random.default_rng(config.random_seed + instance.seed)
    start_time = time.perf_counter()

    cut_pool: list[BendersCut] = []
    iteration_logs: list[IterationLog] = []
    candidate_history: list[dict[str, object]] = []

    incumbent_objective: float | None = None
    incumbent_design: list[int] | None = None
    last_certified_lb = 0.0
    best_gap = None
    previous_gap = inf
    time_to_first_feasible = None
    time_to_5pct_gap = None
    time_to_1pct_gap = None
    time_to_proof = None
    exact_corrections = 0
    total_generated_cuts = 0
    total_retained_cuts = 0
    no_improvement_rounds = 0
    core_point = [0.5] * instance.n_facilities
    termination_reason = "max_iterations"
    last_exact_design: list[int] | None = None

    for iteration in range(1, config.max_iterations + 1):
        iteration_start = time.perf_counter()
        exact_master_required = (
            iteration == 1
            or iteration % max(1, config.exact_correction_frequency) == 0
            or no_improvement_rounds >= config.stagnation_trigger
            or not cut_pool
        )
        master_solve_seconds = 0.0
        candidate_generation_seconds = 0.0
        exact_anchor_eval_seconds = 0.0
        exact_anchor_scenario_solve_seconds = 0.0
        heuristic_eval_seconds = 0.0
        heuristic_scenario_solve_seconds = 0.0
        anchor_expected_recourse: float | None = None
        anchor_gap_density_raw = 0.0
        search_activated = False
        search_activation_policy = "candidate_search_off"
        search_activation_top_mass = None
        search_activation_top_mass_raw_share = None
        search_activation_effective_support_ratio = None

        exact_master_solution = None
        if exact_master_required:
            master_solve_start = time.perf_counter()
            exact_master_solution = solve_master(
                instance, cut_pool, multicut=config.multicut
            )
            master_solve_seconds = time.perf_counter() - master_solve_start
            exact_corrections += 1
            last_certified_lb = exact_master_solution.objective_value
            last_exact_design = exact_master_solution.design
            core_point = [
                0.5 * current + 0.5 * chosen
                for current, chosen in zip(core_point, exact_master_solution.design, strict=True)
            ]

        anchor_design = (
            exact_master_solution.design
            if exact_master_solution is not None
            else incumbent_design or last_exact_design or [0] * instance.n_facilities
        )

        anchor_proposal = CandidateProposal(
            design=list(anchor_design),
            source="exact_rmp" if exact_master_required else "carry_forward",
            surrogate_score=float(last_certified_lb),
        )
        should_search = (
            config.use_candidate_search
            and iteration <= config.search_max_iterations
            and (previous_gap == inf or previous_gap > config.search_gap_threshold)
        )
        if not config.use_candidate_search:
            search_activation_policy = "candidate_search_off"
        elif iteration > config.search_max_iterations:
            search_activation_policy = "search_iteration_cap"
        elif previous_gap != inf and previous_gap <= config.search_gap_threshold:
            search_activation_policy = "search_gap_gate"
        else:
            search_activation_policy = "search_requested"
        mandatory_cuts: list[BendersCut] = []
        heuristic_cuts: list[BendersCut] = []
        best_iteration_objective = None
        candidate_sources: list[str] = []
        improved_this_iteration = False
        selected_candidates: list[CandidateProposal] = [anchor_proposal]
        eligible_candidates: list[CandidateProposal] = []

        # Always evaluate the exact-master candidate first.
        for proposal in selected_candidates:
            anchor_eval_start = time.perf_counter()
            expected_recourse, generated_cuts, scenario_details = _build_cuts_for_design(
                instance=instance,
                config=config,
                design=proposal.design,
                iteration=iteration,
                source=proposal.source,
                alternative_point=core_point,
            )
            anchor_eval_wall_seconds = time.perf_counter() - anchor_eval_start
            anchor_eval_scenario_solve_seconds = _scenario_solve_seconds(scenario_details)
            exact_anchor_eval_seconds += anchor_eval_wall_seconds
            exact_anchor_scenario_solve_seconds += anchor_eval_scenario_solve_seconds
            anchor_expected_recourse = expected_recourse

            first_stage = float(
                np.dot(np.array(instance.opening_costs, dtype=float), np.array(proposal.design))
            )
            total_objective = first_stage + expected_recourse
            best_iteration_objective = (
                total_objective
                if best_iteration_objective is None
                else min(best_iteration_objective, total_objective)
            )

            candidate_sources.append(proposal.source)
            proposal_metadata = dict(proposal.metadata)
            anchor_history_index = len(candidate_history)
            candidate_history.append(
                {
                    "iteration": iteration,
                    "source": proposal.source,
                    "design": proposal.design,
                    "surrogate_score": proposal.surrogate_score,
                    "true_objective": total_objective,
                    "first_stage_cost": first_stage,
                    "expected_recourse": expected_recourse,
                    "scenario_details": scenario_details,
                    "evaluated_scenarios": len(scenario_details),
                    "proposal_rank": 0,
                    "proposal_family": proposal_metadata.get("proposal_family", proposal.source),
                    "anchor_hamming_distance": proposal_metadata.get(
                        "anchor_hamming_distance", 0
                    ),
                    "flip_set": proposal_metadata.get("flip_set", []),
                    "flip_priority_sum": proposal_metadata.get("flip_priority_sum", 0.0),
                    "harvest_proxy": proposal_metadata.get("harvest_proxy", 0.0),
                    "geometry_pair_proxy": proposal_metadata.get("geometry_pair_proxy", 0.0),
                    "geometry_pair_rank": proposal_metadata.get("geometry_pair_rank"),
                    "geometry_pair_distance": proposal_metadata.get(
                        "geometry_pair_distance"
                    ),
                    "geometry_pair_scenarios": proposal_metadata.get(
                        "geometry_pair_scenarios", []
                    ),
                    "targeting_score": proposal_metadata.get(
                        "targeting_score", proposal.surrogate_score
                    ),
                    "selected_for_exact_probe": True,
                    "effective_candidate_eval_limit": None,
                    "candidate_eval_policy": "exact_full",
                    "candidate_eval_density_ratio": None,
                    "second_candidate_target_gap_ratio": None,
                    "second_candidate_harvest_advantage": None,
                    "second_candidate_geometry_advantage": None,
                    "activation_support_ratio": None,
                    "activation_support_ratio_ceiling": None,
                    "activation_support_ratio_policy": "exact_full",
                    "activation_top_mass_share": None,
                    "activation_top_mass_share_floor": None,
                    "activation_effective_support_ratio": None,
                    "activation_effective_support_ratio_ceiling": None,
                    "effective_scenario_probe_limit": instance.n_scenarios,
                    "probe_limit_policy": "exact_full",
                    "probe_ranking_mode": config.probe_ranking_mode,
                    "probe_economy_early_stop": False,
                    "probe_economy_stop_reason": None,
                    "probe_economy_realized_gain": 0.0,
                    "probe_economy_target_gain": None,
                    "probe_economy_realized_gap_mass": 0.0,
                    "probe_economy_scheduled_gap_mass": 0.0,
                    "scenario_probe_support_ratio": None,
                    "candidate_eval_wall_seconds": anchor_eval_wall_seconds,
                    "scenario_solve_time_seconds": anchor_eval_scenario_solve_seconds,
                    "search_activation_policy": search_activation_policy,
                    "search_activation_top_mass": None,
                    "search_activation_top_mass_raw_share": None,
                    "search_activation_effective_support_ratio": None,
                    "anchor_expected_recourse": anchor_expected_recourse,
                    "anchor_gap_density": None,
                    "anchor_gap_density_raw": None,
                    "anchor_gap_density_threshold": None,
                    "anchor_gap_density_metric": config.heuristic_anchor_gap_density_metric,
                    "second_probe_gate_passed": False,
                    "second_probe_selected_rank": None,
                    "second_probe_directional_distance": None,
                    "second_probe_support_jaccard_distance": None,
                    "scenario_priority_head": [],
                }
            )

            if incumbent_objective is None or total_objective + config.tolerance < incumbent_objective:
                incumbent_objective = total_objective
                incumbent_design = list(proposal.design)
                improved_this_iteration = True
                if time_to_first_feasible is None:
                    time_to_first_feasible = time.perf_counter() - start_time

            if proposal.source in {"exact_rmp", "carry_forward"}:
                mandatory_cuts.extend(generated_cuts)
            else:
                heuristic_cuts.extend(
                    _rank_heuristic_cuts(
                        instance,
                        generated_cuts,
                        config.max_heuristic_cuts_per_candidate,
                    )
                )

        anchor_gap_by_scenario: dict[str, float] = {}
        expected_anchor_underestimation = 0.0
        anchor_gap_density = 0.0
        effective_density_threshold = 0.0
        if config.multicut and mandatory_cuts:
            anchor_gap_by_scenario = _compute_anchor_gap_profile(
                instance=instance,
                base_cuts=cut_pool,
                exact_anchor_cuts=mandatory_cuts,
                anchor_design=anchor_design,
            )
            expected_anchor_underestimation = sum(
                scenario.probability * anchor_gap_by_scenario.get(scenario.scenario_id, 0.0)
                for scenario in instance.scenarios
            )
            anchor_gap_density, anchor_gap_density_raw = _anchor_gap_density(
                instance=instance,
                config=config,
                expected_anchor_underestimation=expected_anchor_underestimation,
                anchor_expected_recourse=anchor_expected_recourse,
            )
            effective_density_threshold = _effective_anchor_gap_density_threshold(
                instance=instance,
                config=config,
            )
            if (
                should_search
                and expected_anchor_underestimation
                < config.heuristic_anchor_gap_activation_threshold
            ):
                should_search = False
                search_activation_policy = "anchor_underestimation_below_threshold"
            if (
                should_search
                and anchor_gap_density
                < effective_density_threshold
            ):
                should_search = False
                search_activation_policy = "anchor_density_below_threshold"
            if should_search:
                (
                    search_concentration_gate_passed,
                    search_activation_policy,
                    search_activation_top_mass,
                    search_activation_effective_support_ratio,
                ) = _passes_search_activation_concentration_gate(
                    instance=instance,
                    config=config,
                    anchor_gap_by_scenario=anchor_gap_by_scenario,
                )
                search_activation_top_mass_raw_share = _probe_top_mass_share(
                    _anchor_gap_mass_scores(instance, anchor_gap_by_scenario),
                    max(1, config.search_activation_top_mass_topk),
                )
                if not search_concentration_gate_passed:
                    should_search = False
            elif search_activation_policy == "search_requested":
                search_activation_policy = "anchor_density_pass"

        if should_search:
            search_activated = True
            if search_activation_policy == "search_requested":
                search_activation_policy = "search_active"
        elif search_activation_policy == "search_requested":
            search_activation_policy = "search_inactive"

        candidate_history[anchor_history_index].update(
            {
                "expected_anchor_underestimation": expected_anchor_underestimation,
                "anchor_gap_density": anchor_gap_density,
                "anchor_gap_density_raw": anchor_gap_density_raw,
                "anchor_gap_density_threshold": effective_density_threshold,
                "anchor_gap_density_metric": config.heuristic_anchor_gap_density_metric,
                "search_activation_policy": search_activation_policy,
                "search_activation_top_mass": search_activation_top_mass,
                "search_activation_top_mass_raw_share": search_activation_top_mass_raw_share,
                "search_activation_effective_support_ratio": (
                    search_activation_effective_support_ratio
                ),
            }
        )

        heuristic_proposals: list[CandidateProposal] = []
        if should_search:
            candidate_generation_start = time.perf_counter()
            heuristic_proposals = generate_candidates(
                instance=instance,
                cuts=[*cut_pool, *mandatory_cuts],
                config=config,
                anchor_design=anchor_design,
                incumbent_design=incumbent_design,
                rng=rng,
                exact_anchor_cuts=mandatory_cuts,
                anchor_gap_by_scenario=anchor_gap_by_scenario,
            )
            candidate_generation_seconds = time.perf_counter() - candidate_generation_start

        unique_candidates: dict[tuple[int, ...], CandidateProposal] = {}
        for proposal in heuristic_proposals:
            existing = unique_candidates.get(proposal.key())
            if existing is None or proposal.surrogate_score < existing.surrogate_score:
                unique_candidates[proposal.key()] = proposal

        ordered_candidates = sorted(
            unique_candidates.values(),
            key=lambda value: _candidate_priority_key(config, value),
        )
        heuristic_reference_score = surrogate_objective(
            instance=instance,
            cuts=[*cut_pool, *mandatory_cuts],
            design=np.array(anchor_design, dtype=int),
            config=config,
        )
        eligible_candidates: list[CandidateProposal] = []
        for proposal in ordered_candidates:
            if proposal.key() == anchor_proposal.key():
                continue
            if config.min_heuristic_surrogate_improvement_ratio > 0.0:
                minimum_required_improvement = (
                    config.min_heuristic_surrogate_improvement_ratio
                    * max(1.0, abs(heuristic_reference_score))
                )
                if (
                    proposal.surrogate_score
                    >= heuristic_reference_score - minimum_required_improvement
                ):
                    continue
            eligible_candidates.append(proposal)

        (
            effective_candidate_eval_limit,
            candidate_eval_policy,
            candidate_eval_metadata,
        ) = _effective_heuristic_eval_limit(
            instance=instance,
            config=config,
            should_search=should_search,
            eligible_candidates=eligible_candidates,
            anchor_gap_density=anchor_gap_density,
            effective_density_threshold=effective_density_threshold,
        )

        for proposal_rank, proposal in enumerate(
            eligible_candidates[:effective_candidate_eval_limit],
            start=1,
        ):
            if config.multicut:
                scored_scenarios = _score_probe_scenarios(
                    instance=instance,
                    config=config,
                    design=proposal.design,
                    anchor_design=anchor_design,
                    exact_anchor_cuts=mandatory_cuts,
                    anchor_gap_by_scenario=anchor_gap_by_scenario,
                )
                scenario_priority = [
                    scenario_index for scenario_index, _, _ in scored_scenarios
                ]
                probe_support_ratio = _probe_support_ratio(scored_scenarios)
                probe_top_mass_share = _probe_top_mass_share(
                    scored_scenarios,
                    max(1, config.activation_top_mass_share_topk),
                )
                probe_top_mass_metric = _top_mass_metric(
                    raw_top_mass_share=probe_top_mass_share,
                    n_scenarios=instance.n_scenarios,
                    topk=max(1, config.activation_top_mass_share_topk),
                    metric=config.activation_top_mass_metric,
                )
                probe_effective_support_ratio = _probe_effective_support_ratio(
                    scored_scenarios
                )
                activation_support_ratio = probe_support_ratio
                (
                    activation_support_gate_passed,
                    activation_support_ratio_policy,
                ) = _passes_activation_support_gate(
                    instance=instance,
                    config=config,
                    probe_support_ratio=probe_support_ratio,
                    probe_top_mass_share=probe_top_mass_share,
                    probe_effective_support_ratio=probe_effective_support_ratio,
                )
                if not activation_support_gate_passed:
                    continue
                effective_probe_limit, probe_limit_policy = _effective_scenario_probe_limit(
                    instance=instance,
                    config=config,
                    should_search=should_search,
                    expected_anchor_underestimation=expected_anchor_underestimation,
                    anchor_gap_density=anchor_gap_density,
                    effective_density_threshold=effective_density_threshold,
                    probe_support_ratio=probe_support_ratio,
                )
                (
                    scenario_priority,
                    effective_probe_limit,
                    probe_limit_policy,
                    second_probe_metadata,
                ) = _select_nonredundant_probe_schedule(
                    instance=instance,
                    config=config,
                    scored_scenarios=scored_scenarios,
                    exact_anchor_cuts=mandatory_cuts,
                    effective_probe_limit=effective_probe_limit,
                    probe_limit_policy=probe_limit_policy,
                )
                scenario_priority_head = [
                    instance.scenarios[scenario_index].scenario_id
                    for scenario_index in scenario_priority[:3]
                ]
                candidate_eval_start = time.perf_counter()
                (
                    expected_recourse,
                    generated_cuts,
                    scenario_details,
                    screened_out,
                    screening_bound,
                    partial_probe_only,
                    probe_economy_metadata,
                ) = (
                    _progressive_heuristic_evaluation(
                        instance=instance,
                        config=config,
                        design=proposal.design,
                        iteration=iteration,
                        source=proposal.source,
                        alternative_point=core_point,
                        cut_pool=[*cut_pool, *mandatory_cuts, *heuristic_cuts],
                        incumbent_objective=incumbent_objective,
                        scenario_order=scenario_priority,
                        scenario_probe_limit=effective_probe_limit,
                        probe_gap_mass_by_scenario={
                            scenario.scenario_id: scenario.probability
                            * anchor_gap_by_scenario.get(scenario.scenario_id, 0.0)
                            for scenario in instance.scenarios
                        },
                    )
                )
                candidate_eval_wall_seconds = time.perf_counter() - candidate_eval_start
                effective_cut_limit = _effective_heuristic_cut_limit(
                    config=config,
                    effective_probe_limit=effective_probe_limit,
                )
            else:
                activation_support_ratio = None
                activation_support_ratio_policy = "single_cut"
                scenario_priority_head = []
                probe_support_ratio = 0.0
                probe_top_mass_share = 0.0
                probe_top_mass_metric = 0.0
                probe_effective_support_ratio = 0.0
                effective_probe_limit = None
                probe_limit_policy = "single_cut"
                second_probe_metadata = {
                    "second_probe_gate_passed": False,
                    "second_probe_selected_rank": None,
                    "second_probe_directional_distance": None,
                    "second_probe_support_jaccard_distance": None,
                }
                probe_economy_metadata = {
                    "probe_economy_early_stop": False,
                    "probe_economy_stop_reason": None,
                    "probe_economy_realized_gain": 0.0,
                    "probe_economy_target_gain": None,
                    "probe_economy_realized_gap_mass": 0.0,
                    "probe_economy_scheduled_gap_mass": 0.0,
                }
                candidate_eval_start = time.perf_counter()
                expected_recourse, generated_cuts, scenario_details = _build_cuts_for_design(
                    instance=instance,
                    config=config,
                    design=proposal.design,
                    iteration=iteration,
                    source=proposal.source,
                    alternative_point=core_point,
                )
                candidate_eval_wall_seconds = time.perf_counter() - candidate_eval_start
                screened_out = False
                screening_bound = None
                partial_probe_only = False
                effective_cut_limit = config.max_heuristic_cuts_per_candidate

            scenario_solve_time_seconds = _scenario_solve_seconds(scenario_details)
            heuristic_eval_seconds += candidate_eval_wall_seconds
            heuristic_scenario_solve_seconds += scenario_solve_time_seconds

            selected_candidates.append(proposal)
            first_stage = float(
                np.dot(np.array(instance.opening_costs, dtype=float), np.array(proposal.design))
            )

            candidate_sources.append(proposal.source)
            proposal_metadata = dict(proposal.metadata)
            if screened_out or expected_recourse is None:
                candidate_history.append(
                    {
                        "iteration": iteration,
                        "source": proposal.source,
                        "design": proposal.design,
                        "surrogate_score": proposal.surrogate_score,
                        "true_objective": None,
                        "first_stage_cost": first_stage,
                        "expected_recourse": None,
                        "scenario_details": scenario_details,
                        "progressive_exact_screening": True,
                        "screened_pruned": screened_out,
                        "partial_cut_harvest": partial_probe_only,
                        "screening_lower_bound": screening_bound,
                        "evaluated_scenarios": len(scenario_details),
                        "expected_anchor_underestimation": expected_anchor_underestimation,
                        "anchor_gap_density": anchor_gap_density,
                        "anchor_gap_density_raw": anchor_gap_density_raw,
                        "anchor_gap_density_threshold": effective_density_threshold,
                        "anchor_gap_density_metric": config.heuristic_anchor_gap_density_metric,
                        "anchor_expected_recourse": anchor_expected_recourse,
                        "proposal_rank": proposal_rank,
                        "proposal_family": proposal_metadata.get(
                            "proposal_family", proposal.source
                        ),
                        "anchor_hamming_distance": proposal_metadata.get(
                            "anchor_hamming_distance", 0
                        ),
                        "flip_set": proposal_metadata.get("flip_set", []),
                        "flip_priority_sum": proposal_metadata.get("flip_priority_sum", 0.0),
                        "harvest_proxy": proposal_metadata.get("harvest_proxy", 0.0),
                        "geometry_pair_proxy": proposal_metadata.get(
                            "geometry_pair_proxy", 0.0
                        ),
                        "geometry_pair_rank": proposal_metadata.get("geometry_pair_rank"),
                        "geometry_pair_distance": proposal_metadata.get(
                            "geometry_pair_distance"
                        ),
                        "geometry_pair_scenarios": proposal_metadata.get(
                            "geometry_pair_scenarios", []
                        ),
                        "targeting_score": proposal_metadata.get(
                            "targeting_score", proposal.surrogate_score
                        ),
                        "selected_for_exact_probe": True,
                        "effective_candidate_eval_limit": effective_candidate_eval_limit,
                        "candidate_eval_policy": candidate_eval_policy,
                        "candidate_eval_density_ratio": candidate_eval_metadata[
                            "candidate_eval_density_ratio"
                        ],
                        "second_candidate_target_gap_ratio": candidate_eval_metadata[
                            "second_candidate_target_gap_ratio"
                        ],
                        "second_candidate_harvest_advantage": candidate_eval_metadata[
                            "second_candidate_harvest_advantage"
                        ],
                        "second_candidate_geometry_advantage": candidate_eval_metadata[
                            "second_candidate_geometry_advantage"
                        ],
                        "activation_support_ratio": activation_support_ratio,
                        "activation_support_ratio_ceiling": (
                            config.activation_support_ratio_ceiling
                            if config.activation_support_ratio_ceiling > 0.0
                            else None
                        ),
                        "activation_support_ratio_policy": activation_support_ratio_policy,
                        "activation_top_mass_share": probe_top_mass_metric,
                        "activation_top_mass_raw_share": probe_top_mass_share,
                        "activation_top_mass_share_floor": (
                            config.activation_top_mass_share_floor
                            if config.activation_top_mass_share_floor > 0.0
                            else None
                        ),
                        "activation_effective_support_ratio": probe_effective_support_ratio,
                        "activation_effective_support_ratio_ceiling": (
                            config.activation_effective_support_ratio_ceiling
                            if config.activation_effective_support_ratio_ceiling > 0.0
                            else None
                        ),
                        "effective_scenario_probe_limit": effective_probe_limit,
                        "probe_limit_policy": probe_limit_policy,
                        "scenario_probe_support_ratio": probe_support_ratio,
                        "probe_ranking_mode": config.probe_ranking_mode,
                        "probe_economy_early_stop": probe_economy_metadata[
                            "probe_economy_early_stop"
                        ],
                        "probe_economy_stop_reason": probe_economy_metadata[
                            "probe_economy_stop_reason"
                        ],
                        "probe_economy_realized_gain": probe_economy_metadata[
                            "probe_economy_realized_gain"
                        ],
                        "probe_economy_target_gain": probe_economy_metadata[
                            "probe_economy_target_gain"
                        ],
                        "probe_economy_realized_gap_mass": probe_economy_metadata[
                            "probe_economy_realized_gap_mass"
                        ],
                        "probe_economy_scheduled_gap_mass": probe_economy_metadata[
                            "probe_economy_scheduled_gap_mass"
                        ],
                        "candidate_eval_wall_seconds": candidate_eval_wall_seconds,
                        "scenario_solve_time_seconds": scenario_solve_time_seconds,
                        "search_activation_policy": search_activation_policy,
                        "search_activation_top_mass": search_activation_top_mass,
                        "search_activation_top_mass_raw_share": (
                            search_activation_top_mass_raw_share
                        ),
                        "search_activation_effective_support_ratio": (
                            search_activation_effective_support_ratio
                        ),
                        "second_probe_gate_passed": second_probe_metadata[
                            "second_probe_gate_passed"
                        ],
                        "second_probe_selected_rank": second_probe_metadata[
                            "second_probe_selected_rank"
                        ],
                        "second_probe_directional_distance": second_probe_metadata[
                            "second_probe_directional_distance"
                        ],
                        "second_probe_support_jaccard_distance": second_probe_metadata[
                            "second_probe_support_jaccard_distance"
                        ],
                        "scenario_priority_head": scenario_priority_head,
                    }
                )
            else:
                total_objective = first_stage + expected_recourse
                best_iteration_objective = (
                    total_objective
                    if best_iteration_objective is None
                    else min(best_iteration_objective, total_objective)
                )

                candidate_history.append(
                    {
                        "iteration": iteration,
                        "source": proposal.source,
                        "design": proposal.design,
                        "surrogate_score": proposal.surrogate_score,
                        "true_objective": total_objective,
                        "first_stage_cost": first_stage,
                        "expected_recourse": expected_recourse,
                        "scenario_details": scenario_details,
                        "progressive_exact_screening": config.multicut,
                        "screened_pruned": False,
                        "partial_cut_harvest": False,
                        "screening_lower_bound": screening_bound,
                        "evaluated_scenarios": len(scenario_details),
                        "expected_anchor_underestimation": expected_anchor_underestimation,
                        "anchor_gap_density": anchor_gap_density,
                        "anchor_gap_density_raw": anchor_gap_density_raw,
                        "anchor_gap_density_threshold": effective_density_threshold,
                        "anchor_gap_density_metric": config.heuristic_anchor_gap_density_metric,
                        "anchor_expected_recourse": anchor_expected_recourse,
                        "proposal_rank": proposal_rank,
                        "proposal_family": proposal_metadata.get(
                            "proposal_family", proposal.source
                        ),
                        "anchor_hamming_distance": proposal_metadata.get(
                            "anchor_hamming_distance", 0
                        ),
                        "flip_set": proposal_metadata.get("flip_set", []),
                        "flip_priority_sum": proposal_metadata.get("flip_priority_sum", 0.0),
                        "harvest_proxy": proposal_metadata.get("harvest_proxy", 0.0),
                        "geometry_pair_proxy": proposal_metadata.get(
                            "geometry_pair_proxy", 0.0
                        ),
                        "geometry_pair_rank": proposal_metadata.get("geometry_pair_rank"),
                        "geometry_pair_distance": proposal_metadata.get(
                            "geometry_pair_distance"
                        ),
                        "geometry_pair_scenarios": proposal_metadata.get(
                            "geometry_pair_scenarios", []
                        ),
                        "targeting_score": proposal_metadata.get(
                            "targeting_score", proposal.surrogate_score
                        ),
                        "selected_for_exact_probe": True,
                        "effective_candidate_eval_limit": effective_candidate_eval_limit,
                        "candidate_eval_policy": candidate_eval_policy,
                        "candidate_eval_density_ratio": candidate_eval_metadata[
                            "candidate_eval_density_ratio"
                        ],
                        "second_candidate_target_gap_ratio": candidate_eval_metadata[
                            "second_candidate_target_gap_ratio"
                        ],
                        "second_candidate_harvest_advantage": candidate_eval_metadata[
                            "second_candidate_harvest_advantage"
                        ],
                        "second_candidate_geometry_advantage": candidate_eval_metadata[
                            "second_candidate_geometry_advantage"
                        ],
                        "activation_support_ratio": activation_support_ratio,
                        "activation_support_ratio_ceiling": (
                            config.activation_support_ratio_ceiling
                            if config.activation_support_ratio_ceiling > 0.0
                            else None
                        ),
                        "activation_support_ratio_policy": activation_support_ratio_policy,
                        "activation_top_mass_share": probe_top_mass_metric,
                        "activation_top_mass_raw_share": probe_top_mass_share,
                        "activation_top_mass_share_floor": (
                            config.activation_top_mass_share_floor
                            if config.activation_top_mass_share_floor > 0.0
                            else None
                        ),
                        "activation_effective_support_ratio": probe_effective_support_ratio,
                        "activation_effective_support_ratio_ceiling": (
                            config.activation_effective_support_ratio_ceiling
                            if config.activation_effective_support_ratio_ceiling > 0.0
                            else None
                        ),
                        "effective_scenario_probe_limit": effective_probe_limit,
                        "probe_limit_policy": probe_limit_policy,
                        "scenario_probe_support_ratio": probe_support_ratio,
                        "probe_ranking_mode": config.probe_ranking_mode,
                        "probe_economy_early_stop": probe_economy_metadata[
                            "probe_economy_early_stop"
                        ],
                        "probe_economy_stop_reason": probe_economy_metadata[
                            "probe_economy_stop_reason"
                        ],
                        "probe_economy_realized_gain": probe_economy_metadata[
                            "probe_economy_realized_gain"
                        ],
                        "probe_economy_target_gain": probe_economy_metadata[
                            "probe_economy_target_gain"
                        ],
                        "probe_economy_realized_gap_mass": probe_economy_metadata[
                            "probe_economy_realized_gap_mass"
                        ],
                        "probe_economy_scheduled_gap_mass": probe_economy_metadata[
                            "probe_economy_scheduled_gap_mass"
                        ],
                        "candidate_eval_wall_seconds": candidate_eval_wall_seconds,
                        "scenario_solve_time_seconds": scenario_solve_time_seconds,
                        "search_activation_policy": search_activation_policy,
                        "search_activation_top_mass": search_activation_top_mass,
                        "search_activation_top_mass_raw_share": (
                            search_activation_top_mass_raw_share
                        ),
                        "search_activation_effective_support_ratio": (
                            search_activation_effective_support_ratio
                        ),
                        "second_probe_gate_passed": second_probe_metadata[
                            "second_probe_gate_passed"
                        ],
                        "second_probe_selected_rank": second_probe_metadata[
                            "second_probe_selected_rank"
                        ],
                        "second_probe_directional_distance": second_probe_metadata[
                            "second_probe_directional_distance"
                        ],
                        "second_probe_support_jaccard_distance": second_probe_metadata[
                            "second_probe_support_jaccard_distance"
                        ],
                        "scenario_priority_head": scenario_priority_head,
                    }
                )

                if (
                    incumbent_objective is None
                    or total_objective + config.tolerance < incumbent_objective
                ):
                    incumbent_objective = total_objective
                    incumbent_design = list(proposal.design)
                    improved_this_iteration = True
                    if time_to_first_feasible is None:
                        time_to_first_feasible = time.perf_counter() - start_time

            heuristic_cuts.extend(
                _rank_heuristic_cuts(
                    instance,
                    generated_cuts,
                    effective_cut_limit,
                )
            )

        if improved_this_iteration:
            no_improvement_rounds = 0
        else:
            no_improvement_rounds += 1

        new_cuts = [*mandatory_cuts, *heuristic_cuts]

        total_generated_cuts += len(new_cuts)
        if config.use_filtering:
            if config.filter_heuristic_cuts_only and config.protect_exact_master_cuts:
                retained_heuristic_cuts = filter_new_cuts(
                    cut_pool + mandatory_cuts,
                    heuristic_cuts,
                    similarity_threshold=config.similarity_threshold,
                    efficacy_threshold=config.efficacy_threshold,
                    max_retained_new_cuts=config.max_retained_new_cuts,
                )
                retained_cuts = [*mandatory_cuts, *retained_heuristic_cuts]
            else:
                retained_cuts = filter_new_cuts(
                    cut_pool,
                    new_cuts,
                    similarity_threshold=config.similarity_threshold,
                    efficacy_threshold=config.efficacy_threshold,
                    max_retained_new_cuts=max(
                        config.max_retained_new_cuts, len(mandatory_cuts)
                    ),
                )
        else:
            retained_cuts = [
                *mandatory_cuts,
                *heuristic_cuts[: config.max_retained_new_cuts],
            ]
        total_retained_cuts += len(retained_cuts)
        cut_pool.extend(retained_cuts)

        gap = _relative_gap(incumbent_objective, last_certified_lb)
        now = time.perf_counter() - start_time
        if gap is not None:
            best_gap = gap if best_gap is None else min(best_gap, gap)
            previous_gap = gap
            if gap <= 0.05 and time_to_5pct_gap is None:
                time_to_5pct_gap = now
            if gap <= 0.01 and time_to_1pct_gap is None:
                time_to_1pct_gap = now
            if gap <= config.tolerance and exact_master_required:
                time_to_proof = now
                termination_reason = "certified_optimality_gap"

        iteration_logs.append(
            IterationLog(
                iteration=iteration,
                exact_master_solved=exact_master_required,
                certified_lower_bound=last_certified_lb,
                incumbent_upper_bound=incumbent_objective,
                relative_gap=gap,
                evaluated_candidates=len(selected_candidates),
                retained_cuts=len(retained_cuts),
                generated_cuts=len(new_cuts),
                feasibility_cuts=0,
                optimality_cuts=len(new_cuts),
                cut_pool_size=len(cut_pool),
                candidate_sources=candidate_sources,
                best_iteration_objective=best_iteration_objective,
                exact_master_design=exact_master_solution.design
                if exact_master_solution is not None
                else None,
                incumbent_design=incumbent_design,
                eligible_heuristic_candidates=len(eligible_candidates),
                selected_heuristic_candidates=max(0, len(selected_candidates) - 1),
                iteration_wall_seconds=time.perf_counter() - iteration_start,
                master_solve_seconds=master_solve_seconds,
                candidate_generation_seconds=candidate_generation_seconds,
                exact_anchor_eval_seconds=exact_anchor_eval_seconds,
                exact_anchor_scenario_solve_seconds=exact_anchor_scenario_solve_seconds,
                heuristic_eval_seconds=heuristic_eval_seconds,
                heuristic_scenario_solve_seconds=heuristic_scenario_solve_seconds,
                search_activated=search_activated,
                search_activation_policy=search_activation_policy,
                search_activation_top_mass=search_activation_top_mass,
                search_activation_top_mass_raw_share=search_activation_top_mass_raw_share,
                search_activation_effective_support_ratio=(
                    search_activation_effective_support_ratio
                ),
                anchor_expected_recourse=anchor_expected_recourse,
                anchor_expected_underestimation=expected_anchor_underestimation,
                anchor_gap_density=anchor_gap_density,
                anchor_gap_density_raw=anchor_gap_density_raw,
                anchor_gap_density_threshold=effective_density_threshold,
                anchor_gap_density_metric=config.heuristic_anchor_gap_density_metric,
            )
        )

        if termination_reason == "certified_optimality_gap":
            break

    # final exact correction if needed for a certified closing bound
    final_gap = _relative_gap(incumbent_objective, last_certified_lb)
    if termination_reason != "certified_optimality_gap":
        final_master = solve_master(instance, cut_pool, multicut=config.multicut)
        exact_corrections += 1
        last_certified_lb = final_master.objective_value
        final_gap = _relative_gap(incumbent_objective, last_certified_lb)
        if final_gap is not None and final_gap <= config.tolerance:
            termination_reason = "final_exact_correction_closed_gap"
            if time_to_proof is None:
                time_to_proof = time.perf_counter() - start_time

    return RunResult(
        run_id=f"{instance.instance_id}__{config.name}",
        instance_id=instance.instance_id,
        config_name=config.name,
        seed=config.random_seed + instance.seed,
        success=incumbent_objective is not None,
        termination_reason=termination_reason,
        incumbent_objective=incumbent_objective,
        certified_lower_bound=last_certified_lb,
        final_gap=final_gap,
        incumbent_design=incumbent_design,
        iterations=len(iteration_logs),
        total_generated_cuts=total_generated_cuts,
        total_retained_cuts=total_retained_cuts,
        time_to_first_feasible=time_to_first_feasible,
        time_to_5pct_gap=time_to_5pct_gap,
        time_to_1pct_gap=time_to_1pct_gap,
        time_to_proof=time_to_proof,
        wall_clock_seconds=time.perf_counter() - start_time,
        exact_corrections=exact_corrections,
        cut_pool_size=len(cut_pool),
        iteration_logs=[log.to_dict() for log in iteration_logs],
        cut_pool=[cut.to_dict() for cut in cut_pool],
        candidate_history=candidate_history,
        config=config.to_dict(),
        instance_metadata={
            **instance.metadata,
            "n_facilities": instance.n_facilities,
            "n_customers": instance.n_customers,
            "n_scenarios": instance.n_scenarios,
        },
    )
