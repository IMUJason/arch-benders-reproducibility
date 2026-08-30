from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np

from .models import BendersCut, CandidateProposal, SolverConfig, SupplyChainInstance


def _scenario_surrogate_value(
    cuts: Iterable[BendersCut], scenario_id: str, design: np.ndarray
) -> float:
    scenario_cuts = [cut.value_at(design.astype(int).tolist()) for cut in cuts if cut.scenario_id == scenario_id]
    if not scenario_cuts:
        return 0.0
    return max(0.0, max(scenario_cuts))


def surrogate_objective(
    instance: SupplyChainInstance,
    cuts: list[BendersCut],
    design: np.ndarray,
    config: SolverConfig,
) -> float:
    first_stage = float(np.dot(np.array(instance.opening_costs, dtype=float), design))
    if config.multicut:
        surrogate_recourse = sum(
            scenario.probability * _scenario_surrogate_value(cuts, scenario.scenario_id, design)
            for scenario in instance.scenarios
        )
    else:
        aggregate_values = [
            cut.value_at(design.astype(int).tolist()) for cut in cuts if cut.scenario_id == "ALL"
        ]
        surrogate_recourse = max(0.0, max(aggregate_values)) if aggregate_values else 0.0

    capacity = float(np.dot(np.array(instance.capacities, dtype=float), design))
    shortfall = max(0.0, instance.expected_total_demand - capacity)
    return first_stage + surrogate_recourse + config.capacity_penalty * shortfall * shortfall


def _flip(design: np.ndarray, index: int) -> np.ndarray:
    candidate = design.copy()
    candidate[index] = 1 - candidate[index]
    return candidate


def _flip_indices(anchor: np.ndarray, design: np.ndarray) -> list[int]:
    return [
        index
        for index, (anchor_value, design_value) in enumerate(
            zip(anchor.astype(int).tolist(), design.astype(int).tolist(), strict=True)
        )
        if anchor_value != design_value
    ]


def _top_gap_support(
    instance: SupplyChainInstance,
    anchor_gap_by_scenario: dict[str, float],
    limit: int | None,
) -> set[str]:
    if limit is None or limit <= 0 or limit >= instance.n_scenarios:
        return {scenario.scenario_id for scenario in instance.scenarios}
    ordered = sorted(
        instance.scenarios,
        key=lambda scenario: (
            scenario.probability * anchor_gap_by_scenario.get(scenario.scenario_id, 0.0),
            anchor_gap_by_scenario.get(scenario.scenario_id, 0.0),
        ),
        reverse=True,
    )
    return {scenario.scenario_id for scenario in ordered[:limit]}


def _compute_harvest_proxy(
    instance: SupplyChainInstance,
    config: SolverConfig,
    design: np.ndarray,
    anchor_design: np.ndarray,
    exact_anchor_cuts: list[BendersCut],
    anchor_gap_by_scenario: dict[str, float],
) -> float:
    exact_cut_by_scenario = {
        cut.scenario_id: cut for cut in exact_anchor_cuts if cut.scenario_id != "ALL"
    }
    support = _top_gap_support(
        instance=instance,
        anchor_gap_by_scenario=anchor_gap_by_scenario,
        limit=config.candidate_top_gap_support,
    )
    delta = design.astype(float) - anchor_design.astype(float)
    proxy = 0.0
    for scenario in instance.scenarios:
        if scenario.scenario_id not in support:
            continue
        reference_cut = exact_cut_by_scenario.get(scenario.scenario_id)
        sensitivity = 0.0
        if reference_cut is not None:
            sensitivity = abs(
                float(np.dot(np.array(reference_cut.coefficients, dtype=float), delta))
            )
        anchor_gap = anchor_gap_by_scenario.get(scenario.scenario_id, 0.0)
        proxy += scenario.probability * (
            config.probe_gap_weight * anchor_gap
            + config.probe_sensitivity_weight * sensitivity
        )
    return float(proxy)


def _compute_facility_flip_scores(
    instance: SupplyChainInstance,
    exact_anchor_cuts: list[BendersCut],
    anchor_gap_by_scenario: dict[str, float],
) -> list[float]:
    probability_by_scenario = {
        scenario.scenario_id: scenario.probability for scenario in instance.scenarios
    }
    scores = np.zeros(instance.n_facilities, dtype=float)
    for cut in exact_anchor_cuts:
        if cut.scenario_id == "ALL":
            continue
        weight = probability_by_scenario.get(cut.scenario_id, 0.0) * anchor_gap_by_scenario.get(
            cut.scenario_id, 0.0
        )
        if weight <= 0.0:
            continue
        scores += weight * np.abs(np.array(cut.coefficients, dtype=float))
    return scores.astype(float).tolist()


def _compute_signed_direction_scores(
    instance: SupplyChainInstance,
    exact_anchor_cuts: list[BendersCut],
    anchor_gap_by_scenario: dict[str, float],
) -> list[float]:
    probability_by_scenario = {
        scenario.scenario_id: scenario.probability for scenario in instance.scenarios
    }
    scores = np.zeros(instance.n_facilities, dtype=float)
    for cut in exact_anchor_cuts:
        if cut.scenario_id == "ALL":
            continue
        weight = probability_by_scenario.get(cut.scenario_id, 0.0) * anchor_gap_by_scenario.get(
            cut.scenario_id, 0.0
        )
        if weight == 0.0:
            continue
        scores += weight * np.array(cut.coefficients, dtype=float)
    return scores.astype(float).tolist()


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


def _score_probe_scenarios(
    instance: SupplyChainInstance,
    config: SolverConfig,
    design: np.ndarray,
    anchor_design: np.ndarray,
    exact_anchor_cuts: list[BendersCut],
    anchor_gap_by_scenario: dict[str, float],
) -> list[tuple[str, float, float]]:
    delta = design.astype(float) - anchor_design.astype(float)
    exact_cut_by_scenario = {
        cut.scenario_id: cut for cut in exact_anchor_cuts if cut.scenario_id != "ALL"
    }
    scored_scenarios: list[tuple[str, float, float]] = []
    for scenario in instance.scenarios:
        reference_cut = exact_cut_by_scenario.get(scenario.scenario_id)
        sensitivity = 0.0
        if reference_cut is not None:
            sensitivity = abs(
                float(np.dot(np.array(reference_cut.coefficients, dtype=float), delta))
            )
        anchor_gap = anchor_gap_by_scenario.get(scenario.scenario_id, 0.0)
        blended = scenario.probability * (
            config.probe_gap_weight * anchor_gap
            + config.probe_sensitivity_weight * sensitivity
        )
        scored_scenarios.append((scenario.scenario_id, blended, scenario.probability))
    return sorted(
        scored_scenarios,
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )


def _effective_geometry_distance_threshold(config: SolverConfig) -> float:
    if config.candidate_geometry_distance_threshold > 0.0:
        return config.candidate_geometry_distance_threshold
    return config.second_probe_directional_distance_threshold


def _geometry_distance_multiplier(distance: float, threshold: float) -> float:
    if threshold <= 0.0:
        return max(1.0, distance)
    return distance / threshold


def _compute_geometry_pair_proxy(
    instance: SupplyChainInstance,
    config: SolverConfig,
    design: np.ndarray,
    anchor_design: np.ndarray,
    exact_anchor_cuts: list[BendersCut],
    anchor_gap_by_scenario: dict[str, float],
) -> tuple[float, int | None, float | None, list[str]]:
    exact_cut_by_scenario = {
        cut.scenario_id: cut for cut in exact_anchor_cuts if cut.scenario_id != "ALL"
    }
    scored_scenarios = _score_probe_scenarios(
        instance=instance,
        config=config,
        design=design,
        anchor_design=anchor_design,
        exact_anchor_cuts=exact_anchor_cuts,
        anchor_gap_by_scenario=anchor_gap_by_scenario,
    )
    if len(scored_scenarios) < 2:
        return 0.0, None, None, []
    threshold = _effective_geometry_distance_threshold(config)
    search_depth = max(2, config.candidate_geometry_pair_search_depth)
    top_scenario_id = scored_scenarios[0][0]
    top_cut = exact_cut_by_scenario.get(top_scenario_id)
    if top_cut is None:
        return 0.0, None, None, []
    best_proxy = 0.0
    best_rank: int | None = None
    best_distance: float | None = None
    best_pair: list[str] = []
    for rank, (scenario_id, scenario_score, _) in enumerate(
        scored_scenarios[1:search_depth], start=2
    ):
        candidate_cut = exact_cut_by_scenario.get(scenario_id)
        if candidate_cut is None:
            continue
        distance = _cut_directional_distance(top_cut, candidate_cut)
        if distance < threshold:
            continue
        proxy = (
            scored_scenarios[0][1] + scenario_score
        ) * _geometry_distance_multiplier(distance, threshold)
        if proxy > best_proxy:
            best_proxy = float(proxy)
            best_rank = rank
            best_distance = float(distance)
            best_pair = [top_scenario_id, scenario_id]
    return best_proxy, best_rank, best_distance, best_pair


def _select_geometry_scenario_pairs(
    instance: SupplyChainInstance,
    config: SolverConfig,
    exact_anchor_cuts: list[BendersCut],
    anchor_gap_by_scenario: dict[str, float],
) -> list[tuple[BendersCut, BendersCut, float]]:
    if config.candidate_geometry_pair_budget <= 0:
        return []
    exact_cut_by_scenario = {
        cut.scenario_id: cut for cut in exact_anchor_cuts if cut.scenario_id != "ALL"
    }
    support = _top_gap_support(
        instance=instance,
        anchor_gap_by_scenario=anchor_gap_by_scenario,
        limit=config.candidate_top_gap_support,
    )
    supported_cuts = [
        exact_cut_by_scenario[scenario.scenario_id]
        for scenario in instance.scenarios
        if scenario.scenario_id in support and scenario.scenario_id in exact_cut_by_scenario
    ]
    scored_pairs: list[tuple[float, BendersCut, BendersCut]] = []
    for first_cut, second_cut in combinations(supported_cuts, 2):
        first_weight = anchor_gap_by_scenario.get(first_cut.scenario_id, 0.0)
        second_weight = anchor_gap_by_scenario.get(second_cut.scenario_id, 0.0)
        if first_weight <= 0.0 and second_weight <= 0.0:
            continue
        distance = _cut_directional_distance(first_cut, second_cut)
        if distance <= _effective_geometry_distance_threshold(config):
            continue
        pair_score = (first_weight + second_weight) * _geometry_distance_multiplier(
            distance,
            _effective_geometry_distance_threshold(config),
        )
        scored_pairs.append((pair_score, first_cut, second_cut))
    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    return [
        (first_cut, second_cut, float(pair_score))
        for pair_score, first_cut, second_cut in scored_pairs[: config.candidate_geometry_pair_budget]
    ]


def _scenario_pair_disagreement_scores(
    anchor_design: np.ndarray,
    first_cut: BendersCut,
    second_cut: BendersCut,
) -> np.ndarray:
    anchor_move_sign = 1.0 - 2.0 * anchor_design.astype(float)
    first = np.array(first_cut.coefficients, dtype=float)
    second = np.array(second_cut.coefficients, dtype=float)
    return np.abs(anchor_move_sign * (first - second))


def generate_candidates(
    instance: SupplyChainInstance,
    cuts: list[BendersCut],
    config: SolverConfig,
    anchor_design: list[int] | np.ndarray,
    incumbent_design: list[int] | None,
    rng: np.random.Generator,
    exact_anchor_cuts: list[BendersCut] | None = None,
    anchor_gap_by_scenario: dict[str, float] | None = None,
) -> list[CandidateProposal]:
    anchor = np.array(anchor_design, dtype=int)
    proposals: dict[tuple[int, ...], CandidateProposal] = {}
    score_cache: dict[tuple[int, ...], float] = {}
    use_structured_targeting = config.candidate_targeting_mode != "baseline"

    def score(design: np.ndarray) -> float:
        key = tuple(design.astype(int).tolist())
        cached = score_cache.get(key)
        if cached is None:
            cached = surrogate_objective(instance, cuts, design.astype(int), config)
            score_cache[key] = cached
        return cached

    def register(design: np.ndarray, source: str, score: float, **metadata: float) -> None:
        design = design.astype(int)
        key = tuple(design.tolist())
        metadata = dict(metadata)
        flip_set = _flip_indices(anchor, design)
        metadata.setdefault("proposal_family", source)
        metadata.setdefault("anchor_hamming_distance", len(flip_set))
        metadata.setdefault("flip_set", flip_set)
        metadata.setdefault("flip_priority_sum", 0.0)
        harvest_proxy = 0.0
        if use_structured_targeting and exact_anchor_cuts and anchor_gap_by_scenario is not None:
            harvest_proxy = _compute_harvest_proxy(
                instance=instance,
                config=config,
                design=design,
                anchor_design=anchor,
                exact_anchor_cuts=exact_anchor_cuts,
                anchor_gap_by_scenario=anchor_gap_by_scenario,
            )
        metadata.setdefault("harvest_proxy", harvest_proxy)
        geometry_pair_proxy = 0.0
        geometry_pair_rank = None
        geometry_pair_distance = None
        geometry_pair_scenarios: list[str] = []
        if use_structured_targeting and exact_anchor_cuts and anchor_gap_by_scenario is not None:
            (
                geometry_pair_proxy,
                geometry_pair_rank,
                geometry_pair_distance,
                geometry_pair_scenarios,
            ) = _compute_geometry_pair_proxy(
                instance=instance,
                config=config,
                design=design,
                anchor_design=anchor,
                exact_anchor_cuts=exact_anchor_cuts,
                anchor_gap_by_scenario=anchor_gap_by_scenario,
            )
        metadata.setdefault("geometry_pair_proxy", geometry_pair_proxy)
        metadata.setdefault("geometry_pair_rank", geometry_pair_rank)
        metadata.setdefault("geometry_pair_distance", geometry_pair_distance)
        metadata.setdefault("geometry_pair_scenarios", geometry_pair_scenarios)
        metadata.setdefault(
            "targeting_score",
            float(
                score
                - config.candidate_harvest_proxy_weight * harvest_proxy
                - config.candidate_geometry_pair_weight * geometry_pair_proxy
            ),
        )
        existing = proposals.get(key)
        proposal = CandidateProposal(
            design=design.tolist(),
            source=source,
            surrogate_score=float(score),
            metadata=metadata,
        )
        if existing is None or proposal.surrogate_score < existing.surrogate_score - 1e-12:
            proposals[key] = proposal
        elif use_structured_targeting and abs(
            proposal.surrogate_score - existing.surrogate_score
        ) <= 1e-12:
            proposal_targeting_score = proposal.metadata.get(
                "targeting_score", proposal.surrogate_score
            )
            existing_targeting_score = existing.metadata.get(
                "targeting_score", existing.surrogate_score
            )
            if proposal_targeting_score < existing_targeting_score - 1e-12:
                proposals[key] = proposal
            elif (
                abs(proposal_targeting_score - existing_targeting_score) <= 1e-12
                and proposal.metadata.get("scenario_pair_base_score", 0.0)
                > existing.metadata.get("scenario_pair_base_score", 0.0)
            ):
                proposals[key] = proposal
            elif (
                abs(proposal_targeting_score - existing_targeting_score) <= 1e-12
                and proposal.metadata.get("geometry_pair_proxy", 0.0)
                > existing.metadata.get("geometry_pair_proxy", 0.0)
            ):
                proposals[key] = proposal
            elif (
                abs(proposal_targeting_score - existing_targeting_score) <= 1e-12
                and abs(
                    proposal.metadata.get("geometry_pair_proxy", 0.0)
                    - existing.metadata.get("geometry_pair_proxy", 0.0)
                )
                <= 1e-12
                and proposal.metadata.get("harvest_proxy", 0.0)
                > existing.metadata.get("harvest_proxy", 0.0)
            ):
                proposals[key] = proposal

    register(
        anchor,
        "anneal_anchor",
        surrogate_objective(instance, cuts, anchor, config),
            accepted_moves=0,
        )

    if incumbent_design is not None:
        incumbent = np.array(incumbent_design, dtype=int)
        register(
            incumbent,
            "incumbent_seed",
            surrogate_objective(instance, cuts, incumbent, config),
            accepted_moves=0,
        )

    seed_designs = [anchor]
    if incumbent_design is not None:
        seed_designs.append(np.array(incumbent_design, dtype=int))

    while len(seed_designs) < min(config.candidate_budget, 4):
        seed_designs.append(rng.integers(0, 2, size=instance.n_facilities, dtype=int))

    for seed_index, initial_design in enumerate(seed_designs):
        current = initial_design.copy()
        current_score = score(current)
        best = current.copy()
        best_score = current_score
        temperature = config.annealing_temperature
        accepted_moves = 0

        for _ in range(config.annealing_steps):
            flip_index = int(rng.integers(0, instance.n_facilities))
            trial = _flip(current, flip_index)
            trial_score = score(trial)
            delta = trial_score - current_score
            if delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 1e-6)):
                current = trial
                current_score = trial_score
                accepted_moves += 1
                if current_score < best_score:
                    best = current.copy()
                    best_score = current_score
            temperature *= config.annealing_cooling

        register(
            best,
            "quantum_inspired_anneal",
            best_score,
            seed_index=seed_index,
            accepted_moves=accepted_moves,
        )

        greedy = best.copy()
        greedy_score = best_score
        for greedy_step in range(config.greedy_refine_steps):
            step_best_design = greedy.copy()
            step_best_score = greedy_score
            step_best_flip = None
            for facility_index in range(instance.n_facilities):
                trial = _flip(greedy, facility_index)
                trial_score = score(trial)
                if trial_score + 1e-9 < step_best_score:
                    step_best_design = trial
                    step_best_score = trial_score
                    step_best_flip = facility_index
            if step_best_flip is None:
                break
            greedy = step_best_design
            greedy_score = step_best_score
            register(
                greedy,
                "greedy_refine",
                greedy_score,
                seed_index=seed_index,
                greedy_step=greedy_step,
                flipped_facility=step_best_flip,
            )

        # one-step deterministic refinement
        for facility_index in range(instance.n_facilities):
            trial = _flip(best, facility_index)
            trial_score = score(trial)
            register(
                trial,
                "local_refine",
                trial_score,
                seed_index=seed_index,
                flipped_facility=facility_index,
            )

    if (
        config.candidate_targeting_mode != "baseline"
        and exact_anchor_cuts
        and anchor_gap_by_scenario is not None
    ):
        flip_scores = _compute_facility_flip_scores(
            instance=instance,
            exact_anchor_cuts=exact_anchor_cuts,
            anchor_gap_by_scenario=anchor_gap_by_scenario,
        )
        ordered_facilities = sorted(
            range(instance.n_facilities),
            key=lambda facility_index: (flip_scores[facility_index], -facility_index),
            reverse=True,
        )
        top_facilities = ordered_facilities[: max(0, config.candidate_flip_pool_size)]
        for facility_index in top_facilities:
            trial = _flip(anchor, facility_index)
            register(
                trial,
                "targeted_local_branch",
                score(trial),
                proposal_family="targeted_local_branch",
                flipped_facility=facility_index,
                flip_priority_sum=float(flip_scores[facility_index]),
            )

        if config.local_branching_radius >= 2 and config.local_branching_pair_budget > 0:
            pair_candidates = sorted(
                combinations(top_facilities, 2),
                key=lambda pair: flip_scores[pair[0]] + flip_scores[pair[1]],
                reverse=True,
            )[: config.local_branching_pair_budget]
            for first_index, second_index in pair_candidates:
                trial = _flip(_flip(anchor, first_index), second_index)
                register(
                    trial,
                    "targeted_local_branch_pair",
                    score(trial),
                    proposal_family="targeted_local_branch_pair",
                    flip_set=[int(first_index), int(second_index)],
                    flip_priority_sum=float(
                        flip_scores[first_index] + flip_scores[second_index]
                    ),
                )

        if config.candidate_disagreement_budget > 0:
            signed_scores = _compute_signed_direction_scores(
                instance=instance,
                exact_anchor_cuts=exact_anchor_cuts,
                anchor_gap_by_scenario=anchor_gap_by_scenario,
            )
            disagreement_facilities = sorted(
                range(instance.n_facilities),
                key=lambda facility_index: abs(signed_scores[facility_index]),
                reverse=True,
            )[: max(2, config.candidate_flip_pool_size)]

            disagreement_count = 0
            max_flips = max(2, config.candidate_disagreement_max_flips)
            for flip_count in range(2, min(max_flips, len(disagreement_facilities)) + 1):
                flip_set = disagreement_facilities[:flip_count]
                trial = anchor.copy()
                for facility_index in flip_set:
                    trial = _flip(trial, facility_index)
                register(
                    trial,
                    "targeted_disagreement",
                    score(trial),
                    proposal_family="targeted_disagreement",
                    flip_set=[int(index) for index in flip_set],
                    flip_priority_sum=float(sum(abs(signed_scores[index]) for index in flip_set)),
                    disagreement_signed_sum=float(sum(signed_scores[index] for index in flip_set)),
                )
                disagreement_count += 1
                if disagreement_count >= config.candidate_disagreement_budget:
                    break

        geometry_pairs = _select_geometry_scenario_pairs(
            instance=instance,
            config=config,
            exact_anchor_cuts=exact_anchor_cuts,
            anchor_gap_by_scenario=anchor_gap_by_scenario,
        )
        for first_cut, second_cut, pair_score in geometry_pairs:
            disagreement_scores = _scenario_pair_disagreement_scores(
                anchor_design=anchor,
                first_cut=first_cut,
                second_cut=second_cut,
            )
            ordered_geometry_facilities = np.argsort(disagreement_scores)[::-1].tolist()
            if not ordered_geometry_facilities:
                continue
            primary_facility = int(ordered_geometry_facilities[0])
            primary_trial = _flip(anchor, primary_facility)
            register(
                primary_trial,
                "targeted_scenario_pair_single",
                score(primary_trial),
                proposal_family="targeted_scenario_pair_single",
                flipped_facility=primary_facility,
                flip_priority_sum=float(disagreement_scores[primary_facility]),
                scenario_pair_ids=[first_cut.scenario_id, second_cut.scenario_id],
                scenario_pair_distance=float(_cut_directional_distance(first_cut, second_cut)),
                scenario_pair_base_score=pair_score,
            )
            if len(ordered_geometry_facilities) >= 2:
                second_facility = int(ordered_geometry_facilities[1])
                pair_trial = _flip(_flip(anchor, primary_facility), second_facility)
                register(
                    pair_trial,
                    "targeted_scenario_pair_pair",
                    score(pair_trial),
                    proposal_family="targeted_scenario_pair_pair",
                    flip_set=[primary_facility, second_facility],
                    flip_priority_sum=float(
                        disagreement_scores[primary_facility]
                        + disagreement_scores[second_facility]
                    ),
                    scenario_pair_ids=[first_cut.scenario_id, second_cut.scenario_id],
                    scenario_pair_distance=float(_cut_directional_distance(first_cut, second_cut)),
                    scenario_pair_base_score=pair_score,
                )

    if use_structured_targeting:
        ordered = sorted(
            proposals.values(),
            key=lambda proposal: (
                proposal.metadata.get("targeting_score", proposal.surrogate_score),
                -proposal.metadata.get("geometry_pair_proxy", 0.0),
                -proposal.metadata.get("harvest_proxy", 0.0),
                proposal.surrogate_score,
            ),
        )
    else:
        ordered = sorted(proposals.values(), key=lambda proposal: proposal.surrogate_score)
    return ordered[: config.candidate_budget]
