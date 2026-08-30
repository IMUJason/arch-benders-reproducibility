from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ScenarioData:
    scenario_id: str
    probability: float
    demands: list[float]
    transport_costs: list[list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioData":
        return cls(**payload)


@dataclass
class SupplyChainInstance:
    instance_id: str
    seed: int
    facility_names: list[str]
    customer_names: list[str]
    opening_costs: list[float]
    capacities: list[float]
    shortage_costs: list[float]
    scenarios: list[ScenarioData]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_facilities(self) -> int:
        return len(self.facility_names)

    @property
    def n_customers(self) -> int:
        return len(self.customer_names)

    @property
    def n_scenarios(self) -> int:
        return len(self.scenarios)

    @property
    def expected_total_demand(self) -> float:
        return sum(
            scenario.probability * sum(scenario.demands) for scenario in self.scenarios
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenarios"] = [scenario.to_dict() for scenario in self.scenarios]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SupplyChainInstance":
        payload = dict(payload)
        payload["scenarios"] = [
            ScenarioData.from_dict(scenario) for scenario in payload["scenarios"]
        ]
        return cls(**payload)


@dataclass
class BendersCut:
    scenario_id: str
    intercept: float
    coefficients: list[float]
    source_iteration: int
    source_candidate: str
    generating_design: list[int]
    recourse_value: float
    efficacy: float
    strengthened: bool
    cut_family: str = "optimality"
    metadata: dict[str, Any] = field(default_factory=dict)

    def value_at(self, design: list[int]) -> float:
        return self.intercept + sum(
            coefficient * variable
            for coefficient, variable in zip(self.coefficients, design, strict=True)
        )

    def signature(self, precision: int = 8) -> tuple[Any, ...]:
        rounded = tuple(round(value, precision) for value in self.coefficients)
        return (self.scenario_id, round(self.intercept, precision), *rounded)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateProposal:
    design: list[int]
    source: str
    surrogate_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[int, ...]:
        return tuple(self.design)


@dataclass
class IterationLog:
    iteration: int
    exact_master_solved: bool
    certified_lower_bound: float
    incumbent_upper_bound: float | None
    relative_gap: float | None
    evaluated_candidates: int
    retained_cuts: int
    generated_cuts: int
    feasibility_cuts: int
    optimality_cuts: int
    cut_pool_size: int
    candidate_sources: list[str]
    best_iteration_objective: float | None
    exact_master_design: list[int] | None
    incumbent_design: list[int] | None
    eligible_heuristic_candidates: int
    selected_heuristic_candidates: int
    iteration_wall_seconds: float
    master_solve_seconds: float
    candidate_generation_seconds: float
    exact_anchor_eval_seconds: float
    exact_anchor_scenario_solve_seconds: float
    heuristic_eval_seconds: float
    heuristic_scenario_solve_seconds: float
    search_activated: bool
    search_activation_policy: str
    search_activation_top_mass: float | None
    search_activation_top_mass_raw_share: float | None
    search_activation_effective_support_ratio: float | None
    anchor_expected_recourse: float | None
    anchor_expected_underestimation: float
    anchor_gap_density: float
    anchor_gap_density_raw: float
    anchor_gap_density_threshold: float
    anchor_gap_density_metric: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SolverConfig:
    name: str
    max_iterations: int = 40
    tolerance: float = 1e-4
    multicut: bool = True
    use_candidate_search: bool = False
    candidate_budget: int = 8
    annealing_steps: int = 80
    annealing_temperature: float = 3.0
    annealing_cooling: float = 0.95
    capacity_penalty: float = 6.0
    exact_correction_frequency: int = 1
    stagnation_trigger: int = 3
    use_filtering: bool = False
    similarity_threshold: float = 0.995
    efficacy_threshold: float = 1e-6
    max_retained_new_cuts: int = 50
    use_strong_cuts: bool = False
    random_seed: int = 0
    heuristic_eval_limit: int = 1
    dynamic_candidate_eval_enabled: bool = False
    dynamic_candidate_eval_base_value: int | None = None
    dynamic_candidate_eval_high_value: int = 2
    dynamic_candidate_eval_low_scenario_cap: int | None = None
    dynamic_candidate_eval_target_gap_ratio_threshold: float = 0.0
    dynamic_candidate_eval_density_ratio_threshold: float = 0.0
    dynamic_candidate_eval_harvest_advantage_threshold: float = 0.0
    dynamic_candidate_eval_geometry_advantage_threshold: float = 0.0
    search_max_iterations: int = 2
    search_gap_threshold: float = 0.02
    greedy_refine_steps: int = 0
    max_heuristic_cuts_per_candidate: int | None = 8
    protect_exact_master_cuts: bool = True
    filter_heuristic_cuts_only: bool = True
    min_heuristic_surrogate_improvement_ratio: float = 0.0
    heuristic_scenario_probe_limit: int | None = None
    heuristic_anchor_gap_activation_threshold: float = 0.0
    heuristic_anchor_gap_density_threshold: float = 0.0
    heuristic_anchor_gap_density_reference_scenarios: int | None = None
    heuristic_anchor_gap_density_scenario_exponent: float = 0.0
    heuristic_anchor_gap_density_metric: str = "raw_per_scenario"
    heuristic_anchor_gap_density_scale_floor: float = 1.0
    search_activation_top_mass_share_floor: float = 0.0
    search_activation_top_mass_topk: int = 1
    search_activation_top_mass_metric: str = "share"
    search_activation_effective_support_ratio_ceiling: float = 0.0
    probe_ranking_mode: str = "weighted_sum"
    probe_gap_weight: float = 1.0
    probe_sensitivity_weight: float = 1.0
    probe_rank_weight: float = 0.0
    probe_overlap_weight: float = 0.0
    candidate_targeting_mode: str = "baseline"
    local_branching_radius: int = 1
    candidate_flip_pool_size: int = 8
    local_branching_pair_budget: int = 6
    candidate_disagreement_budget: int = 0
    candidate_disagreement_max_flips: int = 2
    candidate_harvest_proxy_weight: float = 0.0
    candidate_top_gap_support: int | None = None
    candidate_geometry_pair_budget: int = 0
    candidate_geometry_pair_weight: float = 0.0
    candidate_geometry_pair_search_depth: int = 8
    candidate_geometry_distance_threshold: float = 0.0
    dynamic_probe_limit_enabled: bool = False
    dynamic_probe_limit_base_value: int | None = None
    dynamic_probe_limit_high_value: int = 2
    dynamic_probe_limit_dormant_value: int = 0
    dynamic_probe_limit_low_scenario_cap: int | None = None
    dynamic_probe_limit_high_density_threshold: float = 0.0
    dynamic_probe_limit_high_density_ratio: float = 1.0
    dynamic_probe_limit_high_underestimation_threshold: float = 0.0
    dynamic_probe_limit_high_underestimation_ratio: float = 1.0
    dynamic_probe_limit_support_ratio_threshold: float = 0.0
    activation_support_ratio_ceiling: float = 0.0
    activation_support_ratio_low_scenario_cap: int | None = None
    activation_top_mass_share_floor: float = 0.0
    activation_top_mass_share_topk: int = 1
    activation_top_mass_metric: str = "share"
    activation_effective_support_ratio_ceiling: float = 0.0
    partial_probe_economy_enabled: bool = False
    partial_probe_min_steps: int = 1
    partial_probe_gain_target_ratio: float = 0.0
    partial_probe_optimism_factor: float = 1.0
    partial_probe_early_accept_ratio: float = 0.0
    second_probe_nonredundancy_enabled: bool = False
    second_probe_directional_distance_threshold: float = 0.0
    second_probe_support_topk: int = 4
    second_probe_search_depth: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    run_id: str
    instance_id: str
    config_name: str
    seed: int
    success: bool
    termination_reason: str
    incumbent_objective: float | None
    certified_lower_bound: float | None
    final_gap: float | None
    incumbent_design: list[int] | None
    iterations: int
    total_generated_cuts: int
    total_retained_cuts: int
    time_to_first_feasible: float | None
    time_to_5pct_gap: float | None
    time_to_1pct_gap: float | None
    time_to_proof: float | None
    wall_clock_seconds: float
    exact_corrections: int
    cut_pool_size: int
    iteration_logs: list[dict[str, Any]]
    cut_pool: list[dict[str, Any]]
    candidate_history: list[dict[str, Any]]
    config: dict[str, Any]
    instance_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
