from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import linprog

from .models import SupplyChainInstance


@dataclass
class ScenarioEvaluation:
    scenario_id: str
    objective_value: float
    intercept: float
    coefficients: list[float]
    alpha: list[float]
    beta: list[float]
    strengthened: bool
    solve_time_seconds: float


def _build_primal(instance: SupplyChainInstance, scenario_index: int, design: np.ndarray):
    scenario = instance.scenarios[scenario_index]
    n_facilities = instance.n_facilities
    n_customers = instance.n_customers
    n_flow = n_facilities * n_customers
    n_variables = n_flow + n_customers

    transport = np.array(scenario.transport_costs, dtype=float).reshape(n_flow)
    shortage = np.array(instance.shortage_costs, dtype=float)
    objective = np.concatenate([transport, shortage])

    demand_equalities = np.zeros((n_customers, n_variables))
    for customer_index in range(n_customers):
        for facility_index in range(n_facilities):
            position = facility_index * n_customers + customer_index
            demand_equalities[customer_index, position] = 1.0
        demand_equalities[customer_index, n_flow + customer_index] = 1.0

    capacity_inequalities = np.zeros((n_facilities, n_variables))
    for facility_index in range(n_facilities):
        start = facility_index * n_customers
        end = start + n_customers
        capacity_inequalities[facility_index, start:end] = 1.0

    capacity_rhs = np.array(instance.capacities, dtype=float) * design
    demand_rhs = np.array(scenario.demands, dtype=float)
    bounds = [(0.0, None)] * n_variables

    return objective, capacity_inequalities, capacity_rhs, demand_equalities, demand_rhs, bounds


def _solve_strong_dual(
    instance: SupplyChainInstance,
    scenario_index: int,
    design: np.ndarray,
    alternative_point: np.ndarray,
    target_value: float,
) -> tuple[float, list[float]] | None:
    scenario = instance.scenarios[scenario_index]
    n_facilities = instance.n_facilities
    n_customers = instance.n_customers
    n_variables = n_customers + n_facilities

    objective = np.concatenate(
        [
            np.array(scenario.demands, dtype=float),
            np.array(instance.capacities, dtype=float) * alternative_point,
        ]
    )

    inequality_rows = []
    inequality_rhs = []
    for facility_index in range(n_facilities):
        for customer_index in range(n_customers):
            row = np.zeros(n_variables)
            row[customer_index] = 1.0
            row[n_customers + facility_index] = 1.0
            inequality_rows.append(row)
            inequality_rhs.append(scenario.transport_costs[facility_index][customer_index])

    for customer_index in range(n_customers):
        row = np.zeros(n_variables)
        row[customer_index] = 1.0
        inequality_rows.append(row)
        inequality_rhs.append(instance.shortage_costs[customer_index])

    equality = np.concatenate(
        [
            np.array(scenario.demands, dtype=float),
            np.array(instance.capacities, dtype=float) * design,
        ]
    )

    result = linprog(
        c=-objective,
        A_ub=np.array(inequality_rows, dtype=float),
        b_ub=np.array(inequality_rhs, dtype=float),
        A_eq=equality.reshape(1, -1),
        b_eq=np.array([target_value], dtype=float),
        bounds=[(None, None)] * n_customers + [(None, 0.0)] * n_facilities,
        method="highs",
    )

    if not result.success:
        return None

    alpha = result.x[:n_customers]
    beta = result.x[n_customers:]
    intercept = float(np.dot(alpha, np.array(scenario.demands, dtype=float)))
    coefficients = (
        beta * np.array(instance.capacities, dtype=float)
    ).astype(float).tolist()
    return intercept, coefficients


def solve_scenario_recourse(
    instance: SupplyChainInstance,
    scenario_index: int,
    design: list[int] | np.ndarray,
    *,
    use_strong_cuts: bool = False,
    alternative_point: list[float] | np.ndarray | None = None,
) -> ScenarioEvaluation:
    solve_start = time.perf_counter()
    design_array = np.array(design, dtype=float)
    (
        objective,
        capacity_inequalities,
        capacity_rhs,
        demand_equalities,
        demand_rhs,
        bounds,
    ) = _build_primal(instance, scenario_index, design_array)

    result = linprog(
        c=objective,
        A_ub=capacity_inequalities,
        b_ub=capacity_rhs,
        A_eq=demand_equalities,
        b_eq=demand_rhs,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"Scenario recourse solve failed for {instance.instance_id} / "
            f"{instance.scenarios[scenario_index].scenario_id}: {result.message}"
        )

    alpha = np.array(result.eqlin.marginals, dtype=float)
    beta = np.array(result.ineqlin.marginals, dtype=float)
    intercept = float(np.dot(alpha, demand_rhs))
    coefficients = (beta * np.array(instance.capacities, dtype=float)).astype(float)

    strengthened = False
    if use_strong_cuts and alternative_point is not None:
        strong_cut = _solve_strong_dual(
            instance=instance,
            scenario_index=scenario_index,
            design=design_array,
            alternative_point=np.array(alternative_point, dtype=float),
            target_value=float(result.fun),
        )
        if strong_cut is not None:
            intercept, coefficients = strong_cut[0], np.array(strong_cut[1], dtype=float)
            strengthened = True

    return ScenarioEvaluation(
        scenario_id=instance.scenarios[scenario_index].scenario_id,
        objective_value=float(result.fun),
        intercept=float(intercept),
        coefficients=coefficients.tolist(),
        alpha=alpha.tolist(),
        beta=beta.tolist(),
        strengthened=strengthened,
        solve_time_seconds=time.perf_counter() - solve_start,
    )
