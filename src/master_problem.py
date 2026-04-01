from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .models import BendersCut, SupplyChainInstance


@dataclass
class MasterSolution:
    design: list[int]
    objective_value: float
    theta_values: list[float]
    success: bool


def solve_master(
    instance: SupplyChainInstance,
    cuts: list[BendersCut],
    *,
    multicut: bool,
) -> MasterSolution:
    n_facilities = instance.n_facilities
    n_theta = instance.n_scenarios if multicut else 1
    n_variables = n_facilities + n_theta

    objective = np.zeros(n_variables)
    objective[:n_facilities] = np.array(instance.opening_costs, dtype=float)
    if multicut:
        objective[n_facilities:] = np.array(
            [scenario.probability for scenario in instance.scenarios], dtype=float
        )
    else:
        objective[n_facilities] = 1.0

    lower_bounds = np.concatenate([np.zeros(n_facilities), np.zeros(n_theta)])
    upper_bounds = np.concatenate([np.ones(n_facilities), np.full(n_theta, np.inf)])
    bounds = Bounds(lower_bounds, upper_bounds)
    integrality = np.concatenate([np.ones(n_facilities), np.zeros(n_theta)])

    constraints: list[LinearConstraint] = []
    if cuts:
        matrix = []
        rhs = []
        for cut in cuts:
            row = np.zeros(n_variables)
            row[:n_facilities] = np.array(cut.coefficients, dtype=float)
            theta_index = (
                instance.scenarios.index(
                    next(
                        scenario
                        for scenario in instance.scenarios
                        if scenario.scenario_id == cut.scenario_id
                    )
                )
                if multicut
                else 0
            )
            row[n_facilities + theta_index] = -1.0
            matrix.append(row)
            rhs.append(-cut.intercept)
        matrix_array = np.array(matrix, dtype=float)
        rhs_array = np.array(rhs, dtype=float)
        constraints.append(
            LinearConstraint(matrix_array, -np.inf * np.ones_like(rhs_array), rhs_array)
        )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints if constraints else None,
    )

    if not result.success:
        raise RuntimeError(
            f"Master solve failed for {instance.instance_id}: {result.message}"
        )

    solution = np.rint(result.x[:n_facilities]).astype(int)
    theta_values = result.x[n_facilities:].astype(float).tolist()
    return MasterSolution(
        design=solution.tolist(),
        objective_value=float(result.fun),
        theta_values=theta_values,
        success=True,
    )
