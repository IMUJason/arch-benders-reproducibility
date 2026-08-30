from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import ScenarioData, SupplyChainInstance


def generate_instance(
    instance_id: str,
    *,
    n_facilities: int,
    n_customers: int,
    n_scenarios: int,
    seed: int,
    demand_volatility: float = 0.20,
    transport_volatility: float = 0.10,
    service_level: float = 1.08,
) -> SupplyChainInstance:
    rng = np.random.default_rng(seed)

    facility_names = [f"F{i + 1}" for i in range(n_facilities)]
    customer_names = [f"C{j + 1}" for j in range(n_customers)]

    facility_xy = rng.uniform(0.0, 100.0, size=(n_facilities, 2))
    customer_xy = rng.uniform(0.0, 100.0, size=(n_customers, 2))
    base_distances = np.linalg.norm(
        facility_xy[:, None, :] - customer_xy[None, :, :], axis=2
    )
    base_transport = 3.0 + base_distances / 14.0

    base_demands = rng.integers(12, 30, size=n_customers).astype(float)
    target_capacity = service_level * base_demands.sum()
    raw_capacity = rng.uniform(0.8, 1.4, size=n_facilities)
    capacities = (raw_capacity / raw_capacity.sum()) * target_capacity

    opening_costs = (
        15.0 * capacities / capacities.mean() + rng.uniform(12.0, 28.0, size=n_facilities)
    )
    shortage_costs = (
        np.max(base_transport, axis=0) + rng.uniform(18.0, 30.0, size=n_customers)
    )

    probabilities = rng.random(n_scenarios)
    probabilities = probabilities / probabilities.sum()

    scenarios: list[ScenarioData] = []
    for scenario_index in range(n_scenarios):
        demand_multiplier = rng.lognormal(
            mean=-0.5 * demand_volatility**2,
            sigma=demand_volatility,
            size=n_customers,
        )
        scenario_demands = (base_demands * demand_multiplier).round(4)
        transport_multiplier = rng.lognormal(
            mean=-0.5 * transport_volatility**2,
            sigma=transport_volatility,
            size=(n_facilities, n_customers),
        )
        scenario_costs = (base_transport * transport_multiplier).round(4)

        scenarios.append(
            ScenarioData(
                scenario_id=f"S{scenario_index + 1}",
                probability=float(probabilities[scenario_index]),
                demands=scenario_demands.tolist(),
                transport_costs=scenario_costs.tolist(),
                metadata={
                    "demand_multiplier_mean": float(demand_multiplier.mean()),
                    "transport_multiplier_mean": float(transport_multiplier.mean()),
                },
            )
        )

    return SupplyChainInstance(
        instance_id=instance_id,
        seed=seed,
        facility_names=facility_names,
        customer_names=customer_names,
        opening_costs=opening_costs.round(4).tolist(),
        capacities=capacities.round(4).tolist(),
        shortage_costs=shortage_costs.round(4).tolist(),
        scenarios=scenarios,
        metadata={
            "model_family": "two_stage_stochastic_supply_chain_network_design",
            "demand_volatility": demand_volatility,
            "transport_volatility": transport_volatility,
            "service_level": service_level,
            "facility_coordinates": facility_xy.round(4).tolist(),
            "customer_coordinates": customer_xy.round(4).tolist(),
        },
    )


def save_instance(instance: SupplyChainInstance, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance.to_dict(), indent=2, ensure_ascii=False))
    return path
