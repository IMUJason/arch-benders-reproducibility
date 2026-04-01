from __future__ import annotations

import math

from .models import BendersCut


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left == right else 0.0
    return numerator / (left_norm * right_norm)


def _is_redundant(
    candidate: BendersCut,
    existing: BendersCut,
    similarity_threshold: float,
) -> bool:
    if candidate.scenario_id != existing.scenario_id:
        return False
    if candidate.signature(precision=10) == existing.signature(precision=10):
        return True
    augmented_left = [candidate.intercept, *candidate.coefficients]
    augmented_right = [existing.intercept, *existing.coefficients]
    return _cosine_similarity(augmented_left, augmented_right) >= similarity_threshold


def filter_new_cuts(
    existing_cuts: list[BendersCut],
    new_cuts: list[BendersCut],
    *,
    similarity_threshold: float,
    efficacy_threshold: float,
    max_retained_new_cuts: int,
) -> list[BendersCut]:
    retained: list[BendersCut] = []
    ordered = sorted(new_cuts, key=lambda cut: cut.efficacy, reverse=True)

    for candidate in ordered:
        if candidate.efficacy < efficacy_threshold:
            continue
        if any(
            _is_redundant(candidate, cut, similarity_threshold)
            for cut in [*existing_cuts, *retained]
        ):
            continue
        retained.append(candidate)
        if len(retained) >= max_retained_new_cuts:
            break

    return retained
