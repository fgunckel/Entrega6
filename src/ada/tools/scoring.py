"""Deterministic scoring utilities for architecture alternatives."""

from ada.schemas import (
    AlternativeAnalysis,
    Criterion,
    CriterionScore,
    validate_criteria_weights,
)


def calculate_weighted_score(
    scores: list[CriterionScore],
    criteria: list[Criterion],
) -> float:
    """Calculate a normalized weighted score between 0 and 1."""
    validate_criteria_weights(criteria)

    expected_weights = {
        criterion.criterion_id: criterion.weight
        for criterion in criteria
    }
    received_ids = [score.criterion_id for score in scores]

    if len(received_ids) != len(set(received_ids)):
        raise ValueError("Duplicate criterion scores are not allowed.")

    if set(received_ids) != set(expected_weights):
        raise ValueError(
            "Scores must contain exactly the configured criteria."
        )

    total = 0.0

    for score in scores:
        expected_weight = expected_weights[score.criterion_id]

        if abs(score.weight - expected_weight) > 1e-6:
            raise ValueError(
                f"Weight mismatch for criterion '{score.criterion_id}'."
            )

        total += score.weight * (score.score / 5.0)

    return round(total, 6)


def rank_alternatives(
    alternatives: list[AlternativeAnalysis],
) -> list[str]:
    """Return alternative IDs ordered by weighted score, highest first."""
    if not alternatives:
        raise ValueError("At least one alternative is required.")

    ids = [alternative.alternative_id for alternative in alternatives]

    if len(ids) != len(set(ids)):
        raise ValueError("Alternative IDs must be unique.")

    ranked = sorted(
        alternatives,
        key=lambda alternative: (
            -alternative.weighted_score,
            alternative.alternative_id,
        ),
    )

    return [alternative.alternative_id for alternative in ranked]