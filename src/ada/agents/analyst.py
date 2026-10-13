"""Analyst agent: compare architecture alternatives with weighted criteria.

The analyst owns only the `analysis_artifact` field of shared state.
The LLM may propose per-criterion scores and rationales, but Python
validates the schema and computes all totals deterministically.
"""

from ada.schemas import (
    AlternativeAnalysis,
    AnalysisArtifact,
    Criterion,
    CriterionScore,
)
from ada.tools.scoring import calculate_weighted_score, rank_alternatives

DEFAULT_ALTERNATIVES = [
    {"alternative_id": "monolith", "name": "Monolito"},
    {"alternative_id": "modular-monolith", "name": "Monolito modular"},
    {"alternative_id": "microservices", "name": "Microservicios"},
]

DEFAULT_CRITERIA = [
    {
        "criterion_id": "dev-speed",
        "name": "Velocidad de desarrollo",
        "weight": 0.25,
        "description": "Rapidez con la que el equipo puede entregar funcionalidad.",
    },
    {
        "criterion_id": "operational-cost",
        "name": "Simplicidad operativa",
        "weight": 0.25,
        "description": "Costo de desplegar, monitorear y depurar el sistema.",
    },
    {
        "criterion_id": "independent-deploys",
        "name": "Despliegues independientes",
        "weight": 0.15,
        "description": "Capacidad de desplegar componentes sin afectar a otros.",
    },
    {
        "criterion_id": "scalability",
        "weight": 0.15,
        "name": "Escalabilidad",
        "description": "Capacidad de escalar componentes según demanda.",
    },
    {
        "criterion_id": "team-alignment",
        "name": "Alineación con el equipo",
        "weight": 0.20,
        "description": "Ajuste entre límites de arquitectura y estructura del equipo.",
    },
]


def build_analysis(
    criteria: list[Criterion],
    alternatives: list[dict],
    scores_by_alternative: dict[str, list[CriterionScore]],
) -> AnalysisArtifact:
    """Build a validated analysis from proposed scores.

    Python recalculates every weighted score with `calculate_weighted_score`
    and derives the ranking with `rank_alternatives`; the LLM never computes
    totals.
    """
    if not alternatives:
        raise ValueError("At least two alternatives are required.")

    analyses: list[AlternativeAnalysis] = []

    for alternative in alternatives:
        alternative_id = alternative["alternative_id"]
        proposed_scores = scores_by_alternative.get(alternative_id)
        if proposed_scores is None:
            raise ValueError(f"Missing proposed scores for alternative '{alternative_id}'.")

        # Accept both CriterionScore instances and plain dicts.
        validated_scores = [
            score if isinstance(score, CriterionScore) else CriterionScore.model_validate(score)
            for score in proposed_scores
        ]

        weighted_score = calculate_weighted_score(validated_scores, criteria)

        analyses.append(
            AlternativeAnalysis(
                alternative_id=alternative_id,
                name=alternative["name"],
                scores=validated_scores,
                weighted_score=weighted_score,
                advantages=alternative.get("advantages", []),
                disadvantages=alternative.get("disadvantages", []),
                risks=alternative.get("risks", []),
                assumptions=alternative.get("assumptions", []),
            )
        )

    ranking = rank_alternatives(analyses)

    best_id = ranking[0]
    best = next(a for a in analyses if a.alternative_id == best_id)
    summary = (
        f"'{best.name}' obtiene el mayor puntaje ponderado ({best.weighted_score:.3f}) "
        f"sobre {len(analyses)} alternativas evaluadas con {len(criteria)} criterios ponderados."
    )

    return AnalysisArtifact(
        alternatives=analyses,
        ranking=ranking,
        data_gaps=[],
        qualitative_summary=summary,
    )
