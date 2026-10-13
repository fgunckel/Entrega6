"""ADA multi-agent StateGraph: supervisor-routed hierarchical workflow.

Topology:
    START -> supervisor -> (researcher | analyst | validator | synthesizer
    | limited_finish | END) -> supervisor -> ...

Each worker writes only its own state fields. The supervisor routes using
structured LLM output with deterministic execution limits.
"""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from ada.agents.analyst import build_analysis
from ada.agents.researcher import retrieve_evidence
from ada.agents.supervisor import route_next_agent
from ada.config import get_settings
from ada.schemas import (
    AnalysisArtifact,
    Criterion,
    FinalReport,
    ResearchArtifact,
    RoutingDecision,
    ValidationResult,
)
from ada.state import AgentState
from ada.tools.scoring import calculate_weighted_score

Worker = Callable[[dict], dict]


def _contribution(agent: str, artifact_type: str, summary: str, step: int) -> dict:
    """Build a contribution record for the audit trail."""
    return {
        "agent": agent,
        "artifact_type": agent,
        "summary": summary,
        "step": step,
    }


def make_supervisor_node(chat_model) -> Worker:
    """Create the supervisor node backed by a chat model (real or fake)."""

    def supervisor_node(state: dict) -> dict:
        decision: RoutingDecision = route_next_agent(state, chat_model)
        return {
            "next_agent": decision.next_agent,
            "routing_reason": decision.reason,
            "task_completed": decision.task_completed,
            "steps": state.get("steps", 0) + 1,
        }

    return supervisor_node


def make_researcher_node(encoder=None, documents=None) -> Worker:
    """Create the researcher node. Owns only `research_artifact`."""

    def researcher_node(state: dict) -> dict:

        artifact = retrieve_evidence(
            query=state.get("user_request", ""),
            encoder=encoder,
            documents=documents,
        )
        return {
            "research_artifact": artifact.model_dump(),
            "contributions": [
                *state.get("contributions", []),
                {
                    "agent": "researcher",
                    "artifact_type": "research",
                    "summary": f"{len(artifact.evidence)} evidencias recuperadas.",
                    "step": state.get("steps", 0),
                },
            ],
        }

    return researcher_node


def _default_criteria() -> list[Criterion]:
    """Return the default weighted criteria for architecture decisions."""
    from ada.agents.analyst import DEFAULT_CRITERIA

    return [Criterion.model_validate(item) for item in DEFAULT_CRITERIA]


def _default_alternatives() -> list[dict]:
    """Return the default alternatives under comparison."""
    from ada.agents.analyst import DEFAULT_ALTERNATIVES

    return [dict(item) for item in DEFAULT_ALTERNATIVES]


def _default_scores() -> dict[str, list[dict]]:
    """Return neutral baseline scores per alternative (deterministic)."""
    return {
        "monolith": [
            {
                "criterion_id": "dev-speed",
                "weight": 0.25,
                "score": 4,
                "rationale": "Base sólida para iterar rápido.",
            },
            {
                "criterion_id": "operational-cost",
                "weight": 0.25,
                "score": 5,
                "rationale": "Un solo despliegue.",
            },
            {
                "criterion_id": "independent-deploys",
                "weight": 0.15,
                "score": 1,
                "rationale": "Despliegue unitario acoplado.",
            },
            {
                "criterion_id": "scalability",
                "weight": 0.15,
                "score": 2,
                "rationale": "Escalado global únicamente.",
            },
            {
                "criterion_id": "team-alignment",
                "weight": 0.20,
                "score": 4,
                "rationale": "Simple para equipos pequeños.",
            },
        ],
        "modular-monolith": [
            {
                "criterion_id": "dev-speed",
                "weight": 0.25,
                "score": 4,
                "rationale": "Módulos con límites claros.",
            },
            {
                "criterion_id": "operational-cost",
                "weight": 0.25,
                "score": 4,
                "rationale": "Despliegue unitario simple.",
            },
            {
                "criterion_id": "independent-deploys",
                "weight": 0.15,
                "score": 2,
                "rationale": "Despliegue acoplado pero módulos aislables.",
            },
            {
                "criterion_id": "scalability",
                "weight": 0.15,
                "score": 3,
                "rationale": "Escalado global con módulos extraíbles.",
            },
            {
                "criterion_id": "team-alignment",
                "weight": 0.20,
                "score": 4,
                "rationale": "Buen ajuste para equipos pequeños/medianos.",
            },
        ],
        "microservices": [
            {
                "criterion_id": "dev-speed",
                "weight": 0.25,
                "score": 2,
                "rationale": "Mayor sobrecarga de coordinación.",
            },
            {
                "criterion_id": "operational-cost",
                "weight": 0.25,
                "score": 2,
                "rationale": "Requiere plataforma distribuida.",
            },
            {
                "criterion_id": "independent-deploys",
                "weight": 0.15,
                "score": 5,
                "rationale": "Despliegue independiente por servicio.",
            },
            {
                "criterion_id": "scalability",
                "weight": 0.15,
                "score": 5,
                "rationale": "Escalado selectivo por servicio.",
            },
            {
                "criterion_id": "team-alignment",
                "weight": 0.20,
                "score": 3,
                "rationale": "Adecuado solo con muchos equipos.",
            },
        ],
    }


def analyst_node(state: dict) -> dict:
    """Analyst node. Owns only `analysis_artifact`."""
    criteria = [Criterion.model_validate(item) for item in state.get("criteria", [])]
    if not criteria:
        criteria = _default_criteria()

    alternatives = state.get("system_profile", {}).get("alternatives")
    if not alternatives:
        alternatives = _default_alternatives()

    proposed = _default_scores()

    artifact = build_analysis(criteria, alternatives, proposed)

    return {
        "analysis_artifact": artifact.model_dump(),
        "contributions": [
            *state.get("contributions", []),
            {
                "agent": "analyst",
                "artifact_type": "analysis",
                "summary": f"Ranking: {' > '.join(artifact.ranking)}.",
                "step": state.get("steps", 0),
            },
        ],
    }


def validator_node(state: dict) -> dict:
    """Deterministic validator node. Owns only `validation_result`."""
    errors: list[str] = []
    warnings: list[str] = []

    research = state.get("research_artifact")
    analysis = state.get("analysis_artifact")

    if not research:
        errors.append("Missing research artifact; cannot validate without evidence.")
    if not analysis:
        errors.append("Missing analysis artifact; cannot validate without comparison.")

    missing_evidence_ids: list[str] = []

    criteria = [Criterion.model_validate(item) for item in state.get("criteria", [])]
    if not criteria:
        criteria = _default_criteria()

    if research and analysis:
        try:
            research_artifact = ResearchArtifact.model_validate(research)
            analysis_artifact = AnalysisArtifact.model_validate(analysis)
        except ValidationError as error:
            errors.append(f"Artifact contract violation: {error}")
        else:
            evidence_ids = {item.document_id for item in research_artifact.evidence}
            cited_ids = {
                evidence_id
                for alternative in analysis_artifact.alternatives
                for score in alternative.scores
                for evidence_id in score.evidence_ids
            }
            missing = sorted(cited_ids - evidence_ids)
            if missing:
                errors.append(
                    f"Analysis cites unknown evidence IDs: {', '.join(missing)}."
                )

            for alternative in analysis_artifact.alternatives:
                try:
                    recalc = calculate_weighted_score(alternative.scores, criteria)
                except ValueError as error:
                    errors.append(
                        f"Scoring error for '{alternative.alternative_id}': {error}"
                    )
                else:
                    if abs(alternative.weighted_score - recalc) > 1e-6:
                        errors.append(
                            f"Weighted score mismatch for '{alternative.alternative_id}': "
                            f"reported {alternative.weighted_score}, calculated {recalc}."
                        )

            if not research_artifact.evidence:
                warnings.append(
                    "Research artifact contains no evidence; "
                    "analysis relies on assumptions."
                )

    is_valid = not errors
    settings = get_settings()
    result = ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        missing_evidence_ids=missing_evidence_ids,
        requires_human_review=(
            not is_valid and state.get("retries", 0) >= settings.max_retries
        ),
    )

    return {
        "validation_result": result.model_dump(),
        "contributions": [
            *state.get("contributions", []),
            {
                "agent": "validator",
                "artifact_type": "validation",
                "summary": "Valid" if is_valid else f"{len(errors)} errores de validación.",
                "step": state.get("steps", 0),
            },
        ],
    }


def synthesizer_node(state: dict) -> dict:
    """Synthesizer node. Owns only `final_report`."""
    validation = state.get("validation_result") or {}
    approved = bool(validation.get("is_valid"))

    research = state.get("research_artifact") or {}
    analysis = state.get("analysis_artifact") or {}

    if not approved:
        return {
            "final_report": FinalReport(
                status="needs_human_review",
                executive_summary="Los artefactos no fueron aprobados por el validador.",
                recommendation="Revisar manualmente los errores de validación antes de decidir.",
                alternatives_considered=["N/A"],
                evidence_ids=[],
                assumptions=[],
                risks_and_limitations=[
                    *(validation.get("errors", []) or []),
                    "Recomendación no definitiva: artefactos no aprobados.",
                ],
                validation_warnings=validation.get("errors", []),
            ).model_dump(),
        }

    evidence_ids = sorted(
        {
            item.get("document_id", "")
            for item in research.get("evidence", [])
            if item.get("document_id")
        }
    )
    alternatives_considered = [
        alternative.get("name", alternative.get("alternative_id", ""))
        for alternative in analysis.get("alternatives", [])
    ]
    ranking = analysis.get("ranking", [])
    best_id = ranking[0] if ranking else ""
    best = next(
        (a for a in analysis.get("alternatives", []) if a.get("alternative_id") == best_id),
        {},
    )

    summary = (
        f"Decisión sobre: {state.get('user_request', '')}. "
        f"Se evaluaron {len(alternatives_considered)} alternativas con criterios ponderados; "
        f"evidencia recuperada del corpus local ({len(evidence_ids)} fuentes)."
    )

    recommendation = (
        f"Recomendación: '{best.get('name', best_id)}' "
        f"(puntaje ponderado {best.get('weighted_score', 0):.3f})."
    )

    return {
        "final_report": FinalReport(
            status="complete",
            executive_summary=summary,
            recommendation=recommendation,
            alternatives_considered=alternatives_considered or ["N/A"],
            evidence_ids=evidence_ids,
            assumptions=analysis.get("data_gaps", []),
            risks_and_limitations=[
                "La evidencia proviene exclusivamente del corpus local versionado.",
            ],
            validation_warnings=validation.get("warnings", []),
        ).model_dump(),
        "contributions": [
            *state.get("contributions", []),
            {
                "agent": "synthesizer",
                "artifact_type": "final_report",
                "summary": "Informe final generado.",
                "step": state.get("steps", 0),
            },
        ],
    }


def route_after_supervisor(state: dict) -> str:
    """Conditional edge: route to the agent chosen by the supervisor."""
    return state["next_agent"]


def build_graph(chat_model, encoder=None, documents=None) -> StateGraph:
    """Build the ADA StateGraph with conditional routing."""
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", make_supervisor_node(chat_model))
    graph.add_node("researcher", make_researcher_node(encoder=encoder, documents=documents))
    graph.add_node("analyst", analyst_node)
    graph.add_node("validator", validator_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("limited_finish", limited_finish_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "validator": "validator",
            "synthesizer": "synthesizer",
            "limited_finish": "limited_finish",
            "END": END,
        },
    )

    # Workers return to the supervisor for the next routing decision.
    for worker in ("researcher", "analyst", "validator"):
        graph.add_edge(worker, "supervisor")

    graph.add_edge("synthesizer", END)
    graph.add_edge("limited_finish", END)

    return graph


def limited_finish_node(state: dict) -> dict:
    """Stop the workflow with a limited result (no fabricated recommendation)."""
    return {
        "final_report": FinalReport(
            status="limited",
            executive_summary=(
                "El flujo terminó antes de completar la decisión: "
                "límite de pasos o error no reparable."
            ),
            recommendation="No se emite recomendación definitiva.",
            alternatives_considered=["N/A"],
            evidence_ids=[],
            assumptions=[],
            risks_and_limitations=[
                "Resultado limitado: el flujo no completó investigación y análisis validados.",
            ],
            validation_warnings=[],
        ).model_dump(),
        "task_completed": True,
    }


def compile_graph(chat_model, encoder=None, documents=None):
    """Build and compile the ADA graph."""
    graph = build_graph(chat_model, encoder=encoder, documents=documents)
    return graph.compile()
