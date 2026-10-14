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

from ada.agents.analyst import build_analysis, derive_scores_from_evidence
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
    """Create the researcher node. Owns only `research_artifact`.

    If `validation_errors` are pending, this is a repair round: the node
    regenerates its own artifact (replacing the invalid previous version)
    and clears the pending errors from state.
    """

    def researcher_node(state: dict) -> dict:
        pending_errors = state.get("validation_errors") or []
        artifact = retrieve_evidence(
            query=state.get("user_request", ""),
            encoder=encoder,
            documents=documents,
        )
        summary = f"{len(artifact.evidence)} evidencias recuperadas."
        if pending_errors:
            summary = f"Reintento tras validación: {summary}"
        return {
            "research_artifact": artifact.model_dump(),
            "validation_errors": [],
            # Invalidación de validación previa: el artefacto cambió, así que
            # una validación anterior aprobada ya no autoriza la síntesis.
            "validation_result": None,
            "contributions": [
                *state.get("contributions", []),
                {
                    "agent": "researcher",
                    "artifact_type": "research",
                    "summary": summary,
                    "step": state.get("steps", 0),
                    "attempt": state.get("retries", 0),
                    "errors": pending_errors,
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


def analyst_node(state: dict) -> dict:
    """Analyst node. Owns only `analysis_artifact`.

    If `validation_errors` are pending, this is a repair round: the node
    regenerates its own artifact (replacing the invalid previous version)
    and clears the pending errors from state.
    """
    pending_errors = state.get("validation_errors") or []
    criteria = [Criterion.model_validate(item) for item in state.get("criteria", [])]
    if not criteria:
        criteria = _default_criteria()

    alternatives = state.get("system_profile", {}).get("alternatives")
    if not alternatives:
        alternatives = _default_alternatives()

    research = state.get("research_artifact") or {}
    proposed = derive_scores_from_evidence(
        user_request=state.get("user_request", ""),
        system_profile=state.get("system_profile", {}) or {},
        criteria=criteria,
        alternatives=alternatives,
        research_artifact=research,
    )

    artifact = build_analysis(criteria, alternatives, proposed)

    summary = f"Ranking: {' > '.join(artifact.ranking)}."
    if pending_errors:
        summary = f"Reintento tras validación: {summary}"

    return {
        "analysis_artifact": artifact.model_dump(),
        "validation_errors": [],
        # Invalidación de validación previa: el artefacto cambió, así que
        # una validación anterior aprobada ya no autoriza la síntesis.
        "validation_result": None,
        "contributions": [
            *state.get("contributions", []),
            {
                "agent": "analyst",
                "artifact_type": "analysis",
                "summary": summary,
                "step": state.get("steps", 0),
                "attempt": state.get("retries", 0),
                "errors": pending_errors,
            },
        ],
    }


def validator_node(state: dict) -> dict:
    """Deterministic validator node. Owns only `validation_result`.

    Validations are deterministic: schema contracts, evidence references,
    scoring coherence. Errors are classified by the specialist responsible
    for fixing them (researcher or analyst).
    """
    errors: list[str] = []
    warnings: list[str] = []
    research_errors: list[str] = []
    analysis_errors: list[str] = []

    research = state.get("research_artifact")
    analysis = state.get("analysis_artifact")

    if not research:
        error = "Missing research artifact; cannot validate without evidence."
        errors.append(error)
        research_errors.append(error)
    if not analysis:
        error = "Missing analysis artifact; cannot validate without comparison."
        errors.append(error)
        analysis_errors.append(error)

    criteria = [Criterion.model_validate(item) for item in state.get("criteria", [])]
    if not criteria:
        criteria = _default_criteria()

    if research and analysis:
        # Validación por separado: cada error de esquema queda asociado al
        # artefacto que falló, no arbitrariamente al Researcher.
        research_artifact = None
        analysis_artifact = None
        try:
            research_artifact = ResearchArtifact.model_validate(research)
        except ValidationError as error:
            error_message = f"Research artifact contract violation: {error}"
            errors.append(error_message)
            research_errors.append(error_message)

        try:
            analysis_artifact = AnalysisArtifact.model_validate(analysis)
        except ValidationError as error:
            error_message = f"Analysis artifact contract violation: {error}"
            errors.append(error_message)
            analysis_errors.append(error_message)

        if research_artifact and analysis_artifact:
            evidence_ids = {item.document_id for item in research_artifact.evidence}
            cited_ids = {
                evidence_id
                for alternative in analysis_artifact.alternatives
                for score in alternative.scores
                for evidence_id in score.evidence_ids
            }
            missing = sorted(cited_ids - evidence_ids)
            if missing:
                error = f"Analysis cites unknown evidence IDs: {', '.join(missing)}."
                errors.append(error)
                analysis_errors.append(error)

            for alternative in analysis_artifact.alternatives:
                try:
                    recalc = calculate_weighted_score(alternative.scores, criteria)
                except ValueError as error:
                    message = (
                        f"Scoring error for '{alternative.alternative_id}': {error}"
                    )
                    errors.append(message)
                    analysis_errors.append(message)
                else:
                    if abs(alternative.weighted_score - recalc) > 1e-6:
                        error = (
                            f"Weighted score mismatch for '{alternative.alternative_id}': "
                            f"reported {alternative.weighted_score}, calculated {recalc}."
                        )
                        errors.append(error)
                        analysis_errors.append(error)

            if not research_artifact.evidence:
                # Evidencia obligatoria: sin evidencia trazable no hay informe
                # válido. Error bloqueante en la lista principal, no un warning.
                error = (
                    "Research artifact contains no evidence; retrieve evidence "
                    "before analysis."
                )
                errors.append(error)
                research_errors.append(error)
                warnings.append(
                    "Analysis relies on assumptions: no evidence was retrieved."
                )

    is_valid = not errors
    settings = get_settings()
    retries = state.get("retries", 0)
    fixable = not is_valid and retries < settings.max_retries and bool(
        research_errors or analysis_errors
    )
    result = ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        requires_human_review=(
            not is_valid and (
                state.get("retries", 0) + 1 >= settings.max_retries
                if fixable
                else state.get("retries", 0) >= settings.max_retries
            )
        ),
        research_errors=research_errors,
        analysis_errors=analysis_errors,
    )

    update: dict = {
        "validation_result": result.model_dump(),
        "contributions": [
            *state.get("contributions", []),
            {
                "agent": "validator",
                "artifact_type": "validation",
                "summary": "Valid" if is_valid else f"{len(errors)} errores de validación.",
                "step": state.get("steps", 0),
                "decision": "valid" if is_valid else "invalid",
                "reason": (
                    "Artefactos consistentes." if is_valid
                    else "Errores de validación detectados."
                ),
                "attempt": retries,
                "errors": errors,
            },
        ],
    }

    if not is_valid and fixable:
        # Incremento explícito y determinístico del contador de reintentos.
        # Los errores específicos se entregan al especialista responsable.
        repair_errors = list(research_errors or analysis_errors)
        result_dict = result.model_dump()
        result_dict["pending_repair_agent"] = (
            "researcher" if research_errors else "analyst"
        )
        update["validation_result"] = result_dict
        update["retries"] = retries + 1
        update["validation_errors"] = repair_errors

    return update


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
    """Conditional edge with deterministic synthesis protection.

    The supervisor (LLM) proposes a route, but this edge enforces the
    deterministic rule: `synthesizer` may only run with an explicit,
    current validation whose `is_valid` is True. If the supervisor picks
    `synthesizer` (or `END`) without an approved, current validation, the
    workflow is routed to the validator instead of the requested node.
    """
    proposed = state["next_agent"]
    validation = state.get("validation_result") or {}
    approved = bool(validation.get("is_valid"))

    if proposed == "synthesizer" and not approved:
        # Protección determinística: no se llama al Synthesizer sin
        # validación vigente aprobada; se completa la validación primero.
        return "validator"

    if proposed == "END":
        report = state.get("final_report") or {}
        if report.get("status") == "complete":
            return END
        # END sin informe exitoso vigente no se presenta como éxito.
        return "validator" if state.get("research_artifact") else "limited_finish"

    return proposed


def route_after_validator(state: dict) -> str:
    """Deterministic conditional edge after the validator.

    - All required artifacts valid -> synthesizer.
    - Fixable errors with retries remaining -> the specialist responsible
      for the affected artifact (it receives the specific errors in state).
    - Retries exhausted or unclassified errors -> limited_finish (safe
      output; human review pending). Never synthesizes an invalid result.
    """
    validation = state.get("validation_result") or {}
    if validation.get("is_valid"):
        return "synthesizer"

    settings = get_settings()
    if state.get("retries", 0) >= settings.max_retries:
        return "limited_finish"

    if validation.get("research_errors"):
        return "researcher"
    if validation.get("analysis_errors"):
        return "analyst"

    return "limited_finish"


def make_validator_router() -> Callable[[dict], str]:
    """Return the conditional-edge function used after the validator."""
    return route_after_validator


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

    # Workers return to the supervisor for the next routing decision,
    # except the validator: its outcome decides deterministically.
    for worker in ("researcher", "analyst"):
        graph.add_edge(worker, "supervisor")

    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "synthesizer": "synthesizer",
            "researcher": "researcher",
            "analyst": "analyst",
            "limited_finish": "limited_finish",
        },
    )

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
