"""Diagnóstico reproducible del flujo real de ADA.

Pruebas A-F con la consulta completa del SPEC de diagnóstico, usando un
chat model mock determinístico (sin red, sin Groq) y el corpus local real.
Escribe el log en artifacts/ada_diagnostic.log.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from ada.agents.analyst import CRITERIA_SIGNALS  # noqa: E402
from ada.decision import build_criteria, extract_decision  # noqa: E402
from ada.graph import compile_graph  # noqa: E402
from ada.retrieval.corpus import load_corpus  # noqa: E402

CONSULTA = (
    "Necesito decidir cómo implementar la comunicación entre microservicios "
    "en una arquitectura .NET desplegada en Azure. Compará Azure Service Bus "
    "y la comunicación HTTP síncrona, evaluando acoplamiento, resiliencia, "
    "latencia, complejidad operativa y escalabilidad. Justificá la "
    "recomendación con evidencia del corpus local e indicá los trade-offs y "
    "los casos en los que elegirías cada alternativa."
)

LOG_PATH = PROJECT / "artifacts" / "ada_diagnostic.log"
_log_lines: list[str] = []


def log(text: str = "") -> None:
    print(text)
    _log_lines.append(text)


def header() -> None:
    log("=== ADA - LOG DE DIAGNÓSTICO REPRODUCIBLE ===")
    log(f"Fecha: {datetime.now(UTC).isoformat(timespec='seconds')}")
    log(f"Consulta: {CONSULTA}")
    log("Modelo: mock determinístico (no depende de Groq ni de red)")
    log(
        "Pruebas de integración real con Groq: NO EJECUTADAS "
        "(GROQ_API_KEY no configurada en este entorno). "
        "Todas las pruebas de este log usan mocks determinísticos."
    )
    log("Corpus: data/corpus/corpus.json (8 documentos reales)")


class FakeChatModel:
    """Mock consciente del flujo: responde según la etapa."""

    def invoke(self, messages, **kwargs):
        return self._respond(messages)

    async def ainvoke(self, messages, **kwargs):
        return self._respond(messages)

    def _respond(self, messages):
        text = ""
        for message in messages:
            content = (
                message[1] if isinstance(message, tuple)
                else getattr(message, "content", "")
            )
            if isinstance(content, str):
                text += " " + content
        if "next_agent" in text or "routing" in text:
            # Supervisor: enruta según el progreso real del estado.
            # Con ambos artefactos, el Validator debe correr: el enrutado
            # posterior al validator es determinista en el grafo.
            if "Analysis artifact present: True" in text:
                return _ai(
                    '{"next_agent": "validator", "reason": "artefactos '
                    'listos para validación", "task_completed": false}'
                )
            if "Research artifact present: True" in text:
                return _ai(
                    '{"next_agent": "analyst", "reason": "evidencia lista", '
                    '"task_completed": false}'
                )
            return _ai(
                '{"next_agent": "researcher", "reason": "sin evidencia aún", '
                '"task_completed": false}'
            )
        return _ai("{}")


def _ai(content: str):
    from langchain_core.messages import AIMessage

    return AIMessage(content=content)


def run_graph(initial_state: dict) -> tuple[dict, dict]:
    """Ejecuta el grafo completo con el corpus real y devuelve trace+estado."""
    documents = load_corpus()
    graph = compile_graph(FakeChatModel(), encoder=None, documents=documents)
    trace: dict[str, dict] = {}
    final_state: dict = {}
    for chunk in graph.stream(initial_state, stream_mode="updates"):
        for node, update in (chunk or {}).items():
            if isinstance(update, dict):
                trace.setdefault(node, {}).update(update)
                final_state.update(update)
    return trace, final_state


def executed_nodes(final_state: dict) -> list[str]:
    """Secuencia de nodos ejecutados, reconstruida de contributions."""
    return [
        entry.get("agent", "?")
        for entry in (final_state.get("contributions") or [])
    ]


def test_a_extraction() -> dict:
    log("\n### A. Extracción de la decisión (extract_decision)")
    log("  Esperado : 2 alternativas (azure-service-bus, sync-http) y 5 criterios")
    spec = extract_decision(CONSULTA)
    alt_ids = [a.alternative_id for a in spec.requested_alternatives]
    log(f"  decision_question: {spec.decision_question[:60]}...")
    log(f"  alternatives_are_explicit: {spec.alternatives_are_explicit}")
    log(f"  alternativas extraídas: {alt_ids}")
    log(f"  criterios extraídos: {spec.requested_criteria}")
    ok = alt_ids == ["azure-service-bus", "sync-http"] and len(spec.requested_criteria) == 5
    log(f"  ESTADO: {'PASS' if ok else 'FAIL'}")
    return spec.model_dump()


def test_b_initial_state(spec_data: dict) -> dict:
    log("\n### B. Inicialización del estado")
    log("  Esperado : criteria con los 5 Criterion extraídos, pesos que suman 1")
    criteria = build_criteria(spec_data["requested_criteria"])
    for criterion in criteria:
        log(
            f"    criterion {criterion.criterion_id}: "
            f"weight={criterion.weight} name='{criterion.name}'"
        )
    total = sum(c.weight for c in criteria)
    log(f"  suma de pesos: {total:.6f}")
    ok = abs(total - 1.0) <= 1e-6
    log(f"  ESTADO: {'PASS' if ok else 'FAIL'}")
    return {
        "user_request": CONSULTA,
        "system_profile": {},
        "criteria": [c.model_dump() for c in criteria],
        "decision_spec": spec_data,
    }


def test_c_nodes(initial_state: dict) -> dict:
    log("\n### C. Ejecución de cada nodo (grafo completo con mock)")
    log("  Esperado : Supervisor enruta; Researcher recupera; Analyst usa criteria "
        "del estado; Validator valida; informe solo con validación vigente")
    trace, final_state = run_graph(initial_state)
    for node, update in trace.items():
        log(f"  {node}: escribe {sorted(update.keys())}")
    nodes = executed_nodes(final_state)
    log(f"  secuencia de nodos ejecutados: {nodes}")
    analysis = (trace.get("analyst") or {}).get("analysis_artifact") or {}
    alt_ids = [a["alternative_id"] for a in analysis.get("alternatives", [])]
    log(f"  Analyst: alternativas producidas {alt_ids}")
    validation = (final_state.get("validation_result") or {})
    log(f"  Validator is_valid: {validation.get('is_valid', 'no ejecutado (null)')}, "
        f"errores: {validation.get('errors', [])}")
    report = (final_state.get("final_report") or {})
    log(f"  final_report.status: {report.get('status', 'NO GENERADO')}")
    ok = (
        bool(alt_ids)
        and "researcher" in nodes
        and "analyst" in nodes
        and "validator" in nodes
    )
    log(f"  ESTADO: {'PASS' if ok else 'FAIL'}")
    return {"trace": trace, "final_state": final_state}


def test_d_scoring(trace: dict) -> None:
    log("\n### D. Scoring (valores por criterio, pesos, puntuaciones y ranking)")
    log("  Esperado : puntajes que reflejan las señales del caso recibido; "
        "sin diferencias inventadas cuando los datos son idénticos; "
        "ranking determinístico")
    analysis = (trace.get("analyst") or {}).get("analysis_artifact") or {}
    alternatives = analysis.get("alternatives") or []
    for alternative in alternatives:
        test_detail_alternative(alternative)
    ranking = analysis.get("ranking")
    log(f"  ranking: {ranking}")
    has_signal = any(
        any(s["score"] != 3 for s in a["scores"]) for a in alternatives
    )
    log(f"  ESTADO: {'PASS' if alternatives and has_signal else 'FAIL'}")


def test_e_evidence(trace: dict) -> None:
    log("\n### E. Evidencia recuperada y respaldo por criterio")
    log("  Esperado : IDs y títulos reales del corpus; criterios sin respaldo "
        "explícitos; sin atribuir contenido no presente")
    research = (trace.get("researcher") or {}).get("research_artifact") or {}
    evidence = research.get("evidence", []) or []
    log(f"  evidencia recuperada: {len(evidence)} ítems")
    for item in evidence:
        log(f"    [{item.get('document_id')}] {item.get('title')}")
    requested = extract_decision(CONSULTA).requested_criteria
    backed = sorted(lexically_backed_criteria())
    without = sorted(set(requested) - set(backed))
    log(f"  criterios con respaldo léxico: {backed}")
    log(f"  criterios SIN respaldo suficiente: {without}")
    ok = len(evidence) > 0 and bool(without)
    log(f"  ESTADO: {'PASS' if ok else 'FAIL'}")


def test_f_validation(trace: dict, final_state: dict) -> None:
    log("\n### F. Validación y finalización (Validator + Synthesizer)")
    log("  Esperado : Validator valida los artefactos; el informe solo se "
        "sintetiza con validación vigente; sin recomendación si no hay "
        "evidencia suficiente")
    validation = (final_state.get("validation_result") or {})
    is_valid = validation.get("is_valid")
    log(f"  is_valid: {is_valid if is_valid is not None else 'no ejecutado (null)'}")
    log(f"  errores: {validation.get('errors', [])}")
    log(f"  warnings: {validation.get('warnings', [])}")
    report = (final_state.get("final_report") or {})
    log(f"  final_report.status: {report.get('status')}")
    recommendation = report.get("recommendation")
    log(f"  recommendation: {recommendation or 'SIN RECOMENDACIÓN DEFINITIVA'}")
    # Coherencia: si el informe existe, su status debe ser coherente con la
    # validación vigente (nunca 'complete' con validación inválida o nula).
    if report:
        status = report.get("status")
        no_definitive = not recommendation or "No se emite" in str(recommendation)
        ok = (
            bool(status)
            and (status == "limited" or validation.get("is_valid") is True)
            and (status != "limited" or no_definitive)
        )
    else:
        ok = False
    log(f"  ESTADO: {'PASS' if ok else 'FAIL'}")


def test_detail(text: str) -> None:
    log(f"  {text}")


def test_detail_alternative(alternative: dict) -> None:
    alt_id = alternative["alternative_id"]
    test_detail(f"alternativa {alt_id}: weighted_score={alternative['weighted_score']}")
    for score in alternative["scores"]:
        flat = flat_score(score)
        criterion = flat_score_criterion(score)
        weight = score.get("weight")
        if flat != 3:
            test_detail(
                f"    {criterion}: score={flat} (weight={weight}) -> señal detectada"
            )
        else:
            test_detail(
                f"    {criterion}: score=3 (weight={weight}) -> sin señales "
                "diferenciadoras en el texto: neutro honesto"
            )


def flat_score(score: dict) -> int:
    return int(score["score"])


def flat_score_criterion(score) -> str:
    return str(score["criterion_id"])


def lexically_backed_criteria() -> set[str]:
    """Criterios cuyo nombre aparece en el contenido de la evidencia local."""
    from ada.retrieval.corpus import load_corpus

    text = " ".join(doc.content.lower() for doc in load_corpus())
    backed = set()
    for criterion_id, signals in CRITERIA_SIGNALS.items():
        for signal in signal_terms(signals):
            if signal.lower() in text:
                backed.add(criterion_id)
                break
    return backed


def signal_terms(signals: dict) -> list[str]:
    return list(signals.get("positive", [])) + list(signals.get("negative", []))


def main() -> int:
    header()
    spec_data = test_a_extraction()
    initial_state = test_b_initial_state(spec_data)
    result = test_c_nodes(initial_state)
    test_d_scoring(result["trace"])
    test_e_evidence(result["trace"])
    test_f_validation(result["trace"], result["final_state"])
    log("\n=== FIN DEL LOG DE DIAGNÓSTICO ===")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
