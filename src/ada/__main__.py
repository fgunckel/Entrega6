"""ADA CLI: run a demonstration architecture decision from the terminal.

Usage:
    uv run python -m ada
    uv run python -m ada --request "Decidir arquitectura para una API .NET"

Requires GROQ_API_KEY in .env for live model calls. Without a key the CLI
prints setup instructions instead of failing with a traceback.
"""

from __future__ import annotations

import argparse
import sys

from ada.config import get_settings
from ada.graph import compile_graph

DEMO_REQUEST = (
    "Decidir si una API .NET debe permanecer como monolito, evolucionar a "
    "monolito modular o migrar a microservicios. El equipo es pequeño, hay "
    "presión por entregar rápido y no se requieren despliegues independientes."
)


def _print_header(title: str) -> None:
    print()
    print(f"=== {title} ===")


def _print_sources(state: dict) -> None:
    research = state.get("research_artifact") or {}
    evidence = research.get("evidence", [])
    print(f"\nFuentes recuperadas ({len(evidence)}):")
    for item in evidence:
        uri = item.get("source_uri") or "(fuente local)"
        print(f"  - [{item.get('document_id')}] {item.get('title')}")
        print(f"    Fuente: {uri}")
        excerpt = (item.get("excerpt") or "")[:120]
        print(f"    Fragmento: {excerpt}...")


def _print_scores(analysis: dict) -> None:
    print("\nPuntuaciones ponderadas:")
    for alternative in analysis.get("alternatives", []):
        print(f"  - {alternative.get('name')}: {alternative.get('weighted_score'):.3f}")
    ranking = analysis.get("ranking", [])
    if ranking:
        print(f"  Ranking: {' > '.join(ranking)}")


def _print_validation(validation: dict) -> None:
    is_valid = validation.get("is_valid")
    print(f"\nValidación: {'aprobada' if is_valid else 'con errores'}")
    for error in validation.get("errors", []):
        print(f"  ERROR: {error}")
    for warning in validation.get("warnings", []):
        print(f"  AVISO: {warning}")


def _print_report(report: dict) -> None:
    print("\n=== INFORME FINAL ===")
    print(f"Estado: {report.get('status')}")
    print(f"Resumen: {report.get('executive_summary')}")
    print(f"Recomendación: {report.get('recommendation')}")
    print(f"Alternativas consideradas: {', '.join(report.get('alternatives_considered', []))}")
    if report.get("evidence_ids"):
        print(f"Evidencias: {', '.join(report['evidence_ids'])}")
    for risk in report.get("risks_and_limitations", []):
        print(f"  Riesgo/Limitación: {risk}" if (risk := risk) else "")


def main() -> int:
    """Run the ADA demonstration scenario."""
    parser = argparse.ArgumentParser(description="ADA - Architecture Decision Assistant")
    parser.add_argument(
        "--request",
        type=str,
        default=None,
        help="Escenario técnico a decidir (por defecto usa el escenario de demo).",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.groq_api_key or not settings.groq_api_key.get_secret_value().strip():
        print(
            "GROQ_API_KEY no está configurada.\n"
            "1. Copia .env.example a .env\n"
            "2. Añade tu clave de Groq en GROQ_API_KEY\n"
            "3. Vuelve a ejecutar: uv run python -m ada"
        )
        return 1

    from ada.config import get_chat_model

    try:
        chat_model = get_chat_model()
    except Exception as error:
        print(f"No se pudo inicializar el modelo: {error}")
        return 1

    user_request = args.request or DEMO_REQUEST

    print("=== ADA - Architecture Decision Assistant ===")
    print(f"Solicitud: {user_request}")

    graph = compile_graph(chat_model)

    initial_state = {
        "user_request": user_request,
        "system_profile": {},
        "criteria": [],
        "next_agent": "researcher",
        "routing_reason": "inicio",
        "steps": 0,
        "retries": 0,
        "task_completed": False,
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as error:
        print(f"Error durante la ejecución del flujo: {error}")
        return 1

    print("\nNodos recorridos:", [c["agent"] for c in result.get("contributions", [])])

    _print_sources(result.get("research_artifact") or {})
    _print_scores(result.get("analysis_artifact") or {})
    _print_validation(result.get("validation_result") or {})
    _print_report(result.get("final_report") or {})

    return 0


if __name__ == "__main__":
    sys.exit(main())
