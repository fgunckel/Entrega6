"""Genera notebooks/demo_delegacion.ipynb (demo del flujo de delegación)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "demo_delegacion.ipynb"
OUT.parent.mkdir(exist_ok=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


cells = []

# ── Portada ──────────────────────────────────────────────────────────────
cells.append(
    md(
        "# ADA — Demo del flujo de delegación\n\n"
        "Notebook que demuestra el **flujo de delegación** del orquestador "
        "multi-agente: una consulta del usuario entra por el **Supervisor**, "
        "que enruta dinámicamente entre especialistas hasta generar el informe "
        "final.\n\n"
        "**Flujo esperado:** `Supervisor → Researcher → Analyst → Validator → "
        "Synthesizer → END`.\n\n"
        "- El chat model se reemplaza por un **router determinístico** "
        "(`FakeChatModel`), igual que en la suite de tests: el notebook no "
        "requiere red ni claves de API.\n"
        "- El corpus técnico y el retriever híbrido (BM25 + embeddings + RRF) "
        "son los reales del proyecto.\n"
        "- Cada celda imprime los eventos reales de LangGraph "
        "(`stream_mode=\"updates\"`)."
    )
)

cells.append(
    md(
        "## 1. Preparación del entorno\n\n"
        "Agregamos `src/` al path (layout de paquete) e importamos los "
        "componentes del orquestador."
    )
)

cells.append(
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "PROJECT = Path.cwd()\n"
        "sys.path.insert(0, str(PROJECT / \"src\"))\n"
        "\n"
        "from ada.decision import build_criteria, extract_decision  # noqa: E402\n"
        "from ada.graph import compile_graph  # noqa: E402\n"
        "\n"
        "print(\"Entorno listo:\", PROJECT)"
    )
)

cells.append(
    md(
        "## 2. El router determinístico (simula al Supervisor LLM)\n\n"
        "El `FakeChatModel` decide **solo el enrutamiento**, leyendo el mismo "
        "contexto que vería el modelo real. Es el mismo enfoque de "
        "`tests/unit/test_graph.py`: reemplaza al LLM sin cambiar el grafo.\n\n"
        "Regla clave: llegar al **Synthesizer** exige que la validación vigente "
        "indique `is_valid=True`. Si la validación falla, el grafo aplica su "
        "regla determinista de reparación o finalización segura "
        "(`route_after_validator`); el mock nunca envía un resultado no "
        "validado al Synthesizer."
    )
)

cells.append(
    code(
        "class FakeChatModel:\n"
        "    \"\"\"Router determinístico: simula las decisiones del Supervisor.\"\"\"\n"
        "\n"
        "    def invoke(self, messages):\n"
        "        last = messages[-1]\n"
        "        text = str(last[1] if isinstance(last, tuple) else last)\n"
        "        if \"Research artifact present: False\" in text:\n"
        "            nxt = \"researcher\"\n"
        "        elif \"Analysis artifact present: False\" in text:\n"
        "            nxt = \"analyst\"\n"
        "        elif \"Validation result: none\" in text:\n"
        "            nxt = \"validator\"\n"
        "        elif \"is_valid=True\" in text:\n"
        "            nxt = \"synthesizer\"\n"
        "        else:\n"
        "            nxt = \"END\"\n"
        "\n"
        "        class Response:\n"
        "            content = (\n"
        "                f'{{\"next_agent\": \"{nxt}\", \"reason\": \"demo\", '\n"
        "                '\"task_completed\": false}'\n"
        "            )\n"
        "\n"
        "        return Response()"
    )
)

cells.append(
    md(
        "## 2. La consulta del usuario\n\n"
        "Escenario de decisión de arquitectura con alternativas explícitas "
        "(monolito modular vs. microservicios) y criterios de evaluación."
    )
)

cells.append(
    code(
        "USER_REQUEST = (\n"
        "    \"Decidir si una API .NET debe ser monolito modular o microservicios\"\n"
        ")\n"
        "print(\"Solicitud:\", USER_REQUEST)"
    )
)

cells.append(
    md(
        "## 3. Estado inicial\n\n"
        "La CLI fija la especificación estructurada de la decisión "
        "(`decision_spec`) y los criterios con pesos normalizados; los nodos "
        "los leen, nunca los inventan."
    )
)

cells.append(
    code(
        "decision_spec = extract_decision(USER_REQUEST).model_dump()\n"
        "requested_criteria = decision_spec[\"requested_criteria\"]\n"
        "criteria = (\n"
        "    [c.model_dump() for c in build_criteria(requested_criteria)]\n"
        "    if requested_criteria\n"
        "    else []\n"
        ")\n"
        "\n"
        "initial_state = {\n"
        "    \"user_request\": USER_REQUEST,\n"
        "    \"system_profile\": {},\n"
        "    \"criteria\": criteria,\n"
        "    \"decision_spec\": decision_spec,\n"
        "    \"next_agent\": \"researcher\",\n"
        "    \"routing_reason\": \"inicio\",\n"
        "    \"steps\": 0,\n"
        "    \"retries\": 0,\n"
        "    \"task_completed\": False,\n"
        "}\n"
        "\n"
        "print(\"Criterios:\", [c[\"criterion_id\"] for c in criteria])"
    )
)

cells.append(
    md(
        "## 3. Ejecución del grafo con eventos en vivo\n\n"
        "Usamos el streaming real de LangGraph. Para cada nodo ejecutado se "
        "muestra: el nodo ejecutado, el próximo agente seleccionado, el motivo "
        "de la decisión, el resultado de la validación y el estado del informe."
    )
)

cells.append(
    code(
        "graph = compile_graph(FakeChatModel(), encoder=None)\n"
        "\n"
        "final_state: dict = {}\n"
        "for chunk in graph.stream(initial_state, stream_mode=\"updates\"):\n"
        "    for node, update in (chunk or {}).items():\n"
        "        if not isinstance(update, dict):\n"
        "            continue\n"
        "        final_state = {**final_state, **update}\n"
        "        print(f\"\\n=== Nodo ejecutado: {node} ===\")\n"
        "        if \"next_agent\" in update:\n"
        "            print(f\"  próximo agente: {update['next_agent']}\")\n"
        "        if \"routing_reason\" in update:\n"
        "            print(f\"  motivo: {update['routing_reason']}\")\n"
        "        if \"validation_result\" in update:\n"
        "            validation = update.get(\"validation_result\")\n"
        "            if isinstance(validation, dict):\n"
        "                print(\n"
        "                    f\"  validación: is_valid={validation.get('is_valid')}, \"\n"
        "                    f\"errores={len(validation.get('errors', []))}\"\n"
        "                )\n"
        "            else:\n"
        "                print(\"  validación: (sin cambios)\")\n"
        "        if \"final_report\" in update:\n"
        "            report = update.get(\"final_report\")\n"
        "            status = report.get(\"status\") if isinstance(report, dict) else None\n"
        "            print(f\"  informe: {status or 'sin cambios'}\")"
    )
)

cells.append(
    md(
        "## 4. Resultado: informe final\n\n"
        "El informe solo se genera si el Validator aprobó los artefactos. "
        "Verificamos el estado final y la recomendación."
    )
)

cells.append(
    code(
        "report = final_state.get(\"final_report\") or {}\n"
        "print(\"status:\", report.get(\"status\"))\n"
        "print(\"recomendación:\", report.get(\"recommendation\"))\n"
        "print(\"alternativas:\", report.get(\"alternatives_considered\"))\n"
        "print(\"fuentes citadas:\", len(report.get(\"evidence_ids\", [])))"
    )
)

cells.append(
    md(
        "## 3. La traza de contribuciones\n\n"
        "El estado compartido registra qué agente contribuyó con qué "
        "información (`contributions`), evitando la pérdida de contexto."
    )
)

cells.append(
    code(
        "for entry in final_state.get(\"contributions\") or []:\n"
        "    print(\n"
        "        f\"paso {entry.get('step')}: {entry.get('agent')} — \"\n"
        "        f\"{entry.get('artifact_type')} — {entry.get('summary')}\"\n"
        "    )"
    )
)

cells.append(
    md(
        "## Resumen\n\n"
        "- El **Supervisor** decidió dinámicamente la secuencia de "
        "especialistas a partir del estado compartido.\n"
        "- **Researcher** recuperó evidencia real del corpus con el retriever "
        "híbrido; **Analyst** construyó la comparación y **Validator** "
        "verificó los contratos antes de que **Synthesizer** generara el "
        "informe.\n"
        "- La traza `contributions` deja evidencia de la delegación en cada "
        "paso.\n\n"
        "Para una demo con el LLM en vivo (requiere `GROQ_API_KEY`): "
        "`uv run python -m ada \"...\"`."
    )
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
)
print("Notebook generado:", OUT)
