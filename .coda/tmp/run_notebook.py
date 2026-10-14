"""Ejecuta notebooks/demo_delegacion.ipynb celda por celda (validación)."""

import io
import json
import traceback
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
nb_path = ROOT / "notebooks" / "demo_delegacion.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

namespace: dict = {}
errors = 0
for index, cell in enumerate(nb["cells"], start=1):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(compile(source, f"<cell {index}>", "exec"), namespace)
        print(f"[OK] celda {index}")
        output = buffer.getvalue().strip()
        if output:
            print("     " + output.replace("\n", "\n     ")[:400])
    except Exception:
        errors += 1
        print(f"[ERROR] celda {index}")
        traceback.print_exc(limit=3)

print()
print("Errores:", errors)
