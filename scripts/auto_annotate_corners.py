"""Deprecated wrapper.

`auto_annotate_corners.py` ha sido reemplazado: el único método soportado para generar
ground-truth de esquinas es ahora `scripts/generate_precise_gt.py`.

Este wrapper invoca el script nuevo con los mismos parámetros. Mantengo el wrapper
por compatibilidad de línea de comandos, pero internamente delega en `generate_precise_gt.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def main(argv: list[str] | None = None) -> int:
    # Reconstruye la llamada para delegar al nuevo generador preciso
    argv = argv or sys.argv[1:]
    # Local path to the new script
    script = Path(__file__).parent / "generate_precise_gt.py"
    if not script.exists():
        print("Error: generate_precise_gt.py no encontrado. Asegúrate de que el repositorio está completo.")
        return 2

    cmd = [sys.executable, str(script)] + list(argv)
    print("[INFO] auto_annotate_corners está obsoleto — ejecutando generate_precise_gt.py")
    print("[DEBUG] comando:", " ".join(cmd))
    try:
        res = subprocess.run(cmd)
        return res.returncode
    except Exception as e:
        print("Error al ejecutar generate_precise_gt:", e)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
