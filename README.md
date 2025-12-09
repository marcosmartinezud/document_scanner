Document Scanner — README breve
===============================

Pipeline ligero para "escanear" fotos de documentos: detecta el contorno principal, corrige perspectiva, mejora la imagen y, opcionalmente, hace OCR. Incluye utilidades para generar ground truth de esquinas y evaluar OCR y geometría.

Instalación rápida (PowerShell)
-------------------------------
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Dev (tests/format/lint):
pip install -r requirements-dev.txt
```
Nota GPU: si usarás EasyOCR con GPU, instala primero la versión de torch adecuada para tu CUDA.

Cómo se organiza el proyecto
----------------------------
- `data/raw/`: imágenes de entrada por dataset (`ocr_test`, `ocr_test_bin`, `scanner_test`, `failed`, ...).
- `data/processed/`: salidas del pipeline por imagen (`entrada_contour`, `entrada_warp`, `entrada_warp_doc`, opcional `.txt`, opcional `.corners.json`).
- `data/ground_truth/`: GT OCR (`.gt.txt`) y esquinas (`.corners.json`).
- `reports/metrics/`: métricas CSV/JSON y gráficos.
- `src/`: código fuente; `scripts/`: utilidades (`evaluate.py`, `generate_precise_gt.py`).

Reglas del pipeline (CLI `python -m src.pipeline`)
--------------------------------------------------
- Si `--input` es archivo → procesa solo esa imagen.
- Si `--input` es carpeta → procesa recursivo esa carpeta.
- Si `--input` es `data/raw` → procesa todas las subcarpetas.
- Comportamiento por carpeta superior:
  - `ocr_test`: escaneo + OCR.
  - `scanner_test`: solo geometría (sin OCR). Guarda `.corners.json`.
  - sufijo `_bin` (`ocr_test_bin`, `scanner_test_bin`): binariza (thr=195) y luego OCR.

Comandos útiles (PowerShell)
----------------------------
```powershell
# 1) Una imagen
python -m src.pipeline --input data/raw/ocr_test/ocr_test_1.jpg --output-dir data/processed

# 2) Una carpeta
python -m src.pipeline --input data/raw/ocr_test --output-dir data/processed

# 3) Toda la raíz
python -m src.pipeline --input data/raw --output-dir data/processed

# 4) Solo geometría (sin OCR), sobrescribiendo la regla por carpeta
python -m src.pipeline --input data/raw/failed --output-dir data/processed/failed
```

Salida por imagen
-----------------
- `entrada_contour.<ext>`: contorno dibujado.
- `entrada_warp.<ext>`: perspectiva corregida.
- `entrada_warp_doc.<ext>`: resultado final (binarizado si `_bin`).
- `<stem>.txt`: OCR cuando aplica.
- `<stem>.corners.json`: esquinas predichas (se genera para `scanner_test`).

Evaluar (scripts/evaluate.py)
-----------------------------
```powershell
# OCR + geometría (datasets por defecto)
python .\scripts\evaluate.py --gt-root data/ground_truth --pred-root data/processed --raw-root data/raw --pred-corners-root data/processed --out-dir reports/metrics

# Solo geometría (scanner_test) y 3 gráficos solicitados
python .\scripts\evaluate.py --geometry-only --pred-corners-root data/processed --out-dir reports/metrics
```
Produce `metrics_detail.csv`, `metrics_summary.json` y (solo en `scanner_test`) los gráficos: `corner_rmse_hist.png`, `polygon_iou_hist.png`, `area_ratio_scatter.png`.

Generar GT de esquinas (opcional)
---------------------------------
```powershell
python .\scripts\generate_precise_gt.py --input data/raw/scanner_test --output data/ground_truth/scanner_test_precise --debug-dir data/ground_truth/scanner_test_debug
```

Pruebas rápidas
---------------
```powershell
pip install -r requirements-dev.txt
pytest -q
```

Notas rápidas
-------------
- Si faltan `.gt.txt` o `.corners.json`, las métricas correspondientes salen `null`.
- `corner_rmse` bajo e `IoU` alto indican buena alineación; `area_ratio` ayuda a detectar recortes excesivos.
- Usa `--no-plots` si no quieres gráficos, o cambia `--out-dir` para separar resultados.
