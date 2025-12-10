Document Scanner
===============================

Pipeline para "escanear" fotos de documentos: detecta el contorno principal, corrige perspectiva, mejora la imagen y, opcionalmente, hace OCR. Incluye utilidades para evaluar OCR y geometría.

Como ejecutar
-------------------------------
Instalar requerimientos:
```powershell
# Nota GPU: si usarás EasyOCR con GPU, instala primero la versión de torch adecuada para tu CUDA
pip install -r requirements.txt
```

Ejecutar el pipeline (ejemplos):
```powershell
# 1) Una imagen
python -m src.pipeline --input data/raw/ocr_test/ocr_test_1.jpg --output-dir data/processed

# 2) Una carpeta
python -m src.pipeline --input data/raw/ocr_test --output-dir data/processed

# 3) Toda la raíz
python -m src.pipeline --input data/raw --output-dir data/processed
```

Evaluar:
```powershell
#Para evaluar únicamente la métrica geométrica añade `--geometry-only` al comando
python .\scripts\evaluate.py --gt-root data/ground_truth --pred-root data/processed --raw-root data/raw --pred-corners-root data/processed --out-dir reports/metrics
```

Cómo se organiza el proyecto
----------------------------
- `data/raw/`: imágenes de entrada por dataset (`ocr_test`, `ocr_test_bin`, `scanner_test`).
- `data/processed/`: salidas del pipeline por imagen (`entrada_contour`, `entrada_warp`, `entrada_warp_doc`, opcional `.txt`, opcional `.corners.json`).
- `data/ground_truth/`: GT OCR (`.gt.txt`) y esquinas (`.corners.json`).
- `reports/metrics/`: métricas CSV/JSON y gráficos.
- `src/`: código fuente; `scripts/`: utilidades (`evaluate.py`).

Salida por imagen
-----------------
- `entrada_contour.<ext>`: contorno dibujado.
- `entrada_warp.<ext>`: perspectiva corregida.
- `entrada_warp_doc.<ext>`: resultado final (binarizado si `_bin`).
- `<stem>.txt`: OCR cuando aplica.
- `<stem>.corners.json`: esquinas predichas (solo para `scanner_test`).


Salida de evaluación
--------------------
- `reports/metrics/metrics_detail.csv` + `metrics_summary.json` (combinado)
- Carpetas por dataset al lado de la anterior: `reports/metrics_ocr_test`, `reports/metrics_ocr_test_bin`, `reports/metrics_scanner_test` — cada una con su `metrics_detail.csv` y `metrics_summary.json`.
- Gráficos: `corner_rmse_hist.png`, `polygon_iou_hist.png`, `area_ratio_scatter.png` se generan en `reports/metrics_scanner_test` (si hay datos de `scanner_test`).