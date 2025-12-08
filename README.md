# Document Scanner — Instrucciones completas

Este repositorio contiene un pipeline para detectar el contorno principal de una foto de documento, corregir la perspectiva, mejorar la imagen y (opcionalmente) extraer texto mediante OCR. Además incluye utilidades para auto-anotar esquinas y evaluar la calidad geométrica de la detección.

Este README describe paso a paso cómo preparar el entorno, ejecutar el pipeline, generar ground-truth de esquinas, y lanzar la evaluación (tanto OCR como evaluación geométrica por separado).

## Requisitos

- Python 3.10+
- Las dependencias principales están en `requirements.txt`. Para desarrollo hay herramientas adicionales en `requirements-dev.txt`.
- En Windows PowerShell se usan los ejemplos de comandos que se muestran más abajo.

### 1) Crear y activar entorno virtual (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Instalar dependencias
```powershell
pip install -r requirements.txt
# Para desarrollo (tests/format/lint):
pip install -r requirements-dev.txt
```

Nota sobre GPU: si piensas usar EasyOCR con GPU instala una versión de `torch` compatible con tu CUDA antes de instalar EasyOCR. Consulta la documentación de PyTorch para la línea de instalación correcta.

## Estructura breve

- `data/raw/` — imágenes de entrada organizadas por dataset (ej.: `ocr_test`, `ocr_test_bin`, `scanner_test`).
- `data/ground_truth/` — ground-truth: archivos `.gt.txt` (OCR) y `.corners.json` (esquinas) por dataset.
- `data/processed/` — salida del pipeline: para cada imagen se crea `data/processed/<dataset>/<stem>/` con imágenes y opcionalmente `<stem>.txt` y `<stem>.corners.json`.
- `reports/metrics/` — salidas de la evaluación: `metrics_detail.csv`, `metrics_summary.json`, gráficos PNG.
- `src/` — código fuente.
 - `scripts/` — utilidades: `generate_precise_gt.py`, `evaluate.py`.

## Ejecutar el pipeline

Descripción: `src.pipeline` procesa imágenes, detecta contornos, aplica homografía y mejora la imagen. Dependiendo del dataset puede ejecutar OCR o binarizar.

Comandos (PowerShell):
```powershell
# Procesar una sola imagen
python -m src.pipeline --input data/raw/ocr_test/ocr_test_1.jpg --output-dir data/processed

# Procesar una carpeta (todas las imágenes dentro)
python -m src.pipeline --input data/raw/ocr_test --output-dir data/processed

# Procesar toda la raíz (todas las subcarpetas)
python -m src.pipeline --input data/raw --output-dir data/processed
```

Salida por imagen (en `data/processed/<dataset>/<stem>/`):
- `entrada_contour.<ext>` — imagen con contorno dibujado.
- `entrada_warp.<ext>` — imagen transformada (perspectiva corregida).
- `entrada_warp_doc.<ext>` — imagen final mejorada (o binarizada si es `_bin`).
- `<stem>.txt` — OCR (solo para datasets con OCR habilitado).
- `<stem>.corners.json` — esquinas predichas (pipeline guarda esquinas predichas para `scanner_test`).

## Auto-anotar esquinas (generar ground-truth)

Si no tienes ground-truth de esquinas, puedes generarlo automáticamente con el generador más preciso `scripts/generate_precise_gt.py`.

Ejemplo (PowerShell):
```powershell
python .\scripts\generate_precise_gt.py --input data/raw/scanner_test --output data/ground_truth/scanner_test_precise --debug-dir data/ground_truth/scanner_test_debug
```

- `--debug-dir` (opcional): guarda overlays JPEG con el polígono dibujado para inspección manual.
- `--overwrite`: sobrescribe JSON existentes.

## Evaluación (`scripts/evaluate.py`)

La herramienta de evaluación genera métricas por imagen y por dataset, además de gráficos en `reports/metrics/`.

Flags importantes:
- `--gt-root`: carpeta raíz de ground-truth (por defecto `data/ground_truth`).
- `--pred-root`: carpeta raíz de predicciones (por defecto `data/processed`).
- `--raw-root`: carpeta raíz con imágenes originales (por defecto `data/raw`).
- `--pred-corners-root`: carpeta donde buscar esquinas predichas (normalmente `data/processed`).
- `--datasets`: lista de datasets a evaluar (por ejemplo `ocr_test ocr_test_bin scanner_test`).
- `--skip-ocr`: no calcula CER/WER (útil si solo quieres métricas geométricas).
- `--geometry-only`: atajo — equivale a `--datasets scanner_test --skip-ocr`.
- `--out-dir`: carpeta de salida para reportes (por defecto `reports/metrics`).

Ejemplos (PowerShell):
```powershell
# Evaluación completa (OCR + geometría cuando aplique)
python .\scripts\evaluate.py --gt-root data/ground_truth --pred-root data/processed --raw-root data/raw --pred-corners-root data/processed --datasets ocr_test ocr_test_bin scanner_test --out-dir reports/metrics

# Solo métricas de geometría (scanner_test) y gráficos
python .\scripts\evaluate.py --geometry-only --pred-corners-root data/processed --out-dir reports/metrics

# Evaluar datasets específicos sin OCR
python .\scripts\evaluate.py --datasets scanner_test --skip-ocr --pred-corners-root data/processed
```

Qué produce la evaluación
- `reports/metrics/metrics_detail.csv`: fila por cada imagen con métricas separadas: `cer`, `wer`, `area_ratio_gt`, `area_ratio_pred`, `area_ratio_diff`, `aspect_ratio_gt`, `aspect_ratio_pred`, `aspect_ratio_diff`, `corner_rmse`, `polygon_iou`, etc.
- `reports/metrics/metrics_summary.json`: resumen por dataset y estadísticas agregadas.
- PNGs con histogramas y scatter plots en `reports/metrics/` (CER/WER por dataset, distribución de IoU, RMSE, área, aspecto, y barras de promedios).

Interpretación rápida
- `corner_rmse` (px): error RMS entre pares de esquinas (GT vs Pred). Valores bajos mejor.
- `polygon_iou` (0..1): IoU de polígono en píxeles; 1.0 indica coincidencia perfecta.
- `area_ratio`: área del polígono / área de la imagen (útil para detectar recortes muy pequeños o incorrectos).

Por qué puede salir todo "perfecto"
- Si tus predicciones en `data/processed/<dataset>/*/*.corners.json` fueron copiadas desde las mismas GT (por ejemplo usando `scripts/generate_precise_gt.py`), la evaluación mostrará RMSE=0 e IoU=1. Esto indica que la evaluación funciona — pero no hay discrepancias reales que medir.

Generar casos con discrepancias (para probar sensibilidad)
- Puedes ejecutar el pipeline sobre un conjunto distinto de imágenes (no las que usaste para auto-annotate) para obtener predicciones que probablemente difieran de las GT.
- Alternativamente, crear versiones perturbadas de las esquinas GT y evaluar contra ellas (scripts de prueba pueden generarse si lo deseas).

## Pruebas y herramientas de desarrollo

- Ejecutar tests (requiere `pytest`):
```powershell
pip install -r requirements-dev.txt
pytest -q
```
- Formateo y linting:
```powershell
black src tests
ruff check src tests
```

## Consejos y resolución de problemas

- Si `metrics_summary.json` muestra `null` en CER/WER significa que no se encontraron parejas GT vs pred (GT o pred faltante). Revisa `data/ground_truth/<dataset>` y `data/processed/<dataset>`.
- Si las métricas geométricas son triviales (RMSE=0, IoU=1) mira si las esquinas predichas son exactamente las mismas que las GT.
- Para evaluar solo geometría usa `--geometry-only` para evitar resultados nulos por falta de `.gt.txt`.

¿Quieres que añada ejemplos de scripts para sintetizar errores (perturbaciones) o que genere overlays por imagen para inspección manual? Puedo añadir utilidades automáticas para ambos flujos.

---

Fin del README.
# Document Scanner — Instrucciones completas

Este repositorio contiene un pipeline para detectar el contorno principal de una foto de documento, corregir la perspectiva, mejorar la imagen y (opcionalmente) extraer texto mediante OCR. Además incluye utilidades para auto-anotar esquinas y evaluar la calidad geométrica de la detección.

Este README describe paso a paso cómo preparar el entorno, ejecutar el pipeline, generar ground-truth de esquinas, y lanzar la evaluación (tanto OCR como evaluación geométrica por separado).

## Requisitos

- Python 3.10+
- Las dependencias principales están en `requirements.txt`. Para desarrollo hay herramientas adicionales en `requirements-dev.txt`.
- En Windows PowerShell se usan los ejemplos de comandos que se muestran más abajo.

### 1) Crear y activar entorno virtual (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Instalar dependencias
```powershell
pip install -r requirements.txt
# Para desarrollo (tests/format/lint):
pip install -r requirements-dev.txt
```

Nota sobre GPU: si piensas usar EasyOCR con GPU instala una versión de `torch` compatible con tu CUDA antes de instalar EasyOCR. Consulta la documentación de PyTorch para la línea de instalación correcta.

## Estructura breve

- `data/raw/` — imágenes de entrada organizadas por dataset (ej.: `ocr_test`, `ocr_test_bin`, `scanner_test`).
- `data/ground_truth/` — ground-truth: archivos `.gt.txt` (OCR) y `.corners.json` (esquinas) por dataset.
- `data/processed/` — salida del pipeline: para cada imagen se crea `data/processed/<dataset>/<stem>/` con imágenes y opcionalmente `<stem>.txt` y `<stem>.corners.json`.
- `reports/metrics/` — salidas de la evaluación: `metrics_detail.csv`, `metrics_summary.json`, gráficos PNG.
- `src/` — código fuente.
- `scripts/` — utilidades: `generate_precise_gt.py`, `evaluate.py`.

## Ejecutar el pipeline

Descripción: `src.pipeline` procesa imágenes, detecta contornos, aplica homografía y mejora la imagen. Dependiendo del dataset puede ejecutar OCR o binarizar.

Comandos (PowerShell):
```powershell
# Procesar una sola imagen
python -m src.pipeline --input data/raw/ocr_test/ocr_test_1.jpg --output-dir data/processed

# Procesar una carpeta (todas las imágenes dentro)
python -m src.pipeline --input data/raw/ocr_test --output-dir data/processed

# Procesar toda la raíz (todas las subcarpetas)
python -m src.pipeline --input data/raw --output-dir data/processed
```

Salida por imagen (en `data/processed/<dataset>/<stem>/`):
- `entrada_contour.<ext>` — imagen con contorno dibujado.
- `entrada_warp.<ext>` — imagen transformada (perspectiva corregida).
- `entrada_warp_doc.<ext>` — imagen final mejorada (o binarizada si es `_bin`).
- `<stem>.txt` — OCR (solo para datasets con OCR habilitado).
- `<stem>.corners.json` — esquinas predichas (pipeline guarda esquinas predichas para `scanner_test`).
1. **Crear y activar el entorno virtual (Windows PowerShell):**
   ```powershell
   python -m venv .venv311
   .\.venv311\Scripts\Activate.ps1
   ```
2. **Instalar dependencias (CPU):**
   ```powershell
   pip install -r requirements.txt
   ```
   Si tienes GPU con CUDA, instala la versión de torch/cuDNN adecuada antes de EasyOCR.
3. **Generar las imágenes de salida:**
    ```powershell
    python -m src.pipeline --input data/raw
    # O equivalente corto
    python -m src --input data/raw
    ```

Modos de ejecución
------------------

El CLI infiere automáticamente el modo a partir de la ruta pasada a `--input`:

- Archivo individual: pasa la ruta completa a una imagen, p. ej. `--input data/raw/ocr_test/ocr_test_1.jpg`.
   - Resultado: se crea `data/processed/<subcarpeta>/<file_stem>/` con los tres archivos de imagen (`entrada_contour`, `entrada_warp`, `entrada_warp_doc`).
   - Si la imagen está dentro de una subcarpeta con funcionalidad OCR se genera además `<file_stem>.txt` (UTF-8).

- Carpeta (subcarpeta): pasa la carpeta a procesar, p. ej. `--input data/raw/ocr_test`.
   - Resultado: procesa recursivamente todas las imágenes dentro de la carpeta y crea una carpeta por imagen en `data/processed/<subcarpeta>/<file_stem>/`.

- Todo (raíz): pasa la raíz `data/raw` (o la omites si es la ruta por defecto) para procesar todas las subcarpetas.

Reglas por nombre de subcarpeta
------------------------------

- `ocr_test`: escaneo + OCR — se guarda el `.txt` con el texto extraído.
- `scanner_test`: solo escaneo — no se ejecuta OCR y no se genera `.txt`.
- subcarpetas que terminan en `_bin` (por ejemplo `ocr_test_bin` o `scanner_test_bin`): escaneo → binarización final con threshold=195 → OCR sobre la imagen binarizada; la imagen final `entrada_warp_doc` se guarda en su versión binarizada y se genera el `.txt`.

Ejemplos (PowerShell)
---------------------

```powershell
# Archivo individual
python -m src.pipeline --input data/raw/ocr_test/ocr_test_1.jpg --output-dir data/processed

# Subcarpeta
python -m src.pipeline --input data/raw/ocr_test --output-dir data/processed

# Todo (todas las subcarpetas bajo data/raw)
python -m src.pipeline --input data/raw --output-dir data/processed

# Evaluar datasets (OCR y geometría) y generar gráficos
# El pipeline solo guarda esquinas predichas para `scanner_test`, así que `--pred-corners-root` solo tiene efecto ahí.
python scripts/evaluate.py --gt-root data/ground_truth --pred-root data/processed --raw-root data/raw --datasets ocr_test ocr_test_bin scanner_test
# Resultados en reports/metrics (metrics_summary.json, metrics_detail.csv, gráficos PNG)
```

Salida esperada
----------------

Cada imagen procesada produce (dentro de `data/processed/<subcarpeta>/<file_stem>/`):

- `entrada_contour.<ext>` — imagen con el contorno dibujado.
- `entrada_warp.<ext>` — imagen con perspectiva corregida.
- `entrada_warp_doc.<ext>` — imagen optimizada (o binarizada si la carpeta termina en `_bin`).
- `<file_stem>.txt` — archivo UTF-8 con el OCR (solo para carpetas con OCR habilitado).



## Estructura del proyecto

- `src/pipeline.py`: CLI que orquesta la ejecución.
- `src/document_pipeline/geometry.py`: detección de contorno y homografía.
- `src/document_pipeline/enhancement.py`: mejoras visuales y blanqueo.
- `src/document_pipeline/ocr.py`: inicialización perezosa del OCR y postprocesado.
- `src/document_pipeline/processing.py`: función `process_document` con el flujo completo.
- `requirements.txt`: dependencias principales.
- `pyproject.toml`: configuración de formateo/linter/tests.

Pruebas y calidad
-----------------

- Ejecutar tests (requiere `pytest`):
   ```powershell
   pip install pytest
   pytest
   ```
- Formateo y linting (requiere `black` y `ruff`, definidos en `pyproject.toml`).
