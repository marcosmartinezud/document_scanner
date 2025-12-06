# Extracción Rápida de Documento

La utilidad detecta el contorno principal de una foto, corrige la perspectiva y genera tres variantes:

- `entrada_contour`: la foto original con el contorno dibujado en verde.
- `entrada_warp`: la misma escena con la perspectiva corregida en bruto.
- `entrada_warp_doc`: versión rectificada con sombras reducidas y fondo blanqueado.

## Uso

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
