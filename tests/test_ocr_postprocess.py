import numpy as np

from src.document_pipeline.ocr import _postprocess_easyocr_results


def test_postprocess_groups_lines_and_columns():
    # Simula detecciones de dos columnas con salto de párrafo.
    results = [
        # Columna izquierda
        ([[0, 0], [10, 0], [10, 10], [0, 10]], "Hola", 0.92),
        ([[12, 0], [25, 0], [25, 10], [12, 10]], "mundo", 0.93),
        ([[0, 20], [10, 20], [10, 30], [0, 30]], "multi-", 0.95),
        ([[12, 20], [25, 20], [25, 30], [12, 30]], "linea", 0.96),
        # Columna derecha (x separado para forzar columna 1)
        ([[200, 5], [220, 5], [220, 15], [200, 15]], "Columna", 0.97),
        ([[200, 22], [220, 22], [220, 32], [200, 32]], "Dos", 0.98),
    ]

    text, cleaned, ordered = _postprocess_easyocr_results(results, return_debug=True)

    assert "Hola mundo" in text
    assert "multilinea" in text  # une el guion final
    assert text.split("\n")[0] == "Hola mundo"
    # Comprobamos que hay dos columnas detectadas
    column_ids = {col for (_y, _x, col, _t) in ordered}
    assert column_ids == {0, 1}
    assert len(cleaned) == 6


def test_postprocess_returns_empty_on_low_confidence():
    results = [([[0, 0], [1, 0], [1, 1], [0, 1]], "ruido", 0.1)]
    assert _postprocess_easyocr_results(results, conf_threshold=0.5) == ""
