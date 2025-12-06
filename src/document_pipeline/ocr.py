"""Utilidades OCR con inicialización perezosa y postproceso.

El lector de EasyOCR se construye bajo demanda para evitar cargas pesadas
en el import del módulo y permitir ejecutar partes del pipeline sin
instalar dependencias de GPU.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _detect_gpu(use_gpu_override: bool | None) -> bool:
    """Decide si usar GPU.

    - Si el usuario indica True/False, se respeta.
    - Si es None, se intenta detectar CUDA via torch; si no está torch o falla,
      se vuelve a CPU.
    """
    if use_gpu_override is not None:
        return bool(use_gpu_override)
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=4)
def _build_reader(languages: tuple[str, ...], use_gpu: bool):
    """Crea y cachea el lector de EasyOCR.

    Se cachea por combinación de idiomas y flag de GPU. Si falla la
    inicialización, se propaga como RuntimeError con contexto.
    """
    try:
        import easyocr
    except Exception as exc:  # pragma: no cover - se ejecuta solo si falta easyocr
        raise RuntimeError("EasyOCR no está disponible. Instala 'easyocr' y 'torch'.") from exc

    try:
        reader = easyocr.Reader(list(languages), gpu=use_gpu, verbose=False)
    except Exception as exc:  # pragma: no cover - inicialización puede fallar en runtime
        raise RuntimeError(f"No se pudo inicializar EasyOCR (idiomas={languages}, gpu={use_gpu}).") from exc

    if use_gpu:
        logger.info("EasyOCR: GPU detectada y activada para inferencia.")
    else:
        logger.info("EasyOCR: GPU no detectada — usando CPU.")
    return reader


def get_ocr_reader(languages: Sequence[str] | None = None, use_gpu: bool | None = None):
    """Devuelve una instancia cacheada de EasyOCR.Reader.

    Parameters
    ----------
    languages : secuencia de str
        Idiomas a usar. Por defecto ('en', 'es').
    use_gpu : bool | None
        Forzar uso de GPU (True/False) o dejar que se autodetecte (None).
    """
    lang_tuple = tuple(languages or ("en", "es"))
    use_gpu_final = _detect_gpu(use_gpu)
    return _build_reader(lang_tuple, use_gpu_final)


def extract_text(
    image_bgr: np.ndarray,
    reader=None,
    conf_threshold: float = 0.30,
    return_debug: bool = False,
):
    """Ejecuta OCR sobre una imagen BGR y devuelve el texto postprocesado."""
    if reader is None:
        reader = get_ocr_reader()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(image_rgb)
    return _postprocess_easyocr_results(results, conf_threshold=conf_threshold, return_debug=return_debug)


def _postprocess_easyocr_results(
    results: Iterable,
    conf_threshold: float = 0.30,
    return_debug: bool = False,
):
    """Agrupa detecciones de EasyOCR respetando columnas y líneas.

    - Filtra por confianza.
    - Ordena por Y, luego por columnas si hay separación suficiente.
    - Junta tokens por línea y agrupa en párrafos por salto vertical.
    """
    if not results:
        return ("", [], []) if return_debug else ""

    cleaned: list[dict] = []
    heights: list[float] = []
    x_values: list[float] = []
    for entry in results:
        if (not isinstance(entry, (list, tuple))) or len(entry) < 3:
            continue
        bbox, text, conf = entry
        try:
            confidence = float(conf) if conf is not None else 1.0
        except Exception:
            confidence = 1.0
        if confidence < conf_threshold:
            continue
        try:
            xs = [float(pt[0]) for pt in bbox]
            ys = [float(pt[1]) for pt in bbox]
        except Exception:
            continue
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        heights.append(y_max - y_min)
        x_values.extend([x_min, x_max])
        cleaned.append(
            {
                "text": str(text).strip(),
                "conf": confidence,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "y_center": (y_min + y_max) / 2.0,
            }
        )

    if not cleaned:
        return ("", [], []) if return_debug else ""

    cleaned.sort(key=lambda it: it["y_center"])
    median_h = float(np.median(heights)) if heights else 12.0
    strict_tol = max(8.0, median_h * 0.35)
    paragraph_gap = max(strict_tol * 4.0, median_h * 2.2)

    line_clusters: list[list[dict]] = []
    cluster_centers: list[float] = []

    for det in cleaned:
        best_idx = None
        best_diff = None
        for idx, center in enumerate(cluster_centers):
            diff = abs(det["y_center"] - center)
            if diff <= strict_tol and (best_diff is None or diff < best_diff):
                best_idx = idx
                best_diff = diff
        if best_idx is None:
            line_clusters.append([det])
            cluster_centers.append(det["y_center"])
        else:
            line_clusters[best_idx].append(det)
            cluster_centers[best_idx] = float(np.mean([d["y_center"] for d in line_clusters[best_idx]]))

    lines = [sorted(cluster, key=lambda it: it["x_min"]) for cluster in line_clusters]

    assembled: list[dict] = []
    for group in lines:
        # Divide el grupo en subgrupos si hay huecos horizontales grandes (posibles columnas).
        split_gap = max(median_h * 2.5, 35.0)
        subgroups: list[list[dict]] = [[]]
        last_xmax = None
        for det in group:
            x_min, x_max = det["x_min"], det["x_max"]
            if last_xmax is not None and (x_min - last_xmax) > split_gap:
                subgroups.append([])
            subgroups[-1].append(det)
            last_xmax = x_max

        for sub in subgroups:
            tokens: list[str] = []
            for det in sub:
                token = det["text"]
                if not token:
                    continue
                if tokens and tokens[-1].endswith("-"):
                    tokens[-1] = tokens[-1][:-1] + token
                else:
                    tokens.append(token)
            if not tokens:
                continue
            text = " ".join(tokens)
            x_center = float(np.mean([(det["x_min"] + det["x_max"]) / 2.0 for det in sub]))
            y_center = float(np.mean([det["y_center"] for det in sub]))
            assembled.append(
                {
                    "text": text,
                    "x_center": x_center,
                    "y_center": y_center,
                }
            )

    if not assembled:
        return ("", cleaned, []) if return_debug else ""

    page_width = (max(x_values) - min(x_values)) if x_values else 1.0
    column_ids = [0] * len(assembled)
    if len(assembled) >= 4:
        x_centers = np.array([line["x_center"] for line in assembled])
        order = np.argsort(x_centers)
        sorted_x = x_centers[order]
        gaps = np.diff(sorted_x)
        if gaps.size:
            median_gap = float(np.median(gaps))
            # Umbral más permisivo para detectar columnas en páginas simples.
            split_threshold = max(median_gap * 0.8, page_width * 0.08, 40.0)
            splits = np.where(gaps > split_threshold)[0]
            if splits.size:
                current_col = 0
                for pos, original_idx in enumerate(order):
                    column_ids[original_idx] = current_col
                    if pos in splits:
                        current_col += 1

    ordered = [
        (line["y_center"], line["x_center"], column_ids[idx], line["text"])
        for idx, line in enumerate(assembled)
    ]
    ordered.sort(key=lambda tpl: (tpl[2], tpl[0], tpl[1]))

    paragraphs: list[list[str]] = []
    current_para: list[str] = []
    prev_y = ordered[0][0]
    prev_col = ordered[0][2]
    for y_center, _x_center, col_id, text in ordered:
        if not current_para:
            current_para.append(text)
            prev_y, prev_col = y_center, col_id
            continue
        gap = y_center - prev_y
        if gap > paragraph_gap or col_id != prev_col:
            paragraphs.append(current_para)
            current_para = [text]
        else:
            current_para.append(text)
        prev_y, prev_col = y_center, col_id
    if current_para:
        paragraphs.append(current_para)

    normalized_lines: list[str] = []
    for para in paragraphs:
        for line in para:
            normalized_lines.append(" ".join(line.split()))
        normalized_lines.append("")
    if normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    result_text = "\n".join(normalized_lines)
    if return_debug:
        return result_text, cleaned, ordered
    return result_text
