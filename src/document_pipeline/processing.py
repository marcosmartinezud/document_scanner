"""Función principal de procesamiento del documento."""
from pathlib import Path
from typing import Iterable, Tuple, Optional
import logging

import cv2
import easyocr
import numpy as np
try:
    import torch
except Exception:
    torch = None

from .enhancement import enhance_document_appearance, whiten_near_white
from .geometry import detect_document_contour, preprocess, warp_perspective

DEFAULT_OUTPUT_DIR = Path("data/processed")


# Inicializa el lector de EasyOCR a nivel de módulo para evitar recrearlo en cada llamada.
# Idiomas por defecto: inglés y español (ajusta según tus necesidades).
try:
    # Detecta si hay CUDA disponible y usa GPU si es posible
    gpu_available = False
    if torch is not None:
        try:
            gpu_available = torch.cuda.is_available()
        except Exception:
            gpu_available = False

    _OCR_READER = easyocr.Reader(['en', 'es'], gpu=gpu_available, verbose=False)
    if gpu_available:
        logging.getLogger(__name__).info('EasyOCR: GPU detectada y activada para inferencia.')
    else:
        logging.getLogger(__name__).info('EasyOCR: GPU no detectada — usando CPU.')
except Exception:
    # Si la inicialización falla (por ejemplo, dependencias no instaladas),
    # dejamos _OCR_READER como None y manejamos el error en tiempo de ejecución.
    _OCR_READER = None


def process_document(
    input_path: Path,
    output_dir: Path | None = None,
    do_ocr: bool = True,
    binarize_threshold: Optional[int] = None,
) -> Tuple[Path, Path, Path, str]:
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen de entrada: {input_path}")

    gray, edges = preprocess(image)
    contour = detect_document_contour(gray, edges)

    overlay = image.copy()
    poly = contour.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(overlay, [poly], True, (0, 255, 0), 3)

    warped = warp_perspective(image, contour)
    enhanced_warp = enhance_document_appearance(warped)
    cleaned_warp = whiten_near_white(enhanced_warp)

    # Posible binarización final (opcional). Si se indica `binarize_threshold`,
    # se aplica threshold al resultado limpio generado por el escáner.
    final_for_ocr = cleaned_warp
    if binarize_threshold is not None:
        gray_for_bin = cv2.cvtColor(cleaned_warp, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray_for_bin, int(binarize_threshold), 255, cv2.THRESH_BINARY)
        bin_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        final_for_ocr = bin_bgr

    # Ejecuta OCR solo si está habilitado
    extracted_text = ""
    if do_ocr:
        # Prepara la imagen procesada para OCR: EasyOCR funciona bien con RGB.
        ocr_rgb = cv2.cvtColor(final_for_ocr, cv2.COLOR_BGR2RGB)
        if _OCR_READER is None:
            raise RuntimeError("EasyOCR reader no está disponible. Asegúrate de instalar 'easyocr' y 'torch'.")
        results = _OCR_READER.readtext(ocr_rgb)
        # results: lista de (bbox, text, confidence)
        extracted_text = _postprocess_easyocr_results(results)

    target_dir = output_dir or DEFAULT_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    extension = input_path.suffix if input_path.suffix else ".jpg"
    contour_path = target_dir / f"entrada_contour{extension}"
    warp_path = target_dir / f"entrada_warp{extension}"
    warp_clean_path = target_dir / f"entrada_warp_doc{extension}"

    if not cv2.imwrite(str(contour_path), overlay):
        raise RuntimeError(f"No se pudo guardar la imagen de contorno en {contour_path}")
    if not cv2.imwrite(str(warp_path), warped):
        raise RuntimeError(f"No se pudo guardar la imagen warp en {warp_path}")
    # Guardamos la versión final que representa el estado "optimizado" del documento.
    # Si se aplicó binarización, `final_for_ocr` contiene la imagen binarizada.
    if not cv2.imwrite(str(warp_clean_path), final_for_ocr):
        raise RuntimeError(f"No se pudo guardar la imagen warp procesada en {warp_clean_path}")

    return contour_path, warp_path, warp_clean_path, extracted_text


def process_batch(
    image_paths: Iterable[Path],
    output_root: Path | None = None,
    input_root: Path | None = None,
) -> list[Tuple[Path, Path, Path, str]]:
    """Procesa una colección de imágenes.

    Si se provee `input_root`, se intentará preservar la primera carpeta relativa bajo
    `input_root` (por ejemplo `ocr_test`) y crear la salida en
    `output_root/<subfolder>/<file_stem>/...`. Si no hay `input_root`, el comportamiento
    antiguo se mantiene: `output_root/<file_stem>/...`.
    """
    results: list[Tuple[Path, Path, Path, str]] = []
    root = (output_root or DEFAULT_OUTPUT_DIR).resolve()

    # (no debug)

    for image_path in image_paths:
        # Resolver rutas para evitar mezclas entre rutas relativas y absolutas
        try:
            image_path = image_path.resolve()
        except Exception:
            image_path = Path(str(image_path))

        stem = image_path.stem or "entrada"

        # Determina la carpeta relativa superior (p. ej. 'ocr_test') si se proporciona input_root.
        # Se intenta de forma robusta usando primero la ruta relativa del propio archivo
        # y como fallback la carpeta padre, para manejar distintos niveles de profundidad.
        if input_root is not None:
            try:
                input_root_resolved = input_root.resolve()
            except Exception:
                input_root_resolved = Path(str(input_root))
            try:
                rel = image_path.relative_to(input_root_resolved)
                top_folder = rel.parts[0] if rel.parts else None
            except Exception:
                try:
                    rel_parent = image_path.parent.relative_to(input_root_resolved)
                    top_folder = rel_parent.parts[0] if rel_parent.parts else None
                except Exception:
                    top_folder = None
        else:
            top_folder = None

        if top_folder:
            target_dir = root / top_folder / stem
        else:
            target_dir = root / stem

        # (no debug)

        # Selecciona el comportamiento según la carpeta superior bajo `input_root`:
        # - 'ocr_test' => escaneo + OCR
        # - 'scanner_test' => solo escaneo (sin OCR)
        # - 'scanner_test_bin' => escaneo + binarización (threshold=195) + OCR
        do_ocr_flag = True
        bin_thresh: Optional[int] = None
        if top_folder:
            name = str(top_folder).lower()
            if name == "ocr_test":
                do_ocr_flag = True
                bin_thresh = None
            elif name == "scanner_test":
                do_ocr_flag = False
                bin_thresh = None
            # Soportamos carpetas llamadas "ocr_test_bin" o "scanner_test_bin"
            # y, por seguridad, cualquier carpeta que termine en "_bin".
            elif name in ("scanner_test_bin", "ocr_test_bin") or name.endswith("_bin"):
                do_ocr_flag = True
                bin_thresh = 195

        result = process_document(image_path, target_dir, do_ocr=do_ocr_flag, binarize_threshold=bin_thresh)
        results.append(result)

    return results


def _postprocess_easyocr_results(
    results: list,
    conf_threshold: float = 0.30,
    return_debug: bool = False,
):
    """Agrupa las detecciones de EasyOCR respetando limites horizontales para evitar mezclas."""

    if not results:
        return ("", [], []) if return_debug else ""

    cleaned = []
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
        cleaned.append({
            "text": str(text).strip(),
            "conf": confidence,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "y_center": (y_min + y_max) / 2.0,
        })

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
        tokens: list[str] = []
        for det in group:
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
        x_center = float(np.mean([(det["x_min"] + det["x_max"]) / 2.0 for det in group]))
        y_center = float(np.mean([det["y_center"] for det in group]))
        assembled.append({
            "text": text,
            "x_center": x_center,
            "y_center": y_center,
        })

    if not assembled:
        return ("", cleaned, []) if return_debug else ""

    page_width = (max(x_values) - min(x_values)) if x_values else 1.0
    column_ids = [0] * len(assembled)
    if len(assembled) >= 6:
        x_centers = np.array([line["x_center"] for line in assembled])
        order = np.argsort(x_centers)
        sorted_x = x_centers[order]
        gaps = np.diff(sorted_x)
        if gaps.size:
            median_gap = float(np.median(gaps))
            split_threshold = max(median_gap * 3.0, page_width * 0.18, 80.0)
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
