"""Función principal de procesamiento del documento"""
from pathlib import Path
from typing import Iterable, Tuple, Optional
import json
import logging

import cv2
import numpy as np

from .enhancement import enhance_document_appearance, whiten_near_white
from .geometry import detect_document_contour, preprocess, warp_perspective
from .ocr import extract_text, get_ocr_reader

DEFAULT_OUTPUT_DIR = Path("data/processed")


def process_document(
    input_path: Path,
    output_dir: Path | None = None,
    do_ocr: bool = True, # flag para aplicar OCR
    binarize_threshold: Optional[int] = None, # flag para aplicar binarización
    write_corners: bool = True,
) -> Tuple[Path, Path, Path, str]:
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen de entrada: {input_path}")

    gray, edges = preprocess(image) # devuelve una imagen en escala de grises y los bordes detectados
    contour = detect_document_contour(gray, edges) # devuelve un array con las coordenadas de las esquinas detectadas

    # Crear una imagen a la que se le dibuja el contorno detectado
    overlay = image.copy()
    poly = contour.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(overlay, [poly], True, (0, 255, 0), 3)

    # Aplicar transformación perspectiva para obtener la vista "escaneada"
    warped = warp_perspective(image, contour)
    enhanced_warp = enhance_document_appearance(warped)
    cleaned_warp = whiten_near_white(enhanced_warp)

    """
    Binarización final opcional. Si se indica "binarize_threshold",
    se aplica threshold al resultado generado por el escáner
    """
    final_for_ocr = cleaned_warp
    if binarize_threshold is not None:
        gray_for_bin = cv2.cvtColor(cleaned_warp, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray_for_bin, int(binarize_threshold), 255, cv2.THRESH_BINARY)
        bin_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        final_for_ocr = bin_bgr

    # Ejecutar OCR solo si está habilitado
    extracted_text = ""
    if do_ocr:
        reader = get_ocr_reader()
        extracted_text = extract_text(final_for_ocr, reader=reader)

    # Preparar directorio de salida
    target_dir = output_dir or DEFAULT_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    extension = input_path.suffix if input_path.suffix else ".jpg"
    contour_path = target_dir / f"entrada_contour{extension}"
    warp_path = target_dir / f"entrada_warp{extension}"
    warp_clean_path = target_dir / f"entrada_warp_doc{extension}"
    corners_json_path = target_dir / f"{input_path.stem}.corners.json"

    # Guardar imágenes de salida
    if not cv2.imwrite(str(contour_path), overlay):
        raise RuntimeError(f"No se pudo guardar la imagen de contorno en {contour_path}")
    if not cv2.imwrite(str(warp_path), warped):
        raise RuntimeError(f"No se pudo guardar la imagen warp en {warp_path}")
    if not cv2.imwrite(str(warp_clean_path), final_for_ocr):
        raise RuntimeError(f"No se pudo guardar la imagen warp procesada en {warp_clean_path}")

    # Guardar las esquinas detectadas para evaluarlas después
    if write_corners:
        corners_payload = {
            "image": input_path.name,
            "corners": [[float(x), float(y)] for [x, y] in contour.tolist()],
        }
        corners_json_path.write_text(json.dumps(corners_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return contour_path, warp_path, warp_clean_path, extracted_text


def process_batch(
    image_paths: Iterable[Path],
    output_root: Path | None = None,
    input_root: Path | None = None,
) -> list[Tuple[Path, Path, Path, str]]:

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

        """
        Determinar la carpeta relativa superior ("ocr_test") si se proporciona input_root.
        Se intenta de forma robusta usando primero la ruta relativa del propio archivo
        y como fallback la carpeta padre, para manejar distintos niveles de profundidad
        """
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

        """
        Seleccionar el comportamiento según la carpeta superior bajo "input_root":
            - "ocr_test" => escaneo + OCR
            - "scanner_test" => solo escaneo (sin OCR)
            - "scanner_test_bin" => escaneo + binarización (threshold=195) + OCR
        """
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
            
            elif name in ("scanner_test_bin", "ocr_test_bin") or name.endswith("_bin"):
                do_ocr_flag = True
                bin_thresh = 195

        write_corners = (top_folder == "scanner_test")
        result = process_document(
            image_path,
            target_dir,
            do_ocr=do_ocr_flag,
            binarize_threshold=bin_thresh,
            write_corners=write_corners,
        )
        results.append(result)

    return results


