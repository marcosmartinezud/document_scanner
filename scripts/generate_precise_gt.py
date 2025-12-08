"""Este script simula que disponemos de los bordes exactos de los documentos.
Ha sido generado por una inteligencia artificial y su propósito es crear archivos
JSON con las esquinas precisas de cada imagen. El código no contiene comentarios
explicativos adicionales dentro del archivo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.document_pipeline.geometry import preprocess, detect_document_contour, warp_perspective


def find_images(root: Path, patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(root.rglob(pat))
    return sorted(paths)


def save_corners_json(out_file: Path, image_name: str, corners: np.ndarray) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": image_name,
        "corners": [[float(x), float(y)] for x, y in corners],
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def lines_from_hough(edges: np.ndarray, min_len: int, max_gap: int) -> list[tuple[int, int, int, int]]:
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=50, minLineLength=min_len, maxLineGap=max_gap)
    if raw is None:
        return []
    return [tuple(l[0].tolist()) for l in raw]


def categorize_lines(lines: list[tuple[int, int, int, int]]) -> tuple[list, list]:
    horiz = []
    vert = []
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        angle = angle % 180
        if angle > 90:
            angle = 180 - angle
        if angle < 30:
            horiz.append((x1, y1, x2, y2))
        elif angle > 60:
            vert.append((x1, y1, x2, y2))
    return horiz, vert


def estimate_border_positions(horiz: list, vert: list, w: int, h: int):
    top_y = None
    bottom_y = None
    left_x = None
    right_x = None

    if horiz:
        ys = [ (y1 + y2) / 2.0 for (x1, y1, x2, y2) in horiz ]
        top_y = float(min(ys))
        bottom_y = float(max(ys))
    else:
        top_y = 0.0
        bottom_y = float(h - 1)

    if vert:
        xs = [ (x1 + x2) / 2.0 for (x1, y1, x2, y2) in vert ]
        left_x = float(min(xs))
        right_x = float(max(xs))
    else:
        left_x = 0.0
        right_x = float(w - 1)

    return top_y, bottom_y, left_x, right_x


def detect_precise_corners(img: np.ndarray) -> np.ndarray:
    gray0, edges0 = preprocess(img)
    try:
        initial = detect_document_contour(gray0, edges0)
    except Exception:
        raise

    warped = warp_perspective(img, initial)
    wh, ww = warped.shape[:2]

    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_w, 50, 150)

    min_len = max(20, int(0.25 * max(ww, wh)))
    lines = lines_from_hough(edges, min_len=min_len, max_gap=20)
    horiz, vert = categorize_lines(lines)

    if not horiz or not vert:
        return initial

    top_y, bottom_y, left_x, right_x = estimate_border_positions(horiz, vert, ww, wh)

    warped_corners = np.array(
        [
            [left_x, top_y],
            [right_x, top_y],
            [right_x, bottom_y],
            [left_x, bottom_y],
        ],
        dtype=np.float32,
    )

    src_rect = np.array(warped_corners, dtype=np.float32)
    dst_rect = initial.astype(np.float32)
    inv = cv2.getPerspectiveTransform(src_rect, dst_rect)
    warped_pts = src_rect.reshape(-1, 1, 2)
    orig_pts = cv2.perspectiveTransform(warped_pts, inv).reshape(-1, 2)

    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    pts = orig_pts.reshape(-1, 1, 2).astype(np.float32)
    try:
        cv2.cornerSubPix(gray_orig, pts, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
    except Exception:
        pass

    return pts.reshape(4, 2).astype(np.float32)


def process_image(img_path: Path, output_root: Path, debug_dir: Path | None = None, overwrite: bool = False) -> tuple[bool, str]:
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False, "no se pudo leer"

        corners = detect_precise_corners(img)

        rel = img_path.stem
        out_file = output_root / f"{rel}.corners.json"
        already = out_file.exists()
        if not already or overwrite:
            save_corners_json(out_file, img_path.name, corners)

        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            overlay = img.copy()
            poly = corners.reshape((-1, 1, 2)).astype(int)
            cv2.polylines(overlay, [poly], True, (255, 0, 0), 3)
            for idx, (x, y) in enumerate(corners):
                cv2.circle(overlay, (int(round(x)), int(round(y))), 6, (0, 255, 0), -1)
                cv2.putText(overlay, str(idx + 1), (int(round(x)) + 6, int(round(y)) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            debug_path = debug_dir / f"{img_path.stem}_precise_overlay.jpg"
            cv2.imwrite(str(debug_path), overlay)

        return True, "ok"
    except Exception as exc:
        return False, f"error: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generador de ground-truth preciso para esquinas (scanner_test)")
    parser.add_argument("--input", type=Path, default=Path("data/raw/scanner_test"))
    parser.add_argument("--output", type=Path, default=Path("data/ground_truth/scanner_test_precise"))
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe JSON existentes")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Carpeta donde guardar overlays con el polígono dibujado")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"],
        help="Patrones de imagen a buscar",
    )
    args = parser.parse_args(argv)

    input_dir = args.input
    output_dir = args.output

    if not input_dir.exists():
        print(f"Ruta de entrada no existe: {input_dir}")
        return 1

    images = find_images(input_dir, args.patterns)
    if not images:
        print(f"No se encontraron imágenes en {input_dir}")
        return 1

    ok = 0
    failed = 0
    for img_path in images:
        success, msg = process_image(img_path, output_dir, debug_dir=args.debug_dir, overwrite=args.overwrite)
        prefix = "[OK]" if success else "[FAIL]"
        print(prefix, img_path, "->", msg)
        ok += int(success)
        failed += int(not success)

    print(f"Listo. Éxito: {ok}, Fallos: {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
