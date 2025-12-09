"""Script para evaluar métricas OCR (CER, WER) y métricas geométricas 
(corner RMSE, IoU) en datasets de imágenes escaneadas.
Genera un resumen en JSON y un CSV detallado por imagen"""
from __future__ import annotations

import argparse
import json
import sys
import csv
from pathlib import Path
from typing import Optional, Iterable

import numpy as np

from src.eval_metrics import cer, wer, polygon_area, aspect_ratio, mean_corner_error


# Leer y normalizar texto ground truth o predicho
def load_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


# Leer esquinas desde archivo JSON
def load_corners(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("corners")


def polygon_iou(poly_a, poly_b, shape_hw: tuple[int, int]) -> Optional[float]:
    """
    IoU entre dos polígonos usando máscaras binarias
    """
    import cv2

    h, w = shape_hw # height, width
    # Evitar máscaras vacías
    if h <= 0 or w <= 0: 
        return None
    
    # Crear máscaras binarias
    a = np.zeros((h, w), dtype=np.uint8)
    b = np.zeros((h, w), dtype=np.uint8)

    # Rellenar polígonos
    cv2.fillPoly(a, [np.array(poly_a, dtype=np.int32)], 1)
    cv2.fillPoly(b, [np.array(poly_b, dtype=np.int32)], 1)

    # Calcular intersección y unión
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0: # Evitar división por cero
        return None
    return float(inter) / float(union)


def find_image(raw_root: Path, stem: str, patterns: Iterable[str]) -> Optional[Path]:
    """
    Buscar imagen en raw_root que coincida con el stem dado
    """
    for pat in patterns:
        for p in raw_root.rglob(pat):
            if p.stem == stem:
                return p
    return None


def evaluate_dataset(
    dataset: str,
    raw_root: Path,
    gt_root: Path,
    pred_root: Path,
    pred_corners_root: Optional[Path],
    eval_geometry: bool,
    skip_ocr: bool,
) -> list[dict]:
    """
    Evaluar un dataset específico y devolver una lista de diccionarios con las métricas por imagen
    """

    # Obtener todos los stems de archivos GT (txt y corners.json)
    stems_set = set()
    for p in gt_root.glob("*.gt.txt"):
        stems_set.add(p.name.replace(".gt.txt", ""))
    for p in gt_root.glob("*.corners.json"):
        stems_set.add(p.name.replace(".corners.json", ""))
    stems = sorted(stems_set)
    rows: list[dict] = []
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]

    for stem in stems:
        row = {"dataset": dataset, "stem": stem}

        if skip_ocr: # omitir cálculo de CER/WER
            row["cer"] = None
            row["wer"] = None
        else: # Calcular CER/WER
            gt_txt = gt_root / f"{stem}.gt.txt" # archivo de ground truth

            # Buscar archivo predicho correspondiente
            def find_pred_txt(root: Path, stem_name: str):
                # Buscar en ubicaciones comunes
                candidates = [root / stem_name / f"{stem_name}.txt", root / f"{stem_name}.txt"]
                for c in candidates:
                    if c.exists():
                        return c
                # Búsqueda más flexible
                import re

                def extract_trailing_number(name: str):
                    m = re.search(r"(\d+)$", name)
                    return m.group(1) if m else None

                target_num = extract_trailing_number(stem_name)
                for p in root.rglob("*.txt"):
                    s = p.stem
                    if s == stem_name or s.endswith(stem_name) or stem_name.endswith(s):
                        return p
                    # intentar coincidir por número final
                    if target_num is not None:
                        s_num = extract_trailing_number(s)
                        if s_num == target_num:
                            return p
                return None

            pred_txt_path = find_pred_txt(pred_root, stem)
            ref = load_text(gt_txt)
            hyp = load_text(pred_txt_path) if pred_txt_path is not None else None
            if ref is not None and hyp is not None:
                row["cer"] = cer(ref, hyp)
                row["wer"] = wer(ref, hyp)
            else:
                row["cer"] = None
                row["wer"] = None

        if eval_geometry: # Calcular métricas geométricas (solo para scanner_test)
            corners_gt = load_corners(gt_root / f"{stem}.corners.json")
            img_path = find_image(raw_root, stem, patterns)
            img = None
            if img_path and img_path.exists():
                import cv2 

                img = cv2.imread(str(img_path))

            if corners_gt is not None and img is not None:
                h, w = img.shape[:2]
                area_img = float(w * h)
                area_poly = polygon_area(corners_gt)
                row["area_ratio_gt"] = area_poly / area_img if area_img > 0 else None
                row["aspect_ratio_gt"] = aspect_ratio(corners_gt)

            # Comparar pred vs GT de esquinas (solo si hay pred_corners_root)
            pred_c = None
            if pred_corners_root is not None:
                for candidate in [
                    pred_corners_root / f"{stem}.corners.json",
                    pred_corners_root / stem / f"{stem}.corners.json",
                ]:
                    if candidate.exists():
                        pred_c = load_corners(candidate)
                        break

            if pred_c is not None:
                row["area_ratio_pred"] = None
                row["aspect_ratio_pred"] = None
                if img is not None:
                    h, w = img.shape[:2]
                    area_img = float(w * h)
                    area_pred = polygon_area(pred_c)
                    row["area_ratio_pred"] = area_pred / area_img if area_img > 0 else None
                    row["aspect_ratio_pred"] = aspect_ratio(pred_c)

                if corners_gt is not None:
                    try:
                        row["corner_rmse"] = mean_corner_error(corners_gt, pred_c)
                    except Exception:
                        row["corner_rmse"] = None

                    if img is not None:
                        # IoU entre polígonos usando máscaras
                        try:
                            row["polygon_iou"] = polygon_iou(corners_gt, pred_c, img.shape[:2])
                        except Exception:
                            row["polygon_iou"] = None

                if row.get("area_ratio_gt") is not None and row.get("area_ratio_pred") is not None:
                    row["area_ratio_diff"] = row["area_ratio_pred"] - row["area_ratio_gt"]
                if row.get("aspect_ratio_gt") is not None and row.get("aspect_ratio_pred") is not None:
                    row["aspect_ratio_diff"] = row["aspect_ratio_pred"] - row["aspect_ratio_gt"]

        rows.append(row)

    return rows


def summarize(rows: list[dict]) -> dict:
    """
    Agregar resultados por dataset y calcular promedios generales
    """
    def avg(key: str, dataset: str | None = None):
        vals = [r[key] for r in rows if r.get(key) is not None and (dataset is None or r.get("dataset") == dataset)]
        return sum(vals) / len(vals) if vals else None

    datasets = sorted({r["dataset"] for r in rows})
    per_ds = {ds: {
        "count": sum(1 for r in rows if r["dataset"] == ds),
        "cer_avg": avg("cer", ds),
        "wer_avg": avg("wer", ds),
        "area_ratio_gt_avg": avg("area_ratio_gt", ds) if ds == "scanner_test" else None,
        "area_ratio_pred_avg": avg("area_ratio_pred", ds) if ds == "scanner_test" else None,
        "area_ratio_diff_avg": avg("area_ratio_diff", ds) if ds == "scanner_test" else None,
        "aspect_ratio_gt_avg": avg("aspect_ratio_gt", ds) if ds == "scanner_test" else None,
        "aspect_ratio_pred_avg": avg("aspect_ratio_pred", ds) if ds == "scanner_test" else None,
        "aspect_ratio_diff_avg": avg("aspect_ratio_diff", ds) if ds == "scanner_test" else None,
        "corner_rmse_avg": avg("corner_rmse", ds) if ds == "scanner_test" else None,
        "polygon_iou_avg": avg("polygon_iou", ds) if ds == "scanner_test" else None,
    } for ds in datasets}

    summary = {
        "overall": {
            "count": len(rows),
            "cer_avg": avg("cer"),
            "wer_avg": avg("wer"),
            # Métricas geométricas globales (solo scanner_test)
            "area_ratio_gt_avg": avg("area_ratio_gt", "scanner_test"),
            "area_ratio_pred_avg": avg("area_ratio_pred", "scanner_test"),
            "area_ratio_diff_avg": avg("area_ratio_diff", "scanner_test"),
            "aspect_ratio_gt_avg": avg("aspect_ratio_gt", "scanner_test"),
            "aspect_ratio_pred_avg": avg("aspect_ratio_pred", "scanner_test"),
            "aspect_ratio_diff_avg": avg("aspect_ratio_diff", "scanner_test"),
            "corner_rmse_avg": avg("corner_rmse", "scanner_test"),
            "polygon_iou_avg": avg("polygon_iou", "scanner_test"),
        },
        "by_dataset": per_ds,
    }

    # Comparaciones específicas entre datasets (si los dos existen)
    if "ocr_test" in per_ds and "ocr_test_bin" in per_ds:
        summary["comparison_ocr_vs_bin"] = {
            "cer_diff": safe_diff(per_ds["ocr_test"].get("cer_avg"), per_ds["ocr_test_bin"].get("cer_avg")),
            "wer_diff": safe_diff(per_ds["ocr_test"].get("wer_avg"), per_ds["ocr_test_bin"].get("wer_avg")),
        }
    return summary


# Función auxiliar para diferencias seguras
def safe_diff(a, b):
    if a is None or b is None:
        return None
    return b - a


# Guardar lista de diccionarios en CSV
def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# Guardar diccionario en JSON
def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# Generar gráficos de métricas usando matplotlib
def plot_metrics(out_dir: Path, summary: dict, rows: list[dict]):
    """
    Genera únicamente tres gráficos geométricos para `scanner_test`:
    - corner_rmse_hist.png
    - polygon_iou_hist.png
    - area_ratio_scatter.png

    No genera gráficos para otros datasets (ocr_test / ocr_test_bin) y
    tampoco genera gráficos cuando el directorio de salida sea
    `metrics_ocr_test` o `metrics_ocr_test_bin`.
    """
    # Saltar gráficos explícitamente en las carpetas pedidas
    if out_dir.name in {"metrics_ocr_test", "metrics_ocr_test_bin"}:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib no disponible; se omiten gráficos")
        return

    # Solo consideramos filas del dataset geométrico
    geo_rows = [r for r in rows if r.get("dataset") == "scanner_test"]
    if not geo_rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    def hist_plot(values, title, fname, bins=20):
        vals = [v for v in values if v is not None]
        if not vals:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals, bins=bins, color="#4c78a8", alpha=0.8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    def scatter_plot(x_vals, y_vals, title, xlab, ylab, fname):
        xs = [x for x, y in zip(x_vals, y_vals) if x is not None and y is not None]
        ys = [y for x, y in zip(x_vals, y_vals) if x is not None and y is not None]
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys, color="#4c78a8", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    # Gráficos solicitados
    hist_plot([r.get("corner_rmse") for r in geo_rows], "Distribución Corner RMSE", "corner_rmse_hist.png")
    hist_plot([r.get("polygon_iou") for r in geo_rows], "Distribución IoU", "polygon_iou_hist.png")
    scatter_plot(
        [r.get("area_ratio_gt") for r in geo_rows],
        [r.get("area_ratio_pred") for r in geo_rows],
        "Área GT vs Pred",
        "Área ratio GT",
        "Área ratio Pred",
        "area_ratio_scatter.png",
    )


def main(argv=None) -> int:
    """
    Definir la CLI y ejecutar la evaluación
    """
    parser = argparse.ArgumentParser(description="Evalúa OCR y geometría por dataset")
    parser.add_argument("--gt-root", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--pred-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--pred-corners-root", type=Path, default=None, help="Carpeta donde están las esquinas predichas; si no se proporciona, no se calcula corner_rmse")
    parser.add_argument("--datasets", nargs="+", default=["ocr_test", "ocr_test_bin", "scanner_test"], help="Datasets a evaluar")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--no-plots", action="store_true", help="No generar gráficos")
    parser.add_argument("--skip-ocr", action="store_true", help="No calcular CER/WER (útil si solo quieres geometría)")
    parser.add_argument("--geometry-only", action="store_true", help="Atajo: evalúa solo scanner_test y desactiva OCR")
    args = parser.parse_args(argv)

    if args.geometry_only:
        args.datasets = ["scanner_test"]
        args.skip_ocr = True

    all_rows: list[dict] = []
    for ds in args.datasets:
        gt_dir = args.gt_root / ds
        pred_dir = args.pred_root / ds
        raw_dir = args.raw_root / ds
        if not gt_dir.exists():
            print(f"[WARN] GT no encontrado: {gt_dir}, se omite {ds}")
            continue
        pred_corners_dir = args.pred_corners_root / ds if args.pred_corners_root else None
        rows = evaluate_dataset(
            ds,
            raw_dir,
            gt_dir,
            pred_dir,
            pred_corners_dir,
            eval_geometry=(ds == "scanner_test"),
            skip_ocr=args.skip_ocr or ds not in {"ocr_test", "ocr_test_bin"},
        )
        all_rows.extend(rows)

    summary = summarize(all_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "metrics_detail.csv", all_rows)
    write_json(args.out_dir / "metrics_summary.json", summary)

    if not args.no_plots:
        plot_metrics(args.out_dir, summary, all_rows)

    print("Resumen:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Detalle CSV: {args.out_dir / 'metrics_detail.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
