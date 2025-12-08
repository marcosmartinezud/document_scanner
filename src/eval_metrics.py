"""Métricas básicas para OCR y geometría."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    # Implementación iterativa O(n*m) espacio O(min(n,m))
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate.

    Devuelve distancia de Levenshtein normalizada por la longitud de la referencia.
    Si la referencia está vacía, devuelve 0.0 si hyp también está vacía, si no 1.0.
    """
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = _levenshtein(list(ref), list(hyp))
    return dist / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate (por espacios)."""
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    dist = _levenshtein(ref_tokens, hyp_tokens)
    return dist / len(ref_tokens)


def mean_corner_error(gt: Iterable[Iterable[float]], pred: Iterable[Iterable[float]]) -> float:
    """Error medio (RMSE) por esquina en píxeles entre dos cuadriláteros ordenados.

    Espera el orden (tl, tr, br, bl) para ambos.
    """
    gt_list = list(gt)
    pr_list = list(pred)
    if len(gt_list) != 4 or len(pr_list) != 4:
        raise ValueError("Se esperan 4 esquinas en gt y pred")
    mse = 0.0
    for (gx, gy), (px, py) in zip(gt_list, pr_list):
        mse += (float(gx) - float(px)) ** 2 + (float(gy) - float(py)) ** 2
    return math.sqrt(mse / 4.0)


def polygon_area(points: Iterable[Iterable[float]]) -> float:
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def aspect_ratio(points: Iterable[Iterable[float]]) -> float:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return float(w) / float(h) if h != 0 else float("inf")
