"""Métricas para evaluar OCR y geometría de documentos"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """
    Calcular la distancia de Levenshtein (minimo de inserciones, borrados y sustituciones)
    entre dos secuencias
    """

    # Asegurar que a es la secuencia más larga
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1)) # Fila inicial

    # Recorrer cada caracter en a construyendo la matriz de costos
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def cer(ref: str, hyp: str) -> float:
    """
    Character Error Rate normalizado entre dos cadenas de texto
    """
    if not ref:
        return 0.0 if not hyp else 1.0 # Ref vacío
    
    dist = _levenshtein(list(ref), list(hyp)) # Convertir a listas de caracteres y calcular distancia
    return dist / len(ref) # Normalizar por la longitud de la referencia


def wer(ref: str, hyp: str) -> float:
    """
    Word Error Rate usando tokenización por espacios entre dos cadenas de texto
    """

    # Tokenizar las cadenas en palabras
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0 # Ref vacía
    
    dist = _levenshtein(ref_tokens, hyp_tokens) # Calcular distancia entre listas de palabras
    return dist / len(ref_tokens) # Normalizar por la cantidad de palabras en la referencia


def mean_corner_error(gt: Iterable[Iterable[float]], pred: Iterable[Iterable[float]]) -> float:
    """
    Calcular el error cuadrático medio entre las esquinas de dos polígonos (gt y pred)
    """

    # Convertir a listas para facilitar el acceso
    gt_list = list(gt)
    pr_list = list(pred)

    if len(gt_list) != 4 or len(pr_list) != 4:
        raise ValueError("Se esperan 4 esquinas en gt y pred") # Validar entrada
    mse = 0.0 # Inicializar error cuadrático medio
    for (gx, gy), (px, py) in zip(gt_list, pr_list): # Iterar sobre las esquinas
        mse += (float(gx) - float(px)) ** 2 + (float(gy) - float(py)) ** 2 # Acumular errores
    return math.sqrt(mse / 4.0) # Retornar la raíz cuadrada del promedio


def polygon_area(points: Iterable[Iterable[float]]) -> float:
    """
    Calcular el área de un polígono definido por una lista de puntos (x, y)
    usando la fórmula del área del polígono (shoelace formula)
    """

    pts = list(points)
    if len(pts) < 3:
        return 0.0 # No es un polígono válido
    
    area = 0.0 # Inicializar área
    for i in range(len(pts)): # Iterar sobre los puntos
        # Obtener coordenadas actuales y siguientes (con wrap-around)
        x1, y1 = pts[i] 
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1 # Acumular área parcial
    return abs(area) / 2.0 # Retornar área absoluta dividida por 2


def aspect_ratio(points: Iterable[Iterable[float]]) -> float:
    """
    Calcular la relación de aspecto (ancho/alto) de un conjunto de puntos
    """

    xs = [p[0] for p in points] # Extraer coordenadas x
    ys = [p[1] for p in points] # Extraer coordenadas y

    w = max(xs) - min(xs) # Calcular ancho
    h = max(ys) - min(ys) # Calcular alto
    return float(w) / float(h) if h != 0 else float("inf") # Retornar relación de aspecto evitando división por cero
