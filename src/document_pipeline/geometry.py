"""Funciones para procesar la geometría de los documentos escaneados"""
from typing import Tuple

import cv2
import numpy as np


def preprocess(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convertir a escala de grises

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Convertir a espacio de color LAB
    l_channel, a_channel, b_channel = cv2.split(lab) # Dividir en canales L, A, B
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # Crear objeto CLAHE
    l_enhanced = clahe.apply(l_channel) # Aplicar CLAHE al canal L para mejorar el contraste
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel)) # Recombinar canales LAB
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR) # Convertir de vuelta a BGR

    enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY) # Convertir a escala de grises
    enhanced_gray_f = enhanced_gray.astype(np.float32) # Convertir a float para blur/normalización
    background = cv2.GaussianBlur(enhanced_gray_f, (0, 0), 21) # Difuminar detalles
    normalized = cv2.normalize(enhanced_gray_f - background, None, 0, 255, cv2.NORM_MINMAX) # Resalta las variaciones locales
    normalized = normalized.astype(np.uint8) # Convertir de vuelta a uint8

    denoised = cv2.bilateralFilter(normalized, d=9, sigmaColor=75, sigmaSpace=75) # Reducir ruido manteniendo bordes
    edges_canny = cv2.Canny(denoised, 40, 120) # Detectar bordes con Canny

    # Crear máscara del documento usando umbral adaptativo y morfología
    adaptive = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        7,
    )
    adaptive = cv2.medianBlur(adaptive, 5) # Suavizar la máscara
    doc_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) # Kernel 7x7 para morfología
    doc_mask = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, doc_kernel, iterations=2) # Cerrar huecos
    mask_edges = cv2.morphologyEx(doc_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) # Extraer bordes de la máscara

    # Combinar bordes de Canny y bordes de la máscara
    edges = cv2.bitwise_or(edges_canny, mask_edges)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1) # Cerrar huecos en los bordes
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) # Engrosar los bordes
    return gray, edges


def detect_document_contour(gray: np.ndarray, edges: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # Encontrar contornos en la imagen de bordes
    if not contours:
        raise ValueError("No se encontraron contornos con los parámetros actuales")

    height, width = gray.shape[:2] # Obtener dimensiones de la imagen
    image_area = float(height * width) # Calcular área de la imagen
    border_margin = max(5, int(0.01 * min(height, width))) # Definir margen para bordes de imagen

    def touches_image_border(contour: np.ndarray) -> bool:
        """
        Verifica si un contorno toca el borde de la imagen
        """
        x, y, w, h = cv2.boundingRect(contour) # Obtener rectángulo delimitador
        return (
            x <= border_margin
            or y <= border_margin
            or (x + w) >= (width - border_margin)
            or (y + h) >= (height - border_margin)
        )

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True) # Ordenar contornos por área (el mayor suele ser el documento)

    # Iniciar la busqueda del mejor contorno candidato
    candidate = None
    max_area = 0.0

    for contour in contours_sorted:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True) # Aproximar contorno a polígono
        if len(approx) != 4: # Solo considerar polígonos con 4 vértices
            continue
        if touches_image_border(approx): # Descarta si toca el borde de la imagen
            continue

        # Ignorar áreas demasiado pequeñas
        area = cv2.contourArea(approx)
        if area < 0.05 * image_area:
            continue

        # Asegurar que el contorno sea convexo
        if not cv2.isContourConvex(approx):
            approx = cv2.convexHull(approx)
            if len(approx) != 4 or touches_image_border(approx):
                continue
            area = cv2.contourArea(approx)
        
        # Seleccionar el contorno con el área más grande
        if area > max_area:
            max_area = area
            candidate = approx

    # Si no se encuentra un contorno adecuado, buscar un rectángulo mínimo
    if candidate is None:
        for contour in contours_sorted:
            if touches_image_border(contour):
                continue
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box_area = cv2.contourArea(box.astype(np.float32))
            if 0.05 * image_area <= box_area <= 0.95 * image_area:
                candidate = box
                break

    # Si aún no se encuentra, usar el contorno más grande
    if candidate is None:
        largest_contour = contours_sorted[0]
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        candidate = box

    return order_corners(candidate.reshape(4, 2))


def order_corners(points: np.ndarray) -> np.ndarray:
    if points.shape != (4, 2):
        raise ValueError("Se esperaban cuatro puntos 2D para el contorno")

    pts_sorted = points[np.argsort(points[:, 0])] # Ordenar por coordenada x de izquierda a derecha
    left = pts_sorted[:2] # Puntos izquierdos
    right = pts_sorted[2:] # Puntos derechos

    left = left[np.argsort(left[:, 1])] # Ordenar puntos izquierdos por coordenada y
    right = right[np.argsort(right[:, 1])] # Ordenar puntos derechos por coordenada y

    # Asignar puntos en orden: superior izquierdo, superior derecho, inferior derecho, inferior izquierdo
    top_left, bottom_left = left
    top_right, bottom_right = right

    # Crear array ordenado de puntos
    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return ordered


def warp_perspective(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    if contour.shape != (4, 2):
        raise ValueError("El contorno debe contener cuatro puntos ordenados")

    tl, tr, br, bl = contour.astype(np.float32) # Extraer puntos de las esquinas

    # Calcular ancho superior e inferior (distancia euclidiana)
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    max_width = int(round(max(width_top, width_bottom))) # Ancho máximo del documento

    # Calcular altura derecha e izquierda (distancia euclidiana)
    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)

    # Calcular altura máxima del documento
    max_height = int(round(max(height_right, height_left)))

    # Revisar que no sean cero
    if max_width == 0 or max_height == 0:
        raise ValueError("El contorno es degenerado; no se puede calcular la perspectiva")

    # Definir puntos de destino para la transformación
    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    # Calcular la matriz de transformación y aplicar la transformación de perspectiva
    transform = cv2.getPerspectiveTransform(contour.astype(np.float32), destination)
    return cv2.warpPerspective(image, transform, (max_width, max_height))
