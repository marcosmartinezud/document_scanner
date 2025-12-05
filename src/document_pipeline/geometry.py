"""Funciones relacionadas con el contorno y la geometría del documento."""
from typing import Tuple

import cv2
import numpy as np


def preprocess(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    enhanced_gray_f = enhanced_gray.astype(np.float32)
    background = cv2.GaussianBlur(enhanced_gray_f, (0, 0), 21)
    normalized = cv2.normalize(enhanced_gray_f - background, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)

    denoised = cv2.bilateralFilter(normalized, d=9, sigmaColor=75, sigmaSpace=75)
    edges_canny = cv2.Canny(denoised, 40, 120)

    adaptive = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        7,
    )
    adaptive = cv2.medianBlur(adaptive, 5)
    doc_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    doc_mask = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, doc_kernel, iterations=2)
    mask_edges = cv2.morphologyEx(doc_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    # Merge structural edges from the mask with Canny to avoid gaps on weak borders.
    edges = cv2.bitwise_or(edges_canny, mask_edges)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return gray, edges


def detect_document_contour(gray: np.ndarray, edges: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No se encontraron contornos con los parámetros actuales.")

    height, width = gray.shape[:2]
    image_area = float(height * width)
    border_margin = max(5, int(0.01 * min(height, width)))

    def touches_image_border(contour: np.ndarray) -> bool:
        x, y, w, h = cv2.boundingRect(contour)
        return (
            x <= border_margin
            or y <= border_margin
            or (x + w) >= (width - border_margin)
            or (y + h) >= (height - border_margin)
        )

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    candidate = None
    max_area = 0.0

    for contour in contours_sorted:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        if touches_image_border(approx):
            continue
        area = cv2.contourArea(approx)
        if area < 0.05 * image_area:
            continue
        if not cv2.isContourConvex(approx):
            approx = cv2.convexHull(approx)
            if len(approx) != 4 or touches_image_border(approx):
                continue
            area = cv2.contourArea(approx)
        if area > max_area:
            max_area = area
            candidate = approx

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

    if candidate is None:
        largest_contour = contours_sorted[0]
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        candidate = box

    return order_corners(candidate.reshape(4, 2))


def order_corners(points: np.ndarray) -> np.ndarray:
    if points.shape != (4, 2):
        raise ValueError("Se esperaban cuatro puntos 2D para el contorno.")

    pts_sorted = points[np.argsort(points[:, 0])]
    left = pts_sorted[:2]
    right = pts_sorted[2:]

    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    top_left, bottom_left = left
    top_right, bottom_right = right

    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return ordered


def warp_perspective(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    if contour.shape != (4, 2):
        raise ValueError("El contorno debe contener cuatro puntos ordenados.")

    tl, tr, br, bl = contour.astype(np.float32)

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(round(max(width_top, width_bottom)))

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = int(round(max(height_right, height_left)))

    if max_width == 0 or max_height == 0:
        raise ValueError("El contorno es degenerado; no se puede calcular la perspectiva.")

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(contour.astype(np.float32), destination)
    return cv2.warpPerspective(image, transform, (max_width, max_height))
