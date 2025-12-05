"""Rutinas de mejora visual del documento."""
import cv2
import numpy as np


def enhance_document_appearance(image: np.ndarray) -> np.ndarray:
    planes = [image] if image.ndim == 2 else list(cv2.split(image))
    kernel = np.ones((7, 7), np.uint8)

    enhanced_planes = []
    for plane in planes:
        dilated = cv2.dilate(plane, kernel)
        background = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, background)
        normalized = cv2.normalize(
            diff,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )
        enhanced_planes.append(normalized)

    if len(enhanced_planes) == 1:
        return cv2.cvtColor(enhanced_planes[0], cv2.COLOR_GRAY2BGR)
    return cv2.merge(enhanced_planes)


def whiten_near_white(image: np.ndarray, threshold: int = 235) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 3)
    result = image.copy()
    result[mask == 255] = 255
    return result
