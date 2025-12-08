"""Funciones para mejorar la apariencia de los documentos escaneados"""
import cv2
import numpy as np


def enhance_document_appearance(image: np.ndarray) -> np.ndarray:
    """
    Estimar y eliminar el fondo para realzar el texto y los detalles
    Procesar cada canal por separado para evitar artefactos de mezlca de canales y mantener la 
    fidelidad del color.
    Resultado: imagen con mayor contraste y claridad
    """
    planes = [image] if image.ndim == 2 else list(cv2.split(image)) # Procesar cada canal por separado
    kernel = np.ones((7, 7), np.uint8) # Kernel para dilatación (matriz de 7x7)

    enhanced_planes = [] # Lista para guardar los canales mejorados
    for plane in planes:
        dilated = cv2.dilate(plane, kernel) # Ensanchar las áreas brillantes
        background = cv2.medianBlur(dilated, 21) # Aplicar desenfoque
        diff = 255 - cv2.absdiff(plane, background) # Valores altos para texto y bajos para fondo
        normalized = cv2.normalize(
            diff,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        ) # Normalizar a rango completo
        enhanced_planes.append(normalized) #Guardar el canal mejoradoS

    # Recombinar los canales mejorados
    if len(enhanced_planes) == 1:
        return cv2.cvtColor(enhanced_planes[0], cv2.COLOR_GRAY2BGR)
    return cv2.merge(enhanced_planes)


def whiten_near_white(image: np.ndarray, threshold: int = 235) -> np.ndarray:
    """
    Convertir píxeles casi blancos a blanco puro para mejorar la apariencia
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convertir a escala de grises
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY) # Crear máscara de píxeles casi blancos
    mask = cv2.medianBlur(mask, 3) # Suavizar la máscara para evitar bordes duros
    result = image.copy() 
    result[mask == 255] = 255 # Eliminar tonos amarillentos o ruido en las zonas blancas del papel
    return result
