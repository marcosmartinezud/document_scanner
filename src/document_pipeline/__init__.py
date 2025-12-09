"""API público para el procesamiento del documento"""
from .processing import process_batch, process_document
from .ocr import get_ocr_reader, extract_text

__all__ = ["process_document", "process_batch", "get_ocr_reader", "extract_text"]
