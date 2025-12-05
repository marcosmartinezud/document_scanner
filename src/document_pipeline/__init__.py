"""API público para el procesamiento del documento."""
from .processing import process_batch, process_document

__all__ = ["process_document", "process_batch"]
