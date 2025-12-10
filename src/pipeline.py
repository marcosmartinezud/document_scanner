"""CLI para el pipeline de documentos"""
from argparse import ArgumentParser
from pathlib import Path
import logging

from .document_pipeline import process_batch, process_document


def _build_parser() -> ArgumentParser:
    # Construir el parser
    parser = ArgumentParser(
        description="Detecta el contorno principal y genera las variantes del documento",
    )
    parser.add_argument(
        "--input",
        required=True, # es obligatorio
        help="Ruta de la imagen o carpeta con imágenes de entrada",
    )
    parser.add_argument(
        "--output-dir",
        default=None, #data/processed por defecto
        help="Directorio raíz donde guardar las salidas (por defecto, data/processed)",
    )
    return parser


def _expand_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        """Aceptar extensiones comunes de imagen y busca recursivamente
        dentro de subcarpetas ("data/raw/ocr_test/...")"""

        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        images: list[Path] = [] # lista para almacenar los paths de las imágenes
        for pattern in patterns:
            images.extend(input_path.rglob(pattern))
        return sorted(images)
    return [input_path]


def main() -> None:
    args = _build_parser().parse_args()
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    input_path = Path(args.input)
    output_root = Path(args.output_dir) if args.output_dir else None
    """
    Definir el modo por la naturaleza de "--input":
        - Si es un archivo: modo "archivo"
        - Si es una carpeta bajo "data/raw" (data/raw/scanner_test): modo "carpeta"
        - Si es la raíz "data/raw" o una carpeta que contiene varias subcarpetas: modo "todo" 
    """
    image_paths = _expand_inputs(input_path)
    if not image_paths:
        raise FileNotFoundError("No se encontraron imágenes en la ruta indicada.")

    RAW_ROOT = Path("data/raw").resolve()

    if input_path.is_file():
        # Archivo individual: determinar la subcarpeta superior (si está bajo data/raw)
        try:
            rel = input_path.resolve().relative_to(RAW_ROOT)
            top_folder = rel.parts[0] if rel.parts else None
        except Exception:
            top_folder = input_path.parent.name

        stem = input_path.stem or "entrada"
        target_dir = (output_root or Path("data/processed")) / (top_folder or input_path.parent.name) / stem

        # Decidir flags según la carpeta superior
        do_ocr_flag = True
        bin_thresh = None
        write_corners = True
        if top_folder:
            name = str(top_folder).lower()
            if name == "ocr_test":
                do_ocr_flag = True
                bin_thresh = None
            elif name == "scanner_test":
                do_ocr_flag = False
                bin_thresh = None
                write_corners = True
            # Para carpetas llamadas "ocr_test_bin" o "scanner_test_bin"
            elif name in ("scanner_test_bin", "ocr_test_bin") or name.endswith("_bin"):
                do_ocr_flag = True
                bin_thresh = 195
            else:
                write_corners = True
        else:
            write_corners = True

        contour_path, warp_path, warp_clean_path, extracted_text = process_document(
            input_path,
            target_dir,
            do_ocr=do_ocr_flag,
            binarize_threshold=bin_thresh,
            write_corners=write_corners,
        )
        # Guardar OCR en archivo si existe
        if extracted_text:
            txt_path = target_dir / f"{stem}.txt"
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(extracted_text, encoding="utf-8")
        logging.info(f"Procesado archivo: {input_path} -> {target_dir}")

    else:
        """Carpeta: si es la raíz "data/raw" procesar todo conservando subcarpetas;
        si es una subcarpeta concreta, también procesar recursivamente"""
        try:
            is_raw_root = input_path.resolve() == RAW_ROOT
        except Exception:
            is_raw_root = False

        if is_raw_root:
            # Procesar todas las imágenes bajo data/raw y preservar la subcarpeta superior
            input_root = RAW_ROOT
            image_paths = _expand_inputs(input_path)
            results = process_batch(image_paths, output_root, input_root)
        else:
            """
            Carpeta específica (data/raw/scanner_test)
            Para preservar el nombre de la subcarpeta en la salida, pasar input_root=RAW_ROOT
            """
            try:
                rel = input_path.resolve().relative_to(RAW_ROOT)
                # Si está bajo RAW_ROOT usar RAW_ROOT como input_root para preservar la carpeta
                input_root = RAW_ROOT
            except Exception:
                # Carpeta fuera de data/raw: no preservar subcarpeta superior
                input_root = None

            image_paths = _expand_inputs(input_path)
            results = process_batch(image_paths, output_root, input_root)

        # Para cada resultado: guardar .txt cuando haya texto OCR
        for (_contour_path, _warp_path, warp_clean_path, extracted_text) in results:
            target_dir = warp_clean_path.parent
            stem = target_dir.name
            if extracted_text:
                txt_path = target_dir / f"{stem}.txt"
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text(extracted_text, encoding="utf-8")
            logging.info(f"Procesado: {target_dir}")


if __name__ == "__main__":
    main()
