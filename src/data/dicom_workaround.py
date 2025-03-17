"""
Script para solucionar problemas de conversión DICOM a NIFTI
"""

import sys
import logging
import argparse
import nibabel as nib
import numpy as np
import pydicom
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_dicom_series_manual(dicom_folder):
    """
    Lee una serie de archivos DICOM manualmente y construye un arreglo 3D.
    Útil cuando otras herramientas de conversión fallan.

    Args:
        dicom_folder: Ruta a la carpeta con archivos DICOM
    Returns:
        Tuple: (arreglo 3D de datos, metadatos)
    """
    dicom_folder = Path(dicom_folder)
    dcm_files = list(dicom_folder.glob("*.dcm"))
    if not dcm_files:
        dcm_files = list(dicom_folder.glob("*.DCM"))

    if not dcm_files:
        raise ValueError(f"No se encontraron archivos DICOM en {dicom_folder}")

    logger.info(f"Leyendo {len(dcm_files)} archivos DICOM")

    # Ordenar archivos por InstanceNumber para preservar el orden correcto de slices
    dcm_sorted = []
    for dcm_file in tqdm(dcm_files, desc="Leyendo archivos DICOM"):
        try:
            dcm = pydicom.dcmread(dcm_file)
            instance_num = getattr(dcm, "InstanceNumber", 0)
            dcm_sorted.append((instance_num, dcm_file, dcm))
        except Exception as e:
            logger.warning(f"Error leyendo {dcm_file}: {e}")

    dcm_sorted.sort(key=lambda x: x[0])

    # Obtener información de la primera imagen para determinar las dimensiones
    if not dcm_sorted:
        raise ValueError("No se pudieron leer archivos DICOM válidos")

    first_dcm = dcm_sorted[0][2]
    cols = getattr(first_dcm, "Columns", 0)
    rows = getattr(first_dcm, "Rows", 0)
    slices = len(dcm_sorted)

    logger.info(f"Dimensiones detectadas: {rows}x{cols}x{slices}")

    if cols <= 0 or rows <= 0 or slices <= 0:
        raise ValueError(f"Dimensiones inválidas: {rows}x{cols}x{slices}")

    # Crear arreglo 3D para almacenar los datos
    try:
        pixel_array = np.zeros((slices, rows, cols), dtype=np.float32)

        # Cargar datos de píxeles
        for i, (_, _, dcm) in enumerate(tqdm(dcm_sorted, desc="Procesando slices")):
            if hasattr(dcm, "pixel_array"):
                pixel_array[i, :, :] = dcm.pixel_array.astype(np.float32)

                # Aplicar transformación de escala si es necesario
                if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
                    pixel_array[i, :, :] = (
                        pixel_array[i, :, :] * dcm.RescaleSlope + dcm.RescaleIntercept
                    )

        # Transponer para tener el orden correcto (sagital, coronal, axial)
        pixel_array = np.transpose(pixel_array, (1, 2, 0))

        # Crear metadatos básicos
        meta = {
            "spacing": [
                1.0,
                1.0,
                1.0,
            ],  # Valor por defecto si no hay información de espaciado
            "orientation": "LAS",  # Orientación por defecto (Left, Anterior, Superior)
        }

        # Intentar extraer información de espaciado si está disponible
        if hasattr(first_dcm, "PixelSpacing"):
            meta["spacing"][0] = float(first_dcm.PixelSpacing[0])
            meta["spacing"][1] = float(first_dcm.PixelSpacing[1])

        if hasattr(first_dcm, "SliceThickness"):
            meta["spacing"][2] = float(first_dcm.SliceThickness)

        return pixel_array, meta

    except Exception as e:
        logger.error(f"Error al construir el arreglo 3D: {e}")
        raise


def save_to_nifti(pixel_array, meta, output_file):
    """
    Guarda un arreglo 3D como archivo NIfTI.

    Args:
        pixel_array: Arreglo 3D de datos
        meta: Diccionario con metadatos
        output_file: Ruta de salida para el archivo NIfTI
    """
    # Crear un objeto affine para definir la orientación y espaciado
    affine = np.eye(4)
    affine[0, 0] = meta["spacing"][0]
    affine[1, 1] = meta["spacing"][1]
    affine[2, 2] = meta["spacing"][2]

    # Crear imagen NIfTI
    nifti_img = nib.Nifti1Image(pixel_array, affine)

    # Guardar archivo
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Comprimir si el nombre termina en .gz
    if output_file.suffix != ".gz":
        output_file = output_file.with_suffix(output_file.suffix + ".gz")

    nib.save(nifti_img, output_file)
    logger.info(f"Archivo NIfTI guardado en: {output_file}")

    return output_file


def convert_problematic_dicom(dicom_folder, output_file):
    """
    Convierte una serie DICOM problemática a NIfTI usando un enfoque manual.

    Args:
        dicom_folder: Ruta a la carpeta con archivos DICOM
        output_file: Ruta de salida para el archivo NIfTI
    Returns:
        Path: Ruta al archivo NIfTI generado
    """
    try:
        pixel_array, meta = read_dicom_series_manual(dicom_folder)

        # Normalizar valores a [0, 1] para facilitar visualización
        if np.max(pixel_array) != np.min(pixel_array):
            pixel_array = (pixel_array - np.min(pixel_array)) / (
                np.max(pixel_array) - np.min(pixel_array)
            )

        output_path = save_to_nifti(pixel_array, meta, output_file)
        return output_path
    except Exception as e:
        logger.error(f"Error en la conversión manual: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convierte DICOM problemáticos a NIfTI"
    )
    parser.add_argument("--input", required=True, help="Carpeta con archivos DICOM")
    parser.add_argument(
        "--output", required=True, help="Ruta de salida para archivo NIfTI"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Mostrar información detallada"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    try:
        output_path = convert_problematic_dicom(args.input, args.output)
        if output_path:
            logger.info(f"Conversión exitosa: {output_path}")
            return 0
        else:
            logger.error("Conversión fallida")
            return 1
    except Exception as e:
        logger.error(f"Error durante la conversión: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
