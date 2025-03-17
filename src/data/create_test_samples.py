"""
Script para crear muestras pequeñas de los datos ADNI para pruebas
"""

import os
import shutil
import argparse
import logging
import random
from pathlib import Path
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_dataset(source_dir, dest_dir, num_subjects=3):
    """
    Crea un conjunto de datos de muestra copiando un número limitado de sujetos

    Args:
        source_dir (str): Directorio fuente con datos ADNI completos
        dest_dir (str): Directorio destino para muestras
        num_subjects (int): Número de sujetos a incluir en la muestra
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Encontrar directorios de sujetos
    subject_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    logger.info(
        f"Encontrados {len(subject_dirs)} directorios de sujetos en {source_dir}"
    )

    if not subject_dirs:
        logger.error("No se encontraron directorios de sujetos")
        return

    # Seleccionar aleatoriamente algunos sujetos
    if len(subject_dirs) <= num_subjects:
        selected_subjects = subject_dirs
    else:
        # Intentar tomar al menos un sujeto de cada diagnóstico si es posible
        # (esto requiere conocimiento previo de los datos)
        selected_subjects = random.sample(subject_dirs, num_subjects)

    # Copiar datos de sujetos seleccionados
    for subject_dir in tqdm(selected_subjects, desc="Copiando sujetos de muestra"):
        dest_subject_dir = dest_path / subject_dir.name

        # Crear directorios para MRI y PET si existen en el origen
        mri_dirs = list((subject_dir).glob("*MPRAGE*"))
        if mri_dirs:
            mri_src = random.choice(mri_dirs)
            mri_dest = dest_subject_dir / mri_src.name
            mri_dest.mkdir(parents=True, exist_ok=True)

            # Encontrar una carpeta con DICOMs y copiar algunos
            for root, dirs, files in os.walk(mri_src):
                dicom_files = [f for f in files if f.lower().endswith(".dcm")]
                if dicom_files:
                    # Copiar hasta 20 archivos DICOM
                    sample_dicoms = dicom_files[: min(20, len(dicom_files))]
                    root_path = Path(root)
                    relative_path = root_path.relative_to(subject_dir)
                    dest_dicom_dir = dest_subject_dir / relative_path
                    dest_dicom_dir.mkdir(parents=True, exist_ok=True)

                    for dicom in sample_dicoms:
                        src_file = root_path / dicom
                        dst_file = dest_dicom_dir / dicom
                        shutil.copy2(src_file, dst_file)
                    break  # Solo necesitamos una carpeta con DICOMs

        # Similar para PET
        pet_dirs = list((subject_dir).glob("*PET*"))
        if pet_dirs:
            pet_src = random.choice(pet_dirs)
            pet_dest = dest_subject_dir / pet_src.name
            pet_dest.mkdir(parents=True, exist_ok=True)

            # Similar proceso de copia que para MRI
            for root, dirs, files in os.walk(pet_src):
                dicom_files = [f for f in files if f.lower().endswith(".dcm")]
                if dicom_files:
                    sample_dicoms = dicom_files[: min(20, len(dicom_files))]
                    root_path = Path(root)
                    relative_path = root_path.relative_to(subject_dir)
                    dest_dicom_dir = dest_subject_dir / relative_path
                    dest_dicom_dir.mkdir(parents=True, exist_ok=True)

                    for dicom in sample_dicoms:
                        src_file = root_path / dicom
                        dst_file = dest_dicom_dir / dicom
                        shutil.copy2(src_file, dst_file)
                    break

    logger.info(
        f"Creado conjunto de datos de muestra con {len(selected_subjects)} sujetos en {dest_dir}"
    )


def main():
    """Función principal para ejecución en línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Crear muestra de datos ADNI para pruebas"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Directorio fuente con datos ADNI completos",
    )
    parser.add_argument(
        "--destination",
        type=str,
        required=True,
        help="Directorio destino para muestras",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        default=3,
        help="Número de sujetos a incluir en la muestra",
    )

    args = parser.parse_args()
    create_sample_dataset(args.source, args.destination, args.subjects)


if __name__ == "__main__":
    main()
