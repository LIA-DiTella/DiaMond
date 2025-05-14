"""
Módulo para convertir archivos DICOM de MRI y PET a formato NIFTI
Basado en el trabajo de Hugo Alberto Massaroli
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import dicom2nifti
from dicom2nifti.exceptions import ConversionError
import argparse
import datetime
import json
import logging
import re
from tqdm import tqdm
import pydicom
import warnings
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)

# Añadir importación para archivos ECAT
try:
    import nibabel.ecat as ecat
except ImportError:
    ecat = None
    logger.warning(
        "Módulo nibabel.ecat no disponible. La conversión de archivos ECAT (.v) no será posible."
    )

# Filtrar advertencias específicas de PyDICOM sobre valores inválidos de UI
warnings.filterwarnings("ignore", message="Invalid value for VR UI")

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Silenciar loggers externos demasiado verbosos por defecto
logging.getLogger("dicom2nifti").setLevel(logging.ERROR)
logging.getLogger("pydicom").setLevel(logging.ERROR)
# Silenciar otros loggers verbosos
logging.getLogger("nibabel").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)


def set_global_log_level(verbose=False):
    """
    Configura el nivel de logging global basado en la verbosidad

    Parameters:
        verbose (bool): Si es True, mostrar logs detallados
    """
    log_level = logging.INFO if verbose else logging.WARNING

    # Configurar logger principal
    logger.setLevel(log_level)

    # Configurar loggers de bibliotecas externas
    logging.getLogger("dicom2nifti").setLevel(logging.ERROR)
    logging.getLogger("pydicom").setLevel(logging.ERROR)
    logging.getLogger("nibabel").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)

    # Configurar handler de la raíz para afectar a todos los loggers
    root_logger = logging.getLogger()
    if not verbose:
        root_logger.setLevel(logging.ERROR)
    else:
        root_logger.setLevel(logging.INFO)


# Actualizar patrones para identificar tipos de escaneo específicos en ADNI
MRI_PATTERNS = {
    "mprage": ["mp-rage", "mprage", "t1", "structural", "accelerated_sagittal_mprage"],
    "localizer": ["localizer"],
    "pd_t2": ["pd_t2", "fse", "t2", "flair", "sagittal_3d_flair"],
    "b1_cal": ["b1", "cal", "b1-calibration"],
    "perfusion": ["perfusion", "pasl", "asl"],
}

PET_PATTERNS = {
    "fdg": [
        "fdg",
        "fluorodeoxyglucose",
        "fluoro-2-deoxy-d-glucose",
        "3d_fdg",
        "pet_wb",
    ],
    "av45": ["av45", "florbetapir", "amyvid"],
    "pib": ["pib", "pittsburgh compound b"],
    "fbb": ["fbb", "florbetaben", "neuraceq"],
}


def identify_scan_folders(base_path, scan_type="mri"):
    """
    Identificar y categorizar carpetas de escaneo ADNI

    Parameters:
        base_path (str): Ruta base que contiene las carpetas ADNI
        scan_type (str): Tipo de escaneo ('mri' o 'pet')
    Returns:
        dict: Diccionario de tipos de escaneo y sus rutas
    """
    patterns = MRI_PATTERNS if scan_type.lower() == "mri" else PET_PATTERNS
    scan_types = {category: [] for category in patterns.keys()}

    # Buscar en la estructura anidada de ADNI
    base_path = Path(base_path)

    # Si es un directorio de sujeto (ej: 003_S_1059), explorar dentro
    if base_path.is_dir():
        # Buscar archivos ECAT (.v) para PET directamente en el directorio base
        if scan_type.lower() == "pet":
            ecat_files = list(base_path.glob("*.v")) + list(base_path.glob("*.V"))
            if ecat_files:
                # Asignar archivos ECAT a la categoría 'fdg' por defecto (se puede refinar más adelante)
                for ecat_file in ecat_files:
                    scan_types["fdg"].append(ecat_file.parent)

        # Buscar carpetas de modalidades dentro de cada sujeto
        for scan_folder in base_path.iterdir():
            if scan_folder.is_dir():
                folder_name = scan_folder.name.lower()

                # Verificar si es un tipo de escaneo relevante según modalidad
                for category, keywords in patterns.items():
                    if any(keyword in folder_name for keyword in keywords):
                        # Buscar la carpeta más profunda que contenga archivos DICOM
                        dicom_folder = find_dicom_folder(scan_folder)
                        if dicom_folder is not None:
                            scan_types[category].append(dicom_folder)
                            break

    return scan_types


def find_dicom_folder(start_path):
    """
    Encuentra la carpeta que contiene archivos DICOM en la estructura anidada.

    Parameters:
        start_path (Path): Ruta inicial para la búsqueda
    Returns:
        Path: Ruta a la carpeta que contiene archivos DICOM, o None
    """
    # Si esta carpeta contiene directamente archivos DICOM, devolverla
    if list(start_path.glob("*.dcm")) or list(start_path.glob("*.DCM")):
        return start_path

    # Comprobar si la carpeta contiene archivos ECAT (.v) para PET
    if list(start_path.glob("*.v")) or list(start_path.glob("*.V")):
        return start_path

    # Si no, buscar en sus subcarpetas
    for subfolder in start_path.iterdir():
        if subfolder.is_dir():
            # Si hay una estructura típica de fecha/ID, buscar ahí
            if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}", subfolder.name):
                for id_folder in subfolder.iterdir():
                    if id_folder.is_dir() and id_folder.name.startswith("I"):
                        if list(id_folder.glob("*.dcm")) or list(
                            id_folder.glob("*.DCM")
                        ):
                            return id_folder

            # Recursivamente buscar en otras subcarpetas
            result = find_dicom_folder(subfolder)
            if result is not None:
                return result

    return None


def read_dicom_header(dicom_folder):
    """
    Leer información del encabezado DICOM para determinar modalidad y características

    Parameters:
        dicom_folder (Path): Ruta a la carpeta DICOM
    Returns:
        dict: Información relevante del encabezado
    """
    try:
        # Encontrar primer archivo DICOM
        dicom_files = list(dicom_folder.glob("*.dcm"))
        if not dicom_files:
            dicom_files = list(dicom_folder.glob("*.DCM"))

        if not dicom_files:
            return None

        # Suprimir advertencias específicas al leer el encabezado
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Invalid value for VR UI")
            # Leer encabezado
            header = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)

        return {
            "modality": getattr(header, "Modality", "Unknown"),
            "series_description": getattr(header, "SeriesDescription", "").lower(),
            "study_date": getattr(header, "StudyDate", ""),
            "num_files": len(dicom_files),
        }
    except Exception as e:
        logger.warning(f"Error al leer encabezado DICOM: {e}")
        return None


def select_best_scan(scan_folders, modality="mri"):
    """
    Determinar qué escaneo utilizar basado en calidad y características

    Parameters:
        scan_folders (list): Lista de rutas de carpetas de escaneo
        modality (str): Modalidad ('mri' o 'pet')
    Returns:
        Path: Ruta al mejor escaneo
    """
    if not scan_folders:
        return None

    # Si solo hay uno, usarlo
    if len(scan_folders) == 1:
        return scan_folders[0]

    # Criterios de clasificación para cada modalidad
    best_folder = scan_folders[0]
    best_score = 0

    for folder in scan_folders:
        header_info = read_dicom_header(folder)
        if not header_info:
            continue

        score = 0
        # Verificar si tiene un número adecuado de imágenes
        if header_info["num_files"] > 100:  # MRI suele tener muchos cortes
            score += 10

        # Verificar si es una repetición (generalmente mejores)
        if "repeat" in folder.name.lower():
            score += 5

        # Específico para MRI - T1 ponderado
        if modality == "mri" and "t1" in header_info["series_description"]:
            score += 3

        # Específico para PET - FDG completo
        if modality == "pet" and "fdg" in header_info["series_description"]:
            score += 3

        if score > best_score:
            best_score = score
            best_folder = folder

    return best_folder


def convert_dicoms_to_nifti(dicom_folder, output_dir, verbose=False):
    """
    Convertir serie DICOM a formato NIfTI

    Parameters:
        dicom_folder (Path): Ruta a la carpeta DICOM
        output_dir (Path): Directorio de salida para el archivo NIfTI
        verbose (bool): Si es True, mostrar logs detallados
    Returns:
        Path: Ruta al archivo NIfTI convertido
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verificar si hay archivos ECAT (.v) en la carpeta
    ecat_files = list(dicom_folder.glob("*.v")) + list(dicom_folder.glob("*.V"))
    if ecat_files and ecat is not None:
        try:
            # Usar el primero si hay varios
            ecat_file = ecat_files[0]
            if verbose:
                logger.info(f"Convirtiendo archivo ECAT: {ecat_file}")

            # Leer archivo ECAT usando nibabel
            ecat_img = ecat.load(str(ecat_file))

            # Extraer la imagen principal usando get_fdata() en lugar de get_data()
            img = np.asanyarray(ecat_img.dataobj)  # Recomendado por nibabel

            # Convertir a NIfTI
            output_filename = output_dir / f"{ecat_file.stem}.nii.gz"
            nifti_img = nib.Nifti1Image(img, ecat_img.affine)
            nib.save(nifti_img, str(output_filename))

            if verbose:
                logger.info(f"Archivo ECAT convertido a NIFTI: {output_filename}")

            return output_filename

        except Exception as e:
            logger.error(f"Error al convertir archivo ECAT: {e}")
            # No lanzar la excepción aquí, continuamos con los otros métodos

    # Establecer nivel de logging temporalmente según verbosidad
    old_dicom2nifti_level = logging.getLogger("dicom2nifti").level
    logging.getLogger("dicom2nifti").setLevel(
        logging.ERROR
    )  # Siempre en ERROR para reducir logs

    # Métodos de conversión por orden de preferencia
    conversion_methods = [
        "dcm2niix",  # Método más robusto: usar dcm2niix si está disponible
        "dicom2nifti_patched",  # Método con parche para RepetitionTime
        "dicom2nifti_workaround",  # Método alternativo con configuración modificada
        "nibabel_direct",  # Método directo usando nibabel como último recurso
    ]

    nifti_file = None
    conversion_errors = []

    for method in conversion_methods:
        if nifti_file is not None:
            break

        try:
            if method == "dicom2nifti_patched":
                # Aplicar parche a la función set_tr_te para manejar la falta de RepetitionTime/EchoTime
                try:
                    import dicom2nifti.common as common

                    # Guardar funciones originales para restaurarlas después
                    original_set_tr_te = common.set_tr_te
                    original_create_affine = common.create_affine

                    # Definir una función de parche que maneje la falta de TR/TE
                    def patched_set_tr_te(nii_image, repetition_time, echo_time):
                        if nii_image.header:
                            # Usar valores predeterminados si no existen
                            tr = 0
                            if repetition_time is not None:
                                try:
                                    tr = float(repetition_time)
                                except (ValueError, TypeError):
                                    pass

                            if echo_time is not None:
                                try:
                                    _ = float(echo_time)
                                except (ValueError, TypeError):
                                    pass

                            nii_image.header.set_xyzt_units(xyz=2, t=8)  # 2=mm, 8=sec
                            # Establecer TR en pixdim[4]
                            nii_image.header["pixdim"][4] = (
                                tr / 1000.0
                            )  # Convertir de ms a s

                    # Definir una función de parche para manejar el error NOT_A_VOLUME
                    def patched_create_affine(dicom_input):
                        try:
                            return original_create_affine(dicom_input)
                        except ConversionError as e:
                            if "NOT_A_VOLUME" in str(e):
                                if verbose:
                                    logger.warning(
                                        "Detectada estructura no volumétrica. Intentando creación de matriz afín alternativa."
                                    )

                                # Crear una matriz afín básica usando información de pixel spacing y orientation
                                dicom_input[0].ImageOrientationPatient
                                spacing = dicom_input[0].PixelSpacing

                                # Matriz básica
                                affine = np.eye(4)
                                # Pixel spacing en los primeros dos elementos
                                affine[0, 0] = spacing[0]
                                affine[1, 1] = spacing[1]

                                # Estimar el tercer elemento (espesor de corte)
                                try:
                                    # Intentar obtener el espesor del corte
                                    if hasattr(dicom_input[0], "SliceThickness"):
                                        slice_thickness = dicom_input[0].SliceThickness
                                    else:
                                        slice_thickness = (
                                            1.0  # Valor predeterminado conservador
                                        )

                                    affine[2, 2] = slice_thickness
                                except Exception as e:
                                    # Si falla, usar un valor predeterminado
                                    affine[2, 2] = 1.0

                                return affine
                            else:
                                # Si es otro tipo de error, propagarlo
                                raise

                    # Aplicar los parches
                    common.set_tr_te = patched_set_tr_te
                    common.create_affine = patched_create_affine

                    try:
                        # Ejecutar la conversión con el parche aplicado
                        dicom2nifti.convert_directory(
                            str(dicom_folder),
                            str(output_dir),
                            compression=True,
                            reorient=True,
                        )

                        # Si llegamos aquí, la conversión fue exitosa
                        nifti_files = list(output_dir.glob("*.nii.gz"))
                        if nifti_files:
                            nifti_file = nifti_files[0]
                            if verbose:
                                logger.info(
                                    f"Conversión exitosa con dicom2nifti_patched: {nifti_file}"
                                )
                            break

                    finally:
                        # Restaurar las funciones originales
                        common.set_tr_te = original_set_tr_te
                        common.create_affine = original_create_affine

                except Exception as e:
                    # Capturar errores específicos del parche
                    error_msg = str(e)
                    conversion_errors.append(f"dicom2nifti_patched: {error_msg}")
                    if verbose:
                        logger.warning(f"Error con dicom2nifti_patched: {error_msg}")

                    raise error_msg

            elif method == "dicom2nifti_workaround":
                # Intentar conversión con dicom2nifti pero usando un enfoque modificado
                try:
                    import dicom2nifti.settings as settings

                    # Modificar configuración para evitar el problema de RepetitionTime
                    old_validate = settings.validate_slicecount
                    old_validate_orientation = settings.validate_orientation
                    old_validate_orthogonal = settings.validate_orthogonal

                    # Desactivar validaciones que pueden causar problemas
                    settings.disable_validate_slicecount()
                    settings.disable_validate_orientation()
                    settings.disable_validate_orthogonal()

                    try:
                        dicom2nifti.convert_directory(
                            str(dicom_folder),
                            str(output_dir),
                            compression=True,
                            reorient=True,
                        )
                    finally:
                        # Restaurar configuración original
                        settings.validate_slicecount = old_validate
                        settings.validate_orientation = old_validate_orientation
                        settings.validate_orthogonal = old_validate_orthogonal

                    # Si llegamos aquí, la conversión fue exitosa
                    nifti_files = list(output_dir.glob("*.nii.gz"))
                    if nifti_files:
                        nifti_file = nifti_files[0]
                        if verbose:
                            logger.info(
                                f"Conversión exitosa con dicom2nifti_workaround: {nifti_file}"
                            )
                        break

                except AttributeError as e:
                    # Error específico de RepetitionTime o EchoTime faltantes
                    error_msg = str(e)
                    conversion_errors.append(f"dicom2nifti_workaround: {error_msg}")
                    if verbose:
                        logger.warning(f"Error con dicom2nifti_workaround: {error_msg}")
                    continue

            elif method == "dcm2niix":
                # Método alternativo usando dcm2niix
                nifti_file = convert_with_dcm2niix(dicom_folder, output_dir, verbose)
                if nifti_file:
                    if verbose:
                        logger.info(f"Conversión exitosa con dcm2niix: {nifti_file}")
                    break
                else:
                    conversion_errors.append(
                        "dcm2niix no pudo convertir los archivos o no está instalado"
                    )

            elif method == "nibabel_direct":
                # Último recurso: intento directo con nibabel
                try:
                    if verbose:
                        logger.info("Intentando conversión directa con nibabel...")

                    # Buscar todos los archivos DICOM
                    dicom_files = list(dicom_folder.glob("*.dcm")) + list(
                        dicom_folder.glob("*.DCM")
                    )

                    if not dicom_files:
                        conversion_errors.append(
                            "No se encontraron archivos DICOM para conversión directa"
                        )
                        continue

                    # Intentar cargar el primer archivo para verificar
                    try:
                        test_dicom = pydicom.dcmread(str(dicom_files[0]))

                        # Verificar si es una imagen compatible
                        if not hasattr(test_dicom, "pixel_array"):
                            conversion_errors.append(
                                "El archivo DICOM no contiene datos de imagen"
                            )
                            continue

                        # Obtener información básica
                        rows = test_dicom.Rows
                        cols = test_dicom.Columns

                        if verbose:
                            logger.info(f"Dimensiones de imagen DICOM: {rows}x{cols}")

                        # Si solo hay un archivo, crear un volumen 2D
                        if len(dicom_files) == 1:
                            pixel_data = test_dicom.pixel_array

                            # Crear un volumen simple con una dimensión Z=1
                            volume = np.expand_dims(pixel_data, axis=0)

                            # Crear matriz afín básica
                            affine = np.eye(4)

                            # Aplicar espaciado de píxeles si está disponible
                            if hasattr(test_dicom, "PixelSpacing"):
                                spacing = test_dicom.PixelSpacing
                                affine[0, 0] = spacing[0]
                                affine[1, 1] = spacing[1]

                            # Guardar como NIfTI
                            output_filename = output_dir / "direct_conversion.nii.gz"
                            nifti_img = nib.Nifti1Image(volume, affine)
                            nib.save(nifti_img, str(output_filename))
                            nifti_file = output_filename

                            if verbose:
                                logger.info(f"Conversión directa exitosa: {nifti_file}")
                            break
                        elif len(dicom_files) > 1:
                            # Crear un volumen 3D
                            pixel_data = np.stack(
                                [pydicom.dcmread(str(f)).pixel_array for f in dicom_files]
                            )

                            # Crear matriz afín básica
                            affine = np.eye(4)

                            # Aplicar espaciado de píxeles si está disponible
                            if hasattr(test_dicom, "PixelSpacing"):
                                spacing = test_dicom.PixelSpacing
                                affine[0, 0] = spacing[0]
                                affine[1, 1] = spacing[1]

                            # Guardar como NIfTI
                            output_filename = output_dir / "direct_conversion.nii.gz"
                            nifti_img = nib.Nifti1Image(pixel_data, affine)
                            nib.save(nifti_img, str(output_filename))
                            nifti_file = output_filename

                            if verbose:
                                logger.info(f"Conversión directa exitosa: {nifti_file}")
                            break

                    except Exception as e:
                        conversion_errors.append(
                            f"Error al cargar DICOM con nibabel: {str(e)}"
                        )
                        continue

                except Exception as e:
                    conversion_errors.append(f"Error en conversión directa: {str(e)}")
                    continue

        except Exception as e:
            error_msg = f"Error con método {method}: {str(e)}"
            conversion_errors.append(error_msg)
            if verbose:
                logger.warning(error_msg)

            # No propagamos la excepción, continuamos con el siguiente método

    # Restaurar nivel de logging
    logging.getLogger("dicom2nifti").setLevel(old_dicom2nifti_level)

    # Verificar si la conversión fue exitosa
    if nifti_file is None:
        error_summary = "\n".join(conversion_errors)
        logger.error(
            "Todos los métodos de conversión fallaron. Se requiere la instalación de dcm2niix para un mejor soporte."
        )
        logger.error(
            "Para instalar dcm2niix, ejecute: conda install -c conda-forge dcm2niix"
        )
        logger.error(
            "O visite: https://github.com/rordenlab/dcm2niix para instrucciones de instalación"
        )
        logger.error(f"Detalles del error:\n{error_summary}")
        raise RuntimeError(
            f"Todos los métodos de conversión fallaron, instale dcm2niix para un mejor soporte.\nDetalles: {error_summary}"
        )

    return nifti_file


def convert_with_dcm2niix(dicom_folder, output_dir, verbose=False):
    """
    Convertir DICOM a NIFTI usando dcm2niix (método alternativo)

    Parameters:
        dicom_folder (Path): Ruta a la carpeta DICOM
        output_dir (Path): Directorio de salida para el archivo NIfTI
        verbose (bool): Si es True, mostrar logs detallados
    Returns:
        Path: Ruta al archivo NIfTI convertido o None si falla
    """
    # Verificar si hay archivos ECAT (.v) en la carpeta
    ecat_files = list(dicom_folder.glob("*.v")) + list(dicom_folder.glob("*.V"))
    if ecat_files:
        # dcm2niix puede manejar archivos ECAT directamente en versiones recientes
        # verificar si la versión de dcm2niix soporta archivos .v
        try:
            # Crear un directorio temporal para evitar mezclarse con otros archivos
            with tempfile.TemporaryDirectory() as temp_dir:
                # Ejecutar dcm2niix con opciones específicas para ECAT
                cmd = [
                    "dcm2niix",
                    "-z",
                    "y",  # comprimir output
                    "-f",
                    "%p_%s",  # formato de nombre: paciente_serie
                    "-v",
                    "n",  # no verbose
                    "-o",
                    temp_dir,  # directorio de output
                    str(ecat_files[0]),  # archivo ECAT de input
                ]

                if verbose:
                    logger.info(f"Ejecutando dcm2niix para ECAT: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
                )

                if result.returncode == 0:
                    # Buscar archivos .nii.gz creados
                    nifti_files = list(Path(temp_dir).glob("*.nii.gz"))
                    if nifti_files:
                        # Mover el archivo .nii.gz al directorio de salida
                        dest_file = Path(output_dir) / nifti_files[0].name
                        shutil.copy2(nifti_files[0], dest_file)
                        return dest_file
                    else:
                        if verbose:
                            logger.warning(
                                "dcm2niix no generó archivos NIFTI para ECAT"
                            )
        except Exception as e:
            if verbose:
                logger.error(f"Error al usar dcm2niix para ECAT: {e}")

    try:
        # Verificar si dcm2niix está instalado
        try:
            result = subprocess.run(
                ["dcm2niix", "-v"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            if result.returncode != 0:
                if verbose:
                    logger.warning("dcm2niix no está disponible en el sistema")
                    logger.info(
                        "Para instalar dcm2niix, ejecute: conda install -c conda-forge dcm2niix"
                    )
                return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            if verbose:
                logger.warning("dcm2niix no está disponible en el sistema")
                logger.info(
                    "Para instalar dcm2niix, ejecute: conda install -c conda-forge dcm2niix"
                )
            return None

        # Crear un directorio temporal para evitar mezclarse con otros archivos
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ejecutar dcm2niix con opciones que lo hacen más tolerante
            cmd = [
                "dcm2niix",
                "-z",
                "y",  # comprimir output
                "-f",
                "%p_%s",  # formato de nombre: paciente_serie
                "-m",
                "y",  # fusionar 2D slices
                "-v",
                "n",  # no verbose
                "-o",
                temp_dir,  # directorio de output
                str(dicom_folder),  # directorio de input
            ]

            if verbose:
                logger.info(f"Ejecutando: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
            )

            if result.returncode == 0:
                # Buscar archivos .nii.gz creados
                nifti_files = list(Path(temp_dir).glob("*.nii.gz"))
                if not nifti_files:
                    if verbose:
                        logger.warning("dcm2niix no generó archivos NIFTI")
                    return None

                # Mover el primer archivo .nii.gz al directorio de salida
                dest_file = Path(output_dir) / nifti_files[0].name
                shutil.copy2(nifti_files[0], dest_file)
                return dest_file
            else:
                if verbose:
                    logger.warning(f"dcm2niix falló: {result.stderr.decode('utf-8')}")
                return None

    except Exception as e:
        if verbose:
            logger.error(f"Error al usar dcm2niix: {e}")
        return None


def verify_nifti_quality(nifti_path):
    """
    Realizar verificaciones básicas de calidad en el NIfTI convertido

    Parameters:
        nifti_path (Path): Ruta al archivo NIfTI
    Returns:
        bool: Si el archivo pasa las verificaciones de calidad
    """
    try:
        img = nib.load(str(nifti_path))
        data = img.get_fdata()

        # Verificaciones básicas
        if data.size == 0:
            logger.warning("Imagen vacía (tamaño 0)")
            return False
        if np.all(data == 0):
            logger.warning("Imagen solo contiene ceros")
            return False
        if np.any(np.isnan(data)):
            logger.warning("Imagen contiene valores NaN")
            return False

        # Verificar dimensiones - ser más tolerante (algunos escaneos son más pequeños)
        # if (
        #     min(data.shape) < 10
        # ):  # Dimensión mínima razonable para cualquier imagen médica
        #     logger.warning(f"Dimensiones de imagen muy pequeñas: {data.shape}")
        #     return False

        # # Si la imagen es pequeña pero aún razonable, solo advertir pero aceptarla
        # if min(data.shape) < 100:
        #     logger.warning(
        #         f"Dimensiones de imagen inusuales pero aceptables: {data.shape}"
        #     )

        # Verificar rango de valores - debe haber contraste en la imagen
        data_range = np.max(data) - np.min(data)
        if data_range < 1e-6:  # Rango muy pequeño indica imagen plana
            logger.warning(f"Imagen con muy poco contraste (rango={data_range})")
            return False

        return True
    except Exception as e:
        logger.error(f"Error al verificar calidad del NIfTI: {e}")
        return False


def process_scan(base_path, output_dir, modality="mri", verbose=False):
    """
    Función principal para organizar y convertir escaneos

    Parameters:
        base_path (str): Ruta a las carpetas de escaneo
        output_dir (str): Directorio de salida para los archivos convertidos
        modality (str): Modalidad ('mri' o 'pet')
        verbose (bool): Si es True, mostrar logs detallados
    Returns:
        dict: Rutas a los archivos convertidos o None si falló
    """
    try:
        base_path = Path(base_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extraer ID de sujeto del nombre de la carpeta
        subject_id = base_path.name  # Ej: "003_S_1059"

        # Verificar archivos ECAT directamente para PET si es aplicable
        if modality.lower() == "pet":
            ecat_files = list(base_path.glob("*.v")) + list(base_path.glob("*.V"))

            if ecat_files and ecat is not None:
                if verbose:
                    logger.info(f"Encontrado archivo ECAT para PET: {ecat_files[0]}")

                # Usar directamente el archivo ECAT
                best_scan = base_path
                scan_category = "fdg"  # Por defecto asignamos FDG, podría refinarse

                # Convertir directamente
                nifti_output_dir = output_dir / "nifti" / modality
                nifti_output_dir.mkdir(parents=True, exist_ok=True)

                # Crear nombre de archivo de salida
                output_filename = f"{subject_id}_{modality.upper()}.nii.gz"
                output_path = nifti_output_dir / output_filename

                # Convertir ECAT a NIfTI
                try:
                    ecat_img = ecat.load(str(ecat_files[0]))
                    # Usar np.asanyarray(img.dataobj) en lugar de get_data()
                    img_data = np.asanyarray(ecat_img.dataobj)
                    nifti_img = nib.Nifti1Image(img_data, ecat_img.affine)
                    nib.save(nifti_img, str(output_path))

                    if verbose:
                        logger.info(f"Archivo ECAT convertido a NIfTI: {output_path}")

                    # Crear estructura organizada
                    conversion_info = {
                        "scan_type": scan_category,
                        "nifti_path": output_path,
                        "original_file": str(ecat_files[0]),
                        "conversion_date": datetime.datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        ),
                    }

                    # Guardar información de organización
                    info_file = output_dir / f"{modality}_scan_info.json"
                    with open(info_file, "w") as f:
                        json.dump(
                            {
                                "source_file": str(ecat_files[0]),
                                "nifti_output": str(output_path),
                                "scan_type": scan_category,
                                "conversion_date": conversion_info["conversion_date"],
                                "file_format": "ECAT",
                            },
                            f,
                            indent=2,
                        )

                    return conversion_info

                except Exception as e:
                    if verbose:
                        logger.error(f"Error al convertir archivo ECAT: {e}")
                    # Continuar con el flujo normal

        # Identificar carpetas de escaneo
        if modality.lower() == "mri":
            scan_types = identify_scan_folders(base_path, "mri")

            # Para MRI, priorizamos MPRAGE
            if scan_types["mprage"]:
                best_scan = select_best_scan(scan_types["mprage"], "mri")
                scan_category = "mprage"
            else:
                if verbose:
                    logger.warning(
                        f"No se encontraron escaneos MRI MPRAGE en {base_path}"
                    )
                return None

        elif modality.lower() == "pet":
            scan_types = identify_scan_folders(base_path, "pet")

            # Para PET, priorizamos FDG
            if scan_types["fdg"]:
                best_scan = select_best_scan(scan_types["fdg"], "pet")
                scan_category = "fdg"
            elif any(scan_types.values()):
                for category, scans in scan_types.items():
                    if scans:
                        best_scan = select_best_scan(scans, "pet")
                        scan_category = category
                        break
            else:
                if verbose:
                    logger.warning(f"No se encontraron escaneos PET en {base_path}")
                raise ValueError(
                    f"No se encontraron escaneos PET - Escaneos: {scan_types}"
                )
                return None
        else:
            raise ValueError(f"Modalidad no soportada: {modality}")

        if not best_scan:
            raise ValueError("No se pudo seleccionar un escaneo para convertir")
            return None

        # Mensaje para modo detallado
        if verbose:
            logger.info(
                f"Convirtiendo {modality} para sujeto {subject_id} desde {best_scan}"
            )

        # Convertir a NIfTI con manejo robusto de errores
        nifti_output_dir = output_dir / "nifti" / modality
        try:
            nifti_file = convert_dicoms_to_nifti(best_scan, nifti_output_dir, verbose)
        except Exception as e:
            if verbose:
                logger.error(f"Error al convertir {modality} para {subject_id}: {e}")
            raise e
            return None

        # Verificar la conversión - con más información de error
        quality_check = verify_nifti_quality(nifti_file)
        if not quality_check:
            if verbose:
                logger.error(
                    f"El NIfTI convertido para {subject_id} ({modality}) no pasó las verificaciones de calidad"
                )

            raise ValueError(
                "El NIfTI convertido no pasó las verificaciones de calidad"
            )
            return None

        # Renombrar el archivo de salida para incluir el ID del sujeto y el tipo de modalidad
        output_filename = f"{subject_id}_{modality.upper()}.nii.gz"
        new_nifti_path = nifti_output_dir / output_filename
        os.rename(nifti_file, new_nifti_path)
        nifti_file = new_nifti_path

        # Crear estructura organizada
        conversion_info = {
            "scan_type": scan_category,
            "nifti_path": nifti_file,
            "original_folder": best_scan,
            "conversion_date": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        # Guardar información de organización
        info_file = output_dir / f"{modality}_scan_info.json"
        with open(info_file, "w") as f:
            json.dump(
                {
                    "source_folder": str(best_scan),
                    "nifti_output": str(nifti_file),
                    "scan_type": scan_category,
                    "conversion_date": conversion_info["conversion_date"],
                },
                f,
                indent=2,
            )

        return conversion_info

    except Exception as e:
        if verbose:
            logger.error(
                f"Error al procesar escaneo {modality} para {base_path.name}: {e}"
            )
        raise e
        return None


def batch_process_directories(base_dirs, output_root, modalities=None, verbose=False):
    """
    Procesa por lotes múltiples directorios de escaneos

    Parameters:
        base_dirs (list): Lista de directorios base con escaneos
        output_root (str): Directorio raíz para salida
        modalities (list): Lista de modalidades a procesar ('mri', 'pet' o ambas)
        verbose (bool): Si es True, mostrar logs detallados
    """
    if modalities is None:
        modalities = ["mri", "pet"]

    results = {"success": 0, "failed": 0, "conversions": [], "errors": []}

    # Establecer nivel de logging global
    set_global_log_level(verbose)

    progress_bar = tqdm(base_dirs, desc="Procesando directorios", disable=not verbose)
    for base_dir in progress_bar:
        base_path = Path(base_dir)
        subject_id = base_path.name  # Ej: "003_S_1059"

        # Actualizar descripción de la barra de progreso solo si es verbose
        if verbose:
            progress_bar.set_description(f"Procesando sujeto {subject_id}")

        # El ID del sujeto ya está en el formato correcto
        subject_output_dir = Path(output_root) / subject_id

        for modality in modalities:
            if verbose:
                progress_bar.set_description(
                    f"Procesando {modality} para sujeto {subject_id}"
                )

            try:
                result = process_scan(base_dir, subject_output_dir, modality, verbose)

                if result:
                    results["success"] += 1
                    results["conversions"].append(
                        {
                            "subject_id": subject_id,
                            "modality": modality,
                            "output_path": str(result["nifti_path"]),
                        }
                    )
                    if verbose:
                        progress_bar.set_description(
                            f"Éxito: {subject_id} ({modality})"
                        )
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(
                    {"subject_id": subject_id, "modality": modality, "error": str(e)}
                )
                if verbose:
                    progress_bar.set_description(f"Falló: {subject_id} ({modality})")

    # Guardar resumen
    summary_path = Path(output_root) / "conversion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        logger.info(
            f"Procesamiento por lotes completado. Éxitos: {results['success']}, Fallos: {results['failed']}"
        )
        logger.info(f"Resumen guardado en: {summary_path}")

    return results


def main():
    """Función principal para ejecutar desde línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Convertir archivos DICOM de MRI y PET a formato NIfTI"
    )

    parser.add_argument(
        "--input", type=str, help="Ruta al directorio de entrada con archivos DICOM"
    )
    parser.add_argument(
        "--output", type=str, help="Directorio para guardar archivos NIfTI"
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["mri", "pet", "both"],
        default="both",
        help="Modalidad a procesar (mri, pet, o both)",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Procesar múltiples directorios como lote"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar mensajes detallados durante el procesamiento",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Reducir mensajes al mínimo"
    )
    # Añadir argumento para especificar manejo de archivos ECAT
    parser.add_argument(
        "--ecat",
        action="store_true",
        help="Buscar y procesar archivos ECAT (.v) para PET",
    )

    args = parser.parse_args()

    # Configurar nivel de verbosidad
    verbose = args.verbose and not args.quiet
    set_global_log_level(verbose)

    # Verificar si se pueden manejar archivos ECAT
    if args.ecat and ecat is None:
        logger.error(
            "El módulo nibabel.ecat no está disponible. No se pueden procesar archivos ECAT (.v)"
        )
        return

    if args.batch:
        # Procesar todos los subdirectorios como sujetos separados
        input_path = Path(args.input)
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]

        # Mostrar barra de progreso global solo si es verbose
        if verbose:
            print(f"Procesando {len(subdirs)} directorios...")

        modalities = ["mri", "pet"] if args.modality == "both" else [args.modality]
        res = batch_process_directories(subdirs, args.output, modalities, verbose)
        # if verbose:
        print(res)
    else:
        # Procesar un solo directorio
        if args.modality == "both" or args.modality == "mri":
            mri_result = process_scan(args.input, args.output, "mri", args.verbose)
            if mri_result:
                logger.info(
                    f"Conversión MRI exitosa. NIfTI en: {mri_result['nifti_path']}"
                )
            else:
                logger.warning("La conversión MRI falló")

        if args.modality == "both" or args.modality == "pet":
            pet_result = process_scan(args.input, args.output, "pet", args.verbose)
            if pet_result:
                logger.info(
                    f"Conversión PET exitosa. NIfTI en: {pet_result['nifti_path']}"
                )
            else:
                logger.warning("La conversión PET falló")


if __name__ == "__main__":
    main()
