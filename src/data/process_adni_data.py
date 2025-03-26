"""
Script para procesar datos ADNI:
1. Organizar los datos convertidos
2. Generar los archivos HDF5 para DiaMond
3. Crear splits de entrenamiento/validación/prueba
"""

import os
import sys
import argparse
import logging
import pandas as pd
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import json
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict
import nibabel as nib

# Importar funciones de los otros módulos
from dicom_converter import set_global_log_level
# from dicom_converter import batch_process_directories, set_global_log_level
# from process_data import create_h5_dataset, load_nifti_file
# split_dataset

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Silenciar loggers externos demasiado verbosos por defecto
logging.getLogger("dicom2nifti").setLevel(logging.ERROR)
logging.getLogger("pydicom").setLevel(logging.ERROR)
logging.getLogger("nibabel").setLevel(logging.ERROR)
logging.getLogger("h5py").setLevel(logging.ERROR)


def load_nifti_file(file_path: str) -> np.ndarray:
    """
    Carga un archivo NIFTI y devuelve sus datos como un array de NumPy.

    Args:
        file_path: Ruta al archivo NIFTI

    Returns:
        Array NumPy con los datos de la imagen
    """
    try:
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)

        # Normalización básica a [0, 1]
        if np.max(data) != np.min(data):
            data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Asegurarse de que los datos tienen la forma correcta para DiaMond (C, H, W, D)
        data = data[np.newaxis, ...]  # Añadir canal si no existe

        return data
    except Exception as e:
        logger.error(f"Error al cargar {file_path}: {e}")
        return None


def create_h5_dataset(
    output_path: str,
    subject_data: List[Dict],
    modalities: List[str] = ["MRI", "PET"],
    verbose: bool = False,
) -> None:
    """
    Crea un archivo HDF5 con los datos de los sujetos.

    Args:
        output_path: Ruta donde se guardará el archivo HDF5
        subject_data: Lista de diccionarios con datos de sujetos
        modalities: Lista de modalidades a incluir
        verbose: Si es True, muestra información adicional durante el proceso
    """

    if verbose:
        print("Creating HDF5 file")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as h5f:
        if verbose:
            logger.debug(f"Creando archivo HDF5 en {output_path}")

        # Crear grupo de estadísticas
        print("Creating stats group")
        stats_group = h5f.create_group("stats")
        stats_group.attrs["num_subjects"] = len(subject_data)

        print("Counting diagnoses")
        # Contar diagnósticos
        all_dx = [s["diagnosis"] for s in subject_data]
        dx_counts = {dx: all_dx.count(dx) for dx in set(all_dx)}
        for dx, count in dx_counts.items():
            stats_group.attrs[f"count_{dx}"] = count

        print("Adding subjects")
        # Añadir cada sujeto
        subject_iterator = tqdm(subject_data, desc=f"Creando {output_path}")
        for i, subject in enumerate(subject_iterator):
            # Usar el RID si está disponible, de lo contrario usar índice
            subject_id = subject.get("rid", f"subject_{i}")

            if not subject_id:
                print("No subject ID found")
                continue

            # Verificar que los datos MRI y PET son arrays de numpy válidos
            has_mri = "mri_data" in subject and isinstance(
                subject["mri_data"], np.ndarray
            )
            has_pet = "pet_data" in subject and isinstance(
                subject["pet_data"], np.ndarray
            )

            if not (has_mri or has_pet):
                print(f"Skipping subject {i}: {subject_id} - no valid image data")
                continue

            subject_iterator.set_postfix_str(f"Subject {subject_id}")

            # Limpiar el id para asegurar que sea un nombre de grupo válido en HDF5
            subject_id = re.sub(r"[^\w]", "_", str(subject_id))

            print(f"Creating group for subject {subject_id}")
            subject_group = h5f.create_group(subject_id)
            subject_group.attrs["RID"] = subject["rid"]
            subject_group.attrs["DX"] = subject["diagnosis"]

            # Añadir datos de MRI si están disponibles
            if "MRI" in modalities and has_mri:
                mri_group = subject_group.create_group("MRI/T1")
                # Make subject["mri_data"] a numpy array of shape (128, 128, 128) if it aint

                if subject["mri_data"].shape != (128, 128, 128):
                    if sum(subject["mri_data"].shape) / 128 == 128:
                        subject["mri_data"] = np.array(
                            subject["mri_data"], dtype=np.float32
                        ).reshape(128, 128, 128)
                    else:
                        # Try resizing to 128x128x128
                        tmp_array = np.zeros((128, 128, 128))
                        if len(subject["mri_data"].shape) == 3:
                            tmp_array[:subject["mri_data"].shape[0], :subject["mri_data"].shape[1], :subject["mri_data"].shape[2]] = subject["mri_data"]
                        else:
                            data = subject["mri_data"]
                            data = data.flatten()
                            
                            if sum(data.shape) > 128**3:
                                tmp_array = data[:128*128*128].reshape(128, 128, 128)
                            else:
                                cube_root = int(round(data.shape[0] ** (1. / 3)))
                                tmp_array = data[:cube_root**3].reshape(cube_root, cube_root, cube_root)

                        subject["mri_data"] = tmp_array

                mri_group.create_dataset(
                    "data", data=subject["mri_data"], dtype=np.float32
                )
                if verbose:
                    logger.debug(f"Añadidos datos MRI para sujeto {subject_id}")

            # Añadir datos de PET si están disponibles
            if "PET" in modalities and has_pet:

                if subject["pet_data"].shape != (128, 128, 128):
                    if sum(subject["pet_data"].shape) / (128**2) == 128:
                        subject["pet_data"] = np.array(
                            subject["pet_data"], dtype=np.float32
                        ).reshape(128, 128, 128)
                    else:
                        # Crop PET data to 128x128x128
                        tmp_array = np.zeros((128, 128, 128))
                        if len(subject["pet_data"].shape) == 3:
                            tmp_array[:subject["pet_data"].shape[0], :subject["pet_data"].shape[1], :subject["pet_data"].shape[2]] = subject["pet_data"]
                        else:
                            data = subject["pet_data"]
                            data = data.flatten()
                            if sum(data.shape) > 128**3:
                                tmp_array = data[:128*128*128].reshape(128, 128, 128)
                            else:
                                cube_root = int(round(data.shape[0] ** (1. / 3)))
                                tmp_array = data[:cube_root**3].reshape(cube_root, cube_root, cube_root)

                        subject["pet_data"] = tmp_array

                pet_group = subject_group.create_group("PET/FDG")
                pet_group.create_dataset(
                    "data", data=subject["pet_data"], dtype=np.float32
                )
                if verbose:
                    logger.debug(f"Añadidos datos PET para sujeto {subject_id}")

    logger.info(f"Archivo HDF5 creado: {output_path}")
    logger.info(f"Total de sujetos: {len(subject_data)}")
    logger.info(f"Distribución de diagnósticos: {dx_counts}")

    if verbose:
        logger.debug("Detalles completos de la creación del archivo:")
        for dx, count in dx_counts.items():
            logger.debug(f"  - {dx}: {count} sujetos")


def split_dataset(
    input_path: str,
    output_dir: str,
    n_splits: int = 5,
    test_size: float = 0.2,
    valid_size: float = 0.15,
    random_seed: int = 42,
    verbose: bool = True,
) -> None:
    """
    Divide un archivo HDF5 en múltiples conjuntos de entrenamiento/validación/prueba.

    Args:
        input_path: Ruta al archivo HDF5 maestro
        output_dir: Directorio donde se guardarán los archivos HDF5 divididos
        n_splits: Número de divisiones para validación cruzada
        test_size: Fracción de datos para prueba
        valid_size: Fracción de datos de entrenamiento para validación
        random_seed: Semilla para reproducibilidad
        verbose: Si es True, muestra información adicional durante el proceso
    """
    os.makedirs(output_dir, exist_ok=True)
    log_level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(log_level)

    try:
        with h5py.File(input_path, "r") as h5f:
            # Obtener IDs de sujetos (excluyendo 'stats' si existe)
            subject_ids = [k for k in h5f.keys() if k != "stats"]

            if verbose:
                logger.info(
                    f"Dividiendo dataset con {len(subject_ids)} sujetos en {n_splits} splits"
                )

            # Extraer diagnósticos para estratificación
            diagnoses = [h5f[sid].attrs.get("DX", "Unknown") for sid in subject_ids]

            # Convertir a DataFrame para facilitar la división
            df = pd.DataFrame({"subject_id": subject_ids, "diagnosis": diagnoses})

            # Primero separar el conjunto de prueba global
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_seed,
                stratify=df["diagnosis"] if len(set(df["diagnosis"])) > 1 else None,
            )

            # Crear conjuntos para cada split de validación cruzada
            for split in range(n_splits):
                split_seed = random_seed + split

                # Para cada split, crear train/val/test
                if len(set(train_val_df["diagnosis"])) > 1:
                    train_df, val_df = train_test_split(
                        train_val_df,
                        test_size=valid_size,
                        random_state=split_seed,
                        stratify=train_val_df["diagnosis"],
                    )
                else:
                    train_df, val_df = train_test_split(
                        train_val_df, test_size=valid_size, random_state=split_seed
                    )

                print("Train DF")
                print(train_df)

                print("Val DF")
                print(val_df)

                print("Test DF")
                print(test_df)

                # Crear archivos HDF5 para cada conjunto
                split_files = [
                    (train_df, f"{split}-train.h5", "entrenamiento"),
                    (val_df, f"{split}-valid.h5", "validación"),
                    (test_df, f"{split}-test.h5", "prueba"),
                ]

                for subset_df, filename, subset_name in split_files:
                    output_path = os.path.join(output_dir, filename)

                    with h5py.File(output_path, "w") as out_f:
                        # Copiar estadísticas si existen
                        if "stats" in h5f:
                            h5f.copy("stats", out_f)

                            # Actualizar estadísticas para este subset
                            if "stats" in out_f:
                                out_f["stats"].attrs["num_subjects"] = len(subset_df)

                                # Actualizar conteo de diagnósticos
                                dx_counts = (
                                    subset_df["diagnosis"].value_counts().to_dict()
                                )
                                for dx, count in dx_counts.items():
                                    out_f["stats"].attrs[f"count_{dx}"] = count

                        # Copiar sujetos
                        for _, row in subset_df.iterrows():
                            subject_id = row["subject_id"]
                            h5f.copy(h5f[subject_id], out_f, name=subject_id)

                    if verbose:
                        logger.info(
                            f"Split {split}, conjunto de {subset_name}: {len(subset_df)} sujetos guardados en {output_path}"
                        )

        if verbose:
            logger.info(
                f"División de dataset completada. Archivos guardados en {output_dir}"
            )

    except Exception as e:
        logger.error(f"Error al dividir el dataset: {e}")
        raise


def find_adni_subjects(base_dir):
    """
    Identifica directorios de sujetos ADNI válidos

    Args:
        base_dir (str): Directorio base con datos ADNI
    Returns:
        list: Lista de rutas a directorios de sujetos
    """
    base_path = Path(base_dir)
    subject_dirs = []

    # En ADNI, las carpetas de sujetos típicamente siguen el patrón XXX_S_XXXX
    for item in base_path.iterdir():
        if item.is_dir():
            # Verificar si es un directorio de sujeto con patrón ADNI
            if re.match(r"\d{3}_S_\d{4}", item.name):
                subject_dirs.append(item)
            else:
                # Verificar si contiene archivos DICOM
                dcm_files = list(item.glob("**/*.dcm")) + list(item.glob("**/*.DCM"))
                if dcm_files:
                    subject_dirs.append(item)

    logger.info(f"Encontrados {len(subject_dirs)} directorios de sujetos ADNI")
    return subject_dirs


def extract_subject_metadata(dicom_dir, output_dir):
    """
    Extrae metadatos de los sujetos basados en archivos DICOM y NIFTI

    Args:
        dicom_dir (str): Directorio con datos DICOM
        output_dir (str): Directorio con datos procesados
    Returns:
        pd.DataFrame: DataFrame con metadatos de sujetos
    """
    metadata = []
    output_path = Path(output_dir)

    # Buscar directorios de sujetos
    subject_dirs = [
        d for d in output_path.glob("*") if d.is_dir() and not d.name.startswith(".")
    ]
    logger.info(
        f"Encontrados {len(subject_dirs)} directorios de sujetos en {output_path}"
    )

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        subject_entry = {
            "subject_id": subject_id,
            "diagnosis": "Unknown",  # Se completará más tarde
        }

        """
        (diamond) nacho@Chonas-M1-MacBook-Pro-266 DiaMond % tree data/processed/109_S_0967
        data/processed/109_S_0967
        ├── mri_scan_info.json
        ├── nifti
        │   ├── mri
        │   │   └── 109_S_0967_MRI.nii.gz
        │   └── pet
        │       └── 109_S_0967_PET.nii.gz
        └── pet_scan_info.json

        3 directories, 4 files
        """

        subject_entry["mri_path"] = None
        subject_entry["pet_path"] = None

        if os.path.exists(output_path / subject_id / "mri_scan_info.json"):
            with open(output_path / subject_id / "mri_scan_info.json") as f:
                mri_info = json.load(f)
                subject_entry["mri_info"] = mri_info["nifti_output"]

        if os.path.exists(output_path / subject_id / "pet_scan_info.json"):
            with open(output_path / subject_id / "pet_scan_info.json") as f:
                pet_info = json.load(f)
                subject_entry["pet_info"] = pet_info["nifti_output"]

        # Buscar archivos NIFTI dentro del directorio de sujeto
        mri_files = list(subject_dir.glob("**/nifti/mri/*.nii.gz"))
        pet_files = list(subject_dir.glob("**/nifti/pet/*.nii.gz"))

        # Si no encontramos en la estructura esperada, intentamos una búsqueda más general
        if not mri_files:
            mri_files = list(subject_dir.glob("**/*MRI*.nii.gz"))
        if not pet_files:
            pet_files = list(subject_dir.glob("**/*PET*.nii.gz"))

        # Buscar también en el directorio nifti general
        if not mri_files:
            mri_files = list(output_path.glob(f"nifti/mri/*{subject_id}*.nii.gz"))
        if not pet_files:
            pet_files = list(output_path.glob(f"nifti/pet/*{subject_id}*.nii.gz"))

        # Agregar rutas a los archivos encontrados
        if mri_files:
            # Guardar ruta relativa al directorio output_dir
            mri_path = os.path.relpath(str(mri_files[0]), output_dir)
            subject_entry["mri_path"] = mri_path
            logger.debug(f"Encontrado MRI para {subject_id}: {mri_path}")

        if pet_files:
            # Guardar ruta relativa al directorio output_dir
            pet_path = os.path.relpath(str(pet_files[0]), output_dir)
            subject_entry["pet_path"] = pet_path
            logger.debug(f"Encontrado PET para {subject_id}: {pet_path}")

        # Buscar archivos JSON de info
        mri_info = list(subject_dir.glob("mri_scan_info.json"))
        pet_info = list(subject_dir.glob("pet_scan_info.json"))

        # Procesar información de los JSON si existen
        for info_file in mri_info + pet_info:
            modality = "mri" if "mri" in info_file.name.lower() else "pet"
            try:
                with open(info_file) as f:
                    info = json.load(f)

                # Extraer información adicional si está disponible
                if "scan_date" in info:
                    subject_entry[f"{modality}_scan_date"] = info["scan_date"]
                if "scanner" in info:
                    subject_entry[f"{modality}_scanner"] = info["scanner"]

                # Si el archivo NIFTI no se encontró antes, intentar usar el del JSON
                if f"{modality}_path" not in subject_entry and "nifti_output" in info:
                    nifti_file = info.get("nifti_output", "")
                    if nifti_file and os.path.exists(nifti_file):
                        rel_path = os.path.relpath(nifti_file, output_dir)
                        subject_entry[f"{modality}_path"] = rel_path
            except Exception as e:
                logger.warning(f"Error al procesar {info_file}: {e}")

        # Solo agregar al metadata si tiene al menos una modalidad
        if "mri_path" in subject_entry or "pet_path" in subject_entry:
            metadata.append(subject_entry)
        else:
            logger.warning(
                f"No se encontraron archivos NIFTI para el sujeto {subject_id}"
            )

    # También buscar en la carpeta de metadatos ADNI si existe
    adni_metadata_dir = Path(dicom_dir).parent / "ADNI Metadata"
    if adni_metadata_dir.exists():
        logger.info(f"Buscando metadatos adicionales en {adni_metadata_dir}")

        # Recorrer la estructura para encontrar archivos XML con metadatos
        for subject_dir in adni_metadata_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            # Buscar archivos XML de metadatos
            xml_files = list(subject_dir.glob("**/*.xml"))
            if xml_files:
                # Verificar si el sujeto ya está en metadata
                existing_entry = next(
                    (item for item in metadata if item["subject_id"] == subject_id),
                    None,
                )

                if existing_entry:
                    # Actualizar la entrada existente
                    existing_entry["has_metadata"] = True
                else:
                    # Crear una nueva entrada
                    entry = {
                        "subject_id": subject_id,
                        "diagnosis": "Unknown",  # Se completará más tarde
                        "has_metadata": True,
                    }
                    metadata.append(entry)

    # Convertir a DataFrame y combinar entradas con el mismo subject_id
    df = pd.DataFrame(metadata)

    # Verificar si el DataFrame está vacío
    if df.empty:
        logger.warning(
            "No se encontraron sujetos. El DataFrame de metadatos está vacío."
        )
        # Crear un DataFrame vacío con las columnas necesarias
        df = pd.DataFrame(columns=["subject_id", "diagnosis", "mri_path", "pet_path"])
        return df

    # Verificar si hay sujetos sin modalidades
    if "mri_path" in df.columns and "pet_path" in df.columns:
        missing_modalities = df[(~df.mri_path.notna()) & (~df.pet_path.notna())]
        if not missing_modalities.empty:
            logger.warning(
                f"{len(missing_modalities)} sujetos sin modalidades MRI ni PET"
            )
            logger.warning(f"Ejemplos: {missing_modalities.subject_id.tolist()[:5]}")

        # Asegurar que no haya duplicados de subject_id
        df = df.drop_duplicates(subset=["subject_id"], keep="first")

        # Imprimir estadísticas
        with_mri = df.mri_path.notna().sum()
        with_pet = df.pet_path.notna().sum()
        with_both = df[df.mri_path.notna() & df.pet_path.notna()].shape[0]

        logger.info(f"Total sujetos: {len(df)}")
        logger.info(f"Con MRI: {with_mri}, Con PET: {with_pet}, Con ambos: {with_both}")
    else:
        logger.warning("No se encontraron columnas de modalidades en el DataFrame")
        # Asegurar que existan las columnas necesarias
        for col in ["mri_path", "pet_path"]:
            if col not in df.columns:
                df[col] = None

    return df


def merge_with_clinical_data(metadata, clinical):
    """
    Combina metadatos extraídos con datos clínicos

    Args:
        metadata (pd.DataFrame): DataFrame con metadatos extraídos
        clinical (pd.DataFrame): DataFrame con datos clínicos
    Returns:
        pd.DataFrame: DataFrame combinado
    """
    try:
        # Asegurarse de que ambos tienen la columna subject_id
        if "subject_id" not in clinical.columns:
            # Para ADNI, buscar columnas como PTID, RID, etc.
            id_columns = [
                col
                for col in clinical.columns
                if any(
                    id_name in col.lower()
                    for id_name in ["ptid", "rid", "id", "subject"]
                )
            ]
            if id_columns:
                clinical = clinical.rename(columns={id_columns[0]: "subject_id"})
            else:
                raise ValueError("No se encontró columna de ID en los datos clínicos")

        # Combinar dataframes
        merged = pd.merge(metadata, clinical, on="subject_id", how="left")

        # Mapear diagnóstico para ADNI (buscar columnas DX, DXCURREN, DXCHANGE, etc.)
        dx_columns = [
            col
            for col in merged.columns
            if any(
                dx_name in col.lower()
                for dx_name in ["dx", "diag", "diagnosis", "diagnostic"]
            )
        ]

        if dx_columns:
            # Usar la primera columna de diagnóstico encontrada
            merged["diagnosis"] = merged[dx_columns[0]]

            # Estandarizar diagnósticos al formato esperado por DiaMond
            dx_mapping = {
                "CN": "CN",  # Control Normal
                "Normal": "CN",
                "NL": "CN",
                "MCI": "MCI",  # Deterioro Cognitivo Leve
                "LMCI": "MCI",
                "EMCI": "MCI",
                "AD": "AD",  # Alzheimer
                "Dementia": "AD",
                "Dementia due to AD": "AD",
            }

            # Aplicar mapeo de diagnósticos estándar
            merged["diagnosis"] = merged["diagnosis"].map(
                lambda x: dx_mapping.get(str(x).strip(), "Unknown")
                if pd.notna(x)
                else "Unknown"
            )

        return merged
    except Exception as e:
        logger.error(f"Error al combinar con datos clínicos: {e}")
        return metadata


def extract_diagnosis_from_xml(xml_path):
    """
    Extrae información de diagnóstico y demografía desde archivos XML de ADNI

    Args:
        xml_path (str): Ruta al archivo XML
    Returns:
        dict: Diccionario con información extraída (diagnosis, age, sex, etc.)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Definir namespace si es necesario - algunos archivos XML de ADNI usan namespaces
        namespaces = (
            {"": "http://ida.loni.usc.edu"}
            if "ida.loni.usc.edu" in ET.tostring(root, encoding="unicode")
            else {}
        )

        # Extraer información clave (con manejo de namespace)
        data = {}

        # Extraer ID del sujeto
        subject_id_elem = root.find(".//subjectIdentifier", namespaces)
        if subject_id_elem is not None:
            data["subject_id"] = subject_id_elem.text

        # Extraer diagnóstico/grupo de investigación
        research_group_elem = root.find(".//researchGroup", namespaces)
        if research_group_elem is not None:
            data["diagnosis"] = research_group_elem.text

        # Extraer sexo
        sex_elem = root.find(".//subjectSex", namespaces)
        if sex_elem is not None:
            data["sex"] = sex_elem.text

        # Extraer edad
        age_elem = root.find(".//subjectAge", namespaces)
        if age_elem is not None:
            data["age"] = age_elem.text

        # Extraer información adicional relevante
        weight_elem = root.find(".//weightKg", namespaces)
        if weight_elem is not None:
            data["weight_kg"] = weight_elem.text

        return data
    except Exception as e:
        logger.warning(f"Error al procesar XML {xml_path}: {e}")
        return {}


def create_clinical_data_from_metadata(metadata_dir):
    """
    Crea un DataFrame con datos clínicos a partir de los archivos XML de metadatos

    Args:
        metadata_dir (str): Ruta al directorio de metadatos
    Returns:
        pd.DataFrame: DataFrame con datos clínicos
    """
    clinical_data = []

    metadata_path = Path(metadata_dir)
    if not metadata_path.exists():
        logger.warning(f"Directorio de metadatos no encontrado: {metadata_dir}")
        return pd.DataFrame()

    # Buscar todos los archivos XML en el directorio de metadatos
    xml_files = list(metadata_path.glob("**/*.xml"))
    logger.info(
        f"Encontrados {len(xml_files)} archivos XML en el directorio de metadatos"
    )

    # Extraer información de cada archivo XML
    for xml_file in tqdm(xml_files, desc="Procesando archivos XML"):
        data = extract_diagnosis_from_xml(xml_file)
        if data and "subject_id" in data and "diagnosis" in data:
            clinical_data.append(data)

    # Consolidar entradas para el mismo sujeto (quedarse con la primera entrada)
    df = pd.DataFrame(clinical_data)
    if not df.empty and "subject_id" in df.columns:
        df = df.drop_duplicates(subset=["subject_id"], keep="first")
        logger.info(f"Datos clínicos creados para {len(df)} sujetos")

    return df


def process_adni_workflow(dicom_dir, output_dir, clinical_csv=None, verbose=False):
    """
    Flujo de trabajo completo para procesar datos ADNI

    Args:
        dicom_dir (str): Directorio con datos DICOM
        output_dir (str): Directorio para salida
        clinical_csv (str): Archivo CSV con datos clínicos (opcional)
        verbose (bool): Si True, mostrar logs detallados
    """

    clinical_csv_df = pd.read_csv(clinical_csv)

    # print(clinical_csv_df)

    # Establecer nivel de log global
    set_global_log_level(verbose)

    # 1. Crear directorios de salida
    output_path = Path(output_dir)
    # output_dir = output_path / "nifti"
    hdf5_dir = output_path / "hdf5"

    print(output_path)
    # os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hdf5_dir, exist_ok=True)

    # 2. Encontrar sujetos ADNI
    # subject_dirs = find_adni_subjects(dicom_dir)

    # 3. Convertir archivos DICOM a NIFTI
    # if verbose:
    # logger.info("Convirtiendo archivos DICOM a NIFTI...")
    # batch_process_directories(
    #     subject_dirs, str(output_dir), modalities=["mri", "pet"], verbose=verbose
    # )

    # If not metadata.csv
    # 4. Extraer metadatos
    if verbose:
        logger.info("Extrayendo metadatos...")

    # Verificar que el directorio nifti existe y contiene datos
    # output_dir = output_path / "nifti"
    # if not os.path.exists(output_dir) or not any(output_dir.glob("*")):
    #     logger.warning(f"El directorio NIFTI {output_dir} no existe o está vacío.")
    #     print(f"AVISO: El directorio {output_dir} no existe o está vacío.")
    #     print("Creando un directorio vacío para continuar...")
    #     os.makedirs(output_dir, exist_ok=True)

    metadata = extract_subject_metadata(dicom_dir, output_dir)

    if metadata.empty:
        logger.warning("No se encontraron sujetos en los datos procesados.")
        print("AVISO: No se pudieron encontrar sujetos en los datos procesados.")
        # Crear un DataFrame mínimo para poder continuar
        test_subjects = [
            {"subject_id": "TEST_SUBJECT_1", "diagnosis": "CN"},
            {"subject_id": "TEST_SUBJECT_2", "diagnosis": "AD"},
            {"subject_id": "TEST_SUBJECT_3", "diagnosis": "MCI"},
        ]
        metadata = pd.DataFrame(test_subjects)

    # 5. Combinar con datos clínicos si están disponibles
    if not clinical_csv_df.empty:
        if verbose:
            logger.info(
                f"Combinando con datos clínicos desde archivo CSV: {clinical_csv}"
            )
        metadata = merge_with_clinical_data(metadata, clinical_csv_df)
    else:
        # Intentar crear datos clínicos a partir de archivos XML
        if verbose:
            logger.info(
                "Archivo de datos clínicos no proporcionado, intentando crear desde XML..."
            )
        adni_metadata_dir = Path(dicom_dir).parent / "ADNI Metadata"

        if adni_metadata_dir.exists():
            clinical_data = create_clinical_data_from_metadata(adni_metadata_dir)
            if not clinical_data.empty:
                # Guardar el CSV generado automáticamente
                auto_clinical_csv = Path(output_dir) / "auto_clinical_data.csv"
                clinical_data.to_csv(auto_clinical_csv, index=False)
                if verbose:
                    logger.info(
                        f"Datos clínicos generados automáticamente guardados en: {auto_clinical_csv}"
                    )

                # Combinar con metadatos
                metadata = pd.merge(
                    metadata, clinical_data, on="subject_id", how="left"
                )

                # Asegurar que la columna diagnóstico se llame 'diagnosis'
                if "diagnosis_y" in metadata.columns:
                    metadata["diagnosis"] = metadata["diagnosis_y"]
                    metadata = metadata.drop(columns=["diagnosis_x", "diagnosis_y"])

    # 6. Guardar metadatos - Asegurar que se guarden en el directorio raíz de salida
    metadata_path = output_path / "metadata.csv"
    os.makedirs(output_path, exist_ok=True)  # Asegurar que el directorio existe
    metadata.to_csv(metadata_path, index=False)

    if verbose:
        logger.info(f"Metadatos guardados en {metadata_path}")
    else:
        print(
            f"Metadatos guardados en {metadata_path}"
        )  # Mensaje esencial incluso en modo silencioso

    # get diagnosis for metadata["diagnosis"] from clinical_csv_df["Group"] where metadata["subject_id"] == clinical_csv_df["Subject"]

    metadata["diagnosis"] = metadata["subject_id"].map(
        lambda x: clinical_csv_df.loc[clinical_csv_df["Subject"] == x, "Group"].values[
            0
        ]
        if x in clinical_csv_df["Subject"].values
        else "Unknown"
    )

    metadata.to_csv(metadata_path, index=False)

    print(f"Subjects: {len(metadata)}")
    print(metadata["diagnosis"].value_counts())

    # 7. Preparar datos para el dataset HDF5
    if verbose:
        logger.info("Preparando datasets...")
        logger.info(f"Estructura del DataFrame de metadatos: {metadata.columns}")
        logger.info(f"Primeras filas: {metadata.head()}")

    # Cargar imágenes y preparar datos para HDF5
    subject_data = []

    # Procesamiento normal de imágenes existentes
    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Cargando datos de imagen",
    ):
        subject_dict = {
            "rid": row["subject_id"],
            "diagnosis": row.get("diagnosis", "Unknown"),
            "mri_data": None,
            "pet_data": None,
            "mri_path": None,
            "pet_path": None,
        }

        """
        (diamond) nacho@Chonas-M1-MacBook-Pro-266 DiaMond % tree data/processed/109_S_0967
        data/processed/109_S_0967
        ├── mri_scan_info.json
        ├── nifti
        │   ├── mri
        │   │   └── 109_S_0967_MRI.nii.gz
        │   └── pet
        │       └── 109_S_0967_PET.nii.gz
        └── pet_scan_info.json

        3 directories, 4 files
        """

        if os.path.exists(output_path / row["subject_id"] / "mri_scan_info.json"):
            with open(output_path / row["subject_id"] / "mri_scan_info.json") as f:
                mri_info = json.load(f)
                subject_dict["mri_info"] = mri_info["nifti_output"]

        if os.path.exists(output_path / row["subject_id"] / "pet_scan_info.json"):
            with open(output_path / row["subject_id"] / "pet_scan_info.json") as f:
                pet_info = json.load(f)
                subject_dict["pet_info"] = pet_info["nifti_output"]

        # Cargar MRI - Asegurar que existe la columna y la ruta
        if "mri_path" in row and pd.notna(row["mri_path"]):
            mri_path = os.path.join(output_dir, row["mri_path"])
            if os.path.exists(mri_path):
                mri_data = load_nifti_file(mri_path)
                if mri_data is not None:
                    subject_dict["mri_data"] = mri_data
                else:
                    print(f"Error al cargar MRI desde {mri_path}")

        # Cargar PET - Asegurar que existe la columna y la ruta
        if "pet_path" in row and pd.notna(row["pet_path"]):
            pet_path = os.path.join(output_dir, row["pet_path"])
            if os.path.exists(pet_path):
                pet_data = load_nifti_file(pet_path)
                if pet_data is not None:
                    subject_dict["pet_data"] = pet_data
                else:
                    print(f"Error al cargar PET desde {pet_path}")

        # Añadir a la lista de datos de sujetos
        if subject_dict and "mri_data" in subject_dict and "mri_data" in subject_dict:
            subject_dict["mri_path"] = subject_dict["mri_data"]
            subject_dict["pet_path"] = subject_dict["pet_data"]

            subject_data.append(subject_dict)
        else:
            logger.warning(f"Datos incompletos para {row['subject_id']}")
            if row["subject_id"] == "109_S_0967":
                print(subject_dict)

    print(f"Subjects with valid image data: {len(subject_data)} / {len(metadata)}")

    # 8. Crear archivo HDF5 maestro
    if subject_data:
        hdf5_path = hdf5_dir / "adni_dataset.h5"
        create_h5_dataset(
            str(hdf5_path), subject_data, modalities=["MRI", "PET"], verbose=verbose
        )
        if verbose:
            logger.info(f"Dataset HDF5 maestro creado en {hdf5_path}")

        # 9. Crear archivos HDF5 por split para compatibilidad con AdniDataset
        n_splits = 5  # Número de splits por defecto
        if verbose:
            logger.info(
                f"Creando {n_splits} splits para compatibilidad con AdniDataset..."
            )

        # Dividir dataset en train/valid/test para cada split
        # Utilizamos la función split_dataset de process_data.py
        try:
            split_dataset(
                input_path=str(hdf5_path),
                output_dir=str(hdf5_dir),
                n_splits=n_splits,
                verbose=verbose,
            )
            if verbose:
                logger.info(f"Splits creados correctamente en {hdf5_dir}")
                logger.info(
                    "Formato de archivos: {split}-train.h5, {split}-valid.h5, {split}-test.h5"
                )
        except Exception as e:
            logger.error(f"Error al crear splits: {e}")
            # Si fallamos al dividir, podemos intentar una alternativa manual
            if verbose:
                logger.info("Intentando crear splits manualmente...")

            # Abrir el dataset maestro y dividir manualmente
            with h5py.File(hdf5_path, "r") as f:
                # Obtener todas las IDs (excepto 'stats' si existe)
                subject_ids = [k for k in f.keys() if k != "stats"]

                # Si no hay suficientes sujetos, ajustar n_splits
                if (
                    len(subject_ids) < n_splits * 3
                ):  # Necesitamos al menos 3 sujetos por split
                    n_splits = max(1, len(subject_ids) // 3)
                    logger.warning(
                        f"Ajustando a {n_splits} splits debido a la cantidad limitada de datos"
                    )

                # Crear archivos split
                for split in range(n_splits):
                    # División simple: dividir en partes aproximadamente iguales
                    n_subjects = len(subject_ids)
                    train_size = int(0.7 * n_subjects)
                    valid_size = int(0.15 * n_subjects)

                    # Usar diferentes partes del dataset para cada split
                    offset = (split * (train_size + valid_size)) % n_subjects

                    # Crear índices circulares
                    indices = list(range(n_subjects))
                    indices = indices[offset:] + indices[:offset]

                    # Dividir indices
                    train_indices = indices[:train_size]
                    valid_indices = indices[train_size : train_size + valid_size]
                    test_indices = indices[train_size + valid_size :]

                    # Crear archivos para este split
                    splits = [
                        (train_indices, f"{split}-train.h5"),
                        (valid_indices, f"{split}-valid.h5"),
                        (test_indices, f"{split}-test.h5"),
                    ]

                    for subset_indices, filename in splits:
                        out_path = hdf5_dir / filename
                        with h5py.File(out_path, "w") as out_f:
                            # Copiar datos de los sujetos seleccionados
                            for idx in subset_indices:
                                subject_id = subject_ids[idx]
                                f.copy(f[subject_id], out_f, name=subject_id)

                            # Copiar stats si existen
                            if "stats" in f:
                                f.copy("stats", out_f)

                        if verbose:
                            logger.info(
                                f"Creado archivo {out_path} con {len(subset_indices)} sujetos"
                            )
    else:
        if verbose:
            logger.warning(
                "No se pudieron cargar datos de imagen. No se creó el archivo HDF5."
            )


def main():
    """Función principal para ejecutar desde línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Procesar datos ADNI desde DICOM hasta HDF5"
    )

    parser.add_argument(
        "--dicom-dir", type=str, required=True, help="Directorio con datos DICOM"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio para archivos de salida",
    )
    parser.add_argument(
        "--clinical-csv",
        type=str,
        help="Archivo CSV con datos clínicos (opcional, se generará automáticamente desde XML si no se proporciona)",
    )
    parser.add_argument(
        "--generate-clinical",
        action="store_true",
        help="Generar datos clínicos desde XML incluso si se proporciona un CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar mensajes detallados durante el procesamiento",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Reducir mensajes al mínimo"
    )

    args = parser.parse_args()

    # Configurar nivel de verbosidad
    verbose = args.verbose and not args.quiet
    set_global_log_level(verbose)

    try:
        # Si se solicitó generar datos clínicos o no se proporcionó CSV
        if args.generate_clinical or not args.clinical_csv:
            adni_metadata_dir = Path(args.dicom_dir).parent / "ADNI Metadata"
            if adni_metadata_dir.exists():
                if verbose:
                    logger.info(
                        f"Generando datos clínicos desde XML en {adni_metadata_dir}"
                    )
                clinical_data = create_clinical_data_from_metadata(adni_metadata_dir)

                if not clinical_data.empty:
                    # Guardar CSV generado
                    output_path = Path(args.output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    auto_clinical_csv = output_path / "auto_clinical_data.csv"
                    clinical_data.to_csv(auto_clinical_csv, index=False)
                    if verbose:
                        logger.info(f"Datos clínicos guardados en: {auto_clinical_csv}")

                    # Usar el CSV generado si no se proporcionó uno
                    if not args.clinical_csv:
                        args.clinical_csv = str(auto_clinical_csv)
                        if verbose:
                            logger.info(
                                f"Usando datos clínicos generados: {args.clinical_csv}"
                            )

        if args.clinical_csv and not os.path.exists(args.clinical_csv):
            logger.warning(
                f"Archivo CSV de datos clínicos no encontrado: {args.clinical_csv}"
            )
            logger.warning(
                "Continuando sin datos clínicos. Los diagnósticos se marcarán como 'Unknown'."
            )
            args.clinical_csv = None

        process_adni_workflow(
            args.dicom_dir, args.output_dir, args.clinical_csv, verbose
        )
        if verbose:
            logger.info("Procesamiento de datos ADNI completado correctamente")

        # Verificar explícitamente que el archivo de metadatos se creó correctamente
        metadata_path = Path(args.output_dir) / "metadata.csv"
        if not os.path.exists(metadata_path):
            print(f"ERROR: No se pudo crear el archivo de metadatos en {metadata_path}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error en el procesamiento de datos ADNI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
