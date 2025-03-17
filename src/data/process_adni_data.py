"""
Script para procesar datos ADNI:
1. Organizar los datos convertidos
2. Generar los archivos HDF5 para DiaMond
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

# Importar funciones de los otros módulos
from dicom_converter import batch_process_directories, set_global_log_level
from process_data import create_h5_dataset, load_nifti_file, split_dataset

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


def extract_subject_metadata(dicom_dir, nifti_dir):
    """
    Extrae metadatos de los sujetos basados en archivos DICOM y NIFTI

    Args:
        dicom_dir (str): Directorio con datos DICOM
        nifti_dir (str): Directorio con archivos NIFTI convertidos
    Returns:
        pd.DataFrame: DataFrame con metadatos de sujetos
    """
    metadata = []
    nifti_path = Path(nifti_dir)

    # Buscar archivos JSON de información creados durante la conversión
    info_files = list(nifti_path.glob("**/mri_scan_info.json")) + list(
        nifti_path.glob("**/pet_scan_info.json")
    )

    # También buscar en la carpeta de metadatos si existe
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
                # Aquí se podrían extraer datos de los XML si es necesario
                # Por ahora solo añadimos el ID de sujeto
                entry = {
                    "subject_id": subject_id,
                    "diagnosis": "Unknown",  # Se completará más tarde
                    "has_metadata": True,
                }
                metadata.append(entry)

    # Procesar archivos de información de conversión
    for info_file in info_files:
        subject_id = info_file.parent.name
        modality = "mri" if "mri" in info_file.name else "pet"

        try:
            with open(info_file) as f:
                info = json.load(f)

            # Buscar archivo NIFTI relacionado
            nifti_file = info.get("nifti_output", "")
            if nifti_file and os.path.exists(nifti_file):
                # Crear o actualizar entrada existente
                existing_entry = next(
                    (item for item in metadata if item["subject_id"] == subject_id),
                    None,
                )

                if existing_entry:
                    existing_entry[f"{modality}_path"] = nifti_file
                else:
                    entry = {
                        "subject_id": subject_id,
                        "diagnosis": "Unknown",  # Se completará más tarde
                        f"{modality}_path": nifti_file,
                    }
                    metadata.append(entry)
        except Exception as e:
            logger.warning(f"Error al procesar {info_file}: {e}")

    # Convertir a DataFrame y combinar entradas con el mismo subject_id
    df = pd.DataFrame(metadata)
    if not df.empty:
        df = df.groupby("subject_id").first().reset_index()

    return df


def merge_with_clinical_data(metadata, clinical_csv):
    """
    Combina metadatos extraídos con datos clínicos

    Args:
        metadata (pd.DataFrame): DataFrame con metadatos extraídos
        clinical_csv (str): Ruta al archivo CSV con datos clínicos
    Returns:
        pd.DataFrame: DataFrame combinado
    """
    try:
        clinical = pd.read_csv(clinical_csv)

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

    print(clinical_csv_df)

    # Establecer nivel de log global
    set_global_log_level(verbose)

    # 1. Crear directorios de salida
    output_path = Path(output_dir)
    nifti_dir = output_path / "nifti"
    hdf5_dir = output_path / "hdf5"

    os.makedirs(nifti_dir, exist_ok=True)
    os.makedirs(hdf5_dir, exist_ok=True)

    # 2. Encontrar sujetos ADNI
    # subject_dirs = find_adni_subjects(dicom_dir)

    # 3. Convertir archivos DICOM a NIFTI
    # if verbose:
    # logger.info("Convirtiendo archivos DICOM a NIFTI...")
    # batch_process_directories(
    #     subject_dirs, str(nifti_dir), modalities=["mri", "pet"], verbose=verbose
    # )

    # If not metadata.csv
    if not os.path.exists(output_path / "metadata.csv"):
        # 4. Extraer metadatos
        if verbose:
            logger.info("Extrayendo metadatos...")
        metadata = extract_subject_metadata(dicom_dir, nifti_dir)

        # 5. Combinar con datos clínicos si están disponibles
        if clinical_csv and os.path.exists(clinical_csv):
            if verbose:
                logger.info(
                    f"Combinando con datos clínicos desde archivo CSV: {clinical_csv}"
                )
            metadata = merge_with_clinical_data(metadata, clinical_csv)
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

    else:
        # Si ya existe un archivo de metadatos, cargarlo
        metadata_path = output_path / "metadata.csv"
        metadata = pd.read_csv(metadata_path)

    # get diagnosis for metadata["diagnosis"] from clinical_csv_df["Group"] where metadata["subject_id"] == clinical_csv_df["Subject"]

    metadata["diagnosis"] = metadata["subject_id"].map(
        lambda x: clinical_csv_df.loc[clinical_csv_df["Subject"] == x, "Group"].values[0]
        if x in clinical_csv_df["Subject"].values
        else "Unknown"
    )

    metadata.to_csv(metadata_path, index=False)

    # 7. Preparar datos para el dataset HDF5
    if verbose:
        logger.info("Preparando datasets...")

    # Cargar imágenes y preparar datos para HDF5
    subject_data = []
    progress_bar = tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Cargando datos de imagen",
        disable=not verbose,
    )
    for _, row in progress_bar:
        subject_dict = {
            "rid": row["subject_id"],
            "diagnosis": row.get("Group", "Unknown"),
        }

        # Cargar MRI
        if "mri_path" in row and pd.notna(row["mri_path"]):
            mri_path = row["mri_path"]
            if os.path.exists(mri_path):
                mri_data = load_nifti_file(mri_path)
                if mri_data is not None:
                    subject_dict["mri_data"] = mri_data

        # Cargar PET
        if "pet_path" in row and pd.notna(row["pet_path"]):
            pet_path = row["pet_path"]
            if os.path.exists(pet_path):
                pet_data = load_nifti_file(pet_path)
                if pet_data is not None:
                    subject_dict["pet_data"] = pet_data

        # Solo incluir si tiene al menos una modalidad
        if "mri_data" in subject_dict or "pet_data" in subject_dict:
            subject_data.append(subject_dict)

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
        """ try:
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
         """
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
