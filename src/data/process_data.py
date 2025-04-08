# UNUSED


import os
import sys
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import logging
import re
from tqdm import tqdm
from typing import Dict, List
from sklearn.model_selection import train_test_split

# Configurar el logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Mapeo de diagnóstico (también definido en adni.py)
DIAGNOSIS_MAP = {"CN": 0, "MCI": 1, "FTD": 1, "Dementia": 2, "AD": 2}
DIAGNOSIS_MAP_binary = {"CN": 0, "Dementia": 1, "AD": 1}


def load_nifti_file(
    file_path: str, target_shape: tuple = (128, 128, 128), normalize_to_mni: bool = True
) -> np.ndarray:
    """
    Carga un archivo NIFTI y devuelve sus datos como un array de NumPy.

    Implementa el procesamiento descrito:
    - Normalización al espacio MNI152 (si normalize_to_mni es True)
    - Tamaño de vóxel de 1.5mm³
    - Extracción de mapas de densidad de materia gris (para MRI)
    - Normalización de intensidad a [0, 1]
    - Redimensionado a 128x128x128

    Args:
        file_path: Ruta al archivo NIFTI
        target_shape: Forma objetivo a la que redimensionar (por defecto 128x128x128)
        normalize_to_mni: Si es True, asume que las imágenes ya están normalizadas a MNI152
                          o intenta normalizarlas usando SimpleITK

    Returns:
        Array NumPy con los datos de la imagen
    """
    try:
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)

        # Si es necesario normalizar a MNI152 y no está ya normalizado
        if normalize_to_mni and not is_normalized_to_mni(img):
            logger.warning(
                f"La imagen {file_path} no está normalizada al espacio MNI152."
            )
            logger.warning("Se recomienda normalizar las imágenes previamente.")
            # Aquí se podría implementar la normalización a MNI152 si es necesario
            # usando SimpleITK, ANTs u otra herramienta

        # Normalización básica a [0, 1] como se especifica en el documento
        if np.max(data) != np.min(data):
            data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Redimensionar al tamaño objetivo (128x128x128)
        if target_shape != data.shape:
            from scipy.ndimage import zoom

            # Calcular factores de zoom
            factors = [t / s for t, s in zip(target_shape, data.shape)]
            # Redimensionar usando interpolación
            data = zoom(data, factors, order=1)

        # Asegurarse de que los datos tienen la forma correcta para DiaMond (C, H, W, D)
        data = data[np.newaxis, ...]  # Añadir canal si no existe

        return data
    except Exception as e:
        logger.error(f"Error al cargar {file_path}: {e}")
        return None


def is_normalized_to_mni(img):
    """
    Verifica de manera básica si una imagen parece estar normalizada al espacio MNI152.

    Esta es una verificación simplificada que considera:
    - Dimensiones aproximadas
    - Orientación del espacio
    - Rango de coordenadas

    Args:
        img: Objeto NiBabel de la imagen

    Returns:
        Boolean indicando si la imagen parece estar en espacio MNI
    """
    # Verificación simplificada - en un caso real se requeriría análisis más detallado
    # Las dimensiones aproximadas del template MNI152 con vóxel de 1.5mm³
    mni_dims = (121, 145, 121)  # Dimensiones aproximadas para MNI152 1.5mm³

    # Tolerancia para las dimensiones (puede variar según implementación)
    tolerance = 10

    # Verificar dimensiones
    img_dims = img.shape

    # Si las dimensiones están en el rango esperado para MNI152
    for i, d in enumerate(img_dims):
        if abs(d - mni_dims[i]) > tolerance:
            return False

    # Verificación adicional: comprobar orientación, resolución, etc.
    # Esta es una verificación muy básica y podría mejorarse

    return True


def extract_gray_matter(data):
    """
    Extrae mapas de densidad de materia gris a partir de datos MRI.

    Para implementar correctamente esta función se necesitaría utilizar
    herramientas como SPM, FSL o CAT12 para la segmentación de tejidos.

    Args:
        data: Datos de imagen MRI como array de NumPy

    Returns:
        Array NumPy con los mapas de densidad de materia gris
    """
    # Aquí iría la implementación real de extracción de materia gris
    # Esta es solo una función placeholder

    logger.warning("La extracción de materia gris no está implementada.")
    logger.warning("Se recomienda realizar este paso de procesamiento previamente.")

    return data


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
                mri_group.create_dataset(
                    "data", data=subject["mri_data"], dtype=np.float32
                )
                if verbose:
                    logger.debug(f"Añadidos datos MRI para sujeto {subject_id}")

            # Añadir datos de PET si están disponibles
            if "PET" in modalities and has_pet:
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


def prepare_dataset_splits(
    metadata: pd.DataFrame,
    data_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    valid_size: float = 0.15,
    random_seed: int = 42,
    n_splits: int = 5,
    verbose: bool = True,
    target_shape: tuple = (128, 128, 128),
) -> None:
    """
    Prepara divisiones de entrenamiento/validación/prueba para validación cruzada.

    Args:
        metadata: DataFrame con metadatos de los sujetos
        data_dir: Directorio con los datos crudos
        output_dir: Directorio donde se guardarán los archivos HDF5
        test_size: Fracción de datos para prueba
        valid_size: Fracción de datos de entrenamiento para validación
        random_seed: Semilla para reproducibilidad
        n_splits: Número de divisiones para validación cruzada
        verbose: Si es True, muestra información adicional durante el proceso
        target_shape: Forma objetivo para todas las imágenes (H, W, D)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Primero separamos el conjunto de prueba
    train_val_df, test_df = train_test_split(
        metadata,
        test_size=test_size,
        random_state=random_seed,
        stratify=metadata["diagnosis"],
    )

    # Crear datasets
    test_subjects = load_and_process_subjects(test_df, data_dir, target_shape)
    create_h5_dataset(
        os.path.join(output_dir, "test.h5"), test_subjects, verbose=verbose
    )

    print("Creating splits")

    # Crear divisiones para validación cruzada
    for split in range(n_splits):
        # Para cada split, usamos una semilla diferente
        split_seed = random_seed + split

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=valid_size,
            random_state=split_seed,
            stratify=train_val_df["diagnosis"],
        )

        train_subjects = load_and_process_subjects(train_df, data_dir, target_shape)
        val_subjects = load_and_process_subjects(val_df, data_dir, target_shape)

        create_h5_dataset(
            os.path.join(output_dir, f"{split}-train.h5"),
            train_subjects,
            verbose=verbose,
        )

        create_h5_dataset(
            os.path.join(output_dir, f"{split}-valid.h5"), val_subjects, verbose=verbose
        )

        create_h5_dataset(
            os.path.join(output_dir, f"{split}-test.h5"), test_subjects, verbose=verbose
        )


def load_and_process_subjects(
    metadata: pd.DataFrame, data_dir: str, target_shape: tuple = (128, 128, 128)
) -> List[Dict]:
    """
    Carga y procesa los datos de imágenes para un conjunto de sujetos.

    Procesamiento específico para el estudio de diagnóstico diferencial AD/FTD:
    - Normalización a MNI152
    - Tamaño de vóxel de 1.5mm³
    - Extracción de mapas de densidad de materia gris para MRI
    - Normalización de intensidad a [0, 1]
    - Redimensionado a 128x128x128

    Args:
        metadata: DataFrame con metadatos de los sujetos
        data_dir: Directorio con los datos crudos
        target_shape: Forma objetivo para todas las imágenes (por defecto 128x128x128)

    Returns:
        Lista de diccionarios con datos de sujetos procesados
    """
    subjects = []

    for _, row in tqdm(
        metadata.iterrows(), total=len(metadata), desc="Processing subjects"
    ):
        subject_dict = {
            "rid": row["subject_id"],
            "diagnosis": row["diagnosis"],
        }

        # Load MRI
        if "mri_path" in row and pd.notna(row["mri_path"]):
            mri_path = os.path.join(data_dir, row["mri_path"])
            if os.path.exists(mri_path):
                mri_data = load_nifti_file(mri_path, target_shape)
                if mri_data is not None:
                    subject_dict["mri_data"] = mri_data

        # Load PET
        if "pet_path" in row and pd.notna(row["pet_path"]):
            pet_path = os.path.join(data_dir, row["pet_path"])
            if os.path.exists(pet_path):
                pet_data = load_nifti_file(pet_path, target_shape)
                if pet_data is not None:
                    subject_dict["pet_data"] = pet_data

        # Include all subjects, even if one modality is missing
        subjects.append(subject_dict)

    return subjects


def process_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Procesa el archivo de metadatos.

    Args:
        metadata_path: Ruta al archivo de metadatos (CSV)

    Returns:
        DataFrame con metadatos procesados
    """
    try:
        df = pd.read_csv(metadata_path)

        # Validar columnas requeridas
        required_cols = ["subject_id", "diagnosis"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Columna requerida '{col}' no encontrada en el archivo de metadatos"
                )

        # Convertir IDs de sujeto al formato estándar si es necesario
        if "subject_id" in df.columns:
            # Asegurarse de que los IDs de sujeto sean strings
            df["subject_id"] = df["subject_id"].astype(str)
            df = df.drop_duplicates(subset="subject_id")

        # Filtrar sujetos sin diagnóstico
        df = df.dropna(subset=["diagnosis"])

        # Verificar que los diagnósticos están en el mapeo
        valid_diagnosis = list(DIAGNOSIS_MAP.keys()) + list(DIAGNOSIS_MAP_binary.keys())
        invalid_dx = df[~df["diagnosis"].isin(valid_diagnosis)]
        if len(invalid_dx) > 0:
            logger.warning(
                f"Encontrados {len(invalid_dx)} sujetos con diagnósticos inválidos."
            )
            logger.warning(f"Diagnósticos válidos: {valid_diagnosis}")
            logger.warning("Excluyendo estos sujetos del procesamiento.")
            df = df[df["diagnosis"].isin(valid_diagnosis)]

        return df

    except Exception as e:
        logger.error(f"Error al procesar el archivo de metadatos: {e}")
        return None


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


def main():
    """Función principal para procesar los datos."""
    import argparse

    parser = argparse.ArgumentParser(description="Procesar datos para DiaMond")
    parser.add_argument(
        "--metadata", type=str, required=True, help="Ruta al archivo de metadatos (CSV)"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directorio con los datos crudos"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio donde se guardarán los archivos HDF5",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fracción de datos para prueba"
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.15,
        help="Fracción de datos de entrenamiento para validación",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Semilla para reproducibilidad"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Número de divisiones para validación cruzada",
    )
    parser.add_argument(
        "--target-shape",
        type=str,
        default="128,128,128",
        help="Forma objetivo para todas las imágenes (H,W,D)",
    )
    parser.add_argument(
        "--voxel-size",
        type=str,
        default="1.5,1.5,1.5",
        help="Tamaño de vóxel en mm³ (X,Y,Z)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Reducir mensajes de salida al mínimo"
    )
    parser.add_argument(
        "--skip-mni-check",
        action="store_true",
        help="Omitir verificación de normalización MNI152",
    )

    args = parser.parse_args()

    # Configurar nivel de logging según el parámetro quiet
    if args.quiet:
        logger.setLevel(logging.WARNING)

    # Procesar target_shape
    target_shape = tuple(map(int, args.target_shape.split(",")))

    # Procesar voxel_size
    voxel_size = tuple(map(float, args.voxel_size.split(",")))

    logger.info(
        f"Usando tamaño objetivo: {target_shape}, tamaño de vóxel: {voxel_size} mm³"
    )

    # Procesar metadatos
    logger.info(f"Procesando metadatos desde {args.metadata}")
    metadata = process_metadata(args.metadata)
    if metadata is None:
        logger.error("Error al procesar metadatos. Abortando.")
        sys.exit(1)

    logger.info("Preparando splits del dataset")
    prepare_dataset_splits(
        metadata=metadata,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        valid_size=args.valid_size,
        random_seed=args.random_seed,
        n_splits=args.n_splits,
        verbose=not args.quiet,
        target_shape=target_shape,
    )

    logger.info("Procesamiento de datos completado.")


if __name__ == "__main__":
    main()
