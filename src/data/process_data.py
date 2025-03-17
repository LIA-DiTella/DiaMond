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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as h5f:
        # Crear grupo de estadísticas
        stats_group = h5f.create_group("stats")
        stats_group.attrs["num_subjects"] = len(subject_data)

        # Contar diagnósticos
        all_dx = [s["diagnosis"] for s in subject_data]
        dx_counts = {dx: all_dx.count(dx) for dx in set(all_dx)}
        for dx, count in dx_counts.items():
            stats_group.attrs[f"count_{dx}"] = count

        # Añadir cada sujeto
        for i, subject in enumerate(tqdm(subject_data, desc=f"Creando {output_path}")):
            # Usar el RID si está disponible, de lo contrario usar índice
            subject_id = subject.get("rid", f"subject_{i}")

            # Limpiar el id para asegurar que sea un nombre de grupo válido en HDF5
            subject_id = re.sub(r"[^\w]", "_", str(subject_id))

            subject_group = h5f.create_group(subject_id)
            subject_group.attrs["RID"] = subject["rid"]
            subject_group.attrs["DX"] = subject["diagnosis"]

            # Añadir datos de MRI si están disponibles
            if "MRI" in modalities and "mri_data" in subject:
                mri_group = subject_group.create_group("MRI/T1")
                mri_group.create_dataset("data", data=subject["mri_data"])
                if verbose:
                    logger.debug(f"Añadidos datos MRI para sujeto {subject_id}")

            # Añadir datos de PET si están disponibles
            if "PET" in modalities and "pet_data" in subject:
                pet_group = subject_group.create_group("PET/FDG")
                pet_group.create_dataset("data", data=subject["pet_data"])
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
    test_subjects = load_and_process_subjects(test_df, data_dir)
    create_h5_dataset(
        os.path.join(output_dir, "test.h5"), test_subjects, verbose=verbose
    )

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

        train_subjects = load_and_process_subjects(train_df, data_dir)
        val_subjects = load_and_process_subjects(val_df, data_dir)

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


def load_and_process_subjects(metadata: pd.DataFrame, data_dir: str) -> List[Dict]:
    """
    Carga y procesa los datos de imágenes para un conjunto de sujetos.

    Args:
        metadata: DataFrame con metadatos de los sujetos
        data_dir: Directorio con los datos crudos

    Returns:
        Lista de diccionarios con datos de sujetos procesados
    """
    subjects = []

    for _, row in tqdm(
        metadata.iterrows(), total=len(metadata), desc="Procesando sujetos"
    ):
        subject_dict = {
            "rid": row["subject_id"],
            "diagnosis": row["diagnosis"],
        }

        # Cargar MRI
        if "mri_path" in row and pd.notna(row["mri_path"]):
            mri_path = os.path.join(data_dir, row["mri_path"])
            if os.path.exists(mri_path):
                mri_data = load_nifti_file(mri_path)
                if mri_data is not None:
                    subject_dict["mri_data"] = mri_data

        # Cargar PET
        if "pet_path" in row and pd.notna(row["pet_path"]):
            pet_path = os.path.join(data_dir, row["pet_path"])
            if os.path.exists(pet_path):
                pet_data = load_nifti_file(pet_path)
                if pet_data is not None:
                    subject_dict["pet_data"] = pet_data

        # Solo incluir si tiene al menos una modalidad
        if "mri_data" in subject_dict or "pet_data" in subject_dict:
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
        with h5py.File(input_path, 'r') as h5f:
            # Obtener IDs de sujetos (excluyendo 'stats' si existe)
            subject_ids = [k for k in h5f.keys() if k != 'stats']
            
            if verbose:
                logger.info(f"Dividiendo dataset con {len(subject_ids)} sujetos en {n_splits} splits")
            
            # Extraer diagnósticos para estratificación
            diagnoses = [h5f[sid].attrs.get('DX', 'Unknown') for sid in subject_ids]
            
            # Convertir a DataFrame para facilitar la división
            df = pd.DataFrame({'subject_id': subject_ids, 'diagnosis': diagnoses})
            
            # Primero separar el conjunto de prueba global
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_seed,
                stratify=df['diagnosis'] if len(set(df['diagnosis'])) > 1 else None
            )
            
            # Crear conjuntos para cada split de validación cruzada
            for split in range(n_splits):
                split_seed = random_seed + split
                
                # Para cada split, crear train/val/test
                if len(set(train_val_df['diagnosis'])) > 1:
                    train_df, val_df = train_test_split(
                        train_val_df,
                        test_size=valid_size,
                        random_state=split_seed,
                        stratify=train_val_df['diagnosis']
                    )
                else:
                    train_df, val_df = train_test_split(
                        train_val_df,
                        test_size=valid_size,
                        random_state=split_seed
                    )
                
                # Crear archivos HDF5 para cada conjunto
                split_files = [
                    (train_df, f"{split}-train.h5", "entrenamiento"),
                    (val_df, f"{split}-valid.h5", "validación"),
                    (test_df, f"{split}-test.h5", "prueba")
                ]
                
                for subset_df, filename, subset_name in split_files:
                    output_path = os.path.join(output_dir, filename)
                    
                    with h5py.File(output_path, 'w') as out_f:
                        # Copiar estadísticas si existen
                        if 'stats' in h5f:
                            h5f.copy('stats', out_f)
                            
                            # Actualizar estadísticas para este subset
                            if 'stats' in out_f:
                                out_f['stats'].attrs['num_subjects'] = len(subset_df)
                                
                                # Actualizar conteo de diagnósticos
                                dx_counts = subset_df['diagnosis'].value_counts().to_dict()
                                for dx, count in dx_counts.items():
                                    out_f['stats'].attrs[f'count_{dx}'] = count
                        
                        # Copiar sujetos
                        for _, row in subset_df.iterrows():
                            subject_id = row['subject_id']
                            h5f.copy(h5f[subject_id], out_f, name=subject_id)
                    
                    if verbose:
                        logger.info(f"Split {split}, conjunto de {subset_name}: {len(subset_df)} sujetos guardados en {output_path}")

        if verbose:
            logger.info(f"División de dataset completada. Archivos guardados en {output_dir}")

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
        "--quiet", action="store_true", help="Reducir mensajes de salida al mínimo"
    )

    args = parser.parse_args()

    # Configurar nivel de logging según el parámetro quiet
    if args.quiet:
        logger.setLevel(logging.WARNING)

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
    )

    logger.info("Procesamiento de datos completado.")


if __name__ == "__main__":
    main()
