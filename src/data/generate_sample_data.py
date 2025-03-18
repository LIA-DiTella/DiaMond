#!/usr/bin/env python3
"""
Script para generar datos sintéticos de prueba para el pipeline DiaMond.
Crea archivos H5 con estructura compatible con el formato esperado.
"""

import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm

# Constantes
DIAGNOSES = ["CN", "MCI", "AD"]
IMAGE_SIZE = 128


def create_sample_brain_volume(size=128, noise_level=0.1):
    """
    Crea un volumen cerebral sintético simple

    Args:
        size: Tamaño del volumen (tamaño x tamaño x tamaño)
        noise_level: Nivel de ruido aleatorio a añadir

    Returns:
        Array NumPy con forma (1, size, size, size)
    """
    # Crear un volumen 3D con forma aproximada de cerebro
    volume = np.zeros((1, size, size, size), dtype=np.float32)
    center = size // 2
    radius = size // 3

    # Crear una forma elipsoide simple
    for i in range(size):
        for j in range(size):
            for k in range(size):
                # Distancia elipsoide al centro
                dist = np.sqrt(
                    ((i - center) / 1.2) ** 2
                    + ((j - center) / 1.0) ** 2
                    + ((k - center) / 0.8) ** 2
                )

                if dist < radius:
                    # Simular intensidad decreciente del centro hacia afuera
                    intensity = (1.0 - dist / radius) * 0.8
                    # Añadir ruido
                    intensity += np.random.normal(0, noise_level)
                    volume[0, i, j, k] = max(0, min(1, intensity))

    return volume


def create_sample_dataset(output_dir, num_subjects=30, n_splits=5, verbose=True):
    """
    Crea un conjunto de datos H5 de ejemplo con la estructura esperada por DiaMond
    """
    os.makedirs(output_dir, exist_ok=True)

    # Distribuir diagnósticos
    diagnoses = np.random.choice(DIAGNOSES, size=num_subjects, p=[0.4, 0.3, 0.3])

    # Crear datos para cada sujeto
    subject_data = []
    for i in tqdm(range(num_subjects), desc="Generando sujetos"):
        subject_id = f"SUBJECT_{i + 1:03d}"
        dx = diagnoses[i]

        # Crear volúmenes MRI y PET sintéticos
        mri_data = create_sample_brain_volume()
        pet_data = create_sample_brain_volume()  # Diferente pero correlacionado con MRI

        # Añadir variación según el diagnóstico
        if dx == "AD":
            # Simular atrofia en AD
            center = IMAGE_SIZE // 2
            for i in range(IMAGE_SIZE):
                for j in range(IMAGE_SIZE):
                    for k in range(20, 40):  # Región específica
                        dist = np.sqrt(
                            (i - center) ** 2 + (j - center) ** 2 + (k - 30) ** 2
                        )
                        if dist < 15:
                            mri_data[0, i, j, k] *= 0.7  # Reducción de volumen
                            pet_data[0, i, j, k] *= 0.5  # Hipometabolismo

        subject_data.append(
            {"id": subject_id, "diagnosis": dx, "mri": mri_data, "pet": pet_data}
        )

    # Dividir en conjuntos (pueden solaparse para este ejemplo)
    np.random.shuffle(subject_data)

    # Crear conjuntos para cada split
    split_size = len(subject_data) // n_splits

    # Crear archivo main
    create_h5_file(os.path.join(output_dir, "adni_dataset.h5"), subject_data, verbose)

    # Crear splits
    for split in range(n_splits):
        # Indices circulares para crear diferentes splits
        start_idx = (split * split_size) % len(subject_data)

        # Train: ~70% de los datos
        train_size = int(len(subject_data) * 0.7)
        train_indices = [(start_idx + i) % len(subject_data) for i in range(train_size)]
        train_data = [subject_data[i] for i in train_indices]

        # Valid: ~15% de los datos
        valid_size = int(len(subject_data) * 0.15)
        valid_start = (start_idx + train_size) % len(subject_data)
        valid_indices = [
            (valid_start + i) % len(subject_data) for i in range(valid_size)
        ]
        valid_data = [subject_data[i] for i in valid_indices]

        # Test: ~15% de los datos
        test_indices = [
            i
            for i in range(len(subject_data))
            if i not in train_indices and i not in valid_indices
        ]
        test_data = [subject_data[i] for i in test_indices]

        # Crear archivos H5
        create_h5_file(
            os.path.join(output_dir, f"{split}-train.h5"), train_data, verbose
        )
        create_h5_file(
            os.path.join(output_dir, f"{split}-valid.h5"), valid_data, verbose
        )
        create_h5_file(os.path.join(output_dir, f"{split}-test.h5"), test_data, verbose)


def create_h5_file(filename, subjects, verbose=True):
    """
    Crea un archivo H5 con la estructura esperada por DiaMond
    """
    with h5py.File(filename, "w") as h5f:
        # Crear grupo de estadísticas
        stats = h5f.create_group("stats")
        stats.attrs["num_subjects"] = len(subjects)

        # Contar diagnósticos
        dx_counts = {}
        for s in subjects:
            dx = s["diagnosis"]
            dx_counts[dx] = dx_counts.get(dx, 0) + 1

        for dx, count in dx_counts.items():
            stats.attrs[f"count_{dx}"] = count

        # Añadir cada sujeto
        for subject in subjects:
            # Crear grupo para el sujeto
            subj_group = h5f.create_group(subject["id"])
            subj_group.attrs["DX"] = subject["diagnosis"]
            subj_group.attrs["RID"] = subject["id"]

            # Añadir datos MRI
            mri_group = subj_group.create_group("MRI/T1")
            mri_group.create_dataset("data", data=subject["mri"])

            # Añadir datos PET
            pet_group = subj_group.create_group("PET/FDG")
            pet_group.create_dataset("data", data=subject["pet"])

    if verbose:
        print(f"Creado archivo {filename} con {len(subjects)} sujetos")
        print(f"Distribución de diagnósticos: {dx_counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Generar datos H5 de muestra para DiaMond"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/hdf5",
        help="Directorio de salida para los archivos H5",
    )
    parser.add_argument(
        "--num-subjects", type=int, default=30, help="Número de sujetos a generar"
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Número de divisiones a crear"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Mostrar información detallada"
    )

    args = parser.parse_args()

    # Asegurar ruta absoluta
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.getcwd(), args.output_dir)

    print(f"Generando {args.num_subjects} sujetos en {args.output_dir}")
    create_sample_dataset(
        args.output_dir,
        num_subjects=args.num_subjects,
        n_splits=args.n_splits,
        verbose=args.verbose,
    )

    print("\n¡Datos generados correctamente!")
    print(f"Para usar estos datos, ejecuta: make train DATASET_PATH={args.output_dir}")


if __name__ == "__main__":
    main()
