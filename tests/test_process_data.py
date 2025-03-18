"""
Pruebas unitarias para el módulo process_data.py
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
from unittest.mock import MagicMock, patch
import pandas as pd
import h5py

# Añadir el directorio src al path para importar módulos
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/data"))
)

# Verificar disponibilidad de dependencias
SKIP_REASON = None
try:
    import nibabel as nib
except ImportError:
    SKIP_REASON = "El módulo nibabel no está instalado"

try:
    from process_data import load_nifti_file, create_h5_dataset, process_metadata
except ImportError as e:
    SKIP_REASON = f"Módulo requerido no está instalado: {e}"
    # Crear mocks para las funciones que vamos a testear
    load_nifti_file = MagicMock(return_value=np.random.rand(1, 100, 100, 100))
    create_h5_dataset = MagicMock()
    process_metadata = MagicMock(
        return_value=pd.DataFrame(
            {"subject_id": ["001", "002", "003"], "diagnosis": ["CN", "AD", "MCI"]}
        )
        if "pandas" in sys.modules
        else None
    )


@unittest.skipIf(SKIP_REASON, SKIP_REASON)
class TestProcessData(unittest.TestCase):
    def setUp(self):
        # Crear directorio temporal para pruebas
        self.test_dir = tempfile.mkdtemp()

        # Crear archivos de prueba
        self.create_test_files()

        # Crear datos de ejemplo para MRI y PET
        mri_data = np.random.rand(1, 10, 10, 10).astype(
            np.float32
        )  # Datos pequeños para test
        pet_data = np.random.rand(1, 10, 10, 10).astype(
            np.float32
        )  # Datos pequeños para test

        # Inicialización de test_subjects para las pruebas
        self.test_subjects = [
            {
                "id": "subject1",
                "rid": "001",  # Añadir RID (Research ID)
                "path": os.path.join(self.test_dir, "subject1.nii.gz"),
                "label": 0,
                "diagnosis": "CN",  # Control Normal
                "mri_data": mri_data,  # Añadir datos MRI
                "pet_data": pet_data,  # Añadir datos PET
            },
            {
                "id": "subject2",
                "rid": "002",  # Añadir RID (Research ID)
                "path": os.path.join(self.test_dir, "subject2.nii.gz"),
                "label": 1,
                "diagnosis": "AD",  # Alzheimer's Disease
                "mri_data": mri_data,  # Añadir datos MRI
                "pet_data": pet_data,  # Añadir datos PET
            },
        ]

        # Crear archivos dummy para las pruebas
        for subject in self.test_subjects:
            # Crear un archivo vacío o con datos mínimos
            with open(subject["path"], "wb") as f:
                f.write(b"\x00" * 100)  # Escribir algunos bytes

    def tearDown(self):
        # Limpiar directorios temporales
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_files(self):
        """Crear archivos de prueba necesarios"""
        # Crear un archivo NIFTI de prueba
        if "nibabel" in sys.modules:
            test_data = np.random.rand(100, 100, 100).astype(np.float32)
            self.test_nifti = os.path.join(self.test_dir, "test.nii.gz")
            nib.save(nib.Nifti1Image(test_data, np.eye(4)), self.test_nifti)
        else:
            self.test_nifti = os.path.join(self.test_dir, "test.nii.gz")
            # Crear un archivo vacío como placeholder
            with open(self.test_nifti, "w") as f:
                f.write("")

        # Crear un archivo CSV de metadatos de prueba
        if "pandas" in sys.modules:
            metadata = {
                "subject_id": ["001", "002", "003"],
                "diagnosis": ["CN", "AD", "MCI"],
                "mri_path": [self.test_nifti, self.test_nifti, self.test_nifti],
                "pet_path": [self.test_nifti, self.test_nifti, self.test_nifti],
            }
            self.metadata_csv = os.path.join(self.test_dir, "metadata.csv")
            pd.DataFrame(metadata).to_csv(self.metadata_csv, index=False)
        else:
            self.metadata_csv = os.path.join(self.test_dir, "metadata.csv")
            # Crear un archivo CSV dummy
            with open(self.metadata_csv, "w") as f:
                f.write("subject_id,diagnosis,mri_path,pet_path\n")
                f.write(f"001,CN,{self.test_nifti},{self.test_nifti}\n")

    def test_load_nifti_file(self):
        """Probar la carga de archivos NIFTI"""
        if not os.path.exists(self.test_nifti) or os.path.getsize(self.test_nifti) == 0:
            self.skipTest("Archivo NIFTI de prueba no disponible")

        data = load_nifti_file(self.test_nifti)
        if not isinstance(data, MagicMock):
            self.assertIsNotNone(data)
            self.assertEqual(len(data.shape), 4)  # Debe incluir dimensión de canal
            self.assertEqual(data.shape[0], 1)  # Un solo canal
            self.assertTrue(np.all((data >= 0) & (data <= 1)))  # Datos normalizados

    def test_create_h5_dataset(self):
        """Probar la creación de archivos HDF5"""
        if "h5py" not in sys.modules:
            self.skipTest("h5py no está disponible")

        # Preparar datos de prueba
        mri_data = np.random.rand(1, 100, 100, 100).astype(np.float32)
        pet_data = np.random.rand(1, 100, 100, 100).astype(np.float32)

        subject_data = [
            {
                "rid": "001",
                "diagnosis": "CN",
                "mri_data": mri_data,
                "pet_data": pet_data,
            },
            {"rid": "002", "diagnosis": "AD", "mri_data": mri_data},
        ]

        output_path = os.path.join(self.test_dir, "test_dataset.h5")

        # Crear el dataset
        create_h5_dataset(output_path, subject_data)

        # Verificaciones
        if not isinstance(create_h5_dataset, MagicMock):
            self.assertTrue(os.path.exists(output_path))

            # Verificar contenido
            with h5py.File(output_path, "r") as h5f:
                self.assertTrue("stats" in h5f)
                self.assertTrue("001" in h5f)
                self.assertTrue("002" in h5f)

                # Verificar atributos
                self.assertEqual(h5f["001"].attrs["DX"], "CN")
                self.assertEqual(h5f["002"].attrs["DX"], "AD")

                # Verificar datos
                self.assertTrue("MRI/T1/data" in h5f["001"])
                self.assertTrue("PET/FDG/data" in h5f["001"])
                self.assertTrue("MRI/T1/data" in h5f["002"])
                self.assertFalse("PET/FDG/data" in h5f["002"])

    def test_process_metadata(self):
        """Probar el procesamiento de metadatos"""
        if "pandas" not in sys.modules:
            self.skipTest("pandas no está disponible")

        result = process_metadata(self.metadata_csv)

        if not isinstance(process_metadata, MagicMock):
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 3)
            self.assertTrue("subject_id" in result.columns)
            self.assertTrue("diagnosis" in result.columns)

            # Verificar que los diagnósticos están filtrados correctamente
            valid_dx = list(set(result["diagnosis"]))
            for dx in valid_dx:
                self.assertTrue(dx in ["CN", "AD", "MCI"])

    def test_create_h5_dataset_verbose_parameter(self):
        """Verificar que el parámetro verbose se maneja correctamente en create_h5_dataset"""
        output_path = os.path.join(self.test_dir, "test_verbose.h5")

        with patch("logging.Logger.debug") as mock_debug:
            # Probar con verbose=True
            create_h5_dataset(output_path, self.test_subjects, verbose=True)
            # Verificar que se hayan llamado los logs de debug
            self.assertTrue(mock_debug.called)

        # Reset mock
        mock_debug.reset_mock()

        output_path2 = os.path.join(self.test_dir, "test_nonverbose.h5")
        with patch("logging.Logger.debug") as mock_debug:
            # Probar con verbose=False (default)
            create_h5_dataset(output_path2, self.test_subjects, verbose=False)
            # Verificar que no se hayan llamado los logs de debug
            self.assertFalse(mock_debug.called)

    def test_dataset_creation(self):
        """Verificar que el dataset H5 se crea correctamente con los sujetos proporcionados"""
        output_path = os.path.join(self.test_dir, "test.h5")
        create_h5_dataset(output_path, self.test_subjects)

        # Verificar que el archivo fue creado
        self.assertTrue(os.path.exists(output_path))

        # Verificar contenido del archivo H5
        with h5py.File(output_path, "r") as h5f:
            # Verificar que están los sujetos
            # Los nombres de grupos deberían coincidir con los RIDs que establecimos
            self.assertIn("001", h5f)  # Cambio de 'subject_001' a '001'
            self.assertIn("002", h5f)  # Cambio de 'subject_002' a '002'

            # Verificar que están los diagnósticos correctos
            self.assertEqual(h5f["001"].attrs["DX"], "CN")
            self.assertEqual(h5f["002"].attrs["DX"], "AD")

            # Verificar que están los datos de MRI y PET
            self.assertIn("MRI/T1/data", h5f["001"])
            self.assertIn("PET/FDG/data", h5f["001"])


if __name__ == "__main__":
    unittest.main()
