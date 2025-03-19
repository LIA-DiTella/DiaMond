"""
Pruebas unitarias para el módulo dicom_converter.py
"""

import os
import sys
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock
import importlib.util  # Añadido para verificar disponibilidad de módulos

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

# Verificar si dicom2nifti está disponible
dicom2nifti_available = importlib.util.find_spec("dicom2nifti") is not None

if dicom2nifti_available:
    try:
        from dicom_converter import (
            find_dicom_folder,
            convert_dicoms_to_nifti,
            verify_nifti_quality,
            process_scan,
        )
    except ImportError:
        SKIP_REASON = "Error al importar funciones de dicom_converter.py"
        find_dicom_folder = convert_dicoms_to_nifti = verify_nifti_quality = (
            process_scan
        ) = None
else:
    SKIP_REASON = "El módulo dicom2nifti no está instalado"
    # Crear mocks para las funciones que vamos a testear
    find_dicom_folder = MagicMock(return_value=Path("/mock/path"))
    convert_dicoms_to_nifti = MagicMock(return_value=Path("/mock/output.nii.gz"))
    verify_nifti_quality = MagicMock(return_value=True)
    process_scan = MagicMock(return_value={"nifti_path": "/mock/output.nii.gz"})


@unittest.skipIf(SKIP_REASON, SKIP_REASON)
class TestDicomConverter(unittest.TestCase):
    def setUp(self):
        # Crear directorio temporal para pruebas
        self.test_dir = tempfile.mkdtemp()

        # Ruta a datos de muestra (ajustar según tu estructura)
        self.sample_data_dir = os.path.join(os.path.dirname(__file__), "sample_data")

        # Crear estructura de carpetas si no existe
        if not os.path.exists(self.sample_data_dir):
            os.makedirs(self.sample_data_dir)

    def tearDown(self):
        # Limpiar directorios temporales
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_find_dicom_folder(self):
        """Verificar que la función encuentra carpetas con archivos DICOM"""
        # Esta prueba requiere datos DICOM de muestra
        # Si tienes un directorio con datos DICOM de muestra, ajusta la ruta
        sample_dicom_dir = os.path.join(self.sample_data_dir, "sample_dicom")

        # Crear un directorio de muestra si no existe
        if not os.path.exists(sample_dicom_dir):
            os.makedirs(sample_dicom_dir)
            dummy_dcm_path = os.path.join(sample_dicom_dir, "test.dcm")
            with open(dummy_dcm_path, "w") as f:
                f.write("dummy dicom content")

        result = find_dicom_folder(Path(sample_dicom_dir))
        self.assertIsNotNone(result, "No se pudo encontrar la carpeta DICOM")
        if not isinstance(
            result, MagicMock
        ):  # Saltar esta comprobación si estamos usando un mock
            self.assertEqual(str(result), sample_dicom_dir)

    @patch("dicom2nifti.convert_directory", create=True)
    def test_convert_dicoms_to_nifti(self, mock_convert):
        """Prueba la conversión de DICOM a NIfTI usando mocking"""
        if "nibabel" in sys.modules:
            # Configurar el mock para simular la conversión exitosa
            def side_effect(input_dir, output_dir, **kwargs):
                # Simular la creación de un archivo nifti
                test_data = np.ones((100, 100, 100), dtype=np.float32)
                test_nifti_path = os.path.join(output_dir, "output.nii.gz")
                nib.save(nib.Nifti1Image(test_data, np.eye(4)), test_nifti_path)

            mock_convert.side_effect = side_effect

        # Crear directorios de prueba
        dicom_folder = Path(os.path.join(self.test_dir, "dicom_test"))
        output_dir = Path(os.path.join(self.test_dir, "nifti_output"))
        dicom_folder.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Ejecutar la función a probar
        result = convert_dicoms_to_nifti(dicom_folder, output_dir)

        # Verificaciones
        if not isinstance(result, MagicMock):  # Si no estamos en modo mock
            self.assertTrue(mock_convert.called)
            self.assertTrue(os.path.exists(result))
            self.assertTrue(str(result).endswith(".nii.gz"))

    @unittest.skipIf(
        "nibabel" not in sys.modules, "Se requiere nibabel para esta prueba"
    )
    def test_verify_nifti_quality(self):
        """Verificar que la función de verificación de calidad funciona correctamente"""
        # Crear un archivo NIFTI de prueba válido con contraste variable
        test_data = (
            np.linspace(1, 100, 100 * 100 * 100)
            .reshape(100, 100, 100)
            .astype(np.float32)
        )
        test_nifti = os.path.join(self.test_dir, "test.nii.gz")
        nib.save(nib.Nifti1Image(test_data, np.eye(4)), test_nifti)

        # Verificar que pasa la verificación de calidad
        result = verify_nifti_quality(test_nifti)
        self.assertTrue(
            result, "La verificación de calidad falló para un archivo NIFTI válido"
        )

        # Crear un archivo NIFTI inválido (muy pequeño)
        invalid_data = np.ones((10, 10, 10), dtype=np.float32)
        invalid_nifti = os.path.join(self.test_dir, "invalid.nii.gz")
        nib.save(nib.Nifti1Image(invalid_data, np.eye(4)), invalid_nifti)

        # Verificar que falla la verificación de calidad
        result = verify_nifti_quality(invalid_nifti)
        self.assertFalse(
            result, "La verificación de calidad pasó para un archivo NIFTI inválido"
        )

    def test_process_scan_with_sample(self):
        """
        Test integrado del proceso completo con datos DICOM reales de ADNI

        Esta prueba utiliza datos DICOM reales del proyecto si están disponibles
        """
        # Buscar carpeta de datos ADNI real
        real_data_path = "../src/data/data/ADNI Data"

        if not os.path.exists(real_data_path):
            self.skipTest("Carpeta de datos ADNI reales no encontrada")

        # Buscar un sujeto válido para utilizar como muestra
        subject_dirs = [
            d
            for d in os.listdir(real_data_path)
            if os.path.isdir(os.path.join(real_data_path, d))
        ]
        if not subject_dirs:
            self.skipTest("No se encontraron directorios de sujetos en los datos ADNI")

        # Usar el primer sujeto disponible (por ejemplo, 003_S_1059 si existe)
        sample_subject = next(
            (s for s in subject_dirs if "003_S_1059" in s), subject_dirs[0]
        )
        sample_subject_dir = os.path.join(real_data_path, sample_subject)

        # Crear un directorio temporal para guardar la salida de conversión
        sample_output_dir = os.path.join(self.test_dir, "output")

        # Probar la conversión con datos reales
        result = process_scan(sample_subject_dir, sample_output_dir, modality="mri")

        if not isinstance(result, MagicMock):  # Si no estamos en modo mock
            self.assertIsNotNone(result, "La conversión falló para datos reales")
            if result:
                self.assertIn(
                    "nifti_path", result, "No se generó la ruta de salida NIFTI"
                )
                self.assertTrue(
                    os.path.exists(result["nifti_path"]),
                    "El archivo NIFTI no se creó correctamente",
                )
                self.assertTrue(
                    verify_nifti_quality(result["nifti_path"]),
                    "El archivo NIFTI generado no pasó las verificaciones de calidad",
                )

                # Verificar metadatos del escaneo
                info_file = os.path.join(sample_output_dir, "mri_scan_info.json")
                self.assertTrue(
                    os.path.exists(info_file),
                    "No se generó el archivo de información de escaneo",
                )

                # Verificar que el ID del sujeto está en el nombre del archivo generado
                self.assertIn(
                    sample_subject,
                    os.path.basename(result["nifti_path"]),
                    "El ID del sujeto no está incluido en el nombre del archivo NIFTI",
                )


if __name__ == "__main__":
    unittest.main()
