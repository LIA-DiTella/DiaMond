import os
import sys
import pytest
import argparse
from zenml.client import Client  # Mover importaciones al inicio
from zenml_pipeline import diamond_test_pipeline  # Mover importaciones al inicio

# Añadir la ruta del directorio actual al PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def run_unit_tests():
    """Ejecuta todas las pruebas unitarias con pytest."""
    print("Ejecutando pruebas unitarias...")
    # Utilizamos el módulo pytest directamente en lugar de cli
    pytest.main(["-xvs"])


def run_zenml_pipeline(dataset_path, model_dir, split=0, num_classes=2):
    """Ejecuta la pipeline ZenML para pruebas de extremo a extremo."""
    print("Ejecutando pipeline ZenML...")

    # Inicializar cliente ZenML
    try:
        client = Client()
        if not client.active_stack:
            print(
                "No se encontró un stack ZenML activo. Por favor ejecuta 'zenml init' primero."
            )
            return False
    except Exception as e:
        print(f"Error al inicializar el cliente ZenML: {e}")
        print("Por favor instala ZenML con 'pip install zenml'")
        return False

    # Ejecutar pipeline
    try:
        pipeline_instance = diamond_test_pipeline(
            dataset_path=dataset_path,
            model_dir=model_dir,
            split=split,
            num_classes=num_classes,
        )

        # Imprimir resultados
        data_status, model_status, inference_status = pipeline_instance.get_results()
        print("\nResultados de la pipeline:")
        print(f"Validación de datos: {data_status}")
        print(f"Validación del modelo: {model_status}")
        print(f"Prueba de inferencia: {inference_status}")

        return True
    except Exception as e:
        print(f"Error al ejecutar la pipeline ZenML: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suite de Pruebas DiaMond")
    parser.add_argument(
        "--unit", action="store_true", help="Ejecutar pruebas unitarias"
    )
    parser.add_argument(
        "--pipeline", action="store_true", help="Ejecutar pipeline ZenML"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Ruta al dataset para pruebas de pipeline",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None, help="Ruta al directorio del modelo"
    )
    parser.add_argument(
        "--split", type=int, default=0, help="División del dataset a utilizar"
    )
    parser.add_argument(
        "--num-classes", type=int, default=2, help="Número de clases (2 o 3)"
    )

    args = parser.parse_args()

    # Si no se especifica ninguna prueba, ejecutar todas
    if not args.unit and not args.pipeline:
        args.unit = True
        args.pipeline = True

    # Ejecutar pruebas
    if args.unit:
        run_unit_tests()

    if args.pipeline:
        if args.dataset is None or args.model_dir is None:
            print(
                "Error: --dataset y --model-dir deben especificarse para las pruebas de pipeline"
            )
        else:
            run_zenml_pipeline(
                args.dataset, args.model_dir, args.split, args.num_classes
            )
