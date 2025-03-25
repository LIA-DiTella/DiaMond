"""
DAG para procesar datos de neuroimagen y entrenar modelos para el proyecto DiaMond.

Este DAG automatiza el flujo de trabajo completo:
1. Convertir datos DICOM a formato NIFTI
2. Procesar los datos NIFTI a HDF5
3. Crear divisiones del conjunto de datos para entrenamiento/validación/prueba
4. Entrenar modelo con los datos procesados
5. Evaluar el modelo entrenado
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
import os
import sys

# Añadir el directorio del proyecto al PYTHONPATH para importar módulos
sys.path.insert(0, "./")

# Valores predeterminados que se pasarán a las tareas
default_args = {
    "owner": "nacho",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2023, 1, 1),
}

# Variables configurables para procesamiento de datos
DATA_DICOM = Variable.get("diamond_dicom_dir", default="/src/data/data/ADNI Data")
DATA_OUTPUT = Variable.get("diamond_output_dir", default="/data/processed")
CLINICAL_DATA = Variable.get(
    "diamond_clinical_data", default="/src/data/data/clinical_data.csv"
)
N_SPLITS = int(Variable.get("diamond_n_splits", default="5"))
VERBOSE = int(Variable.get("diamond_verbose", default="0"))
TARGET_SHAPE = Variable.get("diamond_target_shape", default="128,128,128")
VOXEL_SIZE = Variable.get("diamond_voxel_size", default="1.5,1.5,1.5")

# Variables configurables para entrenamiento
DATASET_PATH = Variable.get("diamond_dataset_path", default="/data/processed/hdf5")
CONFIG = Variable.get("diamond_config", default="/config/config.yaml")
MODEL_DIR = Variable.get("diamond_model_dir", default="/models")
IMG_SIZE = int(Variable.get("diamond_img_size", default="128"))
BATCH_SIZE = int(Variable.get("diamond_batch_size", default="8"))
NUM_EPOCHS = int(Variable.get("diamond_num_epochs", default="100"))
LR = float(Variable.get("diamond_learning_rate", default="0.001"))
NUM_CLASSES = int(Variable.get("diamond_num_classes", default="3"))
TRAIN_SPLIT = int(Variable.get("diamond_train_split", default="0"))
USE_WANDB = Variable.get("diamond_use_wandb", default="0") == "1"
WANDB_KEY = Variable.get("diamond_wandb_key", default="")
WANDB_PROJECT = Variable.get("diamond_wandb_project", default="DiaMond")
WANDB_ENTITY = Variable.get("diamond_wandb_entity", default="pardo")

# Crear el DAG
dag = DAG(
    "diamond_complete_pipeline",
    default_args=default_args,
    description="Pipeline completo: procesamiento de datos y entrenamiento para DiaMond",
    schedule_interval=None,  # Solo se ejecuta manualmente
    catchup=False,
    tags=["diamond", "neuroimagen", "procesamiento", "entrenamiento"],
)


# Función para verificar que las rutas existen
def check_paths():
    """Verificar que las rutas necesarias existen antes de iniciar el pipeline."""
    paths_to_check = [DATA_DICOM, os.path.dirname(DATA_OUTPUT), CLINICAL_DATA]
    for path in paths_to_check:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ruta no encontrada: {path}")
    # Crear directorios de salida si no existen
    os.makedirs(DATA_OUTPUT, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_OUTPUT, "hdf5"), exist_ok=True)
    return "Todas las rutas verificadas correctamente."


# Función para decidir si usar wandb o no
def decide_wandb_branch():
    """Decide qué rama seguir según si se usa wandb o no."""
    if USE_WANDB and WANDB_KEY:
        return "train_with_wandb"
    else:
        return "train_without_wandb"


# Tarea para verificar rutas
check_paths_task = PythonOperator(
    task_id="check_paths",
    python_callable=check_paths,
    dag=dag,
)

# Tarea para ejecutar pruebas de código
run_tests_task = BashOperator(
    task_id="run_tests",
    bash_command="cd /src && make test",
    dag=dag,
)

# Tarea para convertir DICOM a NIFTI
dicom_to_nifti_task = BashOperator(
    task_id="dicom_to_nifti",
    bash_command=f"python /src/data/dicom_converter.py "
    f'--input "{DATA_DICOM}" '
    f'--output "{DATA_OUTPUT}" '
    f"--modality both "
    f"--batch",
    dag=dag,
)

# Tarea para procesar datos NIFTI a HDF5
process_to_hdf5_task = BashOperator(
    task_id="process_to_hdf5",
    bash_command=f"python /src/data/process_adni_data.py "
    f'--dicom-dir "{DATA_DICOM}" '
    f'--output-dir "{DATA_OUTPUT}" '
    f'--clinical-data "{CLINICAL_DATA}" '
    f"--target-shape {TARGET_SHAPE} "
    f"--voxel-size {VOXEL_SIZE} "
    f"--verbose {VERBOSE}",
    dag=dag,
)

# Tarea para crear divisiones de datos
create_splits_task = BashOperator(
    task_id="create_splits",
    bash_command=f"python /src/data/create_splits.py "
    f'--data-dir "{DATA_OUTPUT}/hdf5" '
    f"--n-splits {N_SPLITS} "
    f'--output-dir "{DATA_OUTPUT}/splits" '
    f"--verbose {VERBOSE}",
    dag=dag,
)

# Tarea para decidir qué rama de entrenamiento seguir
branch_wandb_task = BranchPythonOperator(
    task_id="branch_wandb_decision",
    python_callable=decide_wandb_branch,
    dag=dag,
)

# Tarea para configurar wandb antes del entrenamiento
setup_wandb_task = BashOperator(
    task_id="setup_wandb",
    bash_command=f"wandb login {WANDB_KEY}",
    dag=dag,
)

# Tarea para entrenar con wandb
train_with_wandb_task = BashOperator(
    task_id="train_with_wandb",
    bash_command=f"python /src/models/train_model.py "
    f'--dataset-path "{DATASET_PATH}" '
    f'--config "{CONFIG}" '
    f'--model-dir "{MODEL_DIR}" '
    f"--img-size {IMG_SIZE} "
    f"--batch-size {BATCH_SIZE} "
    f"--epochs {NUM_EPOCHS} "
    f"--learning-rate {LR} "
    f"--num-classes {NUM_CLASSES} "
    f"--train-split {TRAIN_SPLIT} "
    f"--use-wandb",
    dag=dag,
)

# Tarea para entrenar sin wandb
train_without_wandb_task = BashOperator(
    task_id="train_without_wandb",
    bash_command=f"python /src/models/train_model.py "
    f'--dataset-path "{DATASET_PATH}" '
    f'--config "{CONFIG}" '
    f'--model-dir "{MODEL_DIR}" '
    f"--img-size {IMG_SIZE} "
    f"--batch-size {BATCH_SIZE} "
    f"--epochs {NUM_EPOCHS} "
    f"--learning-rate {LR} "
    f"--num-classes {NUM_CLASSES} "
    f"--train-split {TRAIN_SPLIT}",
    dag=dag,
)

# Punto de unión después del entrenamiento
join_training_task = DummyOperator(
    task_id="join_training",
    trigger_rule=TriggerRule.ONE_SUCCESS,
    dag=dag,
)

# Tarea para evaluar el modelo
evaluate_model_task = BashOperator(
    task_id="evaluate_model",
    bash_command=f"python /src/models/evaluate_model.py "
    f'--dataset-path "{DATASET_PATH}" '
    f'--model-dir "{MODEL_DIR}" '
    f"--train-split {TRAIN_SPLIT} "
    f"--batch-size {BATCH_SIZE} "
    f"--img-size {IMG_SIZE} "
    f"--num-classes {NUM_CLASSES}",
    dag=dag,
)

# Tarea final para indicar que todo ha terminado
pipeline_completed_task = DummyOperator(
    task_id="pipeline_completed",
    dag=dag,
)

# Configurar dependencias entre tareas
(
    check_paths_task
    >> run_tests_task
    >> dicom_to_nifti_task
    >> process_to_hdf5_task
    >> create_splits_task
    >> branch_wandb_task
)

# Ramas del flujo según si se usa wandb o no
branch_wandb_task >> setup_wandb_task >> train_with_wandb_task >> join_training_task
branch_wandb_task >> train_without_wandb_task >> join_training_task

# Continuar con la evaluación después de unir las ramas
join_training_task >> evaluate_model_task >> pipeline_completed_task
