.PHONY: env data clean test test-unit test-pipeline install install-dev data-nifti data-hdf5 data-splits data typecheck typecheck-strict test-all dev-setup airflow-clean

env:
	@if conda env list | grep -q "diamond"; then \
		echo "Environment exists,"; \
		echo "To update environment run: make env-update"; \
		conda activate diamond; \
	else \
		echo "Creating new environment..."; \
		conda create --name diamond python=3.12 --yes --force 2>&1 | tee logs/conda_create.txt && \
		conda env update --file environment.yml --prune 2>&1 | tee logs/conda_env_update.txt && \
		conda activate diamond && \
		python -m ipykernel install --user --name diamond --display-name "Python (diamond)" --yes 2>&1 | tee logs/ipykernel_install.txt && \
		echo "Environment created,"; \
	fi

env-update:
	conda env update -f environment.yml --prune

# Variables para el procesamiento de datos
DATA_DICOM ?= $(shell pwd)/src/data/data/ADNI\ Data
DATA_OUTPUT ?= $(shell pwd)/data/processed
CLINICAL_DATA ?= $(shell pwd)/src/data/data/clinical_data.csv
N_SPLITS ?= 5
VERBOSE ?= 0

# Variables para entrenamiento
DATASET_PATH ?= $(shell pwd)/data/processed/hdf5
CONFIG ?= $(shell pwd)/config/config.yaml
MODEL_DIR ?= $(shell pwd)/models
IMG_SIZE ?= 128
BATCH_SIZE ?= 8
NUM_EPOCHS ?= 100
LR ?= 0.001
NUM_CLASSES ?= 2
WANDB ?= 1
WANDB_KEY ?= 5a7fe6c819f2c71810e3e0addddfd660c3d71300
WANDB_PROJECT ?= DiaMond
WANDB_ENTITY ?= pardo

# Reglas para el procesamiento de datos

install:
	pip install -r requirements.txt 2>&1 | tee logs/requirements_install.txt

install-dev:
	pip install -r requirements-dev.txt 2>&1 | tee logs/requirements_dev_install.txt

test-unit:
	@echo "Ejecutando pruebas unitarias..."
	@cd tests && python run_tests.py --unit

test-pipeline:
	@if [ -z "$(DATASET)" ] || [ -z "$(MODEL_DIR)" ]; then \
		echo "Error: Se requieren variables DATASET y MODEL_DIR."; \
		echo "Uso: make test-pipeline DATASET=/ruta/al/dataset MODEL_DIR=/ruta/al/modelo [SPLIT=0] [NUM_CLASSES=2]"; \
		exit 1; \
	fi
	@echo "Ejecutando pipeline de pruebas..."
	@cd tests && python run_tests.py --pipeline --dataset $(DATASET) --model-dir $(MODEL_DIR) \
		$(if $(SPLIT),--split $(SPLIT),) \
		$(if $(NUM_CLASSES),--num-classes $(NUM_CLASSES),)

test:
	@if [ -n "$(DATASET)" ] && [ -n "$(MODEL_DIR)" ]; then \
		echo "Ejecutando pruebas unitarias y pipeline..."; \
		$(MAKE) test-unit && $(MAKE) test-pipeline DATASET=$(DATASET) MODEL_DIR=$(MODEL_DIR) \
			$(if $(SPLIT),SPLIT=$(SPLIT),) \
			$(if $(NUM_CLASSES),NUM_CLASSES=$(NUM_CLASSES),); \
	else \
		echo "Ejecutando solo pruebas unitarias. Para pipeline, proporciona DATASET y MODEL_DIR."; \
		$(MAKE) test-unit; \
	fi

# Variables para el procesamiento de datos ADNI
data-nifti:
	@echo "Convirtiendo datos DICOM a NIFTI..."
	python src/data/dicom_converter.py \
		--input $(DATA_DICOM) \
		--output $(DATA_OUTPUT) \
		--modality both \
		--batch \

data-hdf5:
	@echo "Procesando datos NIFTI a HDF5..."
	python src/data/process_adni_data.py \
		--dicom-dir $(DATA_DICOM) \
		--output-dir $(DATA_OUTPUT) \
		--clinical-csv $(CLINICAL_DATA) \
		$(if $(VERBOSE),--verbose,--quiet)

# # Nueva regla separada para crear divisiones del dataset
# data-split:
# 	@echo "Creando divisiones de conjunto de datos..."
# 	@if [ -f "$(DATA_OUTPUT)/metadata.csv" ]; then \
# 		python src/data/process_data.py \
# 			--metadata $(DATA_OUTPUT)/metadata.csv \
# 			--data-dir $(DATA_OUTPUT)/nifti \
# 			--output-dir $(DATA_OUTPUT)/hdf5 \
# 			--n-splits $(N_SPLITS) \
# 			$(if $(VERBOSE),--quiet); \
# 	else \
# 		echo "ERROR: Archivo de metadatos no encontrado en $(DATA_OUTPUT)/metadata.csv"; \
# 		echo "El proceso anterior puede haber fallado."; \
# 		exit 1; \
# 	fi

# # Regla para dividir con parámetros específicos
# data-split-custom:
# 	@echo "Creando divisiones de conjunto de datos con parámetros personalizados..."
# 	@if [ -z "$(METADATA)" ] || [ -z "$(NIFTI_DIR)" ] || [ -z "$(OUTPUT_DIR)" ]; then \
# 		echo "Error: Se requieren variables METADATA, NIFTI_DIR y OUTPUT_DIR."; \
# 		echo "Uso: make data-split-custom METADATA=/ruta/al/metadata.csv NIFTI_DIR=/ruta/a/nifti OUTPUT_DIR=/ruta/salida [N_SPLITS=5]"; \
# 		exit 1; \
# 	fi
# 	@python src/data/process_data.py \
# 		--metadata $(METADATA) \
# 		--data-dir $(NIFTI_DIR) \
# 		--output-dir $(OUTPUT_DIR) \
# 		--n-splits $(N_SPLITS) \
# 		$(if $(VERBOSE),--verbose,--quiet)

# # Modificar la regla data-splits para que solo llame a data-split
# data-splits: data-hdf5 data-split
			
# Flujo de trabajo completo
data: data-nifti data-hdf5
	@echo "Procesamiento de datos completado"

# Generar datos sintéticos para probar el pipeline
generate-sample-data:
	@echo "Generando datos sintéticos para pruebas..."
	python src/data/generate_sample_data.py --output-dir $(DATASET_PATH) --verbose

# Comando para entrenamiento básico
train:
	@echo "Iniciando entrenamiento con configuración predeterminada..."
	python src/train.py \
			--config $(CONFIG) \
			--dataset_path $(DATASET_PATH) \
			--img_size $(IMG_SIZE) \
			--batch_size $(BATCH_SIZE) \
			--num_epochs $(NUM_EPOCHS) \
			--learning_rate $(LR) \
			--num_classes $(NUM_CLASSES) \
			--model_dir $(MODEL_DIR) \
			$(if $(filter 1,$(WANDB)),--wandb --wandb_key $(WANDB_KEY) --wandb_project $(WANDB_PROJECT) --wandb_entity $(WANDB_ENTITY),)

# Comando para entrenamiento con configuración personalizada
train-custom:
	@echo "Iniciando entrenamiento con configuración personalizada desde $(CONFIG)..."
	python src/train.py --config $(CONFIG)

# Comando para verificar el dataset
check-dataset:
	@echo "Verificando dataset en $(DATASET_PATH)..."
	python src/check_dataset.py --dataset_dir $(DATASET_PATH)

# Comando para entrenamiento con configuración personalizada y wandb
train-wandb: check-dataset
	@echo "Iniciando entrenamiento con configuración personalizada y wandb..."
	python src/train.py \
			--config $(CONFIG) \
			--wandb \
			--wandb_key $(WANDB_KEY) \
			--wandb_project $(WANDB_PROJECT) \
			--wandb_entity $(WANDB_ENTITY) \
			--model_dir $(MODEL_DIR)

# Comando para evaluación/test
test-model:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: Se requiere variable MODEL_PATH."; \
		echo "Uso: make test-model MODEL_PATH=/ruta/al/modelo.pth [DATASET_PATH=/ruta/al/dataset.h5]"; \
		exit 1; \
	fi
	@echo "Evaluando modelo $(MODEL_PATH)..."
	python src/train.py \
		--dataset_path $(DATASET_PATH) \
		--img_size $(IMG_SIZE) \
		--model_path $(MODEL_PATH) \
		--test

clean:
	@echo "Limpiando archivos generados..."
	rm -rf data/processed/*
	rm -rf data/interim/*
	rm -rf data/external/*
	rm -rf data/raw/*
	rm -rf models/*
	rm -rf logs/*
	@echo "Archivos limpiados"

# Verificación básica de tipos con mypy
typecheck:
	@echo "Ejecutando verificación de tipos con mypy..."
	@mypy src

# Verificación estricta de tipos
typecheck-strict:
	@echo "Ejecutando verificación estricta de tipos..."
	@mypy --disallow-untyped-defs --disallow-incomplete-defs src

lint:
	@echo "Ejecutando verificación de estilo con ruff...""
	@ruff lint .

# Incluir typecheck en las pruebas
test-all: typecheck test lint
	@echo "Todas las pruebas y verificaciones completadas"

# Instalar dependencias de desarrollo (incluye mypy)
dev-setup:
	@pip install mypy types-requests pandas-stubs

# Comandos relacionados con Airflow
airflow-init:
	@echo "Inicializando Airflow..."
	@if [ ! -d "$(HOME)/airflow" ]; then \
		mkdir -p $(HOME)/airflow; \
		mkdir -p $(HOME)/airflow/dags; \
		mkdir -p $(HOME)/airflow/logs; \
		mkdir -p $(HOME)/airflow/plugins; \
		cp -r dags/* $(HOME)/airflow/dags/; \
		export AIRFLOW_HOME=$(HOME)/airflow; \
		pip install apache-airflow; \
		airflow db init; \
		airflow users create \
			--username admin \
			--firstname Admin \
			--lastname User \
			--role Admin \
			--email ipardo@mail.utdt.edu \
			--password admin; \
		echo 'export AIRFLOW_HOME=$(HOME)/airflow' >> ~/.bashrc; \
		echo "Airflow inicializado. Por favor, reinicie su terminal o ejecute 'source ~/.bashrc'"; \
	else \
		echo "El directorio de Airflow ya existe en $(HOME)/airflow"; \
		cp -r dags/* $(HOME)/airflow/dags/; \
		echo "DAGs copiados a $(HOME)/airflow/dags/"; \
	fi

airflow-start:
	@echo "Iniciando servidor web de Airflow..."
	@export AIRFLOW_HOME=$(HOME)/airflow; 
	@echo "Airflow path: $(HOME)/airflow"; \
	airflow webserver -p 8081

airflow-scheduler:
	@echo "Iniciando scheduler de Airflow..."
	airflow scheduler -D

airflow-stop:
	@echo "Deteniendo procesos de Airflow..."
	@pkill -f "airflow webserver" || true
	@pkill -f "airflow scheduler" || true
	@echo "Procesos de Airflow detenidos."

airflow-trigger:
	@echo "Ejecutando el DAG de procesamiento de datos..."
	@export AIRFLOW_HOME=$(HOME)/airflow; \
	airflow dags trigger diamond_data_processing_pipeline

airflow-status:
	@echo "Verificando estado del DAG..."
	@export AIRFLOW_HOME=$(HOME)/airflow; \
	airflow dags list

# Variables para Airflow
set-airflow-vars:
	@echo "Configurando variables de Airflow..."
	@export AIRFLOW_HOME=$(HOME)/airflow; \
	airflow variables set diamond_dicom_dir "$(DATA_DICOM)" && \
	airflow variables set diamond_output_dir "$(DATA_OUTPUT)" && \
	airflow variables set diamond_clinical_data "$(CLINICAL_DATA)" && \
	airflow variables set diamond_n_splits "$(N_SPLITS)" && \
	airflow variables set diamond_verbose "$(VERBOSE)" && \
	airflow variables set diamond_target_shape "$(IMG_SIZE),$(IMG_SIZE),$(IMG_SIZE)" && \
	airflow variables set diamond_voxel_size "1.5,1.5,1.5" && \
	airflow variables set diamond_dataset_path "$(DATASET_PATH)" && \
	airflow variables set diamond_config "$(CONFIG)" && \
	airflow variables set diamond_model_dir "$(MODEL_DIR)" && \
	airflow variables set diamond_img_size "$(IMG_SIZE)" && \
	airflow variables set diamond_batch_size "$(BATCH_SIZE)" && \
	airflow variables set diamond_num_epochs "$(NUM_EPOCHS)" && \
	airflow variables set diamond_learning_rate "$(LR)" && \
	airflow variables set diamond_num_classes "$(NUM_CLASSES)" && \
	airflow variables set diamond_train_split "0" && \
	airflow variables set diamond_use_wandb "$(WANDB)" && \
	airflow variables set diamond_wandb_key "$(WANDB_KEY)" && \
	airflow variables set diamond_wandb_project "$(WANDB_PROJECT)" && \
	airflow variables set diamond_wandb_entity "$(WANDB_ENTITY)" && \
	echo "Variables configuradas correctamente."

# Comando para ejecutar el pipeline completo en Airflow
airflow-run-pipeline:
	@echo "Ejecutando el pipeline completo de DiaMond (procesamiento + entrenamiento)..."
	@export AIRFLOW_HOME=$(HOME)/airflow; \
	airflow dags trigger diamond_complete_pipeline

# Comando para monitorear la ejecución del DAG
airflow-monitor:
	@echo "Abriendo la interfaz web de Airflow para monitorear el progreso..."
	@python -m webbrowser "http://localhost:8080/dags/diamond_complete_pipeline/grid"

# Comando para acceder remotamente a la interfaz de Airflow desde un cliente
airflow-tunnel:
	@echo "Creando túnel SSH para acceder a la interfaz de Airflow..."
	@echo "Usa http://localhost:8080 en tu navegador para acceder a la interfaz"
	@echo "Presiona Ctrl+C para cerrar el túnel cuando termines"
	@read -p "Ingresa el usuario y servidor (formato: usuario@servidor): " SSH_TARGET && \
	ssh -L 8080:localhost:8080 $$SSH_TARGET -N

# Comando para limpiar archivos y configuración de Airflow
airflow-clean:
	@echo "Limpiando archivos temporales y configuración de Airflow..."
	@$(MAKE) airflow-stop
	@rm -f $(HOME)/airflow/airflow.db
	@rm -f $(HOME)/airflow/*.pid
	@rm -f $(HOME)/airflow/*.log
	@rm -f $(HOME)/airflow/webserver_config.py
	@rm -rf $(HOME)/airflow/logs/*
	@echo "¿Deseas re-inicializar la base de datos de Airflow? [y/N]: "
	@read -r REINIT && \
	if [ "$$REINIT" = "y" ] || [ "$$REINIT" = "Y" ]; then \
		export AIRFLOW_HOME=$(HOME)/airflow; \
		echo "Re-inicializando base de datos de Airflow..."; \
		airflow db reset -y; \
		airflow db init; \
		echo "Base de datos re-inicializada correctamente"; \
	fi
	@rm -rf $(HOME)/airflow
	@echo "Limpieza de Airflow completada"
