.PHONY: env data clean test test-unit test-pipeline install install-dev

env:
	@if conda env list | grep -q "diamond"; then \
		echo "Environment exists,"; \
		echo "To update environment run: make env-update"; \
		echo "Run: conda activate diamond"; \
	else \
		echo "Creating new environment..."; \
		conda create --name diamond python=3.12 --force && \
		conda env update --file environment.yml --prune && \
		conda activate diamond && \
		python -m ipykernel install --user --name diamond --display-name "Python (diamond)" && \
		make install && \
		make install-dev && \
		echo "Environment created,"; \
	fi

env-update:
	conda env update -f environment.yml --prune

data:
	bash scripts/prepare_data.sh

clean:
	rm -rf data/raw/*
	rm -rf data/processed/*
	conda env remove --name diamond

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

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
