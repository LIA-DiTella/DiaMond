# Estructura de Carpetas para Datos ADNI

## Estructura Actual del Proyecto

```bash
src/data/data/
├── ADNI Data/           # Datos originales DICOM
│   ├── 003_S_1059/      # ID del sujeto
│   │   ├── B1-calibration_Body/
│   │   │   └── 2006-11-09_17_40_00.0/
│   │   │       └── I29247/
│   │   │           └── [archivos DICOM .dcm]
│   │   ├── B1-calibration_Head/
│   │   ├── MPRAGE/      # Imágenes estructurales T1
│   │   ├── MPRAGE_Repeat/
│   │   └── PET_WB/      # Imágenes PET
│   ├── 003_S_1122/
│   └── ...
├── ADNI Dataset/        # Datos procesados (en formato .v)
│   └── ...
└── ADNI Metadata/       # Metadatos en XML
    └── ...
```

## Estructura Procesada (Tras Conversión)

```bash
/ruta/a/datos/adni_processed/
├── nifti/
│   ├── 003_S_1059/      # Corresponde al ID del sujeto
│   │   └── ...
└── ...
```

## Instrucciones de Uso

### 1. Conversión DICOM a NIFTI

```bash
python dicom_converter.py --input /src/data/data/ADNI\ Data/ --output /ruta/a/datos/adni_processed/ --modality both --batch
```

### 2. Procesamiento completo (DICOM a HDF5)

```bash
# Si tienes un archivo CSV con datos clínicos:/nacho/Desktop/code/DiaMond/src/data/data/ADNI\ Data/ --output-dir /ruta/a/datos/adni_processed/ --clinical-csv /src/data/data/ADNI\ Metadata/clinical_data.csv
python process_adni_data.py --dicom-dir /src/data/data/ADNI\ Data/ --output-dir /ruta/a/datos/adni_processed/ --clinical-csv /ruta/a/tu/archivo/clinical_data.csv

# Generación automática de datos clínicos a partir de XML:
python process_adni_data.py --dicom-dir /src/data/data/ADNI\ Data/ --output-dir /ruta/a/datos/adni_processed/ --generate-clinical
bash
# Sin datos clínicos (se intentará generarlos automáticamente):python process_data.py --metadata /ruta/a/datos/adni_processed/metadata.csv --data-dir /ruta/a/datos/adni_processed/nifti/ --output-dir /ruta/a/datos/adni_processed/hdf5/ --n-splits 5
python process_adni_data.py --dicom-dir /src/data/data/ADNI\ Data/ --output-dir /ruta/a/datos/adni_processed/
```

mendaciones para Pruebas

### 3. Creación de divisiones de conjuntos (train/valid/test)

Crear un conjunto de datos pequeño para pruebas

```bash
python process_data.py --metadata /ruta/a/datos/adni_processed/metadata.csv --data-dir /ruta/a/datos/adni_processed/nifti/ --output-dir /ruta/a/datos/adni_processed/hdf5/ --n-splits 5njunto ADNI durante las pruebas, utilice el script `create_test_samples.py`:
```

## Recomendaciones para Pruebaspython create_test_samples.py --source src/data/data/ADNI\ Data/ --destination tests/sample_data/adni_mini --subjects 2

### Crear un conjunto de datos pequeño para pruebas

ebas unitarias
Para evitar procesar todo el conjunto ADNI durante las pruebas, utilice el script `create_test_samples.py`:
pruebas unitarias utilizan mocks y pequeños conjuntos de datos para verificar funciones específicas:

````bash
python create_test_samples.py --source src/data/data/ADNI\ Data/ --destination tests/sample_data/adni_mini --subjects 2
```make test-unit

### Pruebas unitarias
jo de trabajo recomendado para depuración
Las pruebas unitarias utilizan mocks y pequeños conjuntos de datos para verificar funciones específicas:
Crear conjunto de datos de muestra pequeño (2-3 sujetos)
```bash2. Procesar manualmente un sujeto para verificar la conversión:
make test-unit
```   python dicom_converter.py --input tests/sample_data/adni_mini/003_S_1059 --output tests/output/ --modality mri

### Flujo de trabajo recomendado para depuraciónNIFTI
r todo el pipeline en el conjunto de muestra
1. Crear conjunto de datos de muestra pequeño (2-3 sujetos)
2. Procesar manualmente un sujeto para verificar la conversión:
   ```bash
   python dicom_converter.py --input tests/sample_data/adni_mini/003_S_1059 --output tests/output/ --modality mri
   ```Cada sujeto (ej. 003_S_1059) tiene su carpeta individual con distintos tipos de escaneos (MPRAGE, PET, etc.) organizados en subcarpetas por fecha y ID de escaneo.
3. Verificar resultado visual de la conversión con algún visor NIFTI
4. Ejecutar todo el pipeline en el conjunto de muestraadata**: El archivo `metadata.csv` contiene información sobre todos los sujetos, incluyendo diagnóstico y rutas a sus archivos NIFTI.
5. Escalar al conjunto completo
os son las divisiones para validación cruzada.
## Notas Importantes

1. **Datos DICOM originales**: Cada sujeto (ej. 003_S_1059) tiene su carpeta individual con distintos tipos de escaneos (MPRAGE, PET, etc.) organizados en subcarpetas por fecha y ID de escaneo.

2. **Archivos de metadata**: El archivo `metadata.csv` contiene información sobre todos los sujetos, incluyendo diagnóstico y rutas a sus archivos NIFTI.3. **HDF5 Datasets**: Los archivos HDF5 son la entrada final para la clase `AdniDataset`. El archivo `adni_dataset.h5` contiene todos los sujetos, mientras que los archivos con prefijos numéricos son las divisiones para validación cruzada.4. **Archivos JSON**: Los archivos `*_scan_info.json` contienen información sobre la conversión de cada escaneo.5. **Verificación de calidad**: Durante el proceso de conversión, se realizan comprobaciones automáticas de calidad de imagen que descartan archivos demasiado pequeños o con valores anómalos.## Datos ClínicosLos datos clínicos se pueden proporcionar de dos maneras:1. **Automáticamente**: El script puede extraer automáticamente información de diagnóstico (researchGroup) y datos demográficos de los archivos XML en el directorio "ADNI Metadata".
2. **Manualmente**: Proporcionando un archivo CSV con al menos:
   - Una columna con IDs de sujeto (que coincidan con los nombres de carpeta)
   - Una columna con diagnósticos (CN, MCI, AD)

Ejemplo de formato CSV esperado:
````
