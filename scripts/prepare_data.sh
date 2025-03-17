#!/bin/bash

# Create directories
mkdir -p data/{raw,processed}/{train,val,test}

# Download sample data (you need to replace with actual ADNI credentials)
echo "Please download ADNI data manually from https://adni.loni.usc.edu/"
echo "Place the downloaded files in data/raw/{train,val,test} directories"

# Process data to HDF5
python src/data/process_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --splits train val test
