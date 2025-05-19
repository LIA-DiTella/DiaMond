# Dataset preview
import wandb
from adni import AdniDataset
import os
# dataset_path = "$(shell pwd)/data/processed/hdf5"
 
dataset_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "hdf5"
)

split = 0

split_train_path = f"{dataset_path}/{split}-train.h5"

train_data = AdniDataset(
    path=split_train_path,
    is_training=True,
    out_class_num=3,
    with_mri=wandb.config.with_mri,
    with_pet=wandb.config.with_pet,
    allow_incomplete_pairs=wandb.config.get(
        "allow_incomplete_pairs", False
    ),
)

print("train_data length: ", len(train_data))
