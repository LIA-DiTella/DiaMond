# Dataset preview
import yaml
import wandb
from adni import AdniDataset
import os
# dataset_path = "$(shell pwd)/data/processed/hdf5"
# CONFIG ?= $(shell pwd)/config/config.yaml

config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

all_rids = []

def observe_dataset(dataset_file_path:str):

    wandb.init(
        project="DiaMond",
        entity="pardo",
        notes="Dataset preview",
        tags=[],
        config=config,
        mode="offline",
        reinit="finish_previous",
        name="dataset_preview",
    )

    data = AdniDataset(
        path=dataset_file_path,
        is_training=True,
        out_class_num=3,
        with_mri=wandb.config.with_mri,
        with_pet=wandb.config.with_pet,
        allow_incomplete_pairs=wandb.config.get("allow_incomplete_pairs", False),
    )

    print("data length: ", len(data))

    all_rids.append(data.rids)

    wandb.finish()

if __name__ == "__main__":

    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "hdf5"
    )

    # split = 0
    # split_train_path = f"{dataset_path}/{split}-train.h5"

    # observe_dataset(split_train_path)

    observe_dataset(f"{dataset_path}/adni_dataset.h5")
