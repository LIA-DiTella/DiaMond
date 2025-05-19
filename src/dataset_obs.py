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

    wandb.finish()
    return data._rid, len(data._rid)

if __name__ == "__main__":

    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "hdf5"
    )

    all_train_rids = []

    for split in range(0, 5):
        split_train_path = f"{dataset_path}/{split}-train.h5"
        split_train_rids, len_train = observe_dataset(split_train_path)

        split_val_path = f"{dataset_path}/{split}-valid.h5"
        split_val_rids, len_val = observe_dataset(split_val_path)

        split_test_path = f"{dataset_path}/{split}-test.h5"
        split_test_rids, len_test = observe_dataset(split_test_path)

        val_intersection = set(split_train_rids).intersection(set(split_val_rids))
        test_intersection = set(split_train_rids).intersection(set(split_test_rids))
        val_test_intersection = set(split_val_rids).intersection(set(split_test_rids))
        print(f"Split {split}:")
        print(f"  Train length: {len_train}")
        print(f"  Val length: {len_val}")
        print(f"  Test length: {len_test}")
        print(f"  Train-Val intersection: {len(val_intersection)}")
        print(f"  Train-Test intersection: {len(test_intersection)}")
        print(f"  Val-Test intersection: {len(val_test_intersection)}")
        all_train_rids.append(split_train_rids)

    for i, rids in enumerate(all_train_rids):
        # check set intersection
        for j, other_rids in enumerate(all_train_rids):
            if i != j:
                set_intersection = set(rids).intersection(set(other_rids))
                print(len(set_intersection), len(rids), len(other_rids))

    flattened_rids = [item for sublist in all_train_rids for item in sublist]
    # check if all rids are unique
    unique_rids = set(flattened_rids)
    print(f"Total unique rids: {len(unique_rids)}")
    print(f"Total rids: {len(flattened_rids)}")
    print(f"Total rids: {len(set(flattened_rids))}")


    original_sources = ["/home/ipardo/ADNI_data/DiaMond%20Set/ADNI", "/home/ipardo/ADNI_data/DiaMond%20Set1/ADNI", "/home/ipardo/ADNI_data/DiaMond%20Set2/ADNI", "/home/ipardo/ADNI_data/DiaMond%20Set4/ADNI", "/home/ipardo/ADNI_data/DiaMond%20Set5/ADNI", "/home/ipardo/ADNI_data/DiaMond%20Set6/ADNI", "storage1/DiaMond/tmp/DiaMond%20Set7/ADNI", "storage2/DiaMond_tmp/DiaMond%20Set3/ADNI"]
    symlinked_sources = "storage1/DiaMond/src/data/data/ADNI Data"

    all_subdirs = []
    for source in original_sources:
        # Check how many subdirectories are in the source directory
        subdirs = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
        all_subdirs.extend(subdirs)

    # Check how many subdirectories are in the symlinked source directory
    symlinked_subdirs = [d for d in os.listdir(symlinked_sources) if os.path.isdir(os.path.join(symlinked_sources, d))]

    # Check if the number of subdirectories is the same
    print(f"Original sources: {len(all_subdirs)}")
    print(f"Symlinked sources: {len(symlinked_subdirs)}")

    # Intersection of subdirectories
    intersection = set(all_subdirs).intersection(set(symlinked_subdirs))
    print(f"Intersection: {len(intersection)}")
    
