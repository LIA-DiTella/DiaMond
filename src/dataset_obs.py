# Dataset preview
import yaml
import wandb
from adni import AdniDataset
import os
# dataset_path = "$(shell pwd)/data/processed/hdf5"
# CONFIG ?= $(shell pwd)/config/config.yaml
import csv

config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

def observe_dataset(dataset_file_path:str):

    wandb.init(
        project="DiaMond",
        entity="pardo",
        notes=f"Dataset preview - {dataset_file_path}",
        tags=[],
        config=config,
        mode="offline",
        reinit="finish_previous",
        name=f"dataset_preview_{os.path.basename(dataset_file_path)}",
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

    wandb.log(
        {
            "dataset_file_path": dataset_file_path,
            "dataset_length": len(data),
            "dataset_rids": data._rid,
            "dataset_rids_length": len(data._rid),
        }
    )

    wandb.finish()
    return data._rid, len(data._rid)

if __name__ == "__main__":

    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "hdf5"
    )

    all_train_rids = []

    for split in range(0, 5):
        split_train_path = f"{dataset_path}/{split}-train.h5"
        if not os.path.exists(split_train_path):
            print(f"Split {split} train path does not exist")
            continue
        split_train_rids, len_train = observe_dataset(split_train_path)

        split_val_path = f"{dataset_path}/{split}-valid.h5"
        if not os.path.exists(split_val_path):
            print(f"Split {split} val path does not exist")
            continue
        split_val_rids, len_val = observe_dataset(split_val_path)

        if not os.path.exists(split_val_path):
            print(f"Split {split} val path does not exist")
            continue
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
    

    paper_used_ids_csv = os.path.join(
        os.path.dirname(__file__), "data", "paper_adni-all-ids.csv"
    )

    with open(paper_used_ids_csv, "r") as file:
        reader = csv.reader(file)
        paper_used_ids = [row[0] for row in reader[1:]]  # Skip header row
        print(f"Paper used ids: {len(paper_used_ids)}")
        print(f"Paper used ids: {paper_used_ids}")

    # intersection of paper used ids and all rids

    intersection = set(paper_used_ids).intersection(set(flattened_rids))
    print(f"Intersection: {len(intersection)}")
    # check if all rids are in the paper used ids
    all_rids_in_paper = set(flattened_rids).issubset(set(paper_used_ids))
    print(f"All rids in paper: {all_rids_in_paper}")
    # check if all paper used ids are in the rids
    all_paper_ids_in_rids = set(paper_used_ids).issubset(set(flattened_rids))
    print(f"All paper ids in rids: {all_paper_ids_in_rids}")

    # dirs in 
    processed_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed"
    )

    # Check how many subdirectories are in the processed directory
    subdirs = os.listdir(processed_path)
    # remove "hdf5" from the list
    subdirs = [d for d in subdirs if d != "hdf5"]
    print(f"Processed path subdirs: {len(subdirs)}")

    # intersection between paper used ids and all rids
    intersection = set(paper_used_ids).intersection(set(flattened_rids))
    print(f"Paper used ids to Proccessed intersection: {len(intersection)}")
