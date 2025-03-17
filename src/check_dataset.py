import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Check dataset files")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument("--splits", type=int, default=5, help="Number of splits")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_dir = args.dataset_dir

    print(f"Checking dataset directory: {dataset_dir}")

    if not os.path.exists(dataset_dir):
        print(f"ERROR: Directory does not exist: {dataset_dir}")
        exit(1)

    # Verificar si es un archivo único o un directorio con splits
    if os.path.isfile(dataset_dir):
        print(f"Dataset is a single file: {dataset_dir}")
        # Verificar si podemos abrirlo (requiere h5py)
        try:
            import h5py

            with h5py.File(dataset_dir, "r") as f:
                keys = list(f.keys())
                print(f"File contains keys: {keys}")
        except ImportError:
            print("h5py not available, skipping file content check")
        except Exception as e:
            print(f"Error opening file: {e}")
    else:
        # Es un directorio, verificar los splits
        for split in range(args.splits):
            train_file = os.path.join(dataset_dir, f"{split}-train.h5")
            valid_file = os.path.join(dataset_dir, f"{split}-valid.h5")
            test_file = os.path.join(dataset_dir, f"{split}-test.h5")

            print(f"Split {split}:")
            print(
                f"  Train file {'exists' if os.path.exists(train_file) else 'MISSING'}: {train_file}"
            )
            print(
                f"  Valid file {'exists' if os.path.exists(valid_file) else 'MISSING'}: {valid_file}"
            )
            print(
                f"  Test file {'exists' if os.path.exists(test_file) else 'MISSING'}: {test_file}"
            )
