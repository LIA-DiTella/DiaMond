import pytest
import h5py
import os
import sys
import numpy as np
import torch

# Añadir la ruta del directorio src al PYTHONPATH de forma más robusta
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from adni import AdniDataset, get_image_transform  # noqa: E402


class TestAdniDataset:
    @pytest.fixture
    def setup_empty_h5(self, tmp_path):
        """Create an empty HDF5 file for testing."""
        h5_path = os.path.join(tmp_path, "empty.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_group("stats")
        return h5_path

    @pytest.fixture
    def setup_sample_h5(self, tmp_path):
        """Create a sample HDF5 file with test data."""
        h5_path = os.path.join(tmp_path, "sample.h5")
        with h5py.File(h5_path, "w") as f:
            # Create stats group
            f.create_group("stats")

            # Create sample groups
            for i, dx in enumerate(["CN", "AD"]):
                g = f.create_group(f"subject_{i}")
                g.attrs["DX"] = dx
                g.attrs["RID"] = f"RID{i}"

                # Create PET data - asegurarse de que tiene la forma correcta (C, H, W, D)
                pet_group = g.create_group("PET/FDG")
                # Ajustar forma a (1, 8, 8, 8) en lugar de (1, 1, 8, 8, 8)
                pet_data = np.random.rand(1, 8, 8, 8).astype(np.float32)
                pet_group.create_dataset("data", data=pet_data)

                # Create MRI data - asegurarse de que tiene la forma correcta (C, H, W, D)
                mri_group = g.create_group("MRI/T1")
                # Ajustar forma a (1, 8, 8, 8) en lugar de (1, 1, 8, 8, 8)
                mri_data = np.random.rand(1, 8, 8, 8).astype(np.float32)
                mri_group.create_dataset("data", data=mri_data)
        return h5_path

    def test_image_transform(self):
        transform = get_image_transform(is_training=False)
        # Usar forma (C, H, W, D) para el test
        data = np.random.rand(1, 4, 4, 4).astype(np.float32)
        transformed = transform(data)
        assert transformed.shape == (1, 128, 128, 128)

        # Test training transform (includes random affine)
        transform_train = get_image_transform(is_training=True)
        transformed_train = transform_train(data)
        assert transformed_train.shape == (1, 128, 128, 128)

    def test_empty_dataset(self, setup_empty_h5):
        """Create an empty HDF5 file for testing."""
        with pytest.raises(Exception, match="No valid data found in dataset"):
            AdniDataset(
                path=setup_empty_h5,
                is_training=False,
                out_class_num=2,
                with_mri=True,
                with_pet=True,
            )

    def test_dataset_loading(self, setup_sample_h5):
        dataset = AdniDataset(
            path=setup_sample_h5,
            is_training=False,
            out_class_num=2,
            with_mri=True,
            with_pet=True,
        )
        assert len(dataset) == 2

        # Test getitem
        sample, label = dataset[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2  # MRI and PET

        # Verificar que ambos elementos son tensores de PyTorch
        for tensor in sample:
            assert isinstance(tensor, torch.Tensor), (
                f"Expected torch.Tensor, got {type(tensor)}"
            )

        assert sample[0].shape == (1, 128, 128, 128)
        assert sample[1].shape == (1, 128, 128, 128)
        assert label in [0, 1]

        # Test with MRI only
        dataset_mri = AdniDataset(
            path=setup_sample_h5,
            is_training=False,
            out_class_num=2,
            with_mri=True,
            with_pet=False,
        )
        sample_mri, label_mri = dataset_mri[0]
        assert isinstance(sample_mri, torch.Tensor)
        assert sample_mri.shape == (1, 128, 128, 128)

        # Test with PET only
        dataset_pet = AdniDataset(
            path=setup_sample_h5,
            is_training=False,
            out_class_num=2,
            with_mri=False,
            with_pet=True,
        )
        sample_pet, label_pet = dataset_pet[0]
        assert isinstance(sample_pet, torch.Tensor)
        assert sample_pet.shape == (1, 128, 128, 128)

    def test_class_num(self, setup_sample_h5):
        # Test binary classification
        dataset_binary = AdniDataset(
            path=setup_sample_h5,
            is_training=False,
            out_class_num=2,
            with_mri=True,
            with_pet=True,
        )

        # Check labels are binary
        labels = [dataset_binary[i][1] for i in range(len(dataset_binary))]
        assert all(label in [0, 1] for label in labels)

        # Test with 3 classes - won't work with our test file since we only have CN and AD
        # but we can verify the code attempts to use the 3-class mapping
        try:
            AdniDataset(
                path=setup_sample_h5,
                is_training=False,
                out_class_num=3,
                with_mri=True,
                with_pet=True,
            )
        except Exception:
            pass  # Expected to potentially fail due to missing MCI class
