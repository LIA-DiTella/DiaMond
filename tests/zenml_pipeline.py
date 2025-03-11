import os
import sys
import torch
from zenml.pipelines import pipeline  # Mover importaciones al inicio
from zenml.steps import step  # Mover importaciones al inicio
from zenml.config import DockerSettings  # Mover importaciones al inicio

# Añadir la ruta del directorio src al PYTHONPATH de forma más robusta
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Ahora importamos desde el directorio src
from DiaMond import DiaMond  # noqa: E402
from adni import AdniDataset  # noqa: E402
from regbn import RegBN  # noqa: E402


@step
def data_validation_step(
    dataset_path: str,
    split: int = 0,
    out_class_num: int = 2,
    with_mri: bool = True,
    with_pet: bool = True,
):
    """Validate that data can be loaded and has expected properties."""
    # Load test dataset
    test_data = AdniDataset(
        path=f"{dataset_path}/{split}-test.h5",
        is_training=False,
        out_class_num=out_class_num,
        with_mri=with_mri,
        with_pet=with_pet,
    )

    # Basic validation checks
    assert len(test_data) > 0, "Test dataset is empty"

    # Validate sample shape and type
    sample, label = test_data[0]

    if with_mri and with_pet:
        mri_data, pet_data = sample
        assert isinstance(mri_data, torch.Tensor), "MRI data should be a torch tensor"
        assert isinstance(pet_data, torch.Tensor), "PET data should be a torch tensor"
        assert mri_data.shape[0] == 1, "MRI data should have 1 channel"
        assert pet_data.shape[0] == 1, "PET data should have 1 channel"
    elif with_mri:
        assert isinstance(sample, torch.Tensor), "MRI data should be a torch tensor"
        assert sample.shape[0] == 1, "MRI data should have 1 channel"
    elif with_pet:
        assert isinstance(sample, torch.Tensor), "PET data should be a torch tensor"
        assert sample.shape[0] == 1, "PET data should have 1 channel"

    # Check label
    assert isinstance(label, int), "Label should be an integer"
    assert label in range(out_class_num), (
        f"Label should be in range 0-{out_class_num - 1}"
    )

    return {"status": "success", "samples": len(test_data)}


@step
def model_validation_step(
    block_size: int = 32,
    image_size: int = 128,
    patch_size: int = 8,
    num_classes: int = 2,
    channels: int = 1,
    dim: int = 512,
    depth: int = 4,
    heads: int = 8,
    mlp_dim: int = 309,
):
    """Validate that the model can be instantiated and forward passes work."""
    # Initialize model
    diamond = DiaMond()
    model_pet, model_mri, model_mp = diamond.body_all(
        PATH_PET=None,
        PATH_MRI=None,
        modality="multi",
        block_size=block_size,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        channels=channels,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    head = diamond.head(
        block_size=block_size,
        image_size=image_size,
        num_classes=num_classes,
        channels=channels,
    )

    # Create dummy inputs
    pet = torch.ones(1, channels, image_size, image_size, image_size)
    mri = torch.ones(1, channels, image_size, image_size, image_size)

    # Test forward passes
    with torch.no_grad():
        latent_pet = model_pet(pet)
        latent_mri = model_mri(mri)
        latent_mp = model_mp(pet, mri)
        output = head(latent_mp)

    # Validate output shapes
    expected_latent_shape = (1, (image_size // block_size) ** 3 * num_classes)
    expected_output_shape = (1, 1) if num_classes == 2 else (1, num_classes)

    assert latent_pet.shape == expected_latent_shape, (
        f"Expected latent shape {expected_latent_shape}, got {latent_pet.shape}"
    )
    assert latent_mri.shape == expected_latent_shape, (
        f"Expected latent shape {expected_latent_shape}, got {latent_mri.shape}"
    )
    assert latent_mp.shape == expected_latent_shape, (
        f"Expected latent shape {expected_latent_shape}, got {latent_mp.shape}"
    )
    assert output.shape == expected_output_shape, (
        f"Expected output shape {expected_output_shape}, got {output.shape}"
    )

    # Test RegBN module
    regbn_kwargs = {
        "gpu": 0,
        "f_num_channels": 192 if num_classes == 3 else 128,
        "g_num_channels": 192 if num_classes == 3 else 128,
        "f_layer_dim": [],
        "g_layer_dim": [],
        "normalize_input": True,
        "normalize_output": True,
        "affine": True,
        "sigma_THR": 0.0,
        "sigma_MIN": 0.0,
    }

    regbn_module = RegBN(**regbn_kwargs)
    output_pet_regbn, output_mri_regbn = regbn_module(
        latent_pet, latent_mri, is_training=True, n_epoch=1, steps_per_epoch=1
    )

    assert output_pet_regbn.shape == expected_latent_shape
    assert output_mri_regbn.shape == expected_latent_shape

    return {
        "status": "success",
        "model_parameters": sum(p.numel() for p in model_mp.parameters()),
    }


@step
def inference_test_step(
    model_dir: str,
    split: int = 0,
    modality: str = "multi",
    num_classes: int = 2,
    dataset_path: str = None,
):
    """Test model inference with a saved model."""
    if not os.path.exists(model_dir):
        return {"status": "skipped", "reason": "Model directory not found"}

    model_path = f"{model_dir}/DiaMond_{modality}_split{split}_bestval.pt"
    if not os.path.exists(model_path):
        return {"status": "skipped", "reason": f"Model file not found: {model_path}"}

    # If dataset path is provided, test on actual data
    if dataset_path:
        # Load model
        diamond = DiaMond()
        model_pet, model_mri, model_mp = diamond.body_all(
            modality=modality,
            block_size=32,
            image_size=128,
            patch_size=8,
            num_classes=num_classes,
            channels=1,
            dim=512,
            depth=4,
            heads=8,
            mlp_dim=309,
        )

        head = diamond.head(
            block_size=32,
            image_size=128,
            num_classes=num_classes,
            channels=1,
        )

        # Load checkpoint
        checkpoint = torch.load(model_path)
        [
            m.load_state_dict(checkpoint["model_state_dict"][i])
            for i, m in enumerate([model_pet, model_mri, model_mp])
        ]
        if head is not None:
            head.load_state_dict(checkpoint["head_state_dict"])

        # Set to eval mode
        model_pet.eval()
        model_mri.eval()
        model_mp.eval()
        head.eval()

        # Setup RegBN
        regbn_kwargs = {
            "gpu": 0,
            "f_num_channels": 192 if num_classes == 3 else 128,
            "g_num_channels": 192 if num_classes == 3 else 128,
            "f_layer_dim": [],
            "g_layer_dim": [],
            "normalize_input": True,
            "normalize_output": True,
            "affine": True,
            "sigma_THR": 0.0,
            "sigma_MIN": 0.0,
        }
        regbn_module = RegBN(**regbn_kwargs)

        # Load a sample from the dataset
        test_data = AdniDataset(
            path=f"{dataset_path}/{split}-test.h5",
            is_training=False,
            out_class_num=num_classes,
            with_mri=True,
            with_pet=True,
        )

        if len(test_data) == 0:
            return {"status": "skipped", "reason": "Test dataset is empty"}

        # Get a sample
        (mri_data, pet_data), label = test_data[0]
        mri_data = mri_data.unsqueeze(0)  # Add batch dimension
        pet_data = pet_data.unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            output_pet = model_pet(pet_data)
            output_mri = model_mri(mri_data)

            output_pet, output_mri = regbn_module(
                output_pet, output_mri, is_training=False
            )

            output_mp = model_mp(pet_data, mri_data)
            output = (output_pet + output_mri + output_mp) / 3

            output = head(output)

        # Check output format
        if num_classes == 2:
            pred = torch.sigmoid(output.squeeze(1)) > 0.5
            pred = pred.int().item()
        else:
            pred = torch.argmax(output, dim=1).item()

        return {
            "status": "success",
            "label": label,
            "prediction": pred,
            "model_loaded": True,
        }

    return {"status": "success", "model_exists": True}


@pipeline(
    enable_cache=False,
    settings={"docker": DockerSettings(required_integrations=["torch"])},
)
def diamond_test_pipeline(
    dataset_path: str, model_dir: str, split: int = 0, num_classes: int = 2
):
    """End-to-end test pipeline for DiaMond."""
    data_status = data_validation_step(
        dataset_path=dataset_path, split=split, out_class_num=num_classes
    )
    model_status = model_validation_step(num_classes=num_classes)
    inference_status = inference_test_step(
        model_dir=model_dir,
        split=split,
        modality="multi",
        num_classes=num_classes,
        dataset_path=dataset_path,
    )

    return data_status, model_status, inference_status
