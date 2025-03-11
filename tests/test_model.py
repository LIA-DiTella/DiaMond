import torch
import os
import sys

# Añadir la ruta del directorio src al PYTHONPATH de forma más robusta
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from DiaMond import DiaMond, Head, MINiT, ViT  # noqa: E402


class TestDiaMondModel:
    def test_head(self):
        head = Head(block_size=32, image_size=128, num_classes=2, channels=1)
        # Test with binary classification
        block_count = 128 // 32  # = 4
        x = torch.rand(2, block_count**3 * 2)  # 64 = (128/32)^3, 2 = num_classes
        out = head(x)
        assert out.shape == (2, 1)

        # Test with multiclass - necesita tensor de entrada con dimensiones diferentes
        head_multi = Head(block_size=32, image_size=128, num_classes=3, channels=1)
        x_multi = torch.rand(2, block_count**3 * 3)  # 64 = (128/32)^3, 3 = num_classes
        out_multi = head_multi(x_multi)
        assert out_multi.shape == (2, 3)

    def test_vit(self):
        vit = ViT(
            modality="mono_mri",
            image_size=32,
            patch_size=8,
            num_classes=2,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            channels=1,
        )

        img = torch.rand(2, 1, 32, 32, 32)  # 2 images, 1 channel, 32x32x32
        block_embedding = torch.rand(2, 65, 64)  # (batch, patches+1, dim)

        out = vit(img, block_embedding=block_embedding)
        assert out.shape == (2, 2)

        # Test multi-modal
        vit_multi = ViT(
            modality="multi",
            image_size=32,
            patch_size=8,
            num_classes=2,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            channels=1,
        )

        img_a = torch.rand(2, 1, 32, 32, 32)
        img_b = torch.rand(2, 1, 32, 32, 32)

        out_multi = vit_multi(img_a, img_b, block_embedding=block_embedding)
        assert out_multi.shape == (2, 2)

    def test_minit(self):
        minit = MINiT(
            block_size=32,
            image_size=128,
            patch_size=8,
            num_classes=2,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            channels=1,
            modality="mono_mri",
        )

        img = torch.rand(2, 1, 128, 128, 128)  # 2 images, 1 channel, 128x128x128

        out = minit(img)
        assert out.shape == (2, 64 * 2)  # 64 = (128/32)^3, 2 = num_classes

        # Test multi-modal
        minit_multi = MINiT(
            block_size=32,
            image_size=128,
            patch_size=8,
            num_classes=2,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            channels=1,
            modality="multi",
        )

        img_a = torch.rand(2, 1, 128, 128, 128)
        img_b = torch.rand(2, 1, 128, 128, 128)

        out_multi = minit_multi(img_a, img_b)
        assert out_multi.shape == (2, 64 * 2)

    def test_diamond(self):
        diamond = DiaMond()
        model_pet, model_mri, model_mp = diamond.body_all(
            modality="multi",
            block_size=32,
            image_size=128,
            patch_size=8,
            num_classes=2,
            channels=1,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
        )

        head = diamond.head(block_size=32, image_size=128, num_classes=2, channels=1)

        pet = torch.rand(2, 1, 128, 128, 128)
        mri = torch.rand(2, 1, 128, 128, 128)

        # Test individual models
        latent_pet = model_pet(pet)
        latent_mri = model_mri(mri)
        latent_mp = model_mp(pet, mri)

        assert latent_pet.shape == (2, 64 * 2)
        assert latent_mri.shape == (2, 64 * 2)
        assert latent_mp.shape == (2, 64 * 2)

        # Test head
        output = head(latent_mp)
        assert output.shape == (2, 1)
