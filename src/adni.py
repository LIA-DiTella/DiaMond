import numpy as np
import torch
import h5py
import pandas as pd
import torchio as tio
import monai.transforms as montrans
import logging

from torch.utils.data import Dataset

try:
    from torchvision import transforms
except ImportError:
    print("torchvision no está instalado. Por favor ejecuta 'pip install torchvision'")
    transforms = None

DIAGNOSIS_MAP = {"CN": 0, "MCI": 1, "FTD": 1, "Dementia": 2, "AD": 2}
DIAGNOSIS_MAP_binary = {"CN": 0, "Dementia": 1, "AD": 1}
LOG = logging.getLogger(__name__)


def get_image_transform(is_training: bool):
    img_transforms = [
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.CropOrPad((128, 128, 128)),
    ]

    if is_training:
        randomAffineWithRot = tio.RandomAffine(
            scales=0.2,
            degrees=90,  # +-90 degree in each dimension
            translation=8,  # +-8 pixels offset in each dimension.
            image_interpolation="linear",
            default_pad_value="otsu",
            p=0.5,
        )
        img_transforms.append(randomAffineWithRot)

    img_transform = montrans.Compose(img_transforms)
    return img_transform


class AdniDataset(Dataset):
    def __init__(
        self,
        path,
        is_training=False,
        out_class_num=3,
        with_mri=True,
        with_pet=True,
    ):
        self.path = path
        self.is_training = is_training
        self.out_class_num = out_class_num
        self.mri_data = []
        self.pet_data = []
        self.labels = []
        self.with_mri = with_mri
        self.with_pet = with_pet

        # Definir transformaciones para las imágenes
        self.transforms = []
        if with_mri:
            self.transforms.append(get_image_transform(is_training))
        if with_pet:
            self.transforms.append(get_image_transform(is_training))

        self._load()

    def _load(self):
        print(f"Loading dataset from {self.path}")
        image_data = []
        diagnosis = []
        rid = []

        with h5py.File(self.path, mode="r") as file:
            for name, group in file.items():
                print(f"Loading subject {name}")

                if name == "stats":
                    continue

                # Verificar que el grupo contiene las modalidades requeridas
                has_mri = "MRI/T1/data" in group
                has_pet = "PET/FDG/data" in group

                print(f"has_mri: {has_mri}, has_pet: {has_pet}")

                # Saltar si faltan modalidades requeridas
                if (self.with_mri and not has_mri) or (self.with_pet and not has_pet):
                    LOG.warning(
                        f"Sujeto {name} no tiene todas las modalidades requeridas, saltando"
                    )
                    continue

                # Verificar diagnóstico
                if "DX" not in group.attrs:
                    LOG.warning(f"Sujeto {name} no tiene atributo DX, saltando")
                    continue

                dx = group.attrs["DX"]
                # Saltamos sujetos MCI cuando usamos clasificación binaria
                if self.out_class_num == 2 and dx == "MCI":
                    continue

                # Manejar casos para FTD que pueden no estar en mapeo estándar
                if dx not in DIAGNOSIS_MAP and dx != "FTD":
                    LOG.warning(
                        f"Diagnóstico desconocido {dx} para sujeto {name}, saltando"
                    )
                    continue

                # Cargar datos de imagen según modalidades requeridas
                print(group)
                if self.with_mri and self.with_pet:
                    mri_data = group["MRI/T1/data"][:]
                    pet_data = group["PET/FDG/data"][:]
                    pet_data = np.nan_to_num(pet_data, copy=False)

                    image_data.append((mri_data, pet_data))
                elif self.with_mri:
                    mri_data = group["MRI/T1/data"][:]
                    image_data.append(mri_data)
                elif self.with_pet:
                    pet_data = group["PET/FDG/data"][:]
                    pet_data = np.nan_to_num(pet_data, copy=False)
                    image_data.append(pet_data)
                else:
                    LOG.warning("No se han definido modalidades para cargar")
                    raise ValueError("No se han definido modalidades para cargar")

                diagnosis.append(dx)
                rid.append(group.attrs.get("RID", name))  # Usar nombre como fallback

        LOG.info("DATASET: %s", self.path)
        LOG.info("SAMPLES: %d", len(image_data))

        print("DATASET: ", self.path)
        print("SAMPLES: ", len(image_data))

        # Verificar si hay datos cargados, y lanzar una excepción si no hay ninguno
        if len(image_data) == 0:
            raise Exception(f"No valid data found in dataset: {self.path}")

        labels, counts = np.unique(diagnosis, return_counts=True)
        LOG.info("Classes: %s", pd.Series(counts, index=labels))

        print("Classes: ", pd.Series(counts, index=labels))

        self._image_data = image_data

        # Mapeo de diagnósticos según número de clases
        if self.out_class_num == 3:
            self._diagnosis = [
                DIAGNOSIS_MAP.get(d, 1) for d in diagnosis
            ]  # 1 (MCI) para casos no mapeados
        elif self.out_class_num == 2:
            self._diagnosis = [
                DIAGNOSIS_MAP_binary.get(d, 0) for d in diagnosis
            ]  # 0 (CN) para casos no mapeados

        self._rid = rid

    def __len__(self) -> int:
        return len(self._image_data)

    def __getitem__(self, index: int):
        label = self._diagnosis[index]
        scans = self._image_data[index]

        assert len(self.transforms) > 0, "No se han definido transformaciones"

        if self.with_mri is True and self.with_pet is True:
            assert len(scans) == len(self.transforms), (
                f"Número de scans ({len(scans)}) no coincide con transformaciones ({len(self.transforms)})"
            )

            sample = []
            for scan, transform in zip(scans, self.transforms):
                # Asegurarse de que el tensor tiene la forma correcta para torchio (C, H, W, D)
                if scan.ndim == 5 and scan.shape[0] == 1:
                    scan = scan[0]

                # Aplicar transformación y convertir a tensor si es necesario
                transformed = transform(scan)
                if not isinstance(transformed, torch.Tensor):
                    transformed = torch.from_numpy(transformed)
                sample.append(transformed)

            sample = tuple(sample)
        elif self.with_mri is True or self.with_pet is True:
            # Asegurarse de que el tensor tiene la forma correcta para torchio (C, H, W, D)
            if scans.ndim == 5 and scans.shape[0] == 1:
                scans = scans[0]

            # Aplicar transformación y convertir a tensor si es necesario
            sample = self.transforms[0](scans)
            if not isinstance(sample, torch.Tensor):
                sample = torch.from_numpy(sample)

        return sample, label
