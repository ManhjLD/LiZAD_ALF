import os

from .constants import DatasetConstants
from .mvtec import MVTecDataset
from .visa import VisaDataset
from .btad import BTADDataset
from .mpdd import MPDDDataset

# Đường dẫn riêng cho từng dataset
DATASET_ROOTS = {
    "btad": "/kaggle/working/datasets/btad",
    "mpdd": "/kaggle/working/datasets/MPDD",
    "mvtec": "/kaggle/working/datasets/mvtec",
    "visa": "/kaggle/working/datasets/visa",
}

dataset_classes = {
    "btad": BTADDataset,
    "mpdd": MPDDDataset,
    "mvtec": MVTecDataset,
    "visa": VisaDataset,
}

dataset_dict = {}

for name, root in DATASET_ROOTS.items():
    constants = DatasetConstants(
        base_path=os.path.dirname(root),
        dataset_name=name,
    )

    dataset_dict[name] = (
        constants.get_class_names(),
        dataset_classes[name],
        root,
    )


def get_data(dataset_name, transform, target_transform, training):
    dataset_name = dataset_name.lower()

    dataset_cls_names, dataset_instance, dataset_root = dataset_dict[dataset_name]

    meta_path = os.path.join(dataset_root, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Không tìm thấy meta.json cho dataset '{dataset_name}': {meta_path}"
        )

    return dataset_instance(
        clsnames=dataset_cls_names,
        transform=transform,
        target_transform=target_transform,
        training=training,
        root=dataset_root,
    )
