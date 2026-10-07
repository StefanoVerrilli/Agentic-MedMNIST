from __future__ import annotations

import numpy as np

from ml import DatasetBundle, index_fingerprint

LABELS = (
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
)


def make_bundle(seed: int = 7) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    sizes = {"train": 27, "val": 18, "test": 18}
    images = {
        split: rng.integers(0, 256, size=(size, 28, 28, 3), dtype=np.uint8)
        for split, size in sizes.items()
    }
    targets = {
        split: np.arange(size, dtype="int64") % 9 for split, size in sizes.items()
    }
    selected = {split: np.arange(size, dtype="int64") for split, size in sizes.items()}
    fingerprints = {
        split: index_fingerprint(split, selected[split], targets[split])
        for split in sizes
    }
    return DatasetBundle(
        dataset="pathmnist",
        n_channels=3,
        n_classes=9,
        labels=LABELS,
        task="multi-class",
        official_sizes={"train": 89996, "val": 10004, "test": 7180},
        images=images,
        targets=targets,
        selected_indices=selected,
        selected_index_sha256=fingerprints,
        data_reference="https://example.invalid/pathmnist.npz",
        archive_md5="a8b06965200029087d5bd730944a56c1",
        license="CC BY 4.0",
        test_domain_note="different center",
    )
