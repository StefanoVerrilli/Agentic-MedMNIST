"""Persisted, typed hand-offs for heavy data; never pickle arbitrary objects."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from contracts import BlobReference, sha256_file
from research import contained_path


def save_blob(bb: Any, name: str, obj: Any, version: int) -> BlobReference:
    import numpy as np
    from ml import DatasetBundle, PreparedData

    stem = bb.blob_dir / f"{name}_v{version:03d}"
    metadata: dict[str, Any] = {}
    if isinstance(obj, DatasetBundle):
        kind, path = "dataset", stem.with_suffix(".npz")
        arrays = {f"{field}_{split}": getattr(obj, field)[split]
                  for field in ("images", "targets", "selected_indices") for split in ("train", "val", "test")}
        metadata = {field.name: getattr(obj, field.name) for field in fields(obj)
                    if field.name not in {"images", "targets", "selected_indices"}}
        np.savez(path, **arrays)
    elif isinstance(obj, PreparedData):
        kind, path = "prepared", stem.with_suffix(".json")
        raw = bb.get("blob_raw_dataset")
        metadata = {"dataset_reference": raw.model_dump(mode="json"),
                    "normalization": obj.normalization, "augmentations": obj.augmentations,
                    "mean": obj.mean, "std": obj.std}
        path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    elif name == "model":
        kind = "model"
        result = bb.get("train_result")
        path = contained_path(bb.root, result.checkpoint_path)
        metadata = {"loader": "PathMNISTLitModule", "device": result.device}
        if sha256_file(path) != result.checkpoint_sha256:
            raise ValueError("model checkpoint checksum mismatch")
    elif isinstance(obj, dict) and all(isinstance(value, np.ndarray) for value in obj.values()):
        kind, path = "arrays", stem.with_suffix(".npz")
        np.savez(path, **obj)
    else:
        raise TypeError(f"no typed persistent blob codec for {name}: {type(obj).__name__}")
    return BlobReference(kind=kind, path=path.relative_to(bb.root).as_posix(),
                         sha256=sha256_file(path), metadata=metadata)


def load_blob(root: Path, ref: BlobReference) -> Any:
    import numpy as np
    from ml import DatasetBundle, PreparedData

    path = contained_path(root, ref.path)
    if sha256_file(path) != ref.sha256:
        raise ValueError(f"blob checksum mismatch: {ref.path}")
    if ref.kind == "dataset":
        with np.load(path, allow_pickle=False) as stored:
            arrays = {field: {split: stored[f"{field}_{split}"] for split in ("train", "val", "test")}
                      for field in ("images", "targets", "selected_indices")}
        metadata = dict(ref.metadata)
        metadata["labels"] = tuple(metadata["labels"])
        return DatasetBundle(**metadata, **arrays)
    if ref.kind == "prepared":
        data = json.loads(path.read_text(encoding="utf-8"))
        return PreparedData(bundle=load_blob(root, BlobReference.model_validate(data["dataset_reference"])),
                            normalization=data["normalization"], augmentations=tuple(data["augmentations"]),
                            mean=tuple(data["mean"]), std=tuple(data["std"]))
    if ref.kind == "model":
        from lightning_components import PathMNISTLitModule
        return PathMNISTLitModule.load_from_checkpoint(str(path), map_location=ref.metadata["device"])
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}
