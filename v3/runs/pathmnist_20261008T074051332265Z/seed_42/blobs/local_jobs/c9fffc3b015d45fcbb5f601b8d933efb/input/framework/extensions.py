"""Reviewable extension specs; no generated Python is ever executed.

New architecture families remain reviewed specifications until a developer
implements a registry entry. New representations can be compositions of the
existing orientation-preserving primitives, promoted with a hash-bound review.
"""
from __future__ import annotations

import json
from pathlib import Path

from contracts import (ApprovedExtension, ExtensionGateEntry, ExtensionGateReport,
                       ExtensionProposal, PriorArtBrief, sha256_file)
from research import canonical_hash

BUILTIN_AUGMENTATIONS = frozenset({"hflip", "vflip", "rotate90", "rotate180", "brightness", "contrast"})
_RECIPES: dict[str, tuple[str, ...]] = {}
_APPROVED: dict[str, ApprovedExtension] = {}


def clear_registry() -> None:
    _RECIPES.clear()
    _APPROVED.clear()


def registry_entries() -> list[ApprovedExtension]:
    return list(_APPROVED.values())


def restore_registry(entries: list[dict] | list[ApprovedExtension]) -> None:
    for data in entries:
        entry = ApprovedExtension.model_validate(data)
        proposal = entry.proposal
        if (proposal.kind != "augmentation" or check_recipe(proposal)
                or canonical_hash(proposal.model_dump(mode="json")) != entry.spec_sha256):
            raise ValueError("invalid approved extension registry entry")
        name = "approved_" + proposal.proposal_id
        if name in _RECIPES and _RECIPES[name] != tuple(proposal.operations):
            raise ValueError("approved extension identifier has conflicting definitions")
        _RECIPES[name] = tuple(proposal.operations)
        _APPROVED[name] = entry


def recipe(name: str) -> tuple[str, ...]:
    if name not in _RECIPES:
        raise ValueError(f"augmentation has not passed the extension gate: {name}")
    return _RECIPES[name]


def allowed_augmentations() -> list[str]:
    return sorted(BUILTIN_AUGMENTATIONS | _RECIPES.keys())


def validate_representation(names: list[str] | tuple[str, ...]) -> None:
    for name in names:
        if name not in BUILTIN_AUGMENTATIONS:
            recipe(name)


def check_recipe(proposal: ExtensionProposal) -> list[str]:
    import numpy as np
    from ml import apply_augmentation

    if not proposal.operations or not proposal.label_preservation_rationale.strip():
        return ["orientation recipe and label-preservation rationale are required"]
    image = np.arange(28 * 28 * 3, dtype="uint16").reshape(28, 28, 3)
    result = image.copy()
    for operation in proposal.operations:
        result = apply_augmentation(result, operation)
    if result.shape != image.shape or result.dtype != image.dtype:
        return ["recipe changes shape or dtype"]
    if not np.array_equal(np.sort(result.reshape(-1)), np.sort(image.reshape(-1))):
        return ["recipe does not preserve pixels"]
    # Geometry tests do not establish label preservation; that requires review.
    return []


def gate_extensions(brief: PriorArtBrief, approvals_path: Path | None = None) -> ExtensionGateReport:
    approvals = {}
    digest = None
    if approvals_path is not None:
        document = json.loads(approvals_path.read_text(encoding="utf-8"))
        approvals = document.get("approvals", {})
        if not isinstance(approvals, dict):
            raise ValueError("extension approvals must be an object")
        digest = sha256_file(approvals_path)
    entries = []
    for proposal in brief.proposals:
        spec_hash = canonical_hash(proposal.model_dump(mode="json"))
        approval = approvals.get(proposal.proposal_id, {})
        reviewed = (approval.get("spec_sha256") == spec_hash
                    and approval.get("decision") == "approve"
                    and bool(str(approval.get("reviewer", "")).strip())
                    and bool(str(approval.get("rationale", "")).strip()))
        errors = check_recipe(proposal) if proposal.kind == "augmentation" else []
        status, reason, name = "pending_review", "A hash-bound independent review is required.", None
        if errors:
            status, reason = "rejected", "; ".join(errors)
        elif reviewed and proposal.kind == "architecture":
            status, reason = "reviewed_spec", "Reviewed specification; implement and test a model registry entry before execution."
        elif reviewed:
            name = "approved_" + proposal.proposal_id
            restore_registry([ApprovedExtension(proposal=proposal, spec_sha256=spec_hash,
                reviewer=str(approval["reviewer"]), rationale=str(approval["rationale"]))])
            status, reason = "approved", "Reviewed orientation composition passed deterministic shape, dtype and pixel-preservation checks."
        entries.append(ExtensionGateEntry(proposal_id=proposal.proposal_id,
            spec_sha256=spec_hash, status=status, reason=reason, registry_name=name,
            reviewer=str(approval["reviewer"]) if reviewed else None))
    return ExtensionGateReport(entries=entries, approvals_sha256=digest)
