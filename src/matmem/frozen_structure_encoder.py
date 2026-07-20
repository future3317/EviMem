"""Frozen, policy-visible crystal representations for protocol transport."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

CHGNET_MODEL_NAME = "0.3.0"
CHGNET_MODEL_SHA256 = "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenCHGNetCrystalEncoder:
    """Expose the frozen CHGNet-0.3.0 pooled crystal feature.

    The encoder consumes only a structure that is already observable under the
    source protocol.  It never fits on, or receives, a target-protocol outcome.
    """

    def __init__(self, *, device: str = "cpu") -> None:
        import chgnet.model.model as chgnet_model_module
        from chgnet.model.model import CHGNet

        checkpoint = (
            Path(chgnet_model_module.module_dir)
            / "../pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar"
        ).resolve()
        checksum = _sha256(checkpoint)
        if checksum != CHGNET_MODEL_SHA256:
            raise ValueError("frozen CHGNet representation checkpoint checksum mismatch")
        self._model = CHGNet.load(
            model_name=CHGNET_MODEL_NAME,
            use_device=device,
            verbose=False,
        )
        # Keep the fixed checkpoint and cutoff.  A tiny number of source
        # structures have no 6 Å neighbours; they remain in the task and are
        # explicitly audited instead of being outcome-blindly deleted.
        self._model.graph_converter.set_isolated_atom_response("warn")
        self._device = device
        self._checkpoint = checkpoint

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "encoder": "CHGNet frozen crystal_fea",
            "model_name": CHGNET_MODEL_NAME,
            "package_version": importlib.metadata.version("chgnet"),
            "checkpoint_filename": self._checkpoint.name,
            "checkpoint_sha256": CHGNET_MODEL_SHA256,
            "device": self._device,
            "structure_source": "policy-visible low-fidelity source configuration",
            "target_structure_used": False,
            "target_outcomes_used": False,
            "isolated_atom_policy": "warn_and_preserve_candidate",
        }

    def encode(self, structures: Sequence[Any], *, batch_size: int = 32) -> np.ndarray:
        if not structures or batch_size < 1:
            raise ValueError("frozen structure encoder requires a nonempty valid batch")
        predictions = self._model.predict_structure(
            list(structures),
            task="e",
            return_crystal_feas=True,
            batch_size=batch_size,
        )
        if not isinstance(predictions, list) or len(predictions) != len(structures):
            raise RuntimeError("CHGNet representation batch returned an unexpected shape")
        rows = np.asarray(
            [tuple(float(value) for value in item["crystal_fea"]) for item in predictions],
            dtype=np.float64,
        )
        if (
            rows.ndim != 2
            or rows.shape[1] == 0
            or not np.isfinite(rows).all()
            or any(not math.isfinite(float(value)) for value in rows.reshape(-1))
        ):
            raise ValueError("CHGNet source embedding contains invalid values")
        return rows
