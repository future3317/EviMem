"""Component-level dataset licensing and task-view gates.

The manifest is deliberately fail closed.  Repository or Hugging Face metadata is
not sufficient evidence for a dataset component: every component required by a
view must have an official license source, a checksum, and explicit training and
redistribution decisions.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetRole(StrEnum):
    CORE_TRAIN = "core_train"
    OOD = "ood"
    SCALE = "scale"
    CASE_STUDY = "case_study"
    AUXILIARY = "auxiliary"


class DataView(StrEnum):
    RETRIEVAL = "retrieval_view"
    ADMISSION = "admission_view"
    UPDATE = "update_view"


class LicenseStatus(StrEnum):
    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"
    BLOCKED = "blocked"


class LicenseComponentName(StrEnum):
    ANNOTATIONS = "annotations"
    SOURCE_TEXT = "source_text"
    CODE = "code"
    DERIVED_ARTIFACTS = "derived_artifacts"


class ComponentLicense(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spdx_identifier: str | None
    source_url: str
    retrieved_at: str
    license_file_checksum: str | None
    attribution_requirements: str
    redistribution_allowed: bool
    training_allowed: bool
    status: LicenseStatus

    @field_validator("license_file_checksum")
    @classmethod
    def _validate_checksum(cls, value: str | None) -> str | None:
        if value is None:
            return None
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise ValueError("license_file_checksum must be a SHA-256 digest")
        return f"sha256:{digest.lower()}"


class ComponentLicenses(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    annotations: ComponentLicense
    source_text: ComponentLicense
    code: ComponentLicense
    derived_artifacts: ComponentLicense

    def get(self, component: LicenseComponentName) -> ComponentLicense:
        return getattr(self, component.value)


class DatasetSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    role: DatasetRole
    source_url: str
    source_revision: str | None = None
    source_checksum: str | None = None
    allowed_splits: tuple[str, ...] = Field(min_length=1)
    adapter: str
    licenses: ComponentLicenses
    enabled_views: tuple[DataView, ...] = ()
    view_components: dict[DataView, tuple[LicenseComponentName, ...]] = Field(
        default_factory=dict
    )
    notes: str = ""

    @field_validator("source_checksum")
    @classmethod
    def _validate_source_checksum(cls, value: str | None) -> str | None:
        if value is None:
            return None
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise ValueError("source_checksum must be a SHA-256 digest")
        return f"sha256:{digest.lower()}"

    def components_for_view(self, view: DataView) -> tuple[LicenseComponentName, ...]:
        return self.view_components.get(view, ())

    def training_blockers(self, view: DataView) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.role not in {DatasetRole.CORE_TRAIN, DatasetRole.AUXILIARY}:
            blockers.append(f"role={self.role.value}")
        if view not in self.enabled_views:
            blockers.append("view_not_enabled")
        required = self.components_for_view(view)
        if not required:
            blockers.append("no_component_policy")
        for name in required:
            license_info = self.licenses.get(name)
            if license_info.status != LicenseStatus.CONFIRMED:
                blockers.append(f"{name.value}:status={license_info.status.value}")
            if not license_info.training_allowed:
                blockers.append(f"{name.value}:training_forbidden")
            if not license_info.license_file_checksum:
                blockers.append(f"{name.value}:license_checksum_missing")
        return tuple(blockers)


class DatasetRegistry:
    def __init__(self, specs: list[DatasetSpec]):
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("dataset manifest contains duplicate names")

    @classmethod
    def load(cls, path: str | Path) -> DatasetRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([DatasetSpec.model_validate(item) for item in payload["datasets"]])

    def get(self, name: str) -> DatasetSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"dataset is absent from the audited manifest: {name}") from exc

    def assert_split_allowed(
        self,
        name: str,
        split: str,
        *,
        view: DataView | str | None = None,
        for_training: bool = False,
    ) -> None:
        spec = self.get(name)
        if split not in spec.allowed_splits:
            raise ValueError(f"{name} split {split!r} is not in its official protocol")
        if for_training:
            if view is None:
                raise ValueError("component license gate requires an explicit data view")
            selected_view = DataView(view)
            blockers = spec.training_blockers(selected_view)
            if blockers:
                raise ValueError(
                    f"{name} {selected_view.value} is blocked for training: "
                    + ", ".join(blockers)
                )

    def assert_training_allowed(self, name: str, view: DataView | str) -> None:
        spec = self.get(name)
        split = "train" if "train" in spec.allowed_splits else spec.allowed_splits[0]
        self.assert_split_allowed(name, split, view=view, for_training=True)

    def audit(self) -> dict[str, object]:
        component_issues: dict[str, list[str]] = {}
        training_ready_views: dict[str, list[str]] = {}
        blocked_views: dict[str, dict[str, list[str]]] = {}
        for spec in self._specs.values():
            issues: list[str] = []
            for component in LicenseComponentName:
                item = spec.licenses.get(component)
                if not item.source_url.strip():
                    issues.append(f"{component.value}:missing_source_url")
                if item.status == LicenseStatus.CONFIRMED and not item.license_file_checksum:
                    issues.append(f"{component.value}:confirmed_without_checksum")
            if issues:
                component_issues[spec.name] = issues
            for view in spec.enabled_views:
                blockers = list(spec.training_blockers(view))
                if blockers:
                    blocked_views.setdefault(spec.name, {})[view.value] = blockers
                else:
                    training_ready_views.setdefault(spec.name, []).append(view.value)
        blocked_core = sorted(
            spec.name
            for spec in self._specs.values()
            if spec.role == DatasetRole.CORE_TRAIN and not training_ready_views.get(spec.name)
        )
        for spec in self._specs.values():
            if spec.role == DatasetRole.CORE_TRAIN and not spec.enabled_views:
                blocked_views.setdefault(spec.name, {})["all_views"] = [
                    "no_enabled_training_view"
                ]
        return {
            "ok": not component_issues,
            "training_ready": not blocked_core,
            "dataset_count": len(self._specs),
            "component_issues": component_issues,
            "training_ready_views": training_ready_views,
            "blocked_views": blocked_views,
            "blocked_core_training": blocked_core,
        }
