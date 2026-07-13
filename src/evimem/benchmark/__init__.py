"""SciMem-Curate benchmark contracts, adapters and metrics."""

from .adapters import (
    ADAPTERS,
    DatasetAdapter,
    EvidenceInferenceAdapter,
    MeasEvalAdapter,
    QasperAdapter,
    SciFactAdapter,
    SciRexAdapter,
)
from .audit import build_reports, leakage_report, sha256_file, source_snapshot, write_reports
from .datasets import (
    ComponentLicense,
    ComponentLicenses,
    DatasetRegistry,
    DatasetRole,
    DatasetSpec,
    DataView,
    LicenseComponentName,
    LicenseStatus,
)
from .episode import (
    BenchmarkEpisode,
    EpisodePrediction,
    MemoryQuery,
    OracleAnnotation,
    ScientificDocument,
)
from .metrics import BenchmarkMetrics, compute_benchmark_metrics
from .stream_builder import build_episode_stream
from .views import (
    AlignedEvidence,
    ConversionOrigin,
    InferenceViewInput,
    OracleViewTarget,
    RejectedConversion,
    ViewSample,
)

__all__ = [
    "ADAPTERS",
    "BenchmarkEpisode",
    "BenchmarkMetrics",
    "AlignedEvidence",
    "ComponentLicense",
    "ComponentLicenses",
    "ConversionOrigin",
    "InferenceViewInput",
    "DatasetAdapter",
    "DatasetRegistry",
    "DatasetRole",
    "DatasetSpec",
    "DataView",
    "EpisodePrediction",
    "EvidenceInferenceAdapter",
    "LicenseStatus",
    "LicenseComponentName",
    "MeasEvalAdapter",
    "MemoryQuery",
    "OracleAnnotation",
    "OracleViewTarget",
    "QasperAdapter",
    "RejectedConversion",
    "SciFactAdapter",
    "SciRexAdapter",
    "ScientificDocument",
    "ViewSample",
    "build_episode_stream",
    "build_reports",
    "compute_benchmark_metrics",
    "leakage_report",
    "sha256_file",
    "source_snapshot",
    "write_reports",
]
