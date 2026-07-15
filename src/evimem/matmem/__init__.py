"""Protocol-aware, bounded residual memory for materials discovery.

This package is deliberately separate from the document-curation memory stack.
It consumes native structure and calculation outcomes, never LLM annotations.
"""

from .acquisition import (
    AcquisitionScore,
    BaseBoundaryAcquisition,
    BoundaryUncertaintyAcquisition,
    OnDemandKNNArchiveAcquisition,
    ProtocolAwareBoundaryAcquisition,
    RetentionAwareBoundaryAcquisition,
    SeededRandomAcquisition,
)
from .active import (
    ActiveDiscoveryEvaluator,
    ActiveDiscoveryMetrics,
    ActiveStep,
    CandidatePoolItem,
)
from .baselines import (
    DeterministicReservoirMemory,
    DiversityBoundedMemory,
    FIFOBoundedMemory,
    FullHistoryMemory,
    ResidualPriorityMemory,
)
from .boundary import (
    BoundaryEstimate,
    BoundaryPotentialValue,
    BoundaryRetentionSelection,
    BoundaryRiskConfig,
    BoundaryRiskPotential,
    BoundaryRiskRetention,
    BoundaryWitness,
    BruteForceRetentionSolver,
)
from .cards import HullSnapshot, MaterialMemoryCard, MaterialQuery, SourceProvenance
from .coreset import CoresetSelection, DecisionAwareOnlineCoreset
from .evaluation import (
    DeploymentStrategy,
    DiscoveryMetrics,
    OnlineDiscoveryEvaluator,
    RiskCoveragePoint,
    ScreeningOutcome,
    StreamEvent,
    risk_coverage_curve,
)
from .identity import CanonicalGroupSplit, MaterialIdentity
from .protocols import (
    CompatibilityKind,
    MatchedResidualPair,
    ProtocolCertificate,
    ProtocolCompatibility,
    ProtocolCompatibilityResolver,
    ProtocolTransportMap,
)
from .residual import ResidualCorrection, ResidualCorrector
from .risk import ConformalCalibration, ProtocolRiskController, RiskDecision, ScreeningDecision

__all__ = [
    "AcquisitionScore",
    "ActiveDiscoveryEvaluator",
    "ActiveDiscoveryMetrics",
    "ActiveStep",
    "BaseBoundaryAcquisition",
    "BoundaryUncertaintyAcquisition",
    "BruteForceRetentionSolver",
    "BoundaryEstimate",
    "BoundaryPotentialValue",
    "BoundaryRetentionSelection",
    "BoundaryRiskConfig",
    "BoundaryRiskPotential",
    "BoundaryRiskRetention",
    "BoundaryWitness",
    "CandidatePoolItem",
    "CompatibilityKind",
    "CanonicalGroupSplit",
    "ConformalCalibration",
    "CoresetSelection",
    "DecisionAwareOnlineCoreset",
    "DeterministicReservoirMemory",
    "DeploymentStrategy",
    "DiscoveryMetrics",
    "DiversityBoundedMemory",
    "FIFOBoundedMemory",
    "FullHistoryMemory",
    "HullSnapshot",
    "MatchedResidualPair",
    "MaterialMemoryCard",
    "MaterialIdentity",
    "MaterialQuery",
    "OnlineDiscoveryEvaluator",
    "OnDemandKNNArchiveAcquisition",
    "ProtocolCertificate",
    "ProtocolAwareBoundaryAcquisition",
    "ProtocolCompatibility",
    "ProtocolCompatibilityResolver",
    "ProtocolRiskController",
    "ProtocolTransportMap",
    "ResidualCorrection",
    "ResidualCorrector",
    "ResidualPriorityMemory",
    "RetentionAwareBoundaryAcquisition",
    "RiskCoveragePoint",
    "RiskDecision",
    "ScreeningDecision",
    "ScreeningOutcome",
    "SeededRandomAcquisition",
    "SourceProvenance",
    "StreamEvent",
    "risk_coverage_curve",
]
