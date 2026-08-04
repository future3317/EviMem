"""Protocol-aware scientific state for closed-loop materials discovery.

This package is deliberately separate from the document-curation memory stack.
It consumes native structure and calculation outcomes, never LLM annotations.

Public symbols are re-exported here for convenience; heavy internal helpers live
in their own submodules and are not imported eagerly by ``import matmem``.
"""

from __future__ import annotations

# Acquisition and coreset policies
from .acquisition import (
    AcquisitionScore,
    FrozenHullDistanceAcquisition,
    PosteriorUncertaintyAcquisition,
    SeededRandomAcquisition,
    SurvivalConditionedAcquisition,
)

# Protocol activation
from .activation import (
    ProtocolActivation,
    ProtocolActivationAudit,
    ProtocolAwareActivator,
)
from .baselines import (
    DiversityBoundedMemory,
    FIFOBoundedMemory,
    GPVarianceOneSwapMemory,
)

# Calibration and scoring
from .calibration_utility import (
    CalibrationUtilityBuilder,
    CalibrationUtilityMatrix,
    ProperPosteriorDivergence,
    ReferencePosteriorSnapshot,
    bernoulli_brier_divergence,
    bernoulli_log_divergence,
    gaussian_kl_divergence,
    reference_decision_regret,
    threshold_weighted_crps_divergence,
)

# Closed-loop execution
from .campaign_gate import CampaignGatedICSARRResult, campaign_gated_ic_sarr

# Identity and structure encoding
from .cards import HullSnapshot, MaterialMemoryCard, MaterialQuery, SourceProvenance
from .configs import MATPES_DEFAULTS, MatPESDefaults
from .coreset import (
    CoresetSelection,
    ExactArchivePosteriorProjectionPlanner,
    FacilityLocationCoresetPlanner,
    JointPosteriorRiskOneSwapPlanner,
    JointPosteriorRiskSelection,
    ObjectiveFidelityCandidate,
    ObjectiveFidelityDiagnostic,
    PosteriorProjectionCandidate,
    PosteriorProjectionOneSwapPlanner,
    PosteriorProjectionScorer,
    PosteriorProjectionSelection,
    StreamingCalibrationCoreset,
    StreamingJointPosteriorRiskCoreset,
    StreamingPosteriorProjectionCoreset,
    compare_facility_and_joint_objectives,
)

# Protocol-aware transport and posterior
from .environment_transport import (
    EnvironmentConditionalProtocolTransportMap,
    EnvironmentTransportPrediction,
    EnvironmentTransportStatus,
    MatchedEnvironmentEnergyPair,
)
from .frozen_structure_encoder import (
    CHGNET_MODEL_NAME,
    CHGNET_MODEL_SHA256,
    FrozenCHGNetCrystalEncoder,
)
from .hull_certificate import (
    ActionValueInterval,
    CertifiedActionSet,
    ClusteredConformalCalibration,
    PhaseEnergyInterval,
    RobustHullDecision,
    RobustHullDecisionCertifier,
    RobustHullDecisionKind,
    certify_epsilon_optimal_actions,
    clustered_conformal_quantile,
)
from .hull_engine import CausalHullEngine

# Hull geometry, transport, posterior, and acquisition (formerly protocol_knowledge_gradient)
from .hull_geometry import FixedCompositionHullTemplate, fixed_composition_hull_membership
from .identity import (
    CanonicalGroupSplit,
    MaterialIdentity,
    StructureArtifactIdentity,
    StructureStage,
    WBMStructureSourceField,
)

# Policy registry and defaults
from .policy_registry import (
    CampaignPolicy,
    ProtocolPolicy,
    requires_protocol_transport,
)
from .posterior import ProtocolTargetEnergyPosterior, protocol_target_energy_posterior
from .protocol_acquisition import (
    ConformalSourceRolloutCalibration,
    ConformalSourceRolloutResult,
    DeltaHullActiveSearchResult,
    DualHorizonSourceRolloutResult,
    HullENSResult,
    IndependentConfirmationSourceRolloutResult,
    ProtocolHullKnowledgeGradientResult,
    ProtocolHullPosteriorSummary,
    ProtocolHullRiskReductionResult,
    SafeHullENSResult,
    SourceRolloutDeltaHullResult,
    conformal_one_deviation_source_rollout,
    constrained_dual_horizon_source_rollout,
    delta_hull_active_search,
    fit_conformal_source_rollout_calibration,
    hull_ens,
    independent_confirmation_source_rollout,
    protocol_hull_knowledge_gradient,
    protocol_hull_posterior_summary,
    protocol_hull_risk_reduction,
    safe_hull_ens,
    source_margin_action_indices,
    source_rollout_delta_hull,
    source_rollout_system_score,
)
from .protocol_closed_loop import (
    AppendOnlyProtocolEventLog,
    ObservableProtocolPhase,
    ObservableProtocolQuery,
    ProtocolActionRecord,
    ProtocolCandidate,
    ProtocolCausalHull,
    ProtocolClosedLoopEvent,
    ProtocolClosedLoopResult,
    ProtocolOracleOutcome,
    ProtocolOracleVault,
    ProtocolPolicyState,
    ProtocolPolicySubprocess,
    ProtocolRevealRecord,
    RevealedProtocolObservation,
    SecureProtocolQueryRunner,
)
from .protocols import (
    CompatibilityKind,
    CompositionAwareProtocolTransportMap,
    MatchedEnergyPair,
    MatchedResidualPair,
    ProtocolCertificate,
    ProtocolCompatibility,
    ProtocolCompatibilityResolver,
    ProtocolTransportMap,
)
from .residual import ResidualCorrection, ResidualCorrector
from .residual_posterior import (
    FixedKernelGPConfig,
    FixedKernelResidualGP,
    ResidualPosterior,
    ResidualPrediction,
)
from .ridge_acquisition import (
    HullInfluenceAcquisitionResult,
    HullMarginSubgradient,
    PredictedFinalHullAcquisitionResult,
    hull_margin_subgradient,
    linear_ridge_hull_influence_acquisition,
    linear_ridge_predicted_final_hull_acquisition,
)
from .risk import ConformalCalibration, ProtocolRiskController, RiskDecision, ScreeningDecision
from .sufficient_state import (
    AllOutcomeLinearGaussianState,
    AllOutcomeTargetCorrectionState,
    SufficientStateUpdate,
)
from .transport import (
    FrozenProtocolRidgeTransport,
    fit_protocol_kernel_transport,
    fit_protocol_ridge_transport,
)

# WBM support
from .wbm import (
    DataAuditFinding,
    DataLicenseAuditReport,
    DataLicenseDecision,
    ExternalDataArtifact,
    FrozenPredictionSOAPCache,
    FrozenPredictionSOAPRecord,
    MPCausalHullBuilder,
    MPPhaseRecord,
    OracleEnergySource,
    SOAPCacheConfig,
    WBMObservableRecord,
    WBMOracleRecord,
    audit_external_data_artifacts,
)
from .wbm_grid import (
    FROZEN_BUDGETS,
    FROZEN_CAPACITIES,
    JOINT_RISK_SENTINELS,
    PRIMARY_STRATEGIES,
    FrozenGridCell,
    GaussianNLLShapleyAttribution,
    PosteriorEvaluationSnapshot,
    PosteriorQueryEvaluation,
    PrequentialCausalEvaluator,
    PrequentialRoundMetrics,
    SelectionEffectRecord,
    aggregate_prequential_prefix,
    frozen_grid_cells,
    gaussian_nll_shapley_attribution,
    paired_system_bootstrap,
    paired_system_improvement_bootstrap,
    reference_headroom_recovery,
)
from .wbm_raw import (
    RAW_WBM_EXPECTED_ENTRY_COUNTS,
    RAW_WBM_FILENAMES,
    WBMRawObservableRecord,
    WBMRawOracleOutcome,
    WBMRawOracleVault,
    WBMRawReleaseReport,
    raw_wbm_records_from_payload,
    validate_raw_wbm_release,
)
from .wbm_secure import (
    AppendOnlyWBMEventLog,
    BudgetPrefixParityRecord,
    CompositionHullState,
    CorrectedPhaseEntry,
    ExactEmulationAudit,
    PersistentFIFOEvidence,
    PolicyQuery,
    PolicyState,
    PolicySubprocess,
    PolicyWitness,
    ReconstructedFIFOEvidence,
    RevealedObservation,
    SecureWBMRunner,
    StreamingCoresetEvidence,
    WBMActionRecord,
    WBMEvent,
    WBMOracleVault,
    WBMPhaseTiming,
    WBMReplayAudit,
    WBMReveal,
    WBMRevealRecord,
    WBMRunResult,
    assert_exact_emulation,
    replay_wbm_event_log,
)

__all__ = sorted(name for name in locals() if not name.startswith("_"))
