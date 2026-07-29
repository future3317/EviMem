"""Named constants for reproducible protocol-aware materials discovery.

Every numerical default, seed offset, and tolerance that affects scientific
outcomes lives here.  Changing a value changes published results; keep them
versioned and never inline them.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Seed offsets for randomized-QMC streams
# -----------------------------------------------------------------------------

SOBOL_BLOCK_OFFSET = 104_729
"""Offset between independent scrambled-Sobol blocks for the same base seed."""

ROUND_SEED_OFFSET = 1_009
"""Offset between policy rounds inside a single campaign seed."""

CAMPAIGN_POLICY_SEED_OFFSET = 700_001
"""Domain-separated seed for campaign-level policy simulation."""

IC_STAGE_TWO_SEED_OFFSET = 1_000_000_007
"""Disjoint offset for IC-SARR stage-two confirmation stream."""

# -----------------------------------------------------------------------------
# Default experiment seeds
# -----------------------------------------------------------------------------

DEFAULT_EXPERIMENT_SEED = 20_270_720
"""Frozen MatPES exploratory campaign seed."""

# -----------------------------------------------------------------------------
# Numerical tolerances
# -----------------------------------------------------------------------------

HULL_NUMERICAL_TOLERANCE = 1e-11
"""Formation-energy threshold for fixed-composition hull stability."""

HULL_SYMMETRY_TOLERANCE = 1e-10
"""Covariance symmetry check tolerance."""

HULL_PSD_TOLERANCE = 1e-8
"""Smallest negative eigenvalue allowed for a covariance matrix."""

PSD_REGULARIZATION = 1e-12
"""Jitter added to covariance diagonals after symmetrization."""

MIN_FEATURE_SCALE = 1e-8
"""Minimum standard deviation used when normalizing transport features."""

MIN_VARIANCE = 1e-15
"""Floor used to avoid division by zero in variance-based conditioning."""

MIN_POSITIVE_DISTANCE = 1e-10
"""Minimum distance considered positive when fitting local kernels."""

KERNEL_JITTER = 1e-10
"""Diagonal jitter added to kernel covariance matrices."""

COVARIANCE_CLIP_EPSILON = 2.220446049250313e-16
"""Machine epsilon used to clip Sobol-derived uniforms away from 0 and 1."""

ENTRY_ENERGY_TOLERANCE = 1e-8
"""Tolerance for corrected phase entry total-energy validation."""

FEATURE_SCALE_FLOOR = 1e-8
"""Floor for feature standard-deviation scaling."""

TRANSPORT_RIDGE_INTERCEPT_FLOOR = 1e-8
"""Ridge penalty applied to the intercept term in protocol transport."""

WITHIN_SYSTEM_VARIANCE_FLOOR = 1e-8
"""Floor for estimated within-system residual variance."""

BETWEEN_SYSTEM_VARIANCE_FLOOR = 1e-8
"""Floor for estimated between-system variance."""

LOCAL_KERNEL_LENGTH_FLOOR = 1e-3
"""Lower bound for local-kernel length-scale optimization."""

LOCAL_KERNEL_VARIANCE_FLOOR = 1e-10
"""Lower bound for local-kernel variance optimization."""

SOURCE_AFFINE_PTP_FLOOR = 1e-12
"""Minimum peak-to-peak source energy for affine source-target fit."""

BUDGET_FLOOR = 1e-12
"""Budget comparison tolerance for unit-cost policies."""

# -----------------------------------------------------------------------------
# Default Monte-Carlo and optimization counts
# -----------------------------------------------------------------------------

DEFAULT_SOBOL_SCRAMBLE_COUNT = 16
"""Default number of independent Sobol scrambles."""

DEFAULT_POSTERIOR_SAMPLE_COUNT = 1_024
"""Default posterior sample count for source-rollout evaluations."""

DEFAULT_FANTASY_COUNT = 3
"""Default Gauss-Hermite fantasy count for two-step objectives."""

DEFAULT_INTEGRATION_CONFIDENCE = 0.95
"""Default simultaneous lower-bound confidence."""

MIN_POSTERIOR_SAMPLE_COUNT = 4
"""Minimum posterior sample count accepted by hull-aware policies."""

MIN_SOBOL_SCRAMBLE_COUNT = 2
"""Minimum number of independent Sobol scrambles."""

MIN_DUAL_HORIZON_BLOCK_SIZE = 2
"""Minimum samples per Sobol block."""

# -----------------------------------------------------------------------------
# Default experimental budgets
# -----------------------------------------------------------------------------

DEFAULT_MATPES_BUDGET = 6
"""Default query budget for MatPES closed-loop experiments."""

# -----------------------------------------------------------------------------
# Default ridge/transport hyperparameters
# -----------------------------------------------------------------------------

DEFAULT_RIDGE_PENALTY = 1.0
"""Default ridge penalty for protocol transport."""

DEFAULT_PRIOR_STANDARD_DEVIATION = 0.1
"""Default prior standard deviation for ridge acquisition."""

DEFAULT_BOUNDARY_TEMPERATURE = 0.05
"""Default boundary temperature for ridge acquisition."""

# -----------------------------------------------------------------------------
# Local kernel defaults
# -----------------------------------------------------------------------------

DEFAULT_LOCAL_KERNEL_LENGTH_SCALE = 1.0
"""Default Matérn-5/2 length scale before optimization."""

LOCAL_KERNEL_QUANTILE_LOWER = 0.05
"""Lower empirical-Bayes quantile for length-scale bound."""

LOCAL_KERNEL_QUANTILE_UPPER = 0.95
"""Upper empirical-Bayes quantile for length-scale bound."""

LOCAL_KERNEL_SIGNAL_VARIANCE_FRACTION = 1e-4
"""Lower signal-variance bound as fraction of total within-system variance."""

LOCAL_KERNEL_VARIANCE_MULTIPLIER = 10.0
"""Upper variance bound multiplier."""

LOCAL_KERNEL_BOUND_ACTIVE_TOLERANCE = 1e-8
"""Tolerance for deciding an L-BFGS-B bound is active."""

LOCAL_KERNEL_OPTIMIZER_MESSAGE = ""
"""Default message when no optimizer message is available."""

# -----------------------------------------------------------------------------
# Acquisition / policy defaults
# -----------------------------------------------------------------------------

DEFAULT_ORACLE_COST = 1.0
"""Default relative cost of a single oracle query."""

DEFAULT_CONFORMAL_ALPHA = 0.1
"""Default miscoverage level for split-conformal rollout calibration."""

DEFAULT_CAMPAIGN_OUTER_SAMPLE_COUNT = 128
"""Default outer posterior sample count for campaign gates."""

DEFAULT_CAMPAIGN_OUTER_SEED = 0
"""Default outer seed for campaign gates."""

DEFAULT_CAMPAIGN_INNER_STAGE_ONE_SAMPLE_COUNT = 64
"""Default stage-one sample count for campaign gate IC-SARR inner loop."""

DEFAULT_CAMPAIGN_INNER_STAGE_TWO_SAMPLE_COUNT = 128
"""Default stage-two sample count for campaign gate IC-SARR inner loop."""

DEFAULT_CAMPAIGN_SOBOL_SCRAMBLE_COUNT = 8
"""Default Sobol scramble count for campaign gates."""

# -----------------------------------------------------------------------------
# Subprocess / serialization
# -----------------------------------------------------------------------------

DEFAULT_POLICY_TIMEOUT_SECONDS = 300
"""Default one-shot policy subprocess timeout."""

PERSISTENT_WORKER_POLL_INTERVAL_SECONDS = 0.05
"""Sleep interval when draining persistent worker streams."""

WORKER_EXIT_GRACE_PERIOD_SECONDS = 5.0
"""Grace period before force-terminating a persistent worker."""
