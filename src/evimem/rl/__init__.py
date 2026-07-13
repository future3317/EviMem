"""Replay and reward substrate for EviMem controllers.

Training-framework adapters belong above this package; publication authority
does not.
"""

from .replay_buffer import TrajectoryReplayBuffer
from .reward import RewardBreakdown, RewardConfig, VerifierShapedReward
from .trajectory import (
    RunAuditEvent,
    RunAuditTrajectory,
    TrajectoryRecorder,
    build_run_trajectory,
)

__all__ = [
    "RewardBreakdown",
    "RewardConfig",
    "RunAuditEvent",
    "RunAuditTrajectory",
    "TrajectoryRecorder",
    "TrajectoryReplayBuffer",
    "VerifierShapedReward",
    "build_run_trajectory",
]
