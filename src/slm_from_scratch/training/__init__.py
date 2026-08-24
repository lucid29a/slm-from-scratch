"""The training loop, and everything pluggable around it.

Importing this package registers every concrete LR-schedule strategy in
``LR_SCHEDULES``.
"""

from __future__ import annotations

from slm_from_scratch.training.callback import (
    Callback,
    CheckpointCallback,
    ConsoleLogger,
    EvalCallback,
    LoggingCallback,
    SampleGenerationCallback,
    StepMetrics,
    TensorBoardLogger,
    ThroughputCallback,
    WandbLogger,
)
from slm_from_scratch.training.checkpoint import CheckpointManager, TrainingState
from slm_from_scratch.training.distributed import (
    DDPStrategy,
    DistributedStrategy,
    SingleDeviceStrategy,
)
from slm_from_scratch.training.gradient import GradientAccumulator, GradientClipper
from slm_from_scratch.training.lr_schedule import (
    LR_SCHEDULES,
    WSD,
    CosineWithWarmup,
    LRSchedule,
    LRScheduleConfig,
)
from slm_from_scratch.training.optimizer import OptimizerConfig, OptimizerFactory
from slm_from_scratch.training.precision import PrecisionPolicy
from slm_from_scratch.training.trainer import Trainer, TrainerConfig

__all__ = [
    "LR_SCHEDULES",
    "WSD",
    "Callback",
    "CheckpointCallback",
    "CheckpointManager",
    "ConsoleLogger",
    "CosineWithWarmup",
    "DDPStrategy",
    "DistributedStrategy",
    "EvalCallback",
    "GradientAccumulator",
    "GradientClipper",
    "LRSchedule",
    "LRScheduleConfig",
    "LoggingCallback",
    "OptimizerConfig",
    "OptimizerFactory",
    "PrecisionPolicy",
    "SampleGenerationCallback",
    "SingleDeviceStrategy",
    "StepMetrics",
    "TensorBoardLogger",
    "ThroughputCallback",
    "Trainer",
    "TrainerConfig",
    "TrainingState",
    "WandbLogger",
]
