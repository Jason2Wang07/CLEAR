"""Action trajectory autoencoder core."""

from .data import (
    ActionNormalization,
    ActionSidecarDataModule,
    ActionSidecarDataset,
    RobustGroupStats,
    action_sidecar_collate,
)
from .context import (
    ActionPriorContextCache,
    PRIOR_CONTEXT_KEYS,
    PRIOR_CONTEXT_SCHEMA_VERSION,
)
from .losses import (
    TrajectoryLossOutput,
    mean_so3_geodesic_error,
    so3_geodesic_error,
    trajectory_reconstruction_loss,
)
from .distributions import (
    DiagonalGaussian,
    clamp_logvar,
    diagonal_gaussian_kl,
    scheduled_kl_weight,
    split_gaussian_parameters,
    standardize_gaussian,
)
from .model import (
    ActionAEConfig,
    ActionAEOutput,
    ActionAEVariant,
    ActionAutoencoder,
    ActionVAEConfig,
    ActionVAEOutput,
    ActionVariationalAutoencoder,
    ConditionalTCNDecoder,
    DeterministicTCNEncoder,
    FiLM,
    FutureAwareActionPrior,
    assert_effect_model_frozen,
    freeze_effect_model,
    prepare_effect_condition,
)


def __getattr__(name: str):
    if name == "ActionAELightningModule":
        from .lightning import ActionAELightningModule

        return ActionAELightningModule
    if name == "ActionVAELightningModule":
        from .vae_lightning import ActionVAELightningModule

        return ActionVAELightningModule
    raise AttributeError(name)

__all__ = [
    "ActionAEConfig",
    "ActionAEOutput",
    "ActionAEVariant",
    "ActionAELightningModule",
    "ActionVAELightningModule",
    "ActionAutoencoder",
    "ActionVAEConfig",
    "ActionVAEOutput",
    "ActionVariationalAutoencoder",
    "ActionNormalization",
    "ActionPriorContextCache",
    "ActionSidecarDataModule",
    "ActionSidecarDataset",
    "ConditionalTCNDecoder",
    "DeterministicTCNEncoder",
    "DiagonalGaussian",
    "FiLM",
    "FutureAwareActionPrior",
    "PRIOR_CONTEXT_KEYS",
    "PRIOR_CONTEXT_SCHEMA_VERSION",
    "RobustGroupStats",
    "TrajectoryLossOutput",
    "assert_effect_model_frozen",
    "action_sidecar_collate",
    "clamp_logvar",
    "diagonal_gaussian_kl",
    "freeze_effect_model",
    "mean_so3_geodesic_error",
    "prepare_effect_condition",
    "scheduled_kl_weight",
    "so3_geodesic_error",
    "trajectory_reconstruction_loss",
    "split_gaussian_parameters",
    "standardize_gaussian",
]
