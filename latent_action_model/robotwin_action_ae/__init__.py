"""RoboTwin dual-arm ActionAE training stack."""

from .data import RoboTwinActionDataModule, RoboTwinActionDataset
from .lightning import RoboTwinActionAELightningModule

__all__ = [
    "RoboTwinActionAELightningModule",
    "RoboTwinActionBetaVAELightningModule",
    "RoboTwinActionVAELightningModule",
    "RoboTwinActionDataModule",
    "RoboTwinActionDataset",
]


def __getattr__(name: str):
    if name == "RoboTwinActionBetaVAELightningModule":
        from .beta_vae_lightning import RoboTwinActionBetaVAELightningModule

        return RoboTwinActionBetaVAELightningModule
    if name == "RoboTwinActionVAELightningModule":
        from .vae_lightning import RoboTwinActionVAELightningModule

        return RoboTwinActionVAELightningModule
    raise AttributeError(name)
