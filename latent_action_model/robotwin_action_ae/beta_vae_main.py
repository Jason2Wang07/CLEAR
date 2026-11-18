"""Lightning CLI entry point for the RoboTwin standard-normal Action beta-VAE."""

from lightning.pytorch.cli import LightningCLI

from .beta_vae_lightning import RoboTwinActionBetaVAELightningModule
from .data import RoboTwinActionDataModule


def main() -> None:
    LightningCLI(
        RoboTwinActionBetaVAELightningModule,
        RoboTwinActionDataModule,
        seed_everything_default=20260829,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
