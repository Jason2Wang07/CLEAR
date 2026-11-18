"""Lightning CLI entry point for the standard-normal Action beta-VAE."""

from lightning.pytorch.cli import LightningCLI

from .beta_vae_lightning import ActionBetaVAELightningModule
from .data import ActionSidecarDataModule


def main() -> None:
    LightningCLI(
        ActionBetaVAELightningModule,
        ActionSidecarDataModule,
        seed_everything_default=20260825,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
