"""Lightning CLI entry point for ActionAE A0/A1 training."""

from lightning.pytorch.cli import LightningCLI

from latent_action_model.action_ae.data import ActionSidecarDataModule
from latent_action_model.action_ae.lightning import ActionAELightningModule


def main() -> None:
    LightningCLI(
        ActionAELightningModule,
        ActionSidecarDataModule,
        seed_everything_default=20260807,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
