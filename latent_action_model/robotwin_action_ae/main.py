"""Lightning CLI entry point for RoboTwin ActionAE A0/A1 training."""

from lightning.pytorch.cli import LightningCLI

from .data import RoboTwinActionDataModule
from .lightning import RoboTwinActionAELightningModule


def main() -> None:
    LightningCLI(
        RoboTwinActionAELightningModule,
        RoboTwinActionDataModule,
        seed_everything_default=20260820,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
