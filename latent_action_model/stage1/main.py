"""Lightning CLI entry point for locked Stage-1 effect fine-tuning."""

from lightning.pytorch.cli import LightningCLI

from latent_action_model.core.effect_stage1_lightning import EffectStage1LAM
from latent_action_model.data_loader.effect_window_datamodule import EffectWindowDataModule


def main() -> None:
    LightningCLI(
        EffectStage1LAM,
        EffectWindowDataModule,
        seed_everything_default=20260803,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
