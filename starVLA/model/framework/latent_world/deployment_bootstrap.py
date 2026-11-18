"""Construct checkpoint-owned modules without opening training assets."""
from dataclasses import fields

import yaml

from latent_action_model.action_ae.beta_vae import ActionBetaVAEConfig, StandardNormalActionVAE
from latent_action_model.libero_stage2.runtime_contract import FrozenActionVAEDecoder
from latent_action_model.robotwin_stage2.runtime_contract import FrozenRoboTwinActionVAEDecoder


def build_checkpoint_action_decoder(config_path, *, robotwin=False):
    """Allocate the original decoder graph; strict policy loading restores it.

    The archived YAML supplies architecture only. Normalization, warm-start
    checkpoints, datasets, and optimizer state are deliberately not opened.
    """
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["model"]
    names = {field.name for field in fields(ActionBetaVAEConfig)}
    architecture = {key: value for key, value in config.items() if key in names}
    architecture.setdefault("variant", "a1")
    architecture.setdefault("trajectory_dim", 14 if robotwin else 7)
    model = StandardNormalActionVAE(ActionBetaVAEConfig(**architecture))
    wrapper = FrozenRoboTwinActionVAEDecoder if robotwin else FrozenActionVAEDecoder
    return wrapper(model)
