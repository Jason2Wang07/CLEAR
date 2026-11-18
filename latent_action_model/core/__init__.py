

from .lam_model import LatentLAMModel, load_latent_action_model

from .vq import VQ


def __getattr__(name: str):
    if name == "VJEPA_LAM":
        from .lam_lightinng import VJEPA_LAM

        return VJEPA_LAM
    raise AttributeError(name)

__all__ = [


    "VJEPA_LAM",

    "LatentLAMModel",


    "VQ",
]
