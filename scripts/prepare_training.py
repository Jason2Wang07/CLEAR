#!/usr/bin/env python3
"""Resolve the released recipes onto a local filesystem without changing hashes."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs" / "reference"


def prepare(benchmark, paths_file, output):
    paths = yaml.safe_load(Path(paths_file).read_text())
    paths = {key: str(Path(value).expanduser().resolve()) for key, value in paths.items()}
    required = {"qwen", "dino", "lam_release", "initial_policy", "effect_checkpoint",
                "action_checkpoint", "action_data", "effect_data", "effect_artifacts",
                "latent_cache", "dataset_parent", "source", "deterministic_checkpoint", "output"}
    if benchmark == "robotwin":
        required.add("sidecar_reference")
    if required - paths.keys():
        raise ValueError(f"Missing paths: {sorted(required - paths.keys())}")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    effect_name = f"effect_stage1_{benchmark}_v1_12s_dino_residual.yaml"
    action_name = ("action_beta_vae_libero_v1_12s_a1.yaml" if benchmark == "libero"
                   else "action_beta_vae_robotwin_v1_12s_dense_s1_a1.yaml")
    ae_name = f"action_tcn_ae_{benchmark}_v1_12s_a1.yaml"
    provenance = {}

    def save(cfg, name, source):
        target = output / name
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        provenance[name] = {
            "reference": str(source.relative_to(ROOT)),
            "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "resolved_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }

    for phase in "wpe":
        name = (f"libero_stage2_release_residual_phase_{phase}.yaml" if benchmark == "libero"
                else f"robotwin_stage2_sft_release_residual_phase_{phase}_h200_8gpu.yaml")
        source = REFERENCE / name
        cfg = yaml.safe_load(source.read_text())
        action = cfg["framework"]["action_model"]
        cfg["framework"]["qwenvl"]["base_vlm"] = paths["qwen"]
        overrides = {
            "lam_ckpt_path": paths["effect_checkpoint"],
            "lam_yaml_path": str(REFERENCE / effect_name),
            "lam_vision_model_id_override": paths["dino"],
            "stage2_effect_teacher_checkpoint_path": paths["effect_checkpoint"],
            "stage2_effect_teacher_config_path": str(REFERENCE / effect_name),
            "stage2_action_decoder_checkpoint_path": paths["action_checkpoint"],
            "stage2_action_decoder_config_path": str(REFERENCE / action_name),
            "stage2_action_decoder_data_root": paths["action_data"],
            "stage2_latent_cache_root": paths["latent_cache"],
            "stage2_latent_normalization_path": str(Path(paths["latent_cache"]) / ("normalization.json" if benchmark == "libero" else "latent_normalization.json")),
        }
        if benchmark == "robotwin":
            overrides["stage2_action_decoder_sidecar_reference_root"] = paths["sidecar_reference"]
        action.update(overrides)
        data = cfg["datasets"]["vla_data"]
        data.update(data_root_dir=paths["dataset_parent"], stage2_source_root=paths["source"])
        if benchmark == "libero":
            data["stage2_paired_cache"] = paths["latent_cache"]
            data["stage2_paired_cache_provenance"] = str(Path(paths["latent_cache"]) / "provenance.json")
        else:
            data["stage2_manifest"] = str(Path(paths["action_data"]) / "manifests")
            data["stage2_paired_cache"] = str(Path(paths["latent_cache"]) / "paired_latents.npz")
        cfg["run_root_dir"] = str(Path(paths["output"]) / "policy" / f"phase-{phase}")
        cfg["run_id"] = f"clear-{benchmark}-{phase}"
        cfg["trackers"] = []
        cfg["wandb_project"] = "clear"
        if phase == "w":
            cfg["trainer"]["pretrained_checkpoint"] = paths["initial_policy"]
        save(cfg, f"phase_{phase}.yaml", source)

    for kind, name in [("effect", effect_name), ("action_ae", ae_name), ("action_vae", action_name)]:
        source = REFERENCE / name
        cfg = yaml.safe_load(source.read_text())
        model = cfg["model"]
        cfg["data"]["data_root"] = paths["effect_data" if kind == "effect" else "action_data"]
        if kind == "effect":
            model.update(vision_model_id=paths["dino"], init_checkpoint=paths["lam_release"],
                         teacher_graph_path=str(Path(paths["effect_artifacts"]) / "train_teacher_graph.npz"),
                         latent_bank_path=str(Path(paths["effect_artifacts"]) / "release_latent_bank.npz"))
        else:
            model["normalization_path"] = str(Path(paths["action_data"]) / "stats/normalization.json")
            if "deterministic_checkpoint_path" in model:
                model["deterministic_checkpoint_path"] = paths["deterministic_checkpoint"]
        run = Path(paths["output"]) / kind
        if kind == "action_vae" and not cfg["trainer"].get("callbacks"):
            # These are supplied by the original RoboTwin training launcher.
            cfg["trainer"]["callbacks"] = [{
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {"dirpath": str(run / "checkpoints"), "filename": "step-{step:07d}",
                              "monitor": "val/loss", "mode": "min", "save_top_k": 1,
                              "save_last": True, "auto_insert_metric_name": False},
            }]
            cfg["trainer"]["logger"] = [{
                "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args": {"save_dir": str(run / "tensorboard"), "name": "beta-vae"},
            }]
        for callback in cfg["trainer"].get("callbacks") or []:
            args = callback.get("init_args", {})
            if "dirpath" in args:
                args["dirpath"] = str(run / "checkpoints")
        for logger in cfg["trainer"].get("logger") or []:
            logger["init_args"]["save_dir"] = str(run)
        cfg["trainer"]["default_root_dir"] = str(run)
        save(cfg, f"{kind}.yaml", source)
    (output / "config_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["libero", "robotwin"], required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.benchmark, args.paths, args.output))


if __name__ == "__main__":
    main()
