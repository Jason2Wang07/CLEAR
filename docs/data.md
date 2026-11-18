# Data preparation

Training uses data supplied under the original benchmark licenses. No robot
demonstrations are included in Git or in the policy downloads.

| Input | Source |
| --- | --- |
| LIBERO official HDF5 demonstrations | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| LIBERO LeRobot policy dataset | [jialei02/libero_merged_no_noops_20hz](https://huggingface.co/datasets/jialei02/libero_merged_no_noops_20hz) |
| RoboTwin merged LeRobot dataset | [jialei02/robotwin_merged](https://huggingface.co/datasets/jialei02/robotwin_merged) |
| Initial LaWM | [jialei02/lawam_lam](https://huggingface.co/jialei02/lawam_lam) |
| Initial benchmark policies | [LIBERO SFT](https://huggingface.co/jialei02/lawam_libero_sft_release), [RoboTwin SFT](https://huggingface.co/jialei02/lawam_robotwin_sft_release) |

Run commands from the repository root. Data converters share sibling packages:

```bash
export PYTHONPATH="$PWD:$PWD/data_processing:${PYTHONPATH:-}"
```

## LIBERO representations

Build aligned effect/action windows with an explicit 24-step horizon at 20 Hz:

```bash
python data_processing/prepare_libero_actionvae.py build \
  --source-root data/libero_official --output data/libero/representation \
  --horizon-actions 24 --window-stride 24 --shard-rows 256
```

The converter preserves episode-disjoint splits, sample IDs, masks, and available
next observations. It does not synthesize missing object/contact annotations.
Use `effect/` for effect training and `action/` for the initial trajectory sidecar.

For the effect neighborhood teacher, run
`latent_action_model.stage1.precompute_teacher` subcommands in this order:
`extract-residuals`, `fit-pca`, `build-graph`. The CLI exposes all input/output
paths. Fit PCA and neighborhoods on the training split only. The selected recipe
uses PCA dimensions 2048 for LIBERO and 8192 for RoboTwin; pass the relevant
dimension to `fit-pca` (see `--help`). Build the initial latent bank with
`latent_action_model.stage1.precompute_latent_bank`, then train the effect model.

After selecting the effect teacher, export its posterior means with
`latent_action_model.stage1.export_effect_posterior_means`. Join them using:

```bash
python data_processing/join_libero_effect_posterior.py \
  --source-root data/libero/representation/action \
  --effect-manifest data/libero/representation/effect/manifests/all.parquet \
  --posterior-cache data/libero/effect_artifacts/effect_posterior_means.npz \
  --output data/libero/action_a1
```

The action-sidecar normalization is fitted on training data. Keep its schema,
provenance, checksums and normalization with the sidecar.

## LIBERO policy teacher cache

The policy uses the merged LeRobot release (273,465 rows), 0.4-second chunks at
20 Hz, and a temporal adapter into the ActionVAE's 24-slot representation. Eight
physical control steps are not 24 executed actions.

```bash
export CLEAR_LIBERO_EFFECT_CHECKPOINT="$PWD/checkpoints/clear/libero/effect/epoch-39.ckpt"
export CLEAR_LIBERO_ACTION_CHECKPOINT="$PWD/checkpoints/clear/libero/action/step-0009889.ckpt"
export CLEAR_LIBERO_ACTION_DATA="$PWD/data/libero/action_a1"
export CLEAR_DINO_PATH="$PWD/checkpoints/base/dinov3-vitb16-pretrain-lvd1689m"
python -m latent_action_model.libero_stage2.baseline_04s_cache \
  --dataset-root data/libero_merged_no_noops_20hz \
  --output-root data/libero/policy_cache
```

The inherited `baseline_04s_cache` module name describes the temporal interface;
the cache contains CLEAR's effect and ActionVAE targets. The writer validates
teacher/config hashes and emits provenance plus an atomic completion marker.

## RoboTwin

Use the explicit dual-arm converters, not the single-arm LIBERO converter:

```bash
python -m robotwin_dataset.converter --source-root data/robotwin_merged \
  --output data/robotwin/effect --horizon-frames 36 --window-stride 36
python -m robotwin_action_sidecar.converter --source-root data/robotwin_merged \
  --effect-root data/robotwin/effect --output data/robotwin/action
```

The released policy uses dense stride-one action windows. The
`robotwin_action_sidecar.dense_converter` entrypoint constructs these windows
using the selected effect checkpoint, its selection report, and frozen compatible
normalization. Run `--help` for the explicit paths and distributed export options.
Then use `latent_action_model.robotwin_stage2.export_teacher_cache` to produce
Gaussian action/effect targets. Training checks the 5,087,247-row dense schema,
episode splits, temporal contract, and teacher identities.

Reference normalization/provenance files accompany released action teachers.
They do not contain demonstration shards or constitute a complete training
dataset. Rebuild the dataset and keep integrity checks enabled.
