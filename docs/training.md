# Training

## Model and stages

The effect model aligns local latent neighborhoods with frozen DINO feature
differences. The conditional TCN autoencoder is then trained on trajectories and
warm-starts a standard-normal beta-VAE. The policy predicts effect and action
latents from current observations and language; posterior teachers are used only
for supervision.

Policy training runs **W → P → E**:

| Phase | Trainable components |
| --- | --- |
| W | Effect prediction head and world decoder |
| P | VLM subset, prediction heads and action residual adapter; world decoder and action expert frozen |
| E | VLM subset, prediction heads, world decoder, residual adapter and action expert |

DINO, posterior teachers, and the trajectory decoder remain frozen. The action
predictor receives detached visual features and effect codes. P and E start from
the previous phase's selected checkpoint with a fresh optimizer and schedule.

## Representation learning

Generate local YAML with `scripts/prepare_training.py` first. The reference YAML
files preserve original bytes because their hashes are part of cache provenance.
Generated configs record both reference and resolved hashes.

```bash
python -m latent_action_model.stage1.main fit --config configs/local/libero/effect.yaml
python -m latent_action_model.action_ae.main fit --config configs/local/libero/action_ae.yaml
python -m latent_action_model.action_ae.beta_vae_main fit --config configs/local/libero/action_vae.yaml
```

For RoboTwin, use the generated `robotwin` configs and replace the action module
prefix with `latent_action_model.robotwin_action_ae`. Build and join the selected
effect posterior into the action sidecar before action training. Set
`deterministic_checkpoint` to the selected autoencoder before beta-VAE training.

| Component | LIBERO | RoboTwin |
| --- | --- | --- |
| Effect model | 40 epochs; selected epoch 39 | 40 epochs; selected epoch 9 |
| Effect global batch | 512 = 2 × 64 × 4 | 512 = 2 × 64 × 4 |
| Deterministic action warm start (archived YAML) | 100 epochs | 100 epochs |
| Action beta-VAE | 10,000 updates | 10,000 updates |
| Teacher temporal window | 1.2 s / 24 steps | 1.2 s / 36 steps |

The released simulation recipes are tied to the released teacher hashes. Training
new teachers is supported by the representation modules, but substituting new
teachers into a locked policy recipe requires regenerating caches and their
provenance. Do not disable the hash checks or relabel a new teacher as the original.

## Policy finetuning

| Benchmark: recorded recipe | W updates | P updates | E updates | Global batch |
| --- | ---: | ---: | ---: | ---: |
| LIBERO | 2,000 | 8,000 | 15,000 | 256 = 4 × 16 × 4 |
| RoboTwin | 4,000 | 16,000 | 30,000 | 1,024 = 8 × 32 × 4 |

These are per-phase optimizer updates after accumulation. Initialize W from the
benchmark's released LaWAM SFT policy. Use `--gpus 4` for the provided LIBERO
configuration and `--gpus 8` for RoboTwin. The launcher checks the global batch
before launching. The distributed training module can also be invoked directly
for multi-node arrangements.

```bash
python scripts/prepare_training.py --benchmark robotwin \
  --paths paths-robotwin.yaml --output configs/local/robotwin
python scripts/train_phases.py --configs configs/local/robotwin \
  --gpus 8 --run-name clear-robotwin
```

Use the same run name to resume. Only complete `steps_*_training_state`
directories restore optimizer, scheduler, and random state. Published model-only
weights are suitable for inference or a new finetuning run, not exact resume.

Every 500 updates, the training objective is evaluated over 20 monitoring
batches. The selected checkpoint need not be the last one. LIBERO follows the
inherited all-row training convention, so its tail monitoring subset is not an
independent held-out set. RoboTwin uses episode-disjoint splits.

## Recorded artifacts and paper differences

The supplied paper and development artifacts differ in these reproducibility
details. This release preserves the recorded artifacts rather than changing
hyperparameters after training:

| Item | Supplied paper supplement | Recorded release artifacts |
| --- | --- | --- |
| RoboTwin W/P/E allocation | 8k / 12k / 30k | 4k / 16k / 30k |
| RoboTwin selected ActionVAE | Step 5,000 | Step 10,000, SHA-256 beginning `fac6da74` |
| RoboTwin deterministic action warm start | 50,000 updates | Archived effect-conditioned recipe specifies 100 epochs |

The original RoboTwin model repository's phase W and P selection histories end
at 4,000 and 16,000 respectively, consistent with the development scripts.
The paper's main implementation section specifies SFT initialization; some
supplement text instead says pretrained initialization. The supplied release
recipes initialize from the benchmark SFT policy.

The policy artifacts selected for publication are LIBERO E/13,500 and RoboTwin
E/28,500. Their complete identities are recorded in `models/manifest.json` and
their phase sidecars. The paper reports 99.1% LIBERO and 93.86%/90.92% RoboTwin
clean/randomized; these results were not rerun during repository preparation.

Real-world reproduction is outside this initial bundle: the paper describes
30k updates per task with task-specific CLEAR teachers, whereas the available
older `real_franka` configurations describe 10k-update LaWAM baselines. They
have not been relabeled as CLEAR recipes.
