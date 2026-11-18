# CLEAR

**Combining Scene Changes and Action Trajectories for Robot Control**

[Models](https://huggingface.co/Jason2Jason/CLEAR) · [Training](docs/training.md) · [Data preparation](docs/data.md) · [Evaluation](docs/evaluation.md)

CLEAR builds on [LaWAM](https://github.com/RLinf/LaWAM) by combining a
geometry-regularized representation of scene changes with an effect-conditioned
ActionVAE representation of demonstrated trajectories. At deployment, the policy
predicts an effect, decodes future visual features, and predicts an action latent
through cross-attention to dense transition features. A zero-initialized residual
adapter injects that latent into a flow-matching action expert.

The implementation uses Qwen3-VL-2B-Instruct, frozen DINOv3 ViT-B/16 features,
32-dimensional effect/action latents, and ten flow integration steps. The original
`starVLA` and `latent_action_model` package names are retained for checkpoint compatibility.

## Results reported in the paper

| Benchmark | Setting | Success rate |
| --- | --- | ---: |
| LIBERO | Four-suite average | **99.1%** |
| RoboTwin | Clean | **93.86%** |
| RoboTwin | Randomized | **90.92%** |

These are the supplied paper's main-table results for one evaluation seed. The
unrounded LIBERO average is 99.05%. This release does not claim a new benchmark
rerun. See [reproduction notes](docs/training.md#recorded-artifacts-and-paper-differences)
for differences between the paper's schedules and the recorded training artifacts.

## Installation

Use Linux, Python 3.10, and a CUDA environment compatible with PyTorch 2.6.

```bash
git clone https://github.com/Jason2Wang07/CLEAR.git
cd CLEAR
conda create -n clear python=3.10 -y
conda activate clear
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e . --no-deps
```

FlashAttention must match your CUDA/PyTorch installation. Install benchmark
simulators in separate environments as described in the evaluation guide.

## Download and serve a policy

Accept the upstream DINOv3 model terms and authenticate with Hugging Face before
downloading that encoder. Base models remain available from their original publishers:

```bash
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir checkpoints/base/Qwen3-VL-2B-Instruct
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir checkpoints/base/dinov3-vitb16-pretrain-lvd1689m
python scripts/download_models.py --benchmark libero --kind policy

export CLEAR_QWEN_PATH="$PWD/checkpoints/base/Qwen3-VL-2B-Instruct"
export CLEAR_DINO_PATH="$PWD/checkpoints/base/dinov3-vitb16-pretrain-lvd1689m"
python -m deployment.model_server.server_policy \
  --ckpt_path checkpoints/clear/libero/policy/final_model/pytorch_model.pt \
  --port 10091
```

Download `--benchmark robotwin` for RoboTwin. Each policy bundle includes its YAML,
dataset normalization statistics, and phase metadata; keep the directory intact.
Inference restores affine normalization and decoder tensors from the full policy
state and does not require training datasets, teacher caches, or separate teacher
checkpoints. The frozen ActionVAE decoder is retained in the graph for checkpoint
compatibility and is not used to produce deployed actions.

## Train

Prepare the benchmark data and teacher assets using [the data guide](docs/data.md).
Copy `configs/paths.example.yaml` to a local file and configure the paths, then:

```bash
python scripts/prepare_training.py --benchmark libero \
  --paths paths.yaml --output configs/local/libero
python scripts/train_phases.py --configs configs/local/libero \
  --gpus 4 --run-name clear-libero --dry-run
python scripts/train_phases.py --configs configs/local/libero \
  --gpus 4 --run-name clear-libero
```

The phase runner preserves preflight validation, checkpoint selection, and exact
optimizer-state resume. Teacher checkpoints and configs remain bound by SHA-256.
Full representation-learning commands and batch schedules are in [training](docs/training.md).

## Repository layout

| Directory | Contents |
| --- | --- |
| `latent_action_model/` | Effect learning, ActionVAE, geometry losses, teacher caches and temporal adapters |
| `starVLA/` | CLEAR policy, flow expert, dataloaders and W/P/E training |
| `data_processing/` | LIBERO/RoboTwin conversion and trajectory sidecars |
| `deployment/` | Policy servers and transport |
| `examples/` | LIBERO and RoboTwin evaluation clients |
| `configs/reference/` | Original, hash-bound recipe snapshots |
| `scripts/` | Portable configuration, training and model download commands |
| `models/manifest.json` | Released file sizes, roles and SHA-256 checksums |
| `tests/` | Model, data-contract and deployment regression tests |

The release focuses on the simulation policies and representation-learning code.
The development directory's unrelated experiments, cloud submission scripts,
datasets, logs, and optimizer snapshots are excluded from Git. The paper's
real-world CLEAR experiments are not presented as reproducible from the older
LaWAM baseline finetuning scripts.

## Acknowledgments and licenses

Built on LaWAM, StarVLA, Qwen3-VL, DINOv3, and the LIBERO/RoboTwin ecosystems.
The code uses MIT and Apache-2.0 components; original file notices are retained.
See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
Pretrained model weights and datasets retain their upstream terms, including the
DINOv3 license.
