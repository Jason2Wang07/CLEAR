# Third-party notices

This repository is derived from [RLinf/LaWAM](https://github.com/RLinf/LaWAM),
using development base commit `1add20a`. The upstream package metadata declares
MIT; its original metadata and author list are retained in
`third_party/LaWAM-pyproject.toml`. The upstream snapshot did not contain a
standalone root license file. Existing copyright and license headers remain
attached to copied files.

| Component | Attribution and terms |
| --- | --- |
| LaWAM / StarVLA policy and training infrastructure | LaWAM authors; StarVLA community; MIT as declared in metadata/source headers |
| `latent_action_model/core/utils/modules.py`, `pos_embs.py` | Meta Platforms, Inc. and affiliates; MIT source headers |
| `starVLA/dataloader/gr00t_lerobot/`, GR00T-derived loader and cross-attention DiT | NVIDIA Corporation and affiliates; Apache-2.0 |
| Flow-matching expert | Hugging Face Inc. and upstream contributors; Apache-2.0 source header |
| Training logging utility | OpenVLA / Prismatic contributors; MIT source attribution |

The Apache-2.0 text is included at `third_party/Apache-2.0.txt`. CLEAR modifies
the inherited policy, representation-learning, data and checkpoint code; these
files are not an unmodified distribution of LaWAM or GR00T.

Qwen3-VL and DINOv3 are separate pretrained models. Obtain base weights from the
original publishers and comply with their model terms. DINOv3's license is
retained at `third_party/DINOv3-LICENSE.md`; it also applies where a released
checkpoint contains DINOv3-derived weights. The root MIT license is a code license,
not a replacement for pretrained model licenses. Benchmark code and datasets
remain under their original licenses and are not vendored here.
