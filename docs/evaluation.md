# Evaluation

Download a complete policy bundle with `scripts/download_models.py`. Set
`CLEAR_QWEN_PATH` and `CLEAR_DINO_PATH` to the local base models as in the README.
The policy process uses the CLEAR environment; simulation runs in a separate
benchmark environment. No future observations or demonstration cache is supplied
to a deployed policy.

## LIBERO

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) following its
instructions. The inherited evaluation stack uses MuJoCo 3.3.2. From CLEAR:

```bash
export LIBERO_HOME=/absolute/path/to/LIBERO
export LIBERO_PYTHON=/absolute/path/to/libero-env/bin/python
export STAR_VLA_PYTHON="$(command -v python)"
SUITES="libero_10 libero_goal libero_object libero_spatial" \
NUM_TRIALS_PER_TASK=50 NUM_WORKERS=1 GPU_IDS="0" \
OUTPUT_ROOT=results/eval/libero LIBERO_CKPT_ALIAS=clear \
bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh \
  checkpoints/clear/libero/policy/final_model/pytorch_model.pt
```

The selected policy is phase E, local update 13,500. LIBERO executes eight
actions per policy query at 20 Hz. The evaluation client retains the benchmark's
initialization and settling behavior; do not change it when comparing results.

## RoboTwin

Install [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) and its robot and
scene assets in a separate simulator environment. Install the lightweight RPC
dependencies there (`websockets==15.0.1`, `msgpack==1.1.2`, `json-numpy==2.1.1`,
`omegaconf==2.3.0`, `rich`, and `accelerate`). Set:

```bash
export ROBOTWIN_PATH=/absolute/path/to/RoboTwin
export ROBOTWIN_PYTHON=/absolute/path/to/robotwin-env/bin/python
ROBOTWIN_TASKS=lift_pot \
bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
  checkpoints/clear/robotwin/policy/final_model/pytorch_model.pt demo_clean
```

Remove `ROBOTWIN_TASKS` to run the configured full task set. Use the benchmark's
`demo_randomized` task configuration for randomized evaluation. The selected
policy is phase E, local update 28,500; its physical action chunk is 36 steps at
30 Hz. Keep clean/randomized conditions and evaluation seeds separate in reports.
For the paper's protocol, set `ROBOTWIN_TEST_NUM=100` (50 tasks, 5,000 rollouts
per condition). The inherited launcher defaults to 14 rollouts per task; that
default is not the paper protocol. For a single-GPU smoke run, also set
`NUM_WORKERS=1 ROBOTWIN_NUM_SLOTS=1 ROBOTWIN_TEST_NUM=1`.

## Validation scope

Repository validation covers representation math, masks and normalization,
checkpoint loading, path relocation, and deployment reconstruction. It does not
rerun the full simulator benchmarks or establish new success rates. Keep model
identity, simulator version, task configuration, seed and trial counts with every
result. A small smoke rollout is useful before starting a complete benchmark.
