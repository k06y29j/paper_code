# CNN accepted q-mix receiver

This directory is the runnable migration of the accepted CNN receiver path.
It preserves the receiver contract:

```text
Layer1: img -> E1 -> z1 -> D1 -> x1
receiver: (z1,x1) -> fixed q2_hat mix -> D2(q2_hat) -> u2_hat
          -> combiner(x1,u2_hat) -> x2_hat
```

The receiver never accepts `img`, `z2`, true `q2`, or oracle indices.  During
training the image becomes a target only after the receiver forward pass.

## Migrated modules

- `train_continuous_q_mix_receiver.py`: fixed train-selected q-mix training;
  only the post-q `D2` and combiner are optimized.
- `eval_continuous_q_mix_receiver.py`: strict read-only valid100 evaluation of
  raw and EMA post-q states, including sender-Layer2 tripwire checks.
- `train_continuous_q_receiver.py`, `train_layer2_vq_nested.py`,
  `contracts.py`, `receiver_models.py`, `vq_modules.py`: runtime dependency
  closure needed to load the CNN Layer1 source and continuous-q generators.

## Accepted e60 artifact and required external weights

Code and launch scripts are owned here.  To avoid duplicating large model
files, the migrated config deliberately references the archived checkpoints
under `MY-V2/jscc-f/explore-2` by default.  `ARTIFACT_ROOT` overrides the
initialization/e60 snapshot paths in the supplied shell scripts; if generator
checkpoints move as well, pass a rewritten compatible `ENSEMBLE_SPEC` because
their exact paths are part of the fixed-mix provenance.

- q-mix config: `configs/accepted_qmix_top3_v9_ema.json`
- strict e60 checkpoint:
  `explore-2/checkpoints-continuous-q-mix/cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1/continuous_q_mix_receiver_final_e60_before_continue.pth`
- source Layer1 checkpoint and the three q generators are recorded inside the
  q-mix config/checkpoint and are loaded read-only.

The accepted EMA full DIV2K valid100 report is `PSNR(x1)=21.771992`,
`PSNR(x2_hat)=22.261801`, and `delta=+0.489809 dB`; raw is `+0.383922 dB`.
The old automatic `0.5 dB` boolean therefore remains false by `0.010191 dB`,
while this version is retained as the user-accepted baseline.

## Commands

Run the exact-hyperparameter fresh reproduction (new output directory by
default, so the accepted archive is never overwritten):

```bash
cd /workspace/yongjia/paper_code/CDDM/MY-V2/jscc-f/cnn-work
GPU=1 setsid -w ./run_cnn_continuous_q_mix_top3_v9mse_ema05fix_e60.sh
```

Strictly re-evaluate the immutable accepted e60 snapshot:

```bash
cd /workspace/yongjia/paper_code/CDDM/MY-V2/jscc-f/cnn-work
GPU=1 ./eval_cnn_continuous_q_mix_top3_v9mse_ema05fix_e60.sh
```

Both entrypoints enforce `RandomCrop(256)+RandomHorizontalFlip` for training
and `CenterCrop(256)` for validation.  The latter processes exactly 100 DIV2K
validation images and reports receiver-only, q-only-D2, combiner-input, crop,
and sender-Layer2-tripwire audits.
