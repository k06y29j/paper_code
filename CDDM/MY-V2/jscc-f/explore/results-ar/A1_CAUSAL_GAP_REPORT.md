# A1: AR teacher / rollout / decoder gap

Date: 2026-07-23

## Scope

This was a read-only analysis experiment over existing checkpoints. It created
no optimizer and performed no training. E2, FSQ, D2, and combiner were kept in
`eval()` with zero trainable parameters. Their state SHA-256 hashes were
compared before and after every valid100 run and all hashes were unchanged.

The deployment target is:

```text
mean psnr_x2_hat >= mean psnr_x1 + 0.5 dB
                    = 21.771992 + 0.5
                    = 22.271992 dB
```

Four reconstructions were measured:

- `teacher_hard`: discrete argmax at every position with the true preceding
  tokens. This is diagnostic and is not a deployable sequence.
- `teacher_soft`: posterior-mean continuous FSQ value with true preceding
  tokens. This is diagnostic and is not a valid discrete deployment result.
- `rollout_hard`: deployable 256-step greedy AR.
- `rollout_soft`: posterior-mean continuous FSQ value under generated-prefix
  states. This is diagnostic and is not a valid discrete deployment result.

## Full valid100 results

| Model | Epoch | x1 | Oracle | Teacher hard | Teacher soft | Rollout hard | Rollout soft |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefix K125 best | 40 | 21.771992 | 22.875583 | 22.417773 | 22.455937 | 21.770519 | 21.779934 |
| Prefix K125 latest | 355 | 21.771992 | 22.875583 | 22.478385 | 22.524825 | 21.137686 | 21.140603 |
| Per-token-y1 K125 best | 5 | 21.771992 | 22.875583 | 21.758854 | 21.742578 | 21.755808 | 21.739832 |
| Per-token-y1 K125 latest | 400 | 21.771992 | 22.875583 | 22.567524 | 22.601412 | 21.430915 | 21.446677 |
| Per-token-y1 K4913 best | 220 | 21.771992 | 22.947540 | 22.581303 | 22.578969 | 21.852955 | 21.866673 |
| Per-token-y1 K4913 latest | 400 | 21.771992 | 22.947540 | 22.609836 | 22.605301 | 21.680032 | 21.689440 |

All mature checkpoints exceed the `x1 + 0.5 dB` target under
`teacher_hard`. Their teacher-hard gains are:

- Prefix K125 epoch 40: `+0.645782 dB`
- Prefix K125 epoch 355: `+0.706393 dB`
- Per-token-y1 K125 epoch 400: `+0.795532 dB`
- Per-token-y1 K4913 epoch 220: `+0.809311 dB`
- Per-token-y1 K4913 epoch 400: `+0.837844 dB`

The deployable rollout gains range only from `-0.634306` to `+0.080963 dB`.

## Causal decomposition

### 1. Exposure bias is the dominant current failure

| Model | Teacher CE | Generated-prefix CE | Teacher joint acc | Rollout joint acc |
|---|---:|---:|---:|---:|
| Prefix K125 epoch 40 | 3.9340 | 5.9202 | 8.71% | 3.63% |
| Prefix K125 epoch 355 | 3.5998 | 7.1661 | 11.71% | 3.80% |
| Per-token-y1 K125 epoch 400 | 3.3242 | 6.0473 | 14.83% | 4.73% |
| Per-token-y1 K4913 epoch 220 | 4.7108 | 7.6573 | 7.55% | 4.02% |
| Per-token-y1 K4913 epoch 400 | 4.5692 | 8.1047 | 8.11% | 4.05% |

Training improves performance on oracle-prefix states while generated-prefix CE
gets much worse. Prefix K125 illustrates this most clearly: from epoch 40 to
epoch 355, teacher CE improves from `3.9340` to `3.5998`, but
generated-prefix CE degrades from `5.9202` to `7.1661`, and rollout PSNR drops
from `21.770519` to `21.137686`.

The K4913 epoch-220 position bins show the same transition:

```text
teacher accuracy:  24.31% (tokens 0:16) -> 6.95% (tokens 192:256)
rollout accuracy:  20.50% (tokens 0:16) -> 2.28% (tokens 192:256)
```

The model can make useful local decisions while it remains near the oracle
trajectory, but its own early errors move it to prefix states that were absent
from CE-only teacher-forced training.

### 2. Hard FSQ argmax is not the main bottleneck

For mature checkpoints, `teacher_soft - teacher_hard` is only:

```text
Prefix K125 epoch 40:       +0.038164 dB
Prefix K125 epoch 355:      +0.046440 dB
Per-token-y1 K125 epoch 400:+0.033888 dB
Per-token-y1 K4913 epoch 220:-0.002334 dB
Per-token-y1 K4913 epoch 400:-0.004535 dB
```

Likewise, rollout-soft improves rollout-hard by at most about `0.016 dB`.
Replacing hard decisions with continuous posterior means therefore cannot
close the roughly `0.42 dB` remaining gap to the requested target.

### 3. Per-token y1 fusion helps oracle-prefix modeling but does not solve rollout

At K125, the mature per-token model reaches `14.83%` teacher joint accuracy
and `+0.795532 dB` teacher-hard reconstruction gain. This is stronger than the
Prefix model, so spatial condition injection is useful. Its rollout still
falls to `4.73%` and `-0.341077 dB`, showing that conditioning architecture is
secondary to generated-history robustness at this stage.

### 4. Exact joint-token accuracy understates useful FSQ prediction

Per-token K125 epoch 400 has `14.83%` teacher joint accuracy, while its three
teacher FSQ digit accuracies are `50.50%`, `49.43%`, and `47.85%`. K4913
epoch 220 has `7.55%` joint accuracy and digit accuracies `36.21%`, `36.51%`,
and `34.95%`. A joint-token miss can therefore retain useful scalar-code
information. This explains why teacher-hard reconstruction can exceed the
PSNR target despite low exact joint accuracy.

## Decision

The A1 decision rule selects generated-prefix/on-policy robustness as the next
intervention:

- `teacher_hard >= x1 + 0.5 dB`: true for every mature checkpoint.
- `rollout_hard >= x1 + 0.5 dB`: false for every checkpoint.
- `teacher_soft` materially better than `teacher_hard`: false.

The immediate problem is therefore not a frozen D2/codebook/combiner ceiling,
nor primarily a hard-quantization ceiling. It is the mismatch between
teacher-forced training states and free-running AR states.

No improvement training was started as part of A1.

## Artifacts

- Script: `MY-V2/jscc-f/explore/diagnose_ar_causal_gap.py`
- Full machine-readable result:
  `MY-V2/jscc-f/explore/results-ar/ar_causal_gap_a1.json`
- Initial smoke failure: scalar FSQ state could not be byte-viewed before
  flattening; it failed before data forward and was fixed.
- Successful smoke: two images, all four decode paths, cache replay, and frozen
  hashes. Smoke quality values were not used as evidence.
