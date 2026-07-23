# Fixed train-selected q-mix receiver smoke evidence

Date: 2026-07-13 (Asia/Shanghai)

These are loader/contract checks only.  They are explicitly **not** full
DIV2K validation results and must not be used for model selection or claimed
as a `delta_rx` result.

## Structural CPU smoke

```bash
conda run --no-capture-output -n cddm_ddnm \
  python MY-V2/jscc-f/explore-2/train_continuous_q_mix_receiver.py --smoke-contract
```

Passed: fixed global q mix shape `[2,32,16,16]`; q-only D2;
`combiner(x1,u2)`; no source-target retention; frozen member gradients absent;
D2 and combiner gradients present; RandomCrop-train / CenterCrop-val contract.

## Real DIV2K one-batch closure

The fixed source was
`results-receiver/continuous_q_d512_crossmodel5_train_simplex_top3.json`
(original member indices `[0,2,4]`, normalized weights
`[0.63209844,0.33582941,0.03207218]`).  It was run with one RandomCrop train
batch and one CenterCrop validation batch on CPU, `num_workers=0`, and
`max_train_batches=max_val_batches=1`.

| single receiver initialization | train delta, 1 image | center-val delta, 1 image | `full_validation` |
|---|---:|---:|---:|
| spec v9 reference D2/residual combiner | +0.244961 dB | +0.352579 dB | 0 |
| v12 D2/hybrid combiner (launcher default) | +2.334747 dB | +0.282099 dB | 0 |

Both runs printed `receiver_only_audit=1` and `fixed_q_mix_audit=1`; partial
validation intentionally skipped full-dataset ablations.  The temporary
checkpoint weights generated for these two smoke runs were removed afterward.
