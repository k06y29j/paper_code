# C36 Tail Coarse-to-Fine Validation

- checkpoint: `/workspace/yongjia/paper_code/CDDM/MY/checkpoints-cvq-fsqshared16-tailfocus-e2e-vq1e3/cvq_c36_snr9_stage3_best.pth`
- SNR: `9` dB
- tail gain total: `2.4599` dB
- first half gain: `1.0177` dB
- second half gain: `1.4422` dB
- monotonic nonnegative steps: `19/20`
- early vs late HF ratio: `0.122078` vs `0.177458`
- ordered minus random subset mean: `2.1029` dB
- ordered minus random permuted slots mean: `4.4358` dB

## Outputs
- `tail_prefix_curve.csv`
- `tail_knockout.csv`
- `tail_frequency_curve.csv`
- `tail_random_order.csv`
- `summary.json`
