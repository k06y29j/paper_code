# C36 Tail Coarse-to-Fine Validation

- checkpoint: `/workspace/yongjia/paper_code/CDDM/MY/checkpoints-cvq/cvq_c36_snr9_stage3_best.pth`
- SNR: `9` dB
- tail gain total: `-0.0006` dB
- first half gain: `0.0027` dB
- second half gain: `-0.0033` dB
- monotonic nonnegative steps: `12/20`
- early vs late HF ratio: `0.021208` vs `0.069372`
- ordered minus random subset mean: `-0.0029` dB
- ordered minus random permuted slots mean: `0.0322` dB

## Outputs
- `tail_prefix_curve.csv`
- `tail_knockout.csv`
- `tail_frequency_curve.csv`
- `tail_random_order.csv`
- `summary.json`
