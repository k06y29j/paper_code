# C36 Tail Coarse-to-Fine Validation

- checkpoint: `/workspace/yongjia/paper_code/CDDM/MY/checkpoints-cvq/cvq_c36_snr9_stage1_best.pth`
- SNR: `9` dB
- tail gain total: `2.6953` dB
- first half gain: `1.2427` dB
- second half gain: `1.4526` dB
- monotonic nonnegative steps: `20/20`
- early vs late HF ratio: `0.130867` vs `0.182767`
- ordered minus random subset mean: `3.1009` dB
- ordered minus random permuted slots mean: `5.3596` dB

## Outputs
- `tail_prefix_curve.csv`
- `tail_knockout.csv`
- `tail_frequency_curve.csv`
- `tail_random_order.csv`
- `summary.json`
