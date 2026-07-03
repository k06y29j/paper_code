# C36 Tail Coarse-to-Fine Validation

- checkpoint: `/workspace/yongjia/paper_code/CDDM/MY/checkpoints-cvq-v2-v01-c36-snr9-k4096/cvq_v2_v01_c36_snr9_k4096_stage1_best.pth`
- SNR: `9` dB
- tail gain total: `1.9276` dB
- first half gain: `0.8548` dB
- second half gain: `1.0728` dB
- monotonic nonnegative steps: `20/20`
- early vs late HF ratio: `0.135357` vs `0.164637`
- ordered minus random subset mean: `-0.0003` dB
- ordered minus random permuted slots mean: `1.4324` dB

## Outputs
- `tail_prefix_curve.csv`
- `tail_knockout.csv`
- `tail_frequency_curve.csv`
- `tail_random_order.csv`
- `summary.json`
