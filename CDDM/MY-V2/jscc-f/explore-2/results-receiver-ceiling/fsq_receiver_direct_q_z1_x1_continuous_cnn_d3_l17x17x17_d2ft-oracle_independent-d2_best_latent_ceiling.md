# FSQ receiver latent-ceiling probe

- Checkpoint: `MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-independent-d2-v5/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_d2ft-oracle_independent-d2_best.pth`
- Epoch: `35`; version: `cnn-fsq-k4913-independent-d2-v5`
- Device: `cuda`
- DIV2K val: `100/100` images, `25` batches; full validation: `True`
- Reproduce: `/home/yongjia/.conda/envs/cddm_ddnm/bin/python /workspace/yongjia/paper_code/CDDM/MY-V2/jscc-f/explore-2/probe_fsq_receiver_latent_ceiling.py --batch-size 4 --num-workers 0 --device auto`

## Definitions

- `sender_oracle`: true hard FSQ q decoded by the frozen sender D2/combiner.
- `receiver_qhat`: qhat predicted strictly from `(z1, x1)`, decoded by receiver D2/combiner.
- `receiver_true_q`: true hard FSQ q decoded by receiver D2/combiner; this is an empirical fixed-decoder latent ceiling, not a deployable input or mathematical upper bound.
- `receiver_mid_q`: `q_mid = 0.5 * (qhat + q_true)`; this continuous diagnostic is not necessarily a valid hard FSQ token.
- Delta is PSNR minus `x1`; gap in the table is `PSNR(receiver_true_q) - PSNR(path)`.

## Validation-set means

| Path | PSNR (dB) | Delta vs x1 (dB) | Gap to receiver true-q (dB) |
|---|---:|---:|---:|
| `x1` | 21.772013 | 0.000000 | 0.756274 |
| `sender_oracle` | 22.743489 | 0.971476 | -0.215203 |
| `receiver_qhat` | 22.031828 | 0.259815 | 0.496459 |
| `receiver_true_q` | 22.528286 | 0.756274 | 0.000000 |
| `receiver_mid_q` | 22.280679 | 0.508666 | 0.247607 |

## Latent gaps

- Receiver true-q minus qhat: `0.496459 dB`
- Receiver mid-q minus qhat: `0.248851 dB`
- Receiver true-q minus mid-q: `0.247607 dB`
- Sender oracle minus receiver qhat: `0.711661 dB`
- qhat/true-q element MSE: `0.003451719`

The entire evaluation loop uses `torch.inference_mode`; all loaded modules are frozen and in eval mode.
