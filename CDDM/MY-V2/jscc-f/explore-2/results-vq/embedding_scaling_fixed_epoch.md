# Fixed-epoch image-VQ embedding scaling

This table compares only runs with the same Layer1/Layer2 architecture, data,
seed (`20260713`), C (`256`), K prefixes, optimizer schedule, and validation
epoch (`25`).  Therefore D is the intended changed representation variable.

`PSNR(x1)=21.771992 dB` on all 100 DIV2K validation images.

| D | K256 gain | K1024 gain | K4096 gain | K256 PPL/K | K1024 PPL/K | K4096 PPL/K |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | +1.044606 | +1.177847 | +1.211962 | 0.138605 | 0.090577 | 0.030370 |
| 1024 | +1.126254 | +1.279032 | +1.283548 | 0.183466 | 0.152008 | 0.049037 |
| D1024 - D512 | +0.081647 | +0.101185 | +0.071585 | +0.044861 | +0.061431 | +0.018667 |

All six operating points have top1 below `0.071`.  Their zero/shuffle drops
are above `1.21/1.35 dB`, respectively, so the PSNR improvement is not an
inactive-codebook artifact.  Within each D, PSNR is strictly increasing with
K; at every fixed K, PSNR is strictly increasing from D512 to D1024.

Evidence logs:

- `logs-vq/cnn-image-vq-c256-d512-oracle-v1.bootstrap.log`, validation epoch 25.
- `logs-vq/cnn-image-vq-c256-d1024-oracle-v1.bootstrap.log`, validation epoch 25.

The independently selected D1024 epoch-60 checkpoint further reaches gains
`+1.383345/+1.547117/+1.650135 dB` for K256/K1024/K4096, with
`oracle_goal_met=1`, K4096 zero drop `4.345397 dB`, and K4096 shuffle drop
`2.650154 dB`.  This later best result is reported separately from the fair
fixed-epoch D comparison above.

## Fixed-epoch grouped channel-VQ embedding scaling

The following comparison holds Layer1/Layer2 CNN, enhanced combiner (width 32,
3 blocks), seed `20260713`, C=256, grouped K prefixes, batch size, data, and
validation epoch (`20`) fixed.  A channel token has shape `[1,D]`; D is not
fixed to the native 16x16 map size.

| D | K512 gain | K1024 gain | K4096 gain | K512 PPL/K | K1024 PPL/K | K4096 PPL/K |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | +0.402444 | +0.508769 | +0.507589 | 0.569036 | 0.366025 | 0.133753 |
| 512 | +0.575905 | +0.727043 | +0.748594 | 0.649909 | 0.442384 | 0.183555 |
| D512 - D256 | +0.173461 | +0.218274 | +0.241005 | +0.080873 | +0.076359 | +0.049802 |

Thus every fixed K improves when D grows from 256 to 512.  D512 is also
strictly increasing with K.  The D256 K1024-to-K4096 change is `-0.001180 dB`,
so that lower-dimensional run is retained as a capacity-limit comparison, not
promoted as the final monotonic operating point.

At D512, K512/K1024/K4096 local perplexities are
`1.9969/3.9300/14.2713`, local top1 values are
`0.5280/0.3063/0.1343`, and shuffle drops are
`0.4105/0.6152/0.6028 dB`.  This rules out a dormant or single-code solution.

Evidence logs:

- `logs-vq/cnn-channel-vq-grouped-c256-d256-enhanced-compare-v1.bootstrap.log`, validation epoch 20.
- `logs-vq/cnn-channel-vq-grouped-c256-d512-enhanced-oracle-v1.bootstrap.log`, validation epoch 20.
