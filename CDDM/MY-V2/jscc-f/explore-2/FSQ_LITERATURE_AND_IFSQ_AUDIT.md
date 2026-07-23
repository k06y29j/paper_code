# K=125 FSQ generation: literature and iFSQ contract audit

Date: 2026-07-13

Scope is deliberately limited to the four established K=125 controls/routes:

1. direct hard prediction;
2. direct continuous `q2_hat` prediction;
3. joint-token autoregressive hard generation;
4. continuous diffusion/flow generation, with hard snapping reported only as an ablation.

Receiver inference is always `z1/x1 -> q2_hat -> D2 -> combiner`; `img`, `z2`,
sender `q2`, and oracle indices are forbidden. Training uses random 256 crops and
horizontal flip; validation uses center 256 crops and the complete DIV2K valid100.

## Verified primary sources

- [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505),
  ICLR 2024. It defines the implicit Cartesian codebook and uses hard FSQ tokens
  with MaskGIT and UViM. Its image experiment uses joint mixed-radix tokens; it
  does not establish a K=125 PSNR result.
- [Official FSQ repository](https://github.com/google-research/google-research/tree/master/fsq).
- [iFSQ: Improving FSQ for Image Generation with 1 Line of Code](https://arxiv.org/abs/2601.17124),
  arXiv v2, 2026-01-27.
- [Tencent-Hunyuan/iFSQ](https://github.com/Tencent-Hunyuan/iFSQ/tree/e78bd7089076a49c5fa76f8feebed28379669e79),
  audited at commit `e78bd7089076a49c5fa76f8feebed28379669e79`.
- [Optimality of FSQ Tokens for Continuous Diffusion for Categorical Data](https://arxiv.org/abs/2606.09962),
  arXiv v1, 2026-06-08. Its categorical-posterior diffusion is directly relevant
  to a fixed FSQ grid, but the theorem covers base-2/base-3 grids and the real
  experiment is TTS, not K=125 image residual reconstruction.
- [FlexTok](https://arxiv.org/abs/2502.13967) and
  [official code](https://github.com/apple/ml-flextok): additional evidence for
  autoregressive joint FSQ indices, but at a much larger vocabulary.
- [TerraMind](https://arxiv.org/abs/2504.11171) and
  [official code](https://github.com/IBM/terramind): conditional hard-token
  prediction evidence; its diffusion decoder is after token prediction and is
  not diffusion over `q2`.
- [BAR](https://arxiv.org/abs/2602.09024) and
  [official code](https://github.com/amazon-far/BAR): excluded from K=125 because
  the released quantizer is binary and assumes `K=2^d`.

None of these papers proves a receiver-only `+0.5 dB` PSNR gain. They support
method contracts only; the threshold must be established on the local strict
validation protocol.

## What iFSQ actually changes

iFSQ replaces the tokenizer bounding activation

```text
tanh(z) = 2*sigmoid(2*z)-1
```

with

```text
2*sigmoid(1.6*z)-1
```

before scalar rounding. This is a sender-tokenizer training change, not a loss
or output activation for an already trained receiver predictor. The released
tokenizer config is `d=4`, `levels=[17,17,17,17]`, and therefore is not an
officially validated K=125 recipe. The official repository nevertheless supports
`d=3`, `levels=[5,5,5]` mechanically.

The current local K=125 oracle is not iFSQ:

- `MY-V2/jscc-f/train-stage3-fsq.py:236` applies learnable `GroupNorm`;
- `MY-V2/jscc-f/train-stage3-fsq.py:251` applies `torch.tanh`;
- its grid is the correct five-level Cartesian grid
  `{-1,-0.5,0,0.5,1}^3`.

Changing only the activation in this frozen checkpoint would change its bins and
break the trained E2/D2/combiner contract. A genuine iFSQ comparison therefore
requires a separate opt-in sender tokenizer and retraining; it cannot be claimed
by post-processing the existing `q2`.

## The present oracle is not codebook-collapsed

The K=125 oracle at epoch 100 reports:

| metric | train | valid100 | maximum |
|---|---:|---:|---:|
| used joint codes | 125 | 125 | 125 |
| joint entropy (bit) | 6.490897 | 6.538245 | 6.965784 |
| joint perplexity | 89.9404 | 92.9411 | 125 |
| mean scalar entropy (bit) | 2.186696 | 2.194806 | 2.321928 |
| scalar level usage | 100% | 100% | 100% |
| tanh saturation fraction | 0.02942 | 0.02987 | -- |

Its strict oracle gain is `+1.104487 dB`. Thus the current receiver bottleneck
is not missing FSQ code usage or insufficient oracle headroom. iFSQ may improve
the marginal code distribution, but that does not imply a reduction in the
conditional uncertainty `H(q2 | z1,x1)` that controls receiver prediction.

## Paper-to-local route mapping

| Route | Direct literature support | Paper-faithful target retained locally | Unsupported additions removed |
|---|---|---|---|
| AR hard | strong | one K=125 token per 16x16 site; raster teacher-forcing CE; receiver-only free rollout | scheduled sampling, q/image/oracle auxiliary losses |
| Flow continuous | strong | Gaussian source; linear velocity; logit-normal time; velocity MSE plus cosine; Euler ODE | zero source, Direct-Q transport anchor, endpoint reconstruction loss |
| Direct hard | control only | exact five-level snap before frozen D2 | no paper claim attached |
| Direct continuous | control only | hard-FSQ `q2` supervision but continuous receiver output | no paper claim attached |

The official iFSQ AR code uses standard teacher forcing and a LlamaGen-Large
Transformer. The local route uses a much smaller conditional masked-convolution
model, so it is a method-level K=125 adaptation rather than an architectural
reproduction. The official continuous release is rectified flow/transport, not
a DDPM: Gaussian start, linear velocity, logit-normal time sampling, cosine
velocity loss, and Euler sampling with default NFE 250.

## Current evidence and experiment decision

Completed strict valid100 controls:

| route | gain over x1 | index accuracy |
|---|---:|---:|
| direct continuous, frozen D2/combiner | `+0.220844 dB` | `0.318099` |
| direct hard, frozen D2/combiner | `+0.254072 dB` | `0.325898` |

Previous scheduled-sampling AR and zero-start flow were stopped because their
public receiver-only validation trends became negative. They are not evidence
against the paper contracts because their training objectives/sources differed
from the released methods.

Two isolated method controls were run and stopped after the pre-declared epoch-15
trend check:

- `run_fsq_ar_k125_ifsq_contract.sh`: joint-token CE only, hard inference,
  frozen D2/combiner;
- `run_fsq_flow_k125_ifsq_contract.sh`: Gaussian linear flow, logit-normal
  time, MSE+cosine velocity loss, continuous inference, frozen D2/combiner.

Their strict valid100 results were:

| route | epoch 1 | epoch 5 | epoch 10 | epoch 15 | decision |
|---|---:|---:|---:|---:|---|
| joint-token AR hard | `-0.376383` | `-0.049212` | `-0.014877` | `-0.067467` dB | stopped; rollout gain and accuracy did not continue improving |
| iFSQ-style continuous flow | `-0.814326` | `-0.268079` | `-0.248606` | `-0.383844` dB | stopped; deployment trend reversed while velocity loss fell |

At epoch 15, AR teacher-forced training accuracy was `0.096758`, whereas the
full receiver-only rollout validation accuracy was `0.036289`. These are not
the same estimator: the former receives true previous tokens and the latter
must recursively consume its own predictions. The full validation
condition-shuffle drop was only `0.024424 dB`, so the deployed AR model had not
learned a sufficiently useful dependence on `z1/x1`. The flow validation index
accuracy was `0.021211`, and its continuous output remained below `x1`.

The corresponding logs are:

- `logs-fsq-generation/cnn-fsq-k125-paper-joint-ar-ce-frozen-v4.log`;
- `logs-fsq-generation/cnn-fsq-k125-paper-flow-cont-gaussian-logitcos-v4.log`.

Because the continuous-flow control remained clearly below the direct controls,
the only next diffusion change retained from the literature audit is the categorical
posterior parameterization from arXiv:2606.09962: predict 125 posterior logits,
map their probabilities through the exact frozen `[125,3]` FSQ grid, and use the
posterior mean for continuous output or terminal argmax/nearest grid for hard
output. This stays inside the existing diffusion-hard/continuous routes; it is
not a fifth engineering branch. It will not be described as an exact paper
reproduction because no complete official schedule/code is currently released
and the theorem does not cover the five-level grid.

## Categorical-posterior control: implementation and result

The opt-in `categorical_diffusion` route now implements that fallback without
changing any existing route defaults:

- training uses `t ~ U[0,1]`, `q_t=alpha_t*q2+sigma_t*epsilon`, one 125-way
  posterior per spatial site, and CE against the exact mixed-radix FSQ index;
- continuous deployment returns the posterior mean on the fixed
  `{-1,-0.5,0,0.5,1}^3` grid; `--hard-fsq` returns terminal posterior argmax;
- public inference performs a complete deterministic DDIM rollout from Gaussian
  noise using only `z1/x1`;
- training single-noise metrics are named `train_teacher_*`, exact public
  rollout diagnostics are `train_deploy_*`, and validation remains unprefixed.

The paper specifies the general `alpha_t/sigma_t` corruption family and DDIM
sampling, but not the concrete schedule. The local route therefore records
`cosine_vp`, with `alpha_t=cos(pi*t/2)` and `sigma_t=sin(pi*t/2)`, as an explicit
assumption. Standard-normal initialization is the default; the paper's empirical
TTS-only `2/3` prior scale remains configurable and is not silently adopted.

The formal continuous and hard controls used the same CE-only training
contract, `NFE=12`, and `N(0,I)` initialization. Both were stopped after their
full receiver-only rollout failed the epoch-15 trend gate:

| deployment output | epoch 1 | epoch 5 | epoch 10 | epoch 15 | best |
|---|---:|---:|---:|---:|---:|
| posterior mean continuous | `-0.035550` | `-0.065157` | `-0.274037` | `-0.147483` dB | epoch 1 |
| terminal argmax hard grid | `-0.057079` | `-0.076720` | `-0.277283` | `-0.132556` dB | epoch 1 |

At epoch 15 the teacher-sampled training index accuracy had increased to about
`17.65%`, while the full DDIM validation accuracy was only `3.50%`. This is a
deployment-closure failure: the denoiser learns the supervised noisy posterior,
but the reverse chain from pure Gaussian noise does not produce a useful
sample-specific `q2_hat` from `z1/x1`.

The paper-reported TTS sampling choice was also checked without retraining. On
the epoch-1 best checkpoints, `prior_scale=2/3` and `NFE=25` gave
`-0.039005 dB` continuous and `-0.066323 dB` hard, both slightly below the
default `N(0,I), NFE=12` results. Thus neither more paper-reported DDIM steps nor
the smaller empirical prior closes the local gap.

The formal launcher is `run_fsq_cdcd_k125_paper.sh`; set `MODE=continuous` or
`MODE=hard`. Logs are:

- `logs-fsq-generation/cnn-fsq-k125-paper-cdcd-cont-cosinevp-n12-v1.log`;
- `logs-fsq-generation/cnn-fsq-k125-paper-cdcd-hard-cosinevp-n12-v1.log`;
- `logs-fsq-generation/eval-cnn-fsq-k125-paper-cdcd-cont-prior067-n25-v1.log`;
- `logs-fsq-generation/eval-cnn-fsq-k125-paper-cdcd-hard-prior067-n25-v1.log`.

No K=125 AR/diffusion route in this audit reaches the required `+0.5 dB`.
The current strict codebook-aware receiver control remains direct hard
prediction at `+0.254072 dB`; the oracle headroom remains `+1.103991 dB` on
this exact validation pass.
