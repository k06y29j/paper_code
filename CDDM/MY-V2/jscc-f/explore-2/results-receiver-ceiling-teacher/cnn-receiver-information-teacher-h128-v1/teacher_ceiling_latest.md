# Training-only receiver-information restoration teacher

**Not a deployment result.** This finite model estimates how much residual can be regressed from receiver-visible `(z1, x1)`. It does not generate `q2_hat`, does not execute receiver `D2/combiner`, and must not be cited as a receiver deployment gain.

- Epoch: 4
- Full DIV2K validation: `1` (100 images)
- PSNR(x1): `21.771992 dB`
- PSNR(teacher): `21.798611 dB`
- Delta vs x1: `+0.026619 dB`
- No-leak audit: `1`
- Crop contract: train `RandomCrop(256)+RandomHorizontalFlip`; validation `CenterCrop(256)`

Permitted use: use the residual/image prediction only as a training-time teacher or auxiliary/distillation target for a separately audited q2/index generator. The latter must still demonstrate `q2_hat -> D2 -> combiner(x1,u2_hat)` on full validation.
