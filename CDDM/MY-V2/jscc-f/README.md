# TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256

This directory implements the two-layer DeepJSCC-f style experiment:

- Layer 1: `z1 = E1(x)`, `x1 = D1(z1)`.
- Layer 2: `z2 = E2(concat(x, x1))`, `u2 = D2(concat(z1, z2))`, `x2_hat = Combiner(x1, u2)`.
- No channel module, no AWGN, and no power normalization are applied.

Typical commands:

```bash
conda run -n cddm_ddnm python MY-V2/jscc-f/train_stage1.py
conda run -n cddm_ddnm python MY-V2/jscc-f/train_stage2.py --variant combiner
conda run -n cddm_ddnm python MY-V2/jscc-f/train_stage2.py --variant no_combiner
conda run -n cddm_ddnm python MY-V2/jscc-f/train_stage2.py --variant residual_input
conda run -n cddm_ddnm python MY-V2/jscc-f/eval.py --layer2-ckpt MY-V2/jscc-f/checkpoints/jscc_f_no-c1_layer2_combiner_best.pth
```

Metrics include `MSE/PSNR/SSIM` for `x1`, `u2`, and the final output, plus `delta_psnr`.
