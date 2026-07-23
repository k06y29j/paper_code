# Layer2 FSQ capacity-ceiling protocol

This note prevents several different relaxations from being reported as the
same “codebook-capacity upper bound”.  Because the decoder always receives
`x1`, a vocabulary size `K` alone does not define a unique mathematical PSNR
ceiling.  We use the following empirical ladder.

| Level | Meaning | Same-rate K bound? |
|---|---|---|
| `Actual_K` | Encoder-selected hard FSQ code | yes |
| `HardOracleSearch_K` | Per-image STE/projected search over the same hard grid and frozen decoder; reachable search estimate, not a global optimum | yes |
| `ContinuousLatentRelax` | Optimize arbitrary `d`-dimensional latent in `[-1,1]` | no, rounding relaxed |
| `ArbitraryU2Relax` | Optimize arbitrary full-resolution `u2` into the frozen combiner | no, decoder/rate relaxed |
| `Source320ContinuousRef` | Original continuous `320x16x16` Layer2 | no, wide architecture reference |

Reproduction entrypoints:

- `probe_layer2_fsq_capacity_ceiling.py`: the five-level ladder.
- `probe_layer2_fsq_gain_ceiling.py`: post-hoc alpha envelope for old blend checkpoints.
- `train_layer2_continuous_direct.py`: independently trained same-`d` continuous control.
- `train_layer2_fsq_adapter.py`: preserves the exact source E2/D2 width-320 contract and inserts `320->d->FSQ->320`.
- `train_layer2_fsq_bitplane_multirate.py`: fixed 7/10/13-bit comparison that increases binary FSQ dimensions rather than only scalar precision.

## Established references

- Frozen source Swin Layer2: `x1=27.565002`, `final=42.273165`, gain
  `+14.708163 dB`; its latent is `320x16x16`, so it is not the K=125/729/4913
  ceiling.
- Old calibrated blend runs at common epoch 100:

  | K | final | gain over x1 | continuous | quantization gap |
  |---:|---:|---:|---:|---:|
  | 125 | 27.751608 | +0.186606 | 27.769592 | 0.017983 |
  | 729 | 27.746336 | +0.181334 | 27.755888 | 0.009552 |
  | 4913 | 27.730556 | +0.165554 | 27.732454 | 0.001898 |

  Rounding is already a small part of the error; the learned continuous `d=3`
  codec is the main bottleneck.
- Full-validation post-hoc blend envelope:

  | K | learned gain | best global-alpha gain | per-image-alpha gain | alpha=1 gain |
  |---:|---:|---:|---:|---:|
  | 125 | +0.186572 | +0.186527 | +0.189274 | -6.052642 |
  | 729 | +0.195947 | +0.196770 | +0.200535 | -6.573134 |
  | 4913 | +0.192499 | +0.192526 | +0.194961 | -6.193298 |

  Thus a strong refined image is not simply hidden behind alpha; the complete
  blend-trained codec co-adapted to alpha about 0.1.
- PCA3 of the frozen 320-channel teacher latent explains about 39.7% of latent
  variance in the initial probe.  On full validation it produced
  `26.048557 dB` continuously and `25.640390/25.936726/26.011272 dB` for
  L=5/9/17.  This is a linear latent-MSE benchmark, not an image-MSE optimum.
- On 20 stratified validation images with the old separate checkpoints:

  | K | Actual gain | HardOracleSearch gain | ContinuousLatent gain | ArbitraryU2 gain |
  |---:|---:|---:|---:|---:|
  | 125 | +0.183647 | +0.196654 | +0.246912 | +4.894742 |
  | 729 | +0.177516 | +0.215299 | +0.240518 | +4.931150 |
  | 4913 | +0.168510 | +0.223370 | +0.230922 | +4.997931 |

  The hard search estimate grows with K, but the large jump only after removing
  the `d=3 -> D2` constraint localizes the dominant gap to representation and
  decoder training rather than code selection.

## Current target

For a frozen decoder/combiner and validation image `i`, the exact same-K hard
oracle is

`H_K(i) = max[q in C_K^(16x16)] PSNR(f(q, x1_i), image_i)`.

It is combinatorial and is not evaluated exactly.  `HardOracleSearch_K` is
therefore reported as a **reachable empirical envelope** (a lower bound on the
unknown exact optimum), never as a mathematical upper bound.  The locally
optimized continuous-latent and arbitrary-u2 relaxations are also reachable
relaxation references; only their unavailable global optima would be strict
upper bounds.

For every rate on the same images, report:

- `actual_gain = mean(PSNR_actual - PSNR_x1)`;
- `reachable_ceiling_gain = mean(max(PSNR_actual, PSNR_hard_search) - PSNR_x1)`;
- `ceiling_headroom = reachable_ceiling_gain - actual_gain`;
- `ceiling_utilization = actual_gain / reachable_ceiling_gain` only when the
  denominator is positive.  If no positive reachable gain is found, report
  `null`, not 100%.  Preserve a negative signed utilization when the envelope
  is positive but the actual codec is below `x1`.

The search always retains the encoder-selected code, so its per-image result
cannot be worse than `Actual_K`.  Shared nested checkpoints are the primary
capacity evidence: unlike separate K-specific decoders, they isolate the code
set while holding the decoder/combiner fixed.  The complete training target is
not merely `final > x1`: actual and reachable gains must grow with capacity,
zero/shuffle drops must each be at least 0.1 dB, and actual gain should close a
large fraction of the reachable envelope.  Precision-only L=5/9/17 and
dimension-growing 7/10/13-bit sweeps are reported separately.

## Residual-combiner checkpoints

The identity-safe residual combiner initializes `final=x1` exactly and learns
an additive correction without a global blend-alpha ceiling.  At epoch 5:

| Shared bitplane rate | final | gain over x1 | drop zero | drop shuffle |
|---:|---:|---:|---:|---:|
| K=128 (7 bits) | 27.735576 | +0.170574 | 0.225676 | 0.255009 |
| K=1024 (10 bits) | 27.804236 | +0.239234 | 0.294336 | 0.351841 |
| K=8192 (13 bits) | 27.871901 | +0.306899 | 0.362001 | 0.458558 |

The validation mean and every validation image are strictly ordered at this
checkpoint (`all_rates_strict_ratio=1.0`), and all rates exceed `x1` while
passing both effectiveness ablations.  Training remained goal-eligible through
epoch 100.  The best three-rate mean occurred at epoch 45, with final PSNR
`27.777931/27.867153/27.951080` and gains
`+0.212929/+0.302151/+0.386078 dB`.

### Epoch-45 goal-best reachable-envelope probe

The final epoch-45 residual bitplane goal-best checkpoint was probed on validation
indices `0,25,50,75,99` with 50 hard, 50 continuous, and 75 arbitrary-u2
optimization steps.  Results are saved in
`results-capacity-ceiling/bitplane-residual-goalbest-e45-5img-h50-c50-u75.json`.

| K | Actual gain | Hard-search gain | Continuous reachable gain | Arbitrary-u2 reachable gain | Actual / hard |
|---:|---:|---:|---:|---:|---:|
| 128 | +0.198518 | +0.198639 | +0.309641 | +4.598966 | 99.94% |
| 1024 | +0.285057 | +0.285859 | +0.466233 | +4.602802 | 99.72% |
| 8192 | +0.361827 | +0.363237 | +0.604584 | +4.606458 | 99.61% |

Actual and inherited hard-search envelopes are strictly increasing on the
five-image mean and on every image.  Adjacent actual gains are
`+0.086539/+0.076770 dB`; hard-envelope gains are
`+0.087220/+0.077378 dB`.  Hard search improves the encoder by only
`0.000121/0.000802/0.001409 dB`, so code selection is not the current
bottleneck for this frozen decoder.  The increasing continuous gap and the
much larger u2 relaxation instead localize remaining headroom to quantized
latent representation and decoder/combiner capacity.  This remains a
five-image local reachable-envelope estimate and should be expanded to more
images/restarts.  The earlier epoch-15 snapshot is retained as
`bitplane-residual-goalbest-5img-h50-c50-u75.json` for trajectory comparison.
