# Conditional BAR receiver generator

`train_stage-fsq-bar.py` generates the frozen Layer2 C=16 binary-FSQ code from
the receiver-visible Layer1 state.  It is a generator for the checkpointed
codec, not another Layer2 tokenizer trainer.

The token contract is `[B, 16, 16, 16]`: 256 spatial positions, each carrying
16 binary FSQ channels.  The outer Transformer is raster causal over the 256
positions.  At each position, the inner Masked Bit Modeling (MBM) head makes
`16 x 2` binary logits and confidence-unmasks the bits in four rounds
(`4,4,4,4` by default).  It is therefore not bitwise autoregression and never
uses a 65536-way classifier.

At deployment the only inputs are `z1`, `x1`, and the previously generated
spatial bit vectors.  `image`, true `z2`, true `q2`, and true FSQ bits are
training-only supervision.

## Frozen combinations

| Combination | Default best checkpoint |
| --- | --- |
| `cnn-cnn` | `jscc_f_c-16_stage3_fsq_l1-cnn_l2-cnn_c16_binary_direct_group_compatible_best.pth` |
| `cnn-bar` | `jscc_f_c-16_stage3_fsq_l1-cnn_l2-bar_c16_native-e1152l27h16m4304-d1024l24h16m4096_binary_direct_group_fresh_best.pth` |
| `swin-swin` | `jscc_f_c-16_stage3_fsq_l1-swin_l2-swin_c16_binary_direct_group_compatible_best.pth` |

The defaults resolve these paths under `MY-V2/jscc-f/checkpoints-fsq-c16`.
An explicit `--checkpoint` is accepted only after strict architecture and
binary-FSQ C=16 contract checks.

## Smoke check

Use a small generator for the functional check; it still restores the actual
frozen tokenizer checkpoint.

```bash
conda run -n cddm_ddnm python MY-V2/jscc-f/train_stage-fsq-bar.py \
  --layer1-arch cnn --layer2-arch cnn --cpu --smoke-shapes \
  --hidden 64 --layers 2 --heads 4 --bit-embedding 8 --condition-blocks 1 \
  --mbm-hidden 32 --mbm-layers 2 --mbm-heads 4 --dropout 0
```

The smoke test verifies strict E1/D1/E2/D2/FSQ/combiner loading, full 256-step
cached raster rollout, `[B*256,16,2]` teacher logits, exact binary
bits-to-FSQ-to-decoder round trip, and equivalence of cached rollout conditions
with the teacher-forced causal pass.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n cddm_ddnm python MY-V2/jscc-f/train_stage-fsq-bar.py \
  --layer1-arch cnn --layer2-arch cnn --version cnn-cnn-bar-ar \
  --batch-size 12 --test-batch 1 --num-workers 8 --val-num-workers 4
```

Replace the two architecture flags with `cnn bar` or `swin swin` for the other
saved combinations.  The native `cnn-bar` tokenizer checkpoint itself is about
2.8 GiB, so use a GPU with enough room for the frozen BAR codec plus the
generator and start with `--batch-size 1` if necessary.
