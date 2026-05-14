训练语义编解码器
#checkpoint 1 : 3 stage, no vae 16 32 32
CUDA_VISIBLE_DEVICES=3 python train/train_sc.py \
  --dataset div2k \
  --grad_accum_steps 4 \
  --compile off \
  --batch_size 32 \
  --log_file log/sc-c36-novae.txt \
  --embed_dim 36

#CDDM 18 16 16
CUDA_VISIBLE_DEVICES=3 python CDDM/test/pipeline.py

CUDA_VISIBLE_DEVICES=1 python test/eval_dm_mse.py \
  --dataset div2k \
  --sc_encoder_ckpt checkpoints-val/sc/sc_encoder_div2k_c16.pth \
  --sc_decoder_ckpt checkpoints-val/sc/sc_decoder_div2k_c16.pth \
  --unet_ckpt checkpoints-val/unet_un/unet_un_div2k_c16_best.pth \
  --batch_size 8 --max_batches 20 --recon_steps 50


  CUDA_VISIBLE_DEVICES=1 python train/train_cc.py --mode random_random --out_channels 12
  CUDA_VISIBLE_DEVICES=1 python train/train_cc.py --mode random_random --out_channels 4

CUDA_VISIBLE_DEVICES=1 python train/train_cc.py --mode random_pinv --out_channels 12
CUDA_VISIBLE_DEVICES=1 python train/train_cc.py --mode random_pinv --out_channels 4

CUDA_VISIBLE_DEVICES=3 nohup python train/train_cc.py \
    --mode trained --out_channels 12 \
    --batch_size 16 --epochs 200 --lr 1e-3 \
    --log_file log/cc/trained_c16to12.txt \
    > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python train/train_cc.py \
    --mode trained --out_channels 4 \
    --batch_size 16 --epochs 200 --lr 1e-3 \
    --log_file log/cc/trained_c16to4.txt \
    > /dev/null 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python train/train_cc_aware.py \
  --out_channels 12 --linear_depth 2 --linear_hidden_channels 16 \
  --train_snr_mode fixed --train_snr_db 12 --train_fading awgn \
  --eval_snr_grid 0 3 6 9 12 15 20 25 \
  --batch_size 16 --epochs 200 --lr 1e-3 \
  --log_file log/cc/aware_awgn_snr12_c12_L2.txt \
  > log/cc/aware_awgn_snr12_c12_L2.stdout 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train/train_cc_aware.py \
  --out_channels 12 --linear_depth 3 --linear_hidden_channels 16 \
  --train_snr_mode fixed --train_snr_db 12 --train_fading awgn \
  --eval_snr_grid 0 3 6 9 12 15 20 25 \
  --batch_size 16 --epochs 200 --lr 1e-3 \
  --log_file log/cc/aware_awgn_snr12_c12_L3.txt \
  > log/cc/aware_awgn_snr12_c12_L3.stdout 2>&1 &



  