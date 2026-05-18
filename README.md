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


dm
语义编码器输出空间的分布：
MEAN="0.261506,-0.480177,0.608077,0.271048,0.441773,0.542745,-0.056660,-0.350853,-0.060972,0.432691,0.648729,0.443262,0.218551,-0.118036,0.498166,-0.158471"

STD="0.611215,0.651810,0.672925,0.472801,0.503495,0.713110,0.827775,0.781240,0.623319,0.895998,0.639930,0.892033,0.678167,0.576835,0.613657,1.016683"



CUDA_VISIBLE_DEVICES=0 conda run -n cddm_ddnm python train/train_unet_un.py \
  --dataset div2k \
  --sc_encoder_ckpt checkpoints-val/sc/sc_encoder_div2k_c16.pth \
  --sc_decoder_ckpt checkpoints-val/sc/sc_decoder_div2k_c16.pth \
  --latent_norm channel \
  --latent_channel_mean "$MEAN" \
  --latent_channel_std "$STD" \
  --min_snr_gamma 0 \
  --epochs 600 \
  --batch_size 32 \
  --eval_every_epochs 5 \
  --eval_eps_t_list 0,100,500,999 \
  --eval_z0_t_list 100,500 \
  --eval_recon_t_starts 100,500 \
  --log_file log/unet_un/A1_channel_vanilla.txt \
  --save_dir checkpoints-val/unet_un/A1_channel_vanilla \
  > /dev/null 2>&1 &



CUDA_VISIBLE_DEVICES=1 conda run -n cddm_ddnm python train/train_unet_un.py \
  --dataset div2k \
  --sc_encoder_ckpt checkpoints-val/sc/sc_encoder_div2k_c16.pth \
  --sc_decoder_ckpt checkpoints-val/sc/sc_decoder_div2k_c16.pth \
  --latent_norm channel \
  --latent_channel_mean "$MEAN" \
  --latent_channel_std "$STD" \
  --min_snr_gamma 5 \
  --epochs 600 \
  --batch_size 32 \
  --eval_every_epochs 5 \
  --eval_eps_t_list 0,100,500,999 \
  --eval_z0_t_list 100,500 \
  --eval_recon_t_starts 100,500 \
  --log_file log/unet_un/A2_channel_minsnr5.txt \
  --save_dir checkpoints-val/unet_un/A2_channel_minsnr5 \
  > /dev/null 2>&1 &



CUDA_VISIBLE_DEVICES=2 conda run -n cddm_ddnm python train/train_unet_un.py \
  --dataset div2k \
  --sc_encoder_ckpt checkpoints-val/sc/sc_encoder_div2k_c16.pth \
  --sc_decoder_ckpt checkpoints-val/sc/sc_decoder_div2k_c16.pth \
  --latent_norm channel \
  --latent_channel_mean "$MEAN" \
  --latent_channel_std "$STD" \
  --noise_schedule cosine \
  --min_snr_gamma 5 \
  --epochs 600 \
  --batch_size 32 \
  --eval_every_epochs 5 \
  --eval_eps_t_list 0,100,500,999 \
  --eval_z0_t_list 100,500 \
  --eval_recon_t_starts 100,500 \
  --log_file log/unet_un/A2_channel_minsnr5_cosine.txt \
  --save_dir checkpoints-val/unet_un/A2_channel_minsnr5_cosine \
  > /dev/null 2>&1 &

  