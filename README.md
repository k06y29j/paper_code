训练语义编解码器
#checkpoint 2 : 4 stage, use vae  72 16 16
CUDA_VISIBLE_DEVICES=3 python train/train_sc.py \
  --dataset div2k \
  --grad_accum_steps 4 \
  --compile off \
  --batch_size 32 \
  --save_dir checkpoints-v2 \
  --eval_every_batches 200 \
  --log_file log/sc-v2.txt

#checkpoint 1 : 3 stage, no vae 16 32 32
CUDA_VISIBLE_DEVICES=4 python train/train_sc.py \
  --dataset div2k \
  --grad_accum_steps 4 \
  --compile off \
  --batch_size 32 \
  --save_dir checkpoints-v1 \
  --eval_every_batches 200 \
  --log_file log/sc-v1.txt