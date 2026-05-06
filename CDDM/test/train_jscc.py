"""

仅训练 JSCC 编码器 + 解码器：图像经 encoder → power 归一化与实/虚拼接到 decoder，不加信道噪声。

超参由 pipeline 中的 config 提供。训练阶段不做验证；按 save_model_freq 保存模型；同时在同目录下额外保存 encoder_epoch{N}.pt / decoder_epoch{N}.pt（可用 config.save_checkpoint_history=False 关闭）。

"""

import os

import sys



import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from tqdm import tqdm



_CDDM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _CDDM_ROOT not in sys.path:

    sys.path.insert(0, _CDDM_ROOT)



from Autoencoder.data.datasets import get_loader

from Autoencoder.net.channel import Channel

from Autoencoder.net.network import JSCC_encoder, JSCC_decoder

from Autoencoder.utils import save_model, seed_torch



def _ckpt_with_epoch(orig_path: str, epoch_done: int) -> str:

    directory, basename = os.path.split(orig_path)

    stem, ext = os.path.splitext(basename)

    name = "{}_epoch{:d}{}".format(stem, epoch_done, ext)

    return os.path.join(directory, name) if directory else name





def _save_jscc_pair(encoder, decoder, encoder_path, decoder_path, epoch_done, save_history):

    enc_dir = os.path.dirname(encoder_path)

    if enc_dir:

        os.makedirs(enc_dir, exist_ok=True)

    save_model(encoder, save_path=encoder_path)

    save_model(decoder, save_path=decoder_path)

    if save_history:

        save_model(encoder, save_path=_ckpt_with_epoch(encoder_path, epoch_done))

        save_model(decoder, save_path=_ckpt_with_epoch(decoder_path, epoch_done))

        print(
            "  [历史] "
            + _ckpt_with_epoch(encoder_path, epoch_done)
            + " | "
            + _ckpt_with_epoch(decoder_path, epoch_done)

        )




def _encoder_to_decoder_input(channel, feature):

    """与有噪训练相同的张量布局，但不经过加噪（复数域直连）。"""

    channel_tx, pwr = channel.complex_normalize(feature, power=1)

    L = channel_tx.shape[2]

    z = channel_tx[:, :, :L // 2, :] + channel_tx[:, :, L // 2 :, :] * 1j

    return torch.cat((torch.real(z), torch.imag(z)), dim=2) * torch.sqrt(pwr)





def train_jscc_seqeratly(config):

    torch.backends.cudnn.benchmark = True

    dev = config.device

    if dev.type == "cuda":

        torch.backends.cuda.matmul.allow_tf32 = True

        torch.backends.cudnn.allow_tf32 = True



    use_amp = bool(getattr(config, "use_amp", True)) and dev.type == "cuda"

    nb = bool(getattr(config, "pin_memory", True))



    encoder = JSCC_encoder(config, config.C).to(dev)

    decoder = JSCC_decoder(config, config.C).to(dev)

    channel = Channel(config)



    train_loader, _ = get_loader(config)

    cur_lr = config.learning_rate

    optimizer_encoder = optim.Adam(encoder.parameters(), lr=cur_lr)

    optimizer_decoder = optim.Adam(decoder.parameters(), lr=cur_lr)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    encoder.train()

    decoder.train()



    seed_torch(config.seed if hasattr(config, "seed") else 1029)

    torch.backends.cudnn.benchmark = True

    if dev.type == "cuda":

        torch.backends.cudnn.deterministic = False



    mse_loss = nn.MSELoss()

    save_ckpt_hist = bool(getattr(config, "save_checkpoint_history", True))

    for e in range(config.epoch):

        with tqdm(train_loader, dynamic_ncols=True) as tqdm_train:

            for input_img, _ in tqdm_train:

                input_img = input_img.to(dev, non_blocking=nb)

                optimizer_encoder.zero_grad(set_to_none=True)

                optimizer_decoder.zero_grad(set_to_none=True)

                with torch.autocast(

                    device_type="cuda" if dev.type == "cuda" else "cpu",

                    dtype=torch.float16,

                    enabled=use_amp and dev.type == "cuda",

                ):

                    feature, _ = encoder(input_img)

                    dec_in = _encoder_to_decoder_input(channel, feature)

                    recon = decoder(dec_in)

                    mse_255 = mse_loss(

                        input_img * 255.0, recon.clamp(0.0, 1.0) * 255.0

                    )

                    rec_loss = F.mse_loss(

                        recon.clamp(0.0, 1.0), input_img, reduction="sum"

                    ) / input_img.shape[0]

                    psnr = (

                        10.0 * (torch.log(255.0 * 255.0 / mse_255) / np.log(10.0))

                    ).item()

                    cbr = feature.numel() / 2.0 / input_img.numel()

                if use_amp:

                    scaler.scale(rec_loss).backward()

                    scaler.step(optimizer_encoder)

                    scaler.step(optimizer_decoder)

                    scaler.update()

                else:

                    rec_loss.backward()

                    optimizer_encoder.step()

                    optimizer_decoder.step()



                tqdm_train.set_postfix(

                    ordered_dict={

                        "epoch": e,

                        "train_psnr": f"{psnr:.2f}",

                        "cbr": f"{cbr:.4f}",

                    }

                )



        if (e + 1) % config.save_model_freq == 0:

            ep_done = e + 1

            print("已保存 checkpoint (epoch {}/{})".format(ep_done, config.epoch))

            _save_jscc_pair(

                encoder,

                decoder,

                config.encoder_path,

                config.decoder_path,

                ep_done,

                save_ckpt_hist,

            )



    if (config.epoch % config.save_model_freq) != 0:

        print("已保存最终 checkpoint (epoch {})".format(config.epoch))

        _save_jscc_pair(

            encoder,

            decoder,

            config.encoder_path,

            config.decoder_path,

            config.epoch,

            save_ckpt_hist,

        )

    print("训练结束")


