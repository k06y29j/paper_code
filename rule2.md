1,PSNR计算方式： (各图之和/张数 ΣPSNR_i/N) ,逐图 PSNR 再平均。
2.环境是cddm_ddnm的conda环境
3.nvdia-smi查看可用GPU
4.如果在SNR 12db下，需要在12db下训练，在12db下测试
5.优化GPU利用率，应设置预加载和num-workers，以及cache_decoded，避免反复处理
6.保证swin编码器输出的分布和信道解码器输出的分布一致，这已经通过AA^T = I_4保证
7.同时可以处理信道噪声和信道编码器带来的维度退化噪声
8.在输入[3 256 256],信道编码器输出[4 16 16],超过基线22.419 db（AWGN 12db），超过0.5db
9.模型和日志保存到checkpoints-codex
10.信道编码器后需要功率归一化
11.训练图片不能固定裁剪成cache_decoded,而是完整图片保存成cache_decoded，每次epoch再随机裁剪
在这个基础上，联合训练SWIN编解码器和信道编解码器