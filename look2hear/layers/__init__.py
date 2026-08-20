# Copyright Look2Hear (https://github.com/JusperLee/Look2Hear). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""layers 包：仅保留 TIGER 推理所需的 activations / normalizations 子模块。

原包在此 eager import 了 cnnlayers/rnnlayers/enc_dec/stft/stft_tfgn，
它们对本项目（TIGER 语音分离推理）无用且拖入 librosa / torch-complex /
typeguard / distutils 等重依赖，已删除。TIGER 通过
`from ..layers import activations, normalizations` 直接引用子模块。
"""
