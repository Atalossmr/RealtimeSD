# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchaudio
import numpy as np


def load_audio(input, ori_fs=None, obj_fs=None):
    if isinstance(input, str):
        wav, fs = torchaudio.load(input)
        wav = wav.mean(dim=0, keepdim=True)
        if obj_fs is not None and fs != obj_fs:
            wav = torchaudio.functional.resample(wav, orig_freq=fs, new_freq=obj_fs)
        return wav
    elif isinstance(input, np.ndarray) or isinstance(input, torch.Tensor):
        wav = torch.from_numpy(input) if isinstance(input, np.ndarray) else input
        if wav.dtype in (torch.int16, torch.int32, torch.int64):
            wav = wav.type(torch.float32)
            wav = wav / 32768
        wav = wav.type(torch.float32)
        assert wav.ndim <= 2
        if wav.ndim == 2:
            if wav.shape[0] > wav.shape[1]:
                wav = torch.transpose(wav, 0, 1)
            wav = wav.mean(dim=0, keepdim=True)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if ori_fs is not None and obj_fs is not None and ori_fs!=obj_fs:
            wav = torchaudio.functional.resample(wav, orig_freq=ori_fs, new_freq=obj_fs)
        return wav
    else:
        return input
