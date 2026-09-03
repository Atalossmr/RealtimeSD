"""Fun-ASR-Nano 模型封装：惰性加载 + 段级转写。

prev_text 语义（funasr/models/fun_asr_nano/model.py + 实测确认）：
prev_text 是"当前输入音频前缀的转写"，拼进 prompt 前缀；模型把 prev 尾部与
音频前缀对齐，跳过已覆盖部分只续写新内容，返回值 = prev_text + 新文本（前缀
逐字保留，经统一的空白归一化）。因此：

- 跨段上下文不可用：prev 与本段音频对不上时，模型判"已转写完"直接 EOS；
- 长段窗切时 prev = 本段已累计文本的尾部，窗口重叠区保证对齐；
- 本封装返回值一律剥离 prev_text 前缀，只给新文本。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from common.models import resolve_modelscope_snapshot
from common.utils.device import resolve_device_name

from .constants import MODELSCOPE_CACHE_DIR


logger = logging.getLogger(__name__)


def _is_complete_model_dir(path: Path) -> bool:
    """模型目录完整性检查：权重 + 配置齐全才算命中。

    防止半成品缓存被当作可用。

    Args:
        path: 候选模型目录。

    Returns:
        model.pt 存在且 config.yaml 或 configuration.json 存在时为 True。
    """

    return (path / "model.pt").is_file() and (
        (path / "config.yaml").is_file() or (path / "configuration.json").is_file()
    )


def _resolve_model_path(model: str) -> str:
    """把模型标识解析为本地路径：本地目录 > 仓库缓存 > ModelScope 下载。

    Args:
        model: 本地模型目录路径或 ModelScope 模型 id。

    Returns:
        可用的本地模型目录路径。
    """

    if Path(model).is_dir():
        return model
    return str(
        resolve_modelscope_snapshot(
            model, MODELSCOPE_CACHE_DIR, _is_complete_model_dir
        )
    )


class FunASRNanoASR:
    """Fun-ASR-Nano 段级转写器（首次 transcribe 时才加载模型）。"""

    def __init__(self, config):
        self.config = config
        self._model = None
        self._kwargs: dict = {}
        self._tokenizer = None

    def _resolve_device(self) -> str:
        # 设备解析单一来源在 common.utils.device（"auto" → "cuda:0"）；
        # FunASR from_pretrained 要字符串，这里取规范设备名。
        return resolve_device_name(str(self.config.device))

    def _load(self) -> None:
        from funasr.models.fun_asr_nano.model import FunASRNano

        model_path = _resolve_model_path(str(self.config.model))
        device = self._resolve_device()
        logger.info(
            "[asr] loading Fun-ASR-Nano from %s (device=%s)", model_path, device
        )
        self._model, self._kwargs = FunASRNano.from_pretrained(
            model=model_path, device=device
        )
        self._model.eval()
        self._tokenizer = self._kwargs["tokenizer"]

    def count_tokens(self, text: str) -> int:
        """prev_text 预算计数用的 token 数。

        模型未加载前退化为按字符数粗略估计（中文约 1 字 1 token），仅用于
        兜底。

        Args:
            text: 待计数文本。

        Returns:
            token 数；tokenizer 未就绪时为字符数。
        """

        if self._tokenizer is None:
            # 模型未加载前的粗略估计（中文约 1 字 1 token），仅用于兜底。
            return len(text)
        return len(self._tokenizer.encode(text))

    def tail_text(self, text: str, max_tokens: int) -> str:
        """取文本尾部，token 数不超过 max_tokens（长段内续写上下文用）。

        Args:
            text: 已累计文本。
            max_tokens: token 预算；<= 0 或 text 为空时返回空串。

        Returns:
            尾部文本；整体不超预算时原样返回。tokenizer 未就绪时按字符数
            截尾兜底。
        """

        if max_tokens <= 0 or not text:
            return ""
        if self._tokenizer is None:
            return text[-max_tokens:]
        token_ids = self._tokenizer.encode(text)
        if len(token_ids) <= max_tokens:
            return text
        return self._tokenizer.decode(token_ids[-max_tokens:])

    @staticmethod
    def _strip_prefix(text: str, prev_text: str) -> str:
        """剥离返回值中的 prev_text 前缀（模型把 prev_text 逐字拼回输出）。

        Args:
            text: 模型输出文本。
            prev_text: 推理时传入的 prev_text。

        Returns:
            剥离前缀后的新文本；输出未以 prev_text 开头时告警并保留完整
            输出。
        """

        if prev_text and text.startswith(prev_text):
            return text[len(prev_text) :]
        if prev_text:
            logger.warning(
                "[asr] output does not start with prev_text, keeping full output "
                "(prev tail=%r, output head=%r)",
                prev_text[-20:],
                text[:20],
            )
        return text

    def transcribe(self, waveform: torch.Tensor, prev_text: str = "") -> str:
        """转写一段音频 [1, T]（16k 单声道），返回新增文本（不含 prev_text）。

        prev_text 必须是本段音频前缀的转写（长段窗切续写场景）；跨段上下文
        语义不成立（模型对齐失败会判转写完成而提前 EOS），不要那样用。

        Args:
            waveform: 音频张量 [1, T]，采样率须与 config.sample_rate 一致。
            prev_text: 本段音频前缀的转写；空串表示无上下文（该键不传给模型）。

        Returns:
            新增文本（已剥离 prev_text 前缀并 strip）。
        """

        if self._model is None:
            self._load()
        audio = waveform.squeeze(0).detach().cpu().float()
        # prev_text 为 None 时模型端 None + str 会 TypeError：空则不传该键。
        extra_kwargs = {"prev_text": prev_text} if prev_text else {}
        infer_start = time.perf_counter()
        with torch.inference_mode():
            results, _ = self._model.inference([audio], **extra_kwargs, **self._kwargs)
        elapsed_ms = (time.perf_counter() - infer_start) * 1000
        text = self._strip_prefix(results[0]["text"], prev_text).strip()
        logger.info(
            "[asr] transcribe: %.2fs audio, %.0fms, prev_tokens=%d, text=%r",
            audio.shape[0] / int(self.config.sample_rate),
            elapsed_ms,
            self.count_tokens(prev_text) if prev_text else 0,
            text[:50],
        )
        return text


__all__ = ["FunASRNanoASR"]
