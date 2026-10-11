"""extract 前端包：embedding 之前的一切（与 extract 平级）。

`ChunkFrontend`：滑窗/commit 区调度 + pyannote segmentation 局部识别 +
local track 构建（track_builder），产出 `PreparedChunk`；embedding 提取
（`common.models.embedding_infer.RegionEmbedder`）与本包的组合在
`diarization.extract.extractor.ChunkExtractor`。
"""

from .frontend import ChunkFrontend, PreparedChunk
from .oracle import OracleFrontend
from .scheduling import ChunkWindow, iter_chunk_schedule
from .track_builder import ChunkTrackBuilder, LocalTrack

__all__ = [
    "ChunkFrontend",
    "PreparedChunk",
    "ChunkTrackBuilder",
    "LocalTrack",
    "ChunkWindow",
    "iter_chunk_schedule",
    "OracleFrontend",
]
