"""chunk 管线的共享数据契约：schema（ChunkArtifacts 等）与 commit 区帧裁剪。

当前消费者为 diarization 包（extract → cluster 的交换单元、RTTM writer）
与 pipeline 组合层；tools/experiments/ 的 oracle 实验脚本也镜像本包口径。
"""

from .frames import iter_commit_frames
from .schema import (
    ChunkArtifacts,
    ChunkDebugInfo,
    ChunkObservation,
    LocalAssignmentDebug,
)

__all__ = [
    "iter_commit_frames",
    "ChunkArtifacts",
    "ChunkDebugInfo",
    "ChunkObservation",
    "LocalAssignmentDebug",
]
