"""跨包数据契约：chunk 管线共享数据结构与 commit 区帧裁剪。

diarization 与 separation 两包互不 import，共享契约统一从本包取。
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
