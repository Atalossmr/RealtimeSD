"""merge 事件消费与同步逻辑。"""

from __future__ import annotations


def handle_pending_speaker_merges(
    *, clusterer, config, speaker_buffers, streaming_logger, commit_time: float
) -> None:
    """消费 clusterer 累计的 speaker merge 事件，并同步到 streaming/audio。"""

    merge_events = clusterer.pop_merge_events()
    if not merge_events:
        return

    for event in merge_events:
        large_id = int(event["large"])
        small_id = int(event["small"])

        if config.enable_speech_separation:
            # merge 时同步合并音频缓存，保证最终导出的 stable 音轨不丢失 small speaker 语音。
            speaker_buffers.merge_speaker_audio(small_id, large_id)

        streaming_logger.handle_speaker_merge(
            large_speaker_id=large_id,
            small_speaker_id=small_id,
            merge_time=float(commit_time),
        )
        streaming_logger.notify_speaker_became_inactive(small_id)
