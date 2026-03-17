"""重叠语音分离处理逻辑。"""

import logging


def process_overlap_segment(
    *,
    start: float,
    end: float,
    config,
    separator,
    embedder,
    clusterer,
    speaker_buffers,
    slice_waveform_by_time,
    total_duration: float,
    logger: logging.Logger,
) -> None:
    """处理重叠段：补全音频、分离、匹配，并覆盖写入说话人音轨。"""

    if not separator:
        logger.debug("[separation] skip overlap segment: separator unavailable")
        return
    actual_duration = end - start
    if actual_duration <= 0:
        logger.debug(
            "[separation] skip invalid overlap range start=%.3f end=%.3f",
            start,
            end,
        )
        return
    need_pad = max(0.0, config.separation_required_duration - actual_duration)
    pad_right = min(need_pad, 0.5 * config.advance_step, total_duration - end)
    remaining_pad = need_pad - pad_right
    pad_left = min(remaining_pad, max(0.0, start))
    if pad_left + pad_right < need_pad:
        extra_right = need_pad - (pad_left + pad_right)
        pad_right += min(extra_right, max(0.0, total_duration - end - pad_right))
    full_start = start - pad_left
    full_end = end + pad_right

    logger.info(
        "[separation] process_overlap start=%.3f end=%.3f duration=%.3f full_start=%.3f full_end=%.3f",
        start,
        end,
        actual_duration,
        full_start,
        full_end,
    )

    audio_3s = slice_waveform_by_time(full_start, full_end)
    if audio_3s.shape[-1] == 0:
        logger.warning(
            "[separation] skip overlap due to empty waveform slice start=%.3f end=%.3f",
            full_start,
            full_end,
        )
        return

    separated = separator.separate(audio_3s)
    logger.debug(
        "[separation] separated_input_samples=%d output_shape=%s",
        int(audio_3s.shape[-1]),
        tuple(int(dim) for dim in separated.shape),
    )

    embeddings = embedder.embed_segments(
        [separated[spk_idx : spk_idx + 1] for spk_idx in range(2)]
    )

    active_speakers = clusterer.get_active_speakers((start, full_end))
    match_result = clusterer.match_separated_embeddings(embeddings, active_speakers)
    if not match_result:
        logger.info(
            "[separation] match_failed active_speakers=%d",
            len(active_speakers),
        )
    else:
        logger.info(
            "[separation] match_success assignments=%s",
            {int(gid): int(idx) for gid, idx in match_result.items()},
        )

    left_crop_scale = 1.0
    right_crop_scale = 1.0

    if match_result and config.enable_speech_separation:
        crop_left_seconds = max(0.0, pad_left * left_crop_scale)
        crop_right_seconds = max(0.0, pad_right * right_crop_scale)

        crop_left_samples = int(round(crop_left_seconds * config.sample_rate))
        crop_right_samples = int(round(crop_right_seconds * config.sample_rate))

        total_samples = int(separated.shape[1])
        keep_start = min(max(0, crop_left_samples), total_samples)
        keep_end = max(keep_start, total_samples - max(0, crop_right_samples))

        write_start = full_start + crop_left_seconds

        for global_id, spk_idx in match_result.items():
            spk_audio = separated[spk_idx, keep_start:keep_end]
            speaker_buffers.append(
                int(global_id),
                spk_audio,
                write_start,
                overwrite=True,
            )

    return
