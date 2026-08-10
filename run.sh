#!/bin/bash

set -euo pipefail

# 用法:
#   bash run.sh [audio_input]
#
# 这个脚本专门运行实时管线。
# 运行参数以 `config.yaml` 为准，
# 脚本只负责补充运行时必须信息和少量常用覆盖项。
#
# 可选环境变量:
#   WITH_ASR=1    ASR 跟随进程（transcribe.py --follow）与管线同时启动，
#                 音频段即出即转，实现类流式 ASR 效果；管线结束后 ASR 收尾
#                 落盘 transcript。要求 config.yaml 中 asr_enabled 或
#                 separation_enabled 为 true（否则没有分段音频可转写）。

if [ -f ./.venv/bin/activate ]; then
    source ./.venv/bin/activate
fi

audio_input=${1:-./datasets/examples/}
config_path=${CONFIG_PATH:-./config/config.yaml}
model_path=${MODEL_PATH:-}
hf_token=${HF_TOKEN:-}
hf_cache_dir=${HF_CACHE_DIR:-}
debug_flag=${DEBUG:-0}
show_rttm_flag=${SHOW_RTTM:-0}
with_asr_flag=${WITH_ASR:-0}
output_root=${OUTPUT_ROOT:-./exp}
run_name=${RUN_NAME:-default}

test_name="common"
basic_dir="${output_root}/${test_name}"
exp_dir="${basic_dir}/${run_name}"
results_file="$basic_dir/results.txt"

# ASR 依赖管线的分段导出，先确认配置里开了导出开关。
if [ "$with_asr_flag" = "1" ]; then
    if ! grep -qE '^asr_enabled:\s*true' "$config_path" \
        && ! grep -qE '^separation_enabled:\s*true' "$config_path"; then
        echo "ERROR: WITH_ASR=1 需要 $config_path 中 asr_enabled 或 separation_enabled 为 true" >&2
        exit 1
    fi
fi

rm -rf "$basic_dir"
mkdir -p "$exp_dir"

echo "Online Speaker Diarization pipeline" > "$results_file"
echo "================================================" >> "$results_file"
echo "audio_input: $audio_input" >> "$results_file"
echo "config_path: $config_path" >> "$results_file"
if [ -n "$model_path" ]; then
    echo "model_path_override: $model_path" >> "$results_file"
fi
if [ -n "$hf_cache_dir" ]; then
    echo "hf_cache_dir_override: $hf_cache_dir" >> "$results_file"
fi
if [ "$with_asr_flag" = "1" ]; then
    echo "with_asr: true" >> "$results_file"
fi
echo "run_name: $run_name" >> "$results_file"
echo "" >> "$results_file"

echo "=========================================="
echo "Running with YAML config"
echo "audio_input=$audio_input"
echo "config_path=$config_path"
echo "output_dir=$exp_dir"
if [ -n "$model_path" ]; then
    echo "model_path override enabled"
fi
if [ -n "$hf_cache_dir" ]; then
    echo "hf_cache_dir override enabled"
fi
if [ "$with_asr_flag" = "1" ]; then
    echo "with_asr enabled"
fi
echo "=========================================="

cmd=(
    python3 pipeline.py
    --wav "$audio_input"
    --output_dir "$exp_dir"
    --config "$config_path"
)

# `model_path` 在 YAML 中通常已经提供；只有用户显式给环境变量时才覆盖。
if [ -n "$model_path" ]; then
    cmd+=(--model_path "$model_path")
fi

# 同理，Hugging Face 相关参数默认跟随 YAML，仅在外部显式覆盖时才传入 CLI。
if [ -n "$hf_cache_dir" ]; then
    cmd+=(--hf_cache_dir "$hf_cache_dir")
fi

if [ -n "$hf_token" ]; then
    cmd+=(--hf_token "$hf_token")
fi

# 调试和额外导出属于运行时开关，保留脚本级覆盖最方便。
if [ "$debug_flag" = "1" ]; then
    cmd+=(--debug --verbose)
fi

if [ "$show_rttm_flag" = "1" ]; then
    cmd+=(--show_rttm)
fi

printf 'Command: ' | tee "$exp_dir/command.log"
printf '%q ' "${cmd[@]}" | tee -a "$exp_dir/command.log"
printf '\n' | tee -a "$exp_dir/command.log"

# 可选：ASR 跟随进程与管线同时启动——轮询 manifest 增量转写（即出即转），
# 管线结束后由 done 哨兵文件通知它收尾落盘。
asr_pid=""
done_file="$exp_dir/.diarization_done"
if [ "$with_asr_flag" = "1" ]; then
    asr_cmd=(
        python3 transcribe.py
        --segments_dir "$exp_dir"
        --config "$config_path"
        --follow
        --done_file "$done_file"
    )
    if [ "$debug_flag" = "1" ]; then
        asr_cmd+=(--verbose)
    fi

    printf 'ASR Command: ' | tee -a "$exp_dir/command.log"
    printf '%q ' "${asr_cmd[@]}" | tee -a "$exp_dir/command.log"
    printf '\n' | tee -a "$exp_dir/command.log"

    "${asr_cmd[@]}" &
    asr_pid=$!
    echo "ASR follow process started (pid=$asr_pid)"
fi

# 管线允许失败：失败后仍要落哨兵放行 ASR 收尾，再按原退出码退出。
set +e
"${cmd[@]}"
pipeline_rc=$?
set -e

if [ "$with_asr_flag" = "1" ]; then
    touch "$done_file"
    wait "$asr_pid"
fi

if [ "$pipeline_rc" -ne 0 ]; then
    exit "$pipeline_rc"
fi

rttm_count=$(python3 - <<'PY' "$exp_dir"
import os
import sys

exp_dir = sys.argv[1]
files = sorted(
    f for f in os.listdir(exp_dir)
    if f.endswith('.streaming.rttm')
)
print(len(files))
PY
)

echo "$run_name -> streaming_rttm_files=$rttm_count | config=$config_path" >> "$results_file"
echo "Result: $run_name -> streaming_rttm_files=$rttm_count"
echo "Pipeline log: $exp_dir/run.log"

if [ "$with_asr_flag" = "1" ]; then
    transcript_count=$(python3 - <<'PY' "$exp_dir"
import os
import sys

exp_dir = sys.argv[1]
files = sorted(
    f for f in os.listdir(exp_dir)
    if f.endswith('.transcript.jsonl')
)
print(len(files))
PY
)

    echo "$run_name -> transcript_files=$transcript_count | config=$config_path" >> "$results_file"
    echo "ASR Result: $run_name -> transcript_files=$transcript_count"
    echo "ASR log: $exp_dir/transcribe.log"
fi
