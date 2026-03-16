#!/bin/bash

set -euo pipefail

# 用法:
#   bash run.sh [audio_input]
#
# 这个脚本专门测试 overlap 版本在线流水线。
# 运行参数尽量以 `online_pipline_overlap_config.yaml` 为准，
# 脚本只负责补充运行时必须信息和少量常用覆盖项。

if [ -f ./.venv/bin/activate ]; then
    source ./.venv/bin/activate
fi

audio_input=${1:-./examples/}
config_path=${CONFIG_PATH:-./config.yaml}
model_path=${MODEL_PATH:-}
hf_token=${HF_TOKEN:-}
hf_cache_dir=${HF_CACHE_DIR:-}
ref_rttm=${REF_RTTM:-}
ref_rttm_dir=${REF_RTTM_DIR:-./datasets/rttm}
debug_flag=${DEBUG:-0}
save_scores_flag=${SAVE_SEGMENTATION_SCORES:-0}
show_rttm_flag=${SHOW_RTTM:-0}
output_root=${OUTPUT_ROOT:-./exp}
run_name=${RUN_NAME:-default}

test_name="pipline_exp"
basic_dir="${output_root}/${test_name}"
exp_dir="${basic_dir}/${run_name}"
results_file="$basic_dir/results.txt"

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
echo "=========================================="

cmd=(
    python3 pipline.py
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

if [ "$save_scores_flag" = "1" ]; then
    cmd+=(--save_segmentation_scores)
fi

if [ "$show_rttm_flag" = "1" ]; then
    cmd+=(--show_rttm)
fi

printf 'Command: ' | tee "$exp_dir/command.log"
printf '%q ' "${cmd[@]}" | tee -a "$exp_dir/command.log"
printf '\n' | tee -a "$exp_dir/command.log"

"${cmd[@]}"

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

echo "$run_name -> streaming_rttm_files=$rttm_count | config=$config_path save_segmentation_scores=$save_scores_flag" >> "$results_file"
echo "Result: $run_name -> streaming_rttm_files=$rttm_count"
echo "Pipeline log: $exp_dir/run.log"
