#!/usr/bin/env bash
#SBATCH --job-name=power-smc-vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/power-smc-%j.out
#SBATCH --error=outputs/slurm/power-smc-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tyliu/ghworkspace/vllm-ps/vllm}"
cd "${REPO_ROOT}"

mkdir -p outputs/slurm outputs/power_smc

MODEL="${MODEL:-/data/shared/models/Qwen2.5-0.5B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
PROMPT_FILE="${PROMPT_FILE:-}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PARTICLES="${PARTICLES:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
ALPHA="${ALPHA:-4.0}"
ALPHA_RAMP_TOKENS="${ALPHA_RAMP_TOKENS:-16}"
ESS_THRESHOLD="${ESS_THRESHOLD:-0.5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
IGNORE_EOS="${IGNORE_EOS:-0}"
STOP_TOKEN_ID="${STOP_TOKEN_ID:-}"

EXTRA_ARGS=()
if [[ -n "${PROMPT_FILE}" ]]; then
  EXTRA_ARGS+=(--prompt-file "${PROMPT_FILE}")
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  EXTRA_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
fi
if [[ "${IGNORE_EOS}" == "1" ]]; then
  EXTRA_ARGS+=(--ignore-eos)
fi
if [[ -n "${STOP_TOKEN_ID}" ]]; then
  IFS=',' read -r -a STOP_TOKEN_IDS <<< "${STOP_TOKEN_ID}"
  for token_id in "${STOP_TOKEN_IDS[@]}"; do
    EXTRA_ARGS+=(--stop-token-id "${token_id}")
  done
fi

POWER_SMC_USE_SOURCE_VLLM=1 scripts/power_smc_env.sh \
  examples/generate/benchmark_power_smc.py \
  --model "${MODEL}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-tokens "${MAX_TOKENS}" \
  --particles "${PARTICLES}" \
  --block-size "${BLOCK_SIZE}" \
  --alpha "${ALPHA}" \
  --alpha-ramp-tokens "${ALPHA_RAMP_TOKENS}" \
  --ess-threshold "${ESS_THRESHOLD}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "${EXTRA_ARGS[@]}" \
  --output-json "outputs/power_smc/benchmark_${SLURM_JOB_ID}.json" \
  --output-md "outputs/power_smc/benchmark_${SLURM_JOB_ID}.md"
