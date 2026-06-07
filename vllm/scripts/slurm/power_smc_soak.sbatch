#!/usr/bin/env bash
#SBATCH --job-name=power-smc-soak
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/power-smc-soak-%j.out
#SBATCH --error=outputs/slurm/power-smc-soak-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tyliu/ghworkspace/vllm-ps/vllm}"
cd "${REPO_ROOT}"

mkdir -p outputs/slurm outputs/power_smc

MODEL="${MODEL:-/data/shared/models/Qwen2.5-0.5B-Instruct}"
PROMPT_FILE="${PROMPT_FILE:-}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
ITERATIONS="${ITERATIONS:-12}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PARTICLES="${PARTICLES:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
ALPHA="${ALPHA:-16.0}"
ALPHA_RAMP_TOKENS="${ALPHA_RAMP_TOKENS:-1}"
ESS_THRESHOLD="${ESS_THRESHOLD:-1.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
IGNORE_EOS="${IGNORE_EOS:-1}"
MAX_GPU_AFTER_DELTA_MIB="${MAX_GPU_AFTER_DELTA_MIB:-512}"

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

POWER_SMC_USE_SOURCE_VLLM=1 scripts/power_smc_env.sh \
  examples/generate/power_smc_soak.py \
  --model "${MODEL}" \
  --num-prompts "${NUM_PROMPTS}" \
  --iterations "${ITERATIONS}" \
  --max-tokens "${MAX_TOKENS}" \
  --particles "${PARTICLES}" \
  --block-size "${BLOCK_SIZE}" \
  --alpha "${ALPHA}" \
  --alpha-ramp-tokens "${ALPHA_RAMP_TOKENS}" \
  --ess-threshold "${ESS_THRESHOLD}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-gpu-after-delta-mib "${MAX_GPU_AFTER_DELTA_MIB}" \
  "${EXTRA_ARGS[@]}" \
  --output-json "outputs/power_smc/soak_${SLURM_JOB_ID}.json" \
  --output-md "outputs/power_smc/soak_${SLURM_JOB_ID}.md"
