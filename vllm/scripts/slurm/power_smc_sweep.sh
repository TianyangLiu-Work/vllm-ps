#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tyliu/ghworkspace/vllm-ps/vllm}"
cd "${REPO_ROOT}"

mkdir -p outputs/power_smc outputs/slurm

MODEL="${MODEL:-/data/shared/models/Qwen2.5-0.5B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
PROMPT_FILE="${PROMPT_FILE:-}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
ALPHA_RAMP_TOKENS="${ALPHA_RAMP_TOKENS:-1}"
ESS_THRESHOLD="${ESS_THRESHOLD:-1.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
IGNORE_EOS="${IGNORE_EOS:-1}"
STOP_TOKEN_ID="${STOP_TOKEN_ID:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
MANIFEST="${MANIFEST:-outputs/power_smc/sweep_${RUN_TAG}.tsv}"
DRY_RUN="${DRY_RUN:-0}"
SEQUENTIAL="${SEQUENTIAL:-1}"

# Each row is: label|max_tokens|particles|alpha.
# Override with SWEEP_CONFIGS=$'name|64|8|4.0\nname2|128|16|16.0'.
DEFAULT_SWEEP_CONFIGS=$'p8_t64_a4|64|8|4.0\np16_t128_a16|128|16|16.0\np32_t128_a16|128|32|16.0'
SWEEP_CONFIGS="${SWEEP_CONFIGS:-${DEFAULT_SWEEP_CONFIGS}}"

printf "run_tag\tlabel\tjob_id\tmodel\tnum_prompts\tprompt_file\tmax_tokens\tparticles\tblock_size\talpha\talpha_ramp_tokens\tess_threshold\tgpu_memory_utilization\tattention_backend\tignore_eos\tstop_token_id\n" > "${MANIFEST}"

previous_job_id=""
while IFS='|' read -r label max_tokens particles alpha; do
  if [[ -z "${label}" ]]; then
    continue
  fi

  export_vars="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${MODEL},NUM_PROMPTS=${NUM_PROMPTS},MAX_TOKENS=${max_tokens},PARTICLES=${particles},BLOCK_SIZE=${BLOCK_SIZE},ALPHA=${alpha},ALPHA_RAMP_TOKENS=${ALPHA_RAMP_TOKENS},ESS_THRESHOLD=${ESS_THRESHOLD},GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION},ATTENTION_BACKEND=${ATTENTION_BACKEND},IGNORE_EOS=${IGNORE_EOS}"
  if [[ -n "${PROMPT_FILE}" ]]; then
    export_vars="${export_vars},PROMPT_FILE=${PROMPT_FILE}"
  fi
  if [[ -n "${STOP_TOKEN_ID}" ]]; then
    export_vars="${export_vars},STOP_TOKEN_ID=${STOP_TOKEN_ID}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    job_id="DRY_RUN"
  else
    sbatch_args=(
      --parsable
      --job-name "psmc-${label}"
      --export="${export_vars}"
    )
    if [[ "${SEQUENTIAL}" == "1" && -n "${previous_job_id}" ]]; then
      sbatch_args+=(--dependency="afterany:${previous_job_id}")
    fi
    job_id="$(sbatch "${sbatch_args[@]}" scripts/slurm/power_smc_benchmark.sbatch)"
    previous_job_id="${job_id}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${RUN_TAG}" \
    "${label}" \
    "${job_id}" \
    "${MODEL}" \
    "${NUM_PROMPTS}" \
    "${PROMPT_FILE}" \
    "${max_tokens}" \
    "${particles}" \
    "${BLOCK_SIZE}" \
    "${alpha}" \
    "${ALPHA_RAMP_TOKENS}" \
    "${ESS_THRESHOLD}" \
    "${GPU_MEMORY_UTILIZATION}" \
    "${ATTENTION_BACKEND}" \
    "${IGNORE_EOS}" \
    "${STOP_TOKEN_ID}" >> "${MANIFEST}"
done <<< "${SWEEP_CONFIGS}"

echo "Wrote sweep manifest: ${MANIFEST}"
