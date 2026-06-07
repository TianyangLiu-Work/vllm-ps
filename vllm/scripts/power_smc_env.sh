#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared environment setup for the Power-SMC vLLM experiments.
#
# This follows the local server README:
#   * use a conda-managed environment under /data/conda_envs
#   * use shared Hugging Face caches under /data/shared
#   * submit GPU work through Slurm instead of running GPU jobs directly

set -euo pipefail

export POWER_SMC_REPO_ROOT="${POWER_SMC_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export POWER_SMC_CONDA_PREFIX="${POWER_SMC_CONDA_PREFIX:-/data/conda_envs/power-smc-vllm}"
export POWER_SMC_PYTHON="${POWER_SMC_PYTHON:-${POWER_SMC_CONDA_PREFIX}/bin/python}"
export POWER_SMC_USE_SOURCE_VLLM="${POWER_SMC_USE_SOURCE_VLLM:-0}"

if [[ "${POWER_SMC_USE_SOURCE_VLLM}" == "1" ]]; then
  export PYTHONPATH="${POWER_SMC_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${POWER_SMC_REPO_ROOT}/examples/generate${PYTHONPATH:+:${PYTHONPATH}}"
fi
export HF_HOME="${HF_HOME:-/data/shared/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION:-12.9}"
export CUDA_HOME="${CUDA_HOME:-${POWER_SMC_CONDA_PREFIX}}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
export CC="${CC:-${CUDA_HOME}/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-${CUDA_HOME}/bin/x86_64-conda-linux-gnu-g++}"
export PATH="${CUDA_HOME}/bin:${PATH}"
cuda_stub_dir="${CUDA_HOME}/targets/x86_64-linux/lib/stubs"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${cuda_stub_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${cuda_stub_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LDFLAGS="-L${cuda_stub_dir}${LDFLAGS:+ ${LDFLAGS}}"

if [[ ! -x "${POWER_SMC_PYTHON}" ]]; then
  echo "POWER_SMC_PYTHON is not executable: ${POWER_SMC_PYTHON}" >&2
  exit 1
fi

exec "${POWER_SMC_PYTHON}" "$@"
