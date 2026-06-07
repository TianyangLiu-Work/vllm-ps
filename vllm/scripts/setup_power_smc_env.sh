#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Conda-managed environment setup/check for the Power-SMC experiments.
# This follows serverREADME.md by using /data/conda_envs for environments
# and leaves GPU execution to Slurm.

set -euo pipefail

CONDA="${CONDA:-/data/conda/bin/conda}"
PREFIX="${POWER_SMC_CONDA_PREFIX:-/data/conda_envs/power-smc-vllm}"
CLONE_FROM="${POWER_SMC_CLONE_FROM:-/data/conda_envs/predictive_clipo}"
PYTHON_VERSION="${POWER_SMC_PYTHON_VERSION:-3.11}"
CHECK_ONLY="${POWER_SMC_ENV_CHECK_ONLY:-0}"
REPO_ROOT="${POWER_SMC_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

PACKAGES=(
  torch
  transformers
  datasets
  numpy
  pandas
  tqdm
  sympy
  pylatexenc
  ruff
  tblib
  vllm
)

BUILD_DEPS=(
  "setuptools>=77.0.3,<81.0.0"
  "setuptools-scm>=8.0"
  "setuptools-rust>=1.9.0"
  "cmake>=3.26.1"
  ninja
  wheel
  jinja2
  packaging
)

if [[ ! -x "${CONDA}" ]]; then
  echo "conda is not executable: ${CONDA}" >&2
  exit 1
fi

if [[ ! -x "${PREFIX}/bin/python" ]]; then
  if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo "missing conda environment: ${PREFIX}" >&2
    exit 1
  fi
  if [[ -x "${CLONE_FROM}/bin/python" ]]; then
    "${CONDA}" create -y -p "${PREFIX}" --clone "${CLONE_FROM}"
  else
    "${CONDA}" create -y -p "${PREFIX}" "python=${PYTHON_VERSION}"
  fi
fi

if [[ "${CHECK_ONLY}" != "1" ]]; then
  "${CONDA}" install -y -p "${PREFIX}" -c nvidia cuda-nvcc=12.9
  "${PREFIX}/bin/python" -m pip install "${BUILD_DEPS[@]}"
  "${PREFIX}/bin/python" -m pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu129 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.11.0+cu129" \
    "torchvision==0.26.0+cu129" \
    "torchaudio==2.11.0+cu129"
  "${PREFIX}/bin/python" -m pip install \
    "setuptools==80.10.2" \
    "numpy==2.3.5" \
    "fsspec==2026.2.0" \
    "cuda-python==12.9.7"
  VLLM_USE_PRECOMPILED=1 \
    VLLM_MAIN_CUDA_VERSION=12.9 \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
    "${PREFIX}/bin/python" -m pip install -e "${REPO_ROOT}" \
    --no-build-isolation --no-deps --force-reinstall
fi

missing="$("${PREFIX}/bin/python" - "${PACKAGES[@]}" <<'PY'
import importlib.util
import sys

missing = [
    package for package in sys.argv[1:]
    if importlib.util.find_spec(package.replace("-", "_")) is None
]
print(" ".join(missing))
PY
)"

if [[ -n "${missing}" ]]; then
  echo "missing Python packages in ${PREFIX}: ${missing}" >&2
  exit 1
fi

"${PREFIX}/bin/python" - <<'PY'
import importlib.util
import sys

packages = [
    "torch",
    "transformers",
    "datasets",
    "numpy",
    "pandas",
    "tqdm",
    "sympy",
    "pylatexenc",
    "ruff",
    "tblib",
    "vllm",
]
missing = [name for name in packages if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing packages: {', '.join(missing)}")

import torch
import vllm

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"vllm={vllm.__version__}")
print(f"vllm_file={vllm.__file__}")
PY
