# vllm-ps — Power-SMC + vLLM

**vllm-ps** integrates **Power-SMC** (Power Sequential Monte Carlo sampling for
LLM reasoning, [arXiv:2602.10273](https://arxiv.org/abs/2602.10273)) into the
**vLLM** inference engine. It contains two co-located codebases:

- `Power-SMC/` — standalone PyTorch/HuggingFace research experiments
- `vllm/` — vLLM fork with Power-SMC as a V1 scheduler-integrated decoding strategy

---

## Project

| Aspect | Location |
|---|---|
| **Standalone experiments** | `Power-SMC/` — `power_samp_math.py`, `power_samp_gsm.py`, `power_samp_gpqa.py` |
| **Core SMC algorithm** | `Power-SMC/smc_samp_utils.py` (memory-optimized + CoW cache) |
| **vLLM Power-SMC config/helpers** | `vllm/vllm/v1/power_smc.py` |
| **vLLM scheduler integration** | `vllm/vllm/v1/core/sched/scheduler.py` |
| **vLLM sampler integration** | `vllm/vllm/v1/sample/sampler.py` (gathers base/proposal logprobs) |
| **Design docs** | `vllm/docs/design/power_smc_completion_audit.md`, `power_smc_kv_cow.md` |
| **Benchmark reports** | `vllm/outputs/power_smc/` — many `.md` files |
| **Slurm sweep infra** | `vllm/scripts/slurm/power_smc_sweep.sh`, `power_smc_benchmark.sbatch` |
| **vLLM AGENTS.md** | `vllm/AGENTS.md` (upstream conventions — read before touching vLLM core) |

**Stack:** Python 3.10+, PyTorch, HuggingFace `transformers` + `datasets`,
vLLM (C++/CUDA/Triton backend), FlashInfer attention, conda (`Power-SMC/`),
uv + venv (`vllm/`).

---

## Commands

### Power-SMC standalone experiments (conda env)

```bash
# Activate via script (env at /data/conda_envs/power-smc-vllm by default)
export POWER_SMC_CONDA_ENV=/path/to/conda_envs/power-smc-vllm
./Power-SMC/scripts/power_smc_env.sh python Power-SMC/power_samp_math.py --model qwen --temperature 0.25 --mcmc_steps 10
./Power-SMC/scripts/power_smc_env.sh python Power-SMC/power_samp_gsm.py --model qwen --temperature 0.25 --mcmc_steps 10
./Power-SMC/scripts/power_smc_env.sh python Power-SMC/power_samp_gpqa.py --model phi
```

### vLLM fork (uv/venv)

```bash
cd vllm
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in

# Run Power-SMC-specific tests
.venv/bin/python -m pytest tests/v1/test_power_smc.py -v
.venv/bin/python -m pytest tests/examples/test_power_smc_helpers.py -v
# All vLLM tests
.venv/bin/python -m pytest tests/v1/ -v

# Lint
pre-commit run --all-files
pre-commit run ruff-check --all-files
```

### Benchmark

```bash
cd vllm
# See scripts/power_smc_env.sh for managed env
# Single run
./scripts/power_smc_env.sh python examples/generate/benchmark_power_smc.py
# Slurm sweep
DRY_RUN=1 ./scripts/slurm/power_smc_sweep.sh    # dry run
./scripts/slurm/power_smc_sweep.sh               # submit
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Power-SMC/ (standalone)                │
│                                                          │
│  constants.py              power_samp_*.py               │
│  ─ prompt templates,      ─ benchmark scripts            │
│    data loaders              (MATH500/GSM8K/GPQA)        │
│                              for naive/MCMC/SMC           │
│  power_samp_utils.py                                     │
│  ─ MCMC, standard +       smc_samp_utils.py              │
│    naive samplers         ─ core SMC algorithm:          │
│                             particles, ESS, resample,    │
│                             CoW KV cache, multi-round    │
│                                                          │
│  grader_utils/ ─ math/GPQA answer graders               │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                vllm/ (fork with Power-SMC V1)             │
│                                                          │
│  vllm/v1/power_smc.py                                    │
│  ─ PowerSMCConfig validation from SamplingParams.extra   │
│  ─ PowerSMCGroupManager: ESS tracking, resample plans   │
│  ─ log-space weight updates, α annealing schedule        │
│                                                          │
│  vllm/v1/core/sched/scheduler.py                         │
│  ─ Power-SMC particle group lifecycle in the scheduler   │
│  ─ Resample at block boundaries with CoW KV aliasing     │
│  ─ Reset/recompute fallback when CoW not safe            │
│                                                          │
│  vllm/v1/sample/sampler.py                               │
│  ─ gather_power_smc_logprobs: base + proposal log-probs  │
│                                                          │
│  vllm/v1/request.py                                      │
│  ─ power_smc_config, power_smc_child_info on Request     │
│                                                          │
│  vllm/v1/outputs.py                                      │
│  ─ PowerSMCLogprobTensors, EngineCoreOutput extensions   │
└──────────────────────────────────────────────────────────┘
```

**Key flow:**
1. User passes `sampling_params.extra_args["power_smc"]` → `PowerSMCConfig` validated
2. Scheduler spawns N particle child requests from one parent
3. Particles decode in blocks; at each block boundary, scheduler computes ESS via `PowerSMCGroupManager`
4. If ESS < threshold → systematic resampling with KV aliasing (CoW) or reset/recompute fallback
5. After final token, weight-weighted final sample selected
6. Diagnostics optionally returned via response metadata

---

## Conventions

- **Code style:** 88-char line limit (ruff), Google-style docstrings (`Args:`/`Returns:`/`Raises:`)
- **No bare pip or system python** in vllm/ — always `uv` or `.venv/bin/python`
- **Power-SMC standalone:** no lint/setup enforced, ad-hoc scripts
- **vLLM contributions:** must read `vllm/AGENTS.md` first; no pure-agent PRs allowed upstream
- **vLLM Power-SMC is a prototype** — not aiming for upstream merge; independent tests
- **Power-SMC sweep/test conda env** at `/data/conda_envs/power-smc-vllm` by default, overridable via `POWER_SMC_CONDA_ENV`
- **Commit trailers:** use `Co-authored-by:` for AI attribution
- **Pre-commit** required in vllm/: install with `pre-commit install`

---

## Notes

<!-- Quick-add space — add ephemeral work notes, open questions, or reminders here -->
