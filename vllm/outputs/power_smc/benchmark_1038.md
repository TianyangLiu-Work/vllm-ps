# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `1`
- Max tokens: `20`
- Particles: `2`
- Block size: `16`
- Alpha: `4.0`
- Attention backend: `FLASHINFER`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 0.441 | 0.441 | 20 | 45.34 |
| baseline_particles | 0.366 | 0.366 | 40 | 109.15 |
| best_of_n | 0.919 | 0.919 | 40 | 43.52 |
| weighted_best_of_n | 0.388 | 0.388 | 40 | 103.06 |
| power_smc_wrapper | 0.389 | 0.389 | 20 | 51.45 |
| power_smc_internal_no_cow | 0.406 | 0.406 | 20 | 49.31 |
| power_smc_internal_cow | 0.401 | 0.401 | 20 | 49.93 |

## Accuracy

| Run | Exact match | Pass@1 | EM rate |
|---|---:|---:|---:|
| baseline_single | 0/1 | 0.000 | 0.000 |
| baseline_particles | 0/1 | 0.000 | 0.000 |
| best_of_n | 0/1 | 0.000 | 0.000 |
| weighted_best_of_n | 0/1 | 0.000 | 0.000 |
| power_smc_wrapper | 0/1 | 0.000 | 0.000 |
| power_smc_internal_no_cow | 0/1 | 0.000 | 0.000 |
| power_smc_internal_cow | 0/1 | 0.000 | 0.000 |

## GPU Memory

| Run | Available | Samples | Before total MiB | Peak total MiB | Peak delta MiB | After total MiB |
|---|---|---:|---:|---:|---:|---:|
| baseline_single | yes | 8 | 74310 | 74312 | 2 | 74312 |
| baseline_particles | yes | 7 | 74312 | 74312 | 0 | 74312 |
| best_of_n | yes | 15 | 74312 | 74312 | 0 | 74312 |
| weighted_best_of_n | yes | 7 | 74312 | 74312 | 0 | 74312 |
| power_smc_wrapper | yes | 7 | 74312 | 74312 | 0 | 74312 |
| power_smc_internal_no_cow | yes | 8 | 74312 | 74312 | 0 | 74312 |
| power_smc_internal_cow | yes | 7 | 74312 | 74312 | 0 | 74312 |

## KV Reuse Mode

| Run | Mode |
|---|---|
| baseline_single | `none` |
| baseline_particles | `none` |
| best_of_n | `none` |
| weighted_best_of_n | `none` |
| power_smc_wrapper | `public_api_prefix_cache` |
| power_smc_internal_no_cow | `scheduler_reset_recompute` |
| power_smc_internal_cow | `scheduler_snapshot_alias_replay_with_reset_fallback` |

## Power-SMC Diagnostics

| Run | Diagnostics | Missing | Mean final ESS | Total resamples | Max resamples | Mean unique ancestors | KV aliases | KV fallbacks | KV aliased tokens | Chosen particles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| power_smc_wrapper | 1/1 | 0 | 1.000 | 1 | 1 | 1.000 | 0 | 0 | 0 | {"1": 1} |
| power_smc_internal_no_cow | 1/1 | 0 | 1.970 | 0 | 0 | - | 0 | 0 | 0 | {"1": 1} |
| power_smc_internal_cow | 1/1 | 0 | 2.000 | 1 | 1 | 2.000 | 2 | 0 | 64 | {"1": 1} |

## Notes

- `baseline_single` is ordinary vLLM sampling with `n=1`.
- `baseline_particles` samples `n=particles` independent completions.
- `best_of_n` selects the independent completion with maximum sampled
  sequence logprob.
- `weighted_best_of_n` samples one independent completion with weights
  proportional to `p(y)^(alpha-1)`.
- `power_smc_wrapper` uses public vLLM APIs with exact `q=p` weights.
- `power_smc_internal_no_cow` uses the V1 engine mode with
  reset/recompute after resampling.
- `power_smc_internal_cow` uses the V1 engine mode with
  power-temperature proposal, sampled-token base/proposal logprobs,
  diagnostics, and replay-safe full-block KV aliasing where safe.
- KV-cache aliasing is implemented for replay-safe full-block
  prefixes; partial-block copy and larger memory-savings benchmarks
  remain future work.
- GPU memory is sampled with `nvidia-smi` and is a node-level
  approximation; it may include other users on shared GPUs.
