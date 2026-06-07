# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `1`
- Max tokens: `96`
- Particles: `8`
- Block size: `16`
- Alpha: `4.0`
- Ignore EOS: `True`
- Attention backend: `FLASHINFER`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 1.720 | 1.720 | 96 | 55.81 |
| baseline_particles | 1.778 | 1.778 | 768 | 431.84 |
| best_of_n | 2.321 | 2.321 | 768 | 330.95 |
| weighted_best_of_n | 1.848 | 1.848 | 768 | 415.49 |
| power_smc_wrapper | 1.963 | 1.963 | 96 | 48.92 |
| power_smc_internal_no_cow | 1.843 | 1.843 | 96 | 52.08 |
| power_smc_internal_cow | 1.863 | 1.863 | 96 | 51.54 |

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
| baseline_single | yes | 26 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 26 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 34 | 49976 | 49976 | 0 | 49976 |
| weighted_best_of_n | yes | 27 | 49976 | 49976 | 0 | 49976 |
| power_smc_wrapper | yes | 29 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_no_cow | yes | 27 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_cow | yes | 28 | 49976 | 49976 | 0 | 49976 |

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

| Run | Diagnostics | Missing | Mean final ESS | Total resamples | Max resamples | Mean unique ancestors | KV aliases | KV fallbacks | KV aliased blocks | KV aliased tokens | Chosen particles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| power_smc_wrapper | 1/1 | 0 | 4.999 | 5 | 5 | 3.000 | 0 | 0 | 0 | 0 | {"2": 1} |
| power_smc_internal_no_cow | 1/1 | 0 | 1.000 | 0 | 0 | - | 0 | 0 | 0 | 0 | {"0": 1} |
| power_smc_internal_cow | 1/1 | 0 | 1.000 | 0 | 0 | - | 0 | 0 | 0 | 0 | {"0": 1} |

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
