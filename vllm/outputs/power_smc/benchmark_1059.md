# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `1`
- Max tokens: `128`
- Particles: `16`
- Block size: `16`
- Alpha: `16.0`
- Ignore EOS: `True`
- Stop token IDs: `[]`
- Attention backend: `FLASHINFER`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 2.190 | 2.190 | 128 | 58.44 |
| baseline_particles | 2.398 | 2.398 | 2048 | 854.14 |
| best_of_n | 3.010 | 3.010 | 2048 | 680.41 |
| weighted_best_of_n | 2.516 | 2.516 | 2048 | 813.91 |
| power_smc_wrapper | 2.697 | 2.697 | 128 | 47.46 |
| power_smc_internal_no_cow | 2.576 | 2.576 | 128 | 49.70 |
| power_smc_internal_cow | 2.502 | 2.502 | 128 | 51.17 |

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
| baseline_single | yes | 32 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 34 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 43 | 49976 | 49976 | 0 | 49976 |
| weighted_best_of_n | yes | 36 | 49976 | 49976 | 0 | 49976 |
| power_smc_wrapper | yes | 38 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_no_cow | yes | 37 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_cow | yes | 36 | 49976 | 49976 | 0 | 49976 |

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

| Run | Diagnostics | Missing | Mean final ESS | Total resamples | Max resamples | Mean unique ancestors | KV aliases | KV fallbacks | KV aliased blocks | KV aliased tokens | KV physical blocks | KV saved blocks | KV saved tokens | KV pool total blocks | KV pool max used | KV pool min free | Mean done | Max done | Particle len range | Finish reasons | Stop reasons | Resample skips | Chosen particles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| power_smc_wrapper | 1/1 | 0 | 15.631 | 7 | 7 | 2.571 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0-0 | {} | {} | {} | {"5": 1} |
| power_smc_internal_no_cow | 1/1 | 0 | 16.000 | 5 | 5 | 8.000 | 0 | 80 | 0 | 0 | 0 | 0 | 0 | 241363 | 39 | 241324 | 16.000 | 16 | 128-128 | {"length": 16} | {} | {"ess_above_threshold": 2, "not_block_boundary": 16, "stale_block_boundary": 106} | {"6": 1} |
| power_smc_internal_cow | 1/1 | 0 | 16.000 | 5 | 5 | 6.800 | 80 | 0 | 320 | 5120 | 145 | 175 | 2800 | 241363 | 44 | 241319 | 16.000 | 16 | 128-128 | {"length": 16} | {} | {"ess_above_threshold": 2, "not_block_boundary": 16, "stale_block_boundary": 106} | {"6": 1} |

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
