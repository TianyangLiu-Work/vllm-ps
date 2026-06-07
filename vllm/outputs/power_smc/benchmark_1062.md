# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `3`
- Max tokens: `128`
- Particles: `32`
- Block size: `16`
- Alpha: `16.0`
- Ignore EOS: `True`
- Stop token IDs: `[]`
- Attention backend: `FLASHINFER`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 2.149 | 2.201 | 384 | 59.57 |
| baseline_particles | 2.377 | 2.407 | 12288 | 1722.88 |
| best_of_n | 2.652 | 2.951 | 12288 | 1544.63 |
| weighted_best_of_n | 2.520 | 2.533 | 12288 | 1625.53 |
| power_smc_wrapper | 2.755 | 2.871 | 384 | 46.46 |
| power_smc_internal_no_cow | 3.090 | 3.116 | 384 | 41.42 |
| power_smc_internal_cow | 2.627 | 2.685 | 384 | 48.72 |

## Accuracy

| Run | Exact match | Pass@1 | EM rate |
|---|---:|---:|---:|
| baseline_single | 0/3 | 0.000 | 0.000 |
| baseline_particles | 0/3 | 0.000 | 0.000 |
| best_of_n | 0/3 | 0.000 | 0.000 |
| weighted_best_of_n | 0/3 | 0.000 | 0.000 |
| power_smc_wrapper | 0/3 | 0.000 | 0.000 |
| power_smc_internal_no_cow | 0/3 | 0.000 | 0.000 |
| power_smc_internal_cow | 0/3 | 0.000 | 0.000 |

## GPU Memory

| Run | Available | Samples | Before total MiB | Peak total MiB | Peak delta MiB | After total MiB |
|---|---|---:|---:|---:|---:|---:|
| baseline_single | yes | 92 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 101 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 111 | 49976 | 49976 | 0 | 49976 |
| weighted_best_of_n | yes | 107 | 49976 | 49976 | 0 | 49976 |
| power_smc_wrapper | yes | 117 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_no_cow | yes | 132 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_cow | yes | 111 | 49976 | 49976 | 0 | 49976 |

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
| power_smc_wrapper | 3/3 | 0 | 25.333 | 21 | 7 | 12.619 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0-0 | {} | {} | {} | {"11": 1, "21": 1, "8": 1} |
| power_smc_internal_no_cow | 3/3 | 0 | 14.434 | 16 | 6 | 18.938 | 0 | 512 | 0 | 0 | 0 | 0 | 0 | 241363 | 130 | 241233 | 32.000 | 32 | 128-128 | {"length": 96} | {} | {"ess_above_threshold": 5, "not_block_boundary": 47, "stale_block_boundary": 317} | {"12": 1, "27": 1, "4": 1} |
| power_smc_internal_cow | 3/3 | 0 | 24.768 | 16 | 6 | 18.562 | 512 | 0 | 2272 | 36352 | 1298 | 974 | 15584 | 241363 | 240 | 241123 | 32.000 | 32 | 128-128 | {"length": 96} | {} | {"ess_above_threshold": 5, "not_block_boundary": 48, "stale_block_boundary": 318} | {"12": 2, "27": 1} |

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
