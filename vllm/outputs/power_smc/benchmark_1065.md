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
| baseline_single | 2.606 | 2.649 | 384 | 49.12 |
| baseline_particles | 2.460 | 2.540 | 12288 | 1664.98 |
| best_of_n | 2.664 | 2.973 | 12288 | 1537.57 |
| weighted_best_of_n | 2.528 | 2.576 | 12288 | 1620.02 |
| power_smc_wrapper | 2.717 | 2.758 | 384 | 47.11 |
| power_smc_internal_no_cow | 2.628 | 2.678 | 384 | 48.71 |
| power_smc_internal_cow | 2.614 | 2.618 | 384 | 48.96 |

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
| baseline_single | yes | 110 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 104 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 113 | 49976 | 49978 | 2 | 49978 |
| weighted_best_of_n | yes | 107 | 49978 | 49978 | 0 | 49978 |
| power_smc_wrapper | yes | 116 | 49978 | 49978 | 0 | 49978 |
| power_smc_internal_no_cow | yes | 113 | 49978 | 49978 | 0 | 49978 |
| power_smc_internal_cow | yes | 110 | 49978 | 49978 | 0 | 49978 |

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
| power_smc_wrapper | 3/3 | 0 | 22.355 | 21 | 7 | 15.190 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0-0 | {} | {} | {} | {"19": 1, "21": 1, "9": 1} |
| power_smc_internal_no_cow | 3/3 | 0 | 32.000 | 13 | 6 | 24.077 | 0 | 416 | 0 | 0 | 0 | 0 | 0 | 241363 | 99 | 241264 | 32.000 | 32 | 128-128 | {"length": 96} | {} | {"ess_above_threshold": 8, "not_block_boundary": 47, "stale_block_boundary": 317} | {"12": 1, "13": 1, "25": 1} |
| power_smc_internal_cow | 3/3 | 0 | 32.000 | 15 | 5 | 26.533 | 480 | 0 | 3040 | 48640 | 2576 | 464 | 7424 | 241363 | 195 | 241168 | 32.000 | 32 | 128-128 | {"length": 96} | {} | {"ess_above_threshold": 6, "not_block_boundary": 48, "stale_block_boundary": 318} | {"12": 3} |

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
