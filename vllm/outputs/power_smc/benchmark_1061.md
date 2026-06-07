# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `3`
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
| baseline_single | 2.243 | 2.318 | 384 | 57.07 |
| baseline_particles | 2.377 | 2.393 | 6144 | 861.70 |
| best_of_n | 2.632 | 2.937 | 6144 | 778.19 |
| weighted_best_of_n | 2.460 | 2.469 | 6144 | 832.64 |
| power_smc_wrapper | 2.651 | 2.662 | 384 | 48.28 |
| power_smc_internal_no_cow | 2.536 | 2.536 | 384 | 50.47 |
| power_smc_internal_cow | 2.537 | 2.544 | 384 | 50.45 |

## Accuracy

| Run | Exact match | Pass@1 | EM rate |
|---|---:|---:|---:|
| baseline_single | 0/3 | 0.000 | 0.000 |
| baseline_particles | 1/3 | 0.333 | 0.333 |
| best_of_n | 0/3 | 0.000 | 0.000 |
| weighted_best_of_n | 0/3 | 0.000 | 0.000 |
| power_smc_wrapper | 1/3 | 0.333 | 0.333 |
| power_smc_internal_no_cow | 0/3 | 0.000 | 0.000 |
| power_smc_internal_cow | 0/3 | 0.000 | 0.000 |

## GPU Memory

| Run | Available | Samples | Before total MiB | Peak total MiB | Peak delta MiB | After total MiB |
|---|---|---:|---:|---:|---:|---:|
| baseline_single | yes | 95 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 101 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 112 | 49976 | 49976 | 0 | 49976 |
| weighted_best_of_n | yes | 104 | 49976 | 49976 | 0 | 49976 |
| power_smc_wrapper | yes | 111 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_no_cow | yes | 107 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_cow | yes | 107 | 49976 | 49976 | 0 | 49976 |

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
| power_smc_wrapper | 3/3 | 0 | 3.929 | 21 | 7 | 6.429 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0-0 | {} | {} | {} | {"15": 1, "3": 1, "4": 1} |
| power_smc_internal_no_cow | 3/3 | 0 | 11.629 | 17 | 6 | 9.294 | 0 | 272 | 0 | 0 | 0 | 0 | 0 | 241363 | 52 | 241311 | 16.000 | 16 | 128-128 | {"length": 48} | {} | {"ess_above_threshold": 4, "not_block_boundary": 48, "stale_block_boundary": 318} | {"12": 2, "6": 1} |
| power_smc_internal_cow | 3/3 | 0 | 13.824 | 16 | 6 | 9.688 | 256 | 0 | 1152 | 18432 | 691 | 461 | 7376 | 241363 | 117 | 241246 | 16.000 | 16 | 128-128 | {"length": 48} | {} | {"ess_above_threshold": 5, "not_block_boundary": 48, "stale_block_boundary": 318} | {"12": 1, "13": 1, "8": 1} |

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
