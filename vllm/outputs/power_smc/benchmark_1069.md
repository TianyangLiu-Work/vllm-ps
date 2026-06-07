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
| baseline_single | 2.210 | 2.309 | 384 | 57.92 |
| baseline_particles | 2.343 | 2.393 | 6144 | 874.15 |
| best_of_n | 2.606 | 2.901 | 6144 | 785.86 |
| weighted_best_of_n | 2.459 | 2.473 | 6144 | 833.00 |
| power_smc_wrapper | 2.618 | 2.670 | 384 | 48.89 |
| power_smc_internal_no_cow | 2.527 | 2.584 | 384 | 50.66 |
| power_smc_internal_cow | 2.486 | 2.503 | 384 | 51.49 |

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
| baseline_single | yes | 93 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 98 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 109 | 49976 | 49976 | 0 | 49976 |
| weighted_best_of_n | yes | 103 | 49976 | 49976 | 0 | 49976 |
| power_smc_wrapper | yes | 108 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_no_cow | yes | 104 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_cow | yes | 103 | 49976 | 49976 | 0 | 49976 |

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

| Run | Diagnostics | Missing | Mean final ESS | Total resamples | Max resamples | Mean unique ancestors | Maybe checks | Boundary checks | KV aliases | KV fallbacks | KV snapshots | KV alias attempts | KV replay tokens | KV identity noops | KV aliased blocks | KV aliased tokens | KV physical blocks | KV saved blocks | KV saved tokens | KV pool total blocks | KV pool max used | KV pool min free | Mean done | Max done | Particle len range | Finish reasons | Stop reasons | Resample skips | Chosen particles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| power_smc_wrapper | 3/3 | 0 | 12.613 | 21 | 7 | 7.810 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0-0 | {} | {} | {} | {"11": 1, "4": 1, "5": 1} |
| power_smc_internal_no_cow | 3/3 | 0 | 16.000 | 17 | 7 | 11.882 | 27 | 21 | 0 | 272 | 0 | 0 | 32672 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16.000 | 16 | 128-128 | {"length": 48} | {} | {"ess_above_threshold": 4, "not_block_boundary": 3, "stale_block_boundary": 3} | {"4": 1, "6": 2} |
| power_smc_internal_cow | 3/3 | 0 | 15.850 | 19 | 7 | 12.632 | 27 | 21 | 304 | 0 | 240 | 304 | 4448 | 0 | 2192 | 35072 | 1734 | 458 | 7328 | 0 | 0 | 0 | 16.000 | 16 | 128-128 | {"length": 48} | {} | {"ess_above_threshold": 2, "not_block_boundary": 3, "stale_block_boundary": 3} | {"4": 1, "5": 1, "6": 1} |

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
