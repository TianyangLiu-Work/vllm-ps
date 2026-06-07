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
| baseline_single | 2.191 | 2.245 | 384 | 58.41 |
| baseline_particles | 2.406 | 2.481 | 6144 | 851.18 |
| best_of_n | 2.639 | 2.982 | 6144 | 776.19 |
| weighted_best_of_n | 2.442 | 2.519 | 6144 | 838.72 |
| power_smc_wrapper | 2.598 | 2.633 | 384 | 49.27 |
| power_smc_internal_no_cow | 2.519 | 2.550 | 384 | 50.82 |
| power_smc_internal_cow | 2.516 | 2.535 | 384 | 50.88 |

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
| baseline_single | yes | 91 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 99 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 110 | 49976 | 49976 | 0 | 49976 |
| weighted_best_of_n | yes | 102 | 49976 | 49976 | 0 | 49976 |
| power_smc_wrapper | yes | 108 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_no_cow | yes | 106 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_cow | yes | 105 | 49976 | 49976 | 0 | 49976 |

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

| Run | Diagnostics | Missing | Mean final ESS | Total resamples | Max resamples | Mean unique ancestors | KV aliases | KV fallbacks | KV snapshots | KV alias attempts | KV replay tokens | KV identity noops | KV aliased blocks | KV aliased tokens | KV physical blocks | KV saved blocks | KV saved tokens | KV pool total blocks | KV pool max used | KV pool min free | Mean done | Max done | Particle len range | Finish reasons | Stop reasons | Resample skips | Chosen particles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| power_smc_wrapper | 3/3 | 0 | 11.000 | 21 | 7 | 8.190 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0-0 | {} | {} | {} | {"10": 1, "15": 1, "4": 1} |
| power_smc_internal_no_cow | 3/3 | 0 | 16.000 | 17 | 7 | 11.765 | 0 | 272 | 0 | 0 | 34688 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16.000 | 16 | 128-128 | {"length": 48} | {} | {"ess_above_threshold": 4, "not_block_boundary": 48, "stale_block_boundary": 318} | {"4": 1, "6": 2} |
| power_smc_internal_cow | 3/3 | 0 | 16.000 | 12 | 5 | 13.750 | 192 | 0 | 165 | 192 | 2800 | 0 | 1264 | 20224 | 1109 | 155 | 2480 | 0 | 0 | 0 | 16.000 | 16 | 128-128 | {"length": 48} | {} | {"ess_above_threshold": 9, "not_block_boundary": 47, "stale_block_boundary": 317} | {"6": 3} |

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
