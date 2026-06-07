# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `1`
- Max tokens: `64`
- Particles: `8`
- Block size: `16`
- Alpha: `16.0`
- Ignore EOS: `True`
- Stop token IDs: `[]`
- Attention backend: `FLASHINFER`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 1.142 | 1.142 | 64 | 56.06 |
| baseline_particles | 1.141 | 1.141 | 512 | 448.61 |
| best_of_n | 1.690 | 1.690 | 512 | 302.93 |
| weighted_best_of_n | 1.214 | 1.214 | 512 | 421.89 |
| power_smc_wrapper | 1.244 | 1.244 | 64 | 51.47 |
| power_smc_internal_no_cow | 1.291 | 1.291 | 64 | 49.57 |
| power_smc_internal_cow | 1.360 | 1.360 | 64 | 47.06 |

## Accuracy

| Run | Exact match | Pass@1 | EM rate |
|---|---:|---:|---:|
| baseline_single | 1/1 | 1.000 | 1.000 |
| baseline_particles | 0/1 | 0.000 | 0.000 |
| best_of_n | 0/1 | 0.000 | 0.000 |
| weighted_best_of_n | 0/1 | 0.000 | 0.000 |
| power_smc_wrapper | 0/1 | 0.000 | 0.000 |
| power_smc_internal_no_cow | 0/1 | 0.000 | 0.000 |
| power_smc_internal_cow | 0/1 | 0.000 | 0.000 |

## GPU Memory

| Run | Available | Samples | Before total MiB | Peak total MiB | Peak delta MiB | After total MiB |
|---|---|---:|---:|---:|---:|---:|
| baseline_single | yes | 17 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 17 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 25 | 49976 | 49976 | 0 | 49976 |
| weighted_best_of_n | yes | 18 | 49976 | 49976 | 0 | 49976 |
| power_smc_wrapper | yes | 19 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_no_cow | yes | 19 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_cow | yes | 21 | 49976 | 49976 | 0 | 49976 |

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
| power_smc_wrapper | 1/1 | 0 | 1.000 | 3 | 3 | 2.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0-0 | {} | {} | {} | {"1": 1} |
| power_smc_internal_no_cow | 1/1 | 0 | 2.730 | 1 | 1 | 5.000 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 241363 | 34 | 241329 | 8.000 | 8 | 64-64 | {"length": 8} | {} | {"ess_above_threshold": 2, "not_block_boundary": 15, "stale_block_boundary": 45} | {"1": 1} |
| power_smc_internal_cow | 1/1 | 0 | 1.481 | 3 | 3 | 5.333 | 24 | 0 | 72 | 1152 | 45 | 27 | 432 | 241363 | 28 | 241335 | 8.000 | 8 | 64-64 | {"length": 8} | {} | {"not_block_boundary": 16, "stale_block_boundary": 46} | {"0": 1} |

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
