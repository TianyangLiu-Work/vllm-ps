# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `1`
- Max tokens: `64`
- Particles: `8`
- Block size: `16`
- Alpha: `16.0`
- Ignore EOS: `True`
- Attention backend: `FLASHINFER`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 1.153 | 1.153 | 64 | 55.50 |
| baseline_particles | 1.213 | 1.213 | 512 | 422.13 |
| best_of_n | 1.808 | 1.808 | 512 | 283.16 |
| weighted_best_of_n | 1.251 | 1.251 | 512 | 409.29 |
| power_smc_wrapper | 1.262 | 1.262 | 64 | 50.70 |
| power_smc_internal_no_cow | 1.267 | 1.267 | 64 | 50.50 |
| power_smc_internal_cow | 1.270 | 1.270 | 64 | 50.41 |

## Accuracy

| Run | Exact match | Pass@1 | EM rate |
|---|---:|---:|---:|
| baseline_single | 1/1 | 1.000 | 1.000 |
| baseline_particles | 0/1 | 0.000 | 0.000 |
| best_of_n | 0/1 | 0.000 | 0.000 |
| weighted_best_of_n | 1/1 | 1.000 | 1.000 |
| power_smc_wrapper | 0/1 | 0.000 | 0.000 |
| power_smc_internal_no_cow | 0/1 | 0.000 | 0.000 |
| power_smc_internal_cow | 0/1 | 0.000 | 0.000 |

## GPU Memory

| Run | Available | Samples | Before total MiB | Peak total MiB | Peak delta MiB | After total MiB |
|---|---|---:|---:|---:|---:|---:|
| baseline_single | yes | 18 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 18 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 27 | 49976 | 49976 | 0 | 49976 |
| weighted_best_of_n | yes | 19 | 49976 | 49976 | 0 | 49976 |
| power_smc_wrapper | yes | 19 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_no_cow | yes | 19 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_cow | yes | 19 | 49976 | 49976 | 0 | 49976 |

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

| Run | Diagnostics | Missing | Mean final ESS | Total resamples | Max resamples | Mean unique ancestors | KV aliases | KV fallbacks | KV aliased blocks | KV aliased tokens | KV physical blocks | KV saved blocks | KV saved tokens | Mean done | Max done | Particle len range | Finish reasons | Resample skips | Chosen particles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| power_smc_wrapper | 1/1 | 0 | 2.056 | 3 | 3 | 1.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0-0 | {} | {} | {"0": 1} |
| power_smc_internal_no_cow | 1/1 | 0 | 5.000 | 3 | 3 | 4.667 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 8.000 | 8 | 64-64 | {"length": 8} | {"not_block_boundary": 16, "stale_block_boundary": 46} | {"4": 1} |
| power_smc_internal_cow | 1/1 | 0 | 1.000 | 3 | 3 | 4.000 | 24 | 0 | 72 | 1152 | 31 | 41 | 656 | 8.000 | 8 | 64-64 | {"length": 8} | {"not_block_boundary": 16, "stale_block_boundary": 46} | {"7": 1} |

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
