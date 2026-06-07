# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `1`
- Max tokens: `16`
- Particles: `8`
- Block size: `16`
- Alpha: `4.0`
- Ignore EOS: `False`
- Stop token IDs: `[13]`
- Attention backend: `FLASHINFER`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 0.362 | 0.362 | 16 | 44.22 |
| baseline_particles | 0.349 | 0.349 | 118 | 337.84 |
| best_of_n | 0.835 | 0.835 | 128 | 153.23 |
| weighted_best_of_n | 0.341 | 0.341 | 119 | 348.60 |
| power_smc_wrapper | 0.317 | 0.317 | 16 | 50.43 |
| power_smc_internal_no_cow | 0.340 | 0.340 | 14 | 41.15 |
| power_smc_internal_cow | 0.338 | 0.338 | 14 | 41.40 |

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
| baseline_single | yes | 7 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 7 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 13 | 49976 | 49976 | 0 | 49976 |
| weighted_best_of_n | yes | 6 | 49976 | 49976 | 0 | 49976 |
| power_smc_wrapper | yes | 6 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_no_cow | yes | 6 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_cow | yes | 6 | 49976 | 49976 | 0 | 49976 |

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
| power_smc_wrapper | 1/1 | 0 | 1.858 | 0 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0-0 | {} | {} | {"6": 1} |
| power_smc_internal_no_cow | 1/1 | 0 | 1.040 | 0 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8.000 | 8 | 14-16 | {"length": 6, "stop": 2} | {"done_particle": 2, "not_block_boundary": 14} | {"7": 1} |
| power_smc_internal_cow | 1/1 | 0 | 1.122 | 0 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8.000 | 8 | 14-16 | {"length": 6, "stop": 2} | {"done_particle": 2, "not_block_boundary": 14} | {"7": 1} |

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
