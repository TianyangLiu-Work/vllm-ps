# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `3`
- Max tokens: `64`
- Particles: `8`
- Block size: `16`
- Alpha: `4.0`
- Ignore EOS: `True`
- Stop token IDs: `[]`
- Attention backend: `FLASHINFER`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 1.083 | 1.131 | 192 | 59.09 |
| baseline_particles | 1.155 | 1.163 | 1536 | 443.27 |
| best_of_n | 1.393 | 1.698 | 1536 | 367.47 |
| weighted_best_of_n | 1.224 | 1.238 | 1536 | 418.23 |
| power_smc_wrapper | 1.311 | 1.324 | 192 | 48.84 |
| power_smc_internal_no_cow | 1.294 | 1.303 | 192 | 49.47 |
| power_smc_internal_cow | 1.297 | 1.307 | 192 | 49.33 |

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
| baseline_single | yes | 47 | 49974 | 49976 | 2 | 49976 |
| baseline_particles | yes | 50 | 49976 | 49976 | 0 | 49976 |
| best_of_n | yes | 59 | 49976 | 49976 | 0 | 49976 |
| weighted_best_of_n | yes | 52 | 49976 | 49976 | 0 | 49976 |
| power_smc_wrapper | yes | 55 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_no_cow | yes | 55 | 49976 | 49976 | 0 | 49976 |
| power_smc_internal_cow | yes | 55 | 49976 | 49976 | 0 | 49976 |

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
| power_smc_wrapper | 3/3 | 0 | 2.593 | 9 | 3 | 4.111 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0-0 | {} | {} | {} | {"2": 3} |
| power_smc_internal_no_cow | 3/3 | 0 | 4.541 | 9 | 3 | 6.889 | 0 | 72 | 0 | 0 | 0 | 0 | 0 | 241363 | 28 | 241335 | 8.000 | 8 | 64-64 | {"length": 24} | {} | {"not_block_boundary": 48, "stale_block_boundary": 138} | {"1": 1, "2": 1, "5": 1} |
| power_smc_internal_cow | 3/3 | 0 | 3.199 | 9 | 3 | 6.667 | 72 | 0 | 384 | 6144 | 312 | 72 | 1152 | 241363 | 37 | 241326 | 8.000 | 8 | 64-64 | {"length": 24} | {} | {"not_block_boundary": 48, "stale_block_boundary": 138} | {"1": 1, "2": 1, "6": 1} |

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
