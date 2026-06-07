# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `3`
- Max tokens: `128`
- Particles: `8`
- Block size: `16`
- Alpha: `4.0`
- Attention backend: `FLASHINFER`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 2.239 | 2.264 | 384 | 57.16 |
| baseline_particles | 2.356 | 2.440 | 3017 | 426.94 |
| best_of_n | 2.714 | 2.959 | 2878 | 353.44 |
| weighted_best_of_n | 2.494 | 2.570 | 2924 | 390.75 |
| power_smc_wrapper | 2.525 | 2.574 | 384 | 50.69 |
| power_smc_internal_no_cow | 2.472 | 2.552 | 279 | 37.63 |
| power_smc_internal_cow | 2.483 | 2.573 | 384 | 51.55 |

## Accuracy

| Run | Exact match | Pass@1 | EM rate |
|---|---:|---:|---:|
| baseline_single | 1/3 | 0.333 | 0.333 |
| baseline_particles | 1/3 | 0.333 | 0.333 |
| best_of_n | 2/3 | 0.667 | 0.667 |
| weighted_best_of_n | 2/3 | 0.667 | 0.667 |
| power_smc_wrapper | 0/3 | 0.000 | 0.000 |
| power_smc_internal_no_cow | 1/3 | 0.333 | 0.333 |
| power_smc_internal_cow | 0/3 | 0.000 | 0.000 |

## GPU Memory

| Run | Available | Samples | Before total MiB | Peak total MiB | Peak delta MiB | After total MiB |
|---|---|---:|---:|---:|---:|---:|
| baseline_single | yes | 93 | 74310 | 74312 | 2 | 74312 |
| baseline_particles | yes | 98 | 74312 | 74312 | 0 | 74312 |
| best_of_n | yes | 114 | 74312 | 74312 | 0 | 74312 |
| weighted_best_of_n | yes | 103 | 74312 | 74312 | 0 | 74312 |
| power_smc_wrapper | yes | 105 | 74312 | 74312 | 0 | 74312 |
| power_smc_internal_no_cow | yes | 103 | 74312 | 74312 | 0 | 74312 |
| power_smc_internal_cow | yes | 103 | 74312 | 74312 | 0 | 74312 |

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

| Run | Diagnostics | Missing | Mean final ESS | Total resamples | Max resamples | Mean unique ancestors | Chosen particles |
|---|---:|---:|---:|---:|---:|---:|---|
| power_smc_wrapper | 3/3 | 0 | 4.244 | 15 | 7 | 1.800 | {"2": 1, "7": 2} |
| power_smc_internal_no_cow | 3/3 | 0 | 2.650 | 2 | 2 | 2.500 | {"0": 1, "3": 1, "6": 1} |
| power_smc_internal_cow | 3/3 | 0 | 1.000 | 0 | 0 | - | {"0": 2, "7": 1} |

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
