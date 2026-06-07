# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `1`
- Max tokens: `8`
- Particles: `2`
- Block size: `16`
- Alpha: `2.0`
- Attention backend: `FLASHINFER`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 0.218 | 0.218 | 8 | 36.73 |
| baseline_particles | 0.150 | 0.150 | 16 | 106.88 |
| best_of_n | 0.642 | 0.642 | 16 | 24.91 |
| weighted_best_of_n | 0.159 | 0.159 | 16 | 100.84 |
| power_smc_wrapper | 0.159 | 0.159 | 8 | 50.25 |
| power_smc_internal | 0.157 | 0.157 | 8 | 50.90 |

## GPU Memory

| Run | Available | Samples | Before total MiB | Peak total MiB | Peak delta MiB | After total MiB |
|---|---|---:|---:|---:|---:|---:|
| baseline_single | yes | 5 | 74310 | 74312 | 2 | 74312 |
| baseline_particles | yes | 4 | 74312 | 74312 | 0 | 74312 |
| best_of_n | yes | 11 | 74312 | 74312 | 0 | 74312 |
| weighted_best_of_n | yes | 4 | 74312 | 74312 | 0 | 74312 |
| power_smc_wrapper | yes | 4 | 74312 | 74312 | 0 | 74312 |
| power_smc_internal | yes | 4 | 74312 | 74312 | 0 | 74312 |

## Power-SMC Diagnostics

| Run | Diagnostics | Missing | Mean final ESS | Total resamples | Max resamples | Mean unique ancestors | Chosen particles |
|---|---:|---:|---:|---:|---:|---:|---|
| power_smc_wrapper | 1/1 | 0 | 1.934 | 0 | 0 | - | {"1": 1} |
| power_smc_internal | 1/1 | 0 | 1.926 | 0 | 0 | - | {"1": 1} |

## Notes

- `baseline_single` is ordinary vLLM sampling with `n=1`.
- `baseline_particles` samples `n=particles` independent completions.
- `best_of_n` selects the independent completion with maximum sampled
  sequence logprob.
- `weighted_best_of_n` samples one independent completion with weights
  proportional to `p(y)^(alpha-1)`.
- `power_smc_wrapper` uses public vLLM APIs with exact `q=p` weights.
- `power_smc_internal` uses the V1 engine mode with power-temperature
  proposal, sampled-token base/proposal logprobs, diagnostics, and
  conservative block-boundary reset/recompute after resampling.
- KV-cache CoW/aliasing is not implemented yet.
- GPU memory is sampled with `nvidia-smi` and is a node-level
  approximation; it may include other users on shared GPUs.
