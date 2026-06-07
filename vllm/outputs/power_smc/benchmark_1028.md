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
| baseline_single | 0.229 | 0.229 | 8 | 34.92 |
| baseline_particles | 0.199 | 0.199 | 16 | 80.21 |
| power_smc_wrapper | 0.634 | 0.634 | 8 | 12.62 |
| power_smc_internal | 0.169 | 0.169 | 8 | 47.25 |

## Power-SMC Diagnostics

| Run | Diagnostics | Missing | Mean final ESS | Total resamples | Max resamples | Mean unique ancestors | Chosen particles |
|---|---:|---:|---:|---:|---:|---:|---|
| power_smc_wrapper | 1/1 | 0 | - | 0 | 0 | - | {} |
| power_smc_internal | 1/1 | 0 | 1.926 | 0 | 0 | - | {"1": 1} |

## Notes

- `baseline_single` is ordinary vLLM sampling with `n=1`.
- `baseline_particles` samples `n=particles` independent completions.
- `power_smc_wrapper` uses public vLLM APIs with exact `q=p` weights.
- `power_smc_internal` uses the V1 engine mode with power-temperature
  proposal, sampled-token base/proposal logprobs, diagnostics, and
  conservative block-boundary reset/recompute after resampling.
- KV-cache CoW/aliasing is not implemented yet.
