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
| baseline_single | 0.581 | 0.581 | 8 | 13.77 |
| baseline_particles | 0.197 | 0.197 | 16 | 81.21 |
| power_smc_wrapper | 1.324 | 1.324 | 8 | 6.04 |
| power_smc_internal | 0.157 | 0.157 | 8 | 50.93 |

## Notes

- `baseline_single` is ordinary vLLM sampling with `n=1`.
- `baseline_particles` samples `n=particles` independent completions.
- `power_smc_wrapper` uses public vLLM APIs with exact `q=p` weights.
- `power_smc_internal` uses the V1 engine mode with power-temperature
  proposal, sampled-token base/proposal logprobs, diagnostics, and
  conservative block-boundary reset/recompute after resampling.
- KV-cache CoW/aliasing is not implemented yet.
