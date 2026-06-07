# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `1`
- Max tokens: `8`
- Particles: `2`
- Block size: `4`
- Alpha: `2.0`

## Throughput

| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|---:|
| baseline_single | 0.302 | 0.302 | 8 | 26.49 |
| baseline_particles | 0.138 | 0.138 | 16 | 115.53 |
| power_smc | 3.083 | 3.083 | 8 | 2.59 |

## Notes

- `baseline_single` is ordinary vLLM sampling with `n=1`.
- `baseline_particles` samples `n=particles` independent completions.
- `power_smc` uses block-level SMC with exact `q=p` importance weights.
- Current Power-SMC path uses public vLLM APIs and prefix-cache reuse;
  adaptive-temperature proposal and KV-cache CoW need engine-level hooks.
