# Power-SMC vLLM Benchmark Report

This report summarizes the current Power-SMC V1 prototype measurements from
Slurm runs on June 7, 2026. The implementation is benchmarked through
`examples/generate/benchmark_power_smc.py` with the managed environment in
`scripts/power_smc_env.sh`.

## Environment

- vLLM source tree: `/home/tyliu/ghworkspace/vllm-ps/vllm`
- Conda environment: `/data/conda_envs/power-smc-vllm`
- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Attention backend: `FLASHINFER`
- GPU mode: Slurm `gpushare:1`
- Engine settings: V1, eager mode, prefix caching enabled,
  `logprobs_mode="raw_logprobs"`

The Power-SMC reference repository is cloned at
`/home/tyliu/ghworkspace/vllm-ps/Power-SMC`.

Broader benchmark sweeps can be submitted with
`scripts/slurm/power_smc_sweep.sh`. The sweep wrapper launches several
`power_smc_benchmark.sbatch` jobs, defaults to sequential Slurm dependencies
to avoid concurrent vLLM KV-cache preallocation on the same shared GPU, and
records a TSV manifest under `outputs/power_smc/sweep_<tag>.tsv` so each result
JSON/Markdown file can be matched back to its configuration. Use `DRY_RUN=1`
to validate manifest generation without submitting jobs.

## Compared Methods

- `baseline_single`: ordinary vLLM sampling with `n=1`.
- `baseline_particles`: ordinary vLLM sampling with `n=particles`.
- `best_of_n`: independent particles, select maximum sampled sequence logprob.
- `weighted_best_of_n`: independent particles, sample by weights proportional
  to `p(y)^(alpha-1)`.
- `power_smc_wrapper`: public API wrapper using repeated vLLM calls and
  exact `q=p` weights.
- `power_smc_internal_no_cow`: V1 internal Power-SMC with reset/recompute after
  resampling.
- `power_smc_internal_cow`: V1 internal Power-SMC with scheduler-level
  full-block KV aliasing where replay-safe.

## Medium Run

Source artifacts:

- JSON: `outputs/power_smc/benchmark_1036.json`
- Markdown: `outputs/power_smc/benchmark_1036.md`
- Slurm log: `outputs/slurm/power-smc-1036.out`

Configuration:

- Prompts: 3 built-in arithmetic/algebra prompts
- Max tokens: 128
- Particles: 8
- Block size: 16
- Alpha: 4.0
- Alpha ramp tokens: 16
- ESS threshold: 0.5

Throughput:

| Run | Mean latency (s) | Generated tokens | tok/s |
|---|---:|---:|---:|
| baseline_single | 2.239 | 384 | 57.16 |
| baseline_particles | 2.356 | 3017 | 426.94 |
| best_of_n | 2.714 | 2878 | 353.44 |
| weighted_best_of_n | 2.494 | 2924 | 390.75 |
| power_smc_wrapper | 2.525 | 384 | 50.69 |
| power_smc_internal_no_cow | 2.472 | 279 | 37.63 |
| power_smc_internal_cow | 2.483 | 384 | 51.55 |

Accuracy:

| Run | Exact match | Pass@1 | EM rate |
|---|---:|---:|---:|
| baseline_single | 1/3 | 0.333 | 0.333 |
| baseline_particles | 1/3 | 0.333 | 0.333 |
| best_of_n | 2/3 | 0.667 | 0.667 |
| weighted_best_of_n | 2/3 | 0.667 | 0.667 |
| power_smc_wrapper | 0/3 | 0.000 | 0.000 |
| power_smc_internal_no_cow | 1/3 | 0.333 | 0.333 |
| power_smc_internal_cow | 0/3 | 0.000 | 0.000 |

Power-SMC diagnostics:

| Run | Diagnostics | Mean final ESS | Total resamples | Max resamples | Mean unique ancestors |
|---|---:|---:|---:|---:|---:|
| power_smc_wrapper | 3/3 | 4.244 | 15 | 7 | 1.800 |
| power_smc_internal_no_cow | 3/3 | 2.650 | 2 | 2 | 2.500 |
| power_smc_internal_cow | 3/3 | 1.000 | 0 | 0 | - |

The medium run verifies that all compared methods complete under the same
benchmark harness and that the generated report includes latency, throughput,
node-level GPU memory samples, exact match, Pass@1, ESS, resampling counts, and
KV reuse mode. In this run, `power_smc_internal_cow` did not cross a resampling
boundary before completion for the sampled trajectories, so it validates the
internal cow mode execution path but not a cow resampling event.

## Particle Scale Smoke Runs

Source artifacts:

- Particles 16 JSON: `outputs/power_smc/benchmark_1041.json`
- Particles 16 Markdown: `outputs/power_smc/benchmark_1041.md`
- Particles 32 JSON: `outputs/power_smc/benchmark_1042.json`
- Particles 32 Markdown: `outputs/power_smc/benchmark_1042.md`

Configuration:

- Prompts: 1
- Max tokens: 32
- Block size: 16
- Alpha: 4.0
- Alpha ramp tokens: 1
- ESS threshold: 1.0
- GPU memory utilization: 0.50

The 16- and 32-particle runs verify that the internal decoding mode is
configurable at the MVP target particle counts and can complete end-to-end
under the same benchmark harness:

| Particles | Run | Mean latency (s) | Generated tokens | tok/s | Total resamples | KV aliases |
|---:|---|---:|---:|---:|---:|---:|
| 16 | baseline_single | 0.613 | 32 | 52.22 | - | - |
| 16 | baseline_particles | 0.616 | 512 | 831.46 | - | - |
| 16 | power_smc_internal_no_cow | 0.633 | 32 | 50.58 | 0 | 0 |
| 16 | power_smc_internal_cow | 0.633 | 32 | 50.52 | 0 | 0 |
| 32 | baseline_single | 0.697 | 32 | 45.94 | - | - |
| 32 | baseline_particles | 0.713 | 1024 | 1436.74 | - | - |
| 32 | power_smc_internal_no_cow | 0.689 | 32 | 46.42 | 0 | 0 |
| 32 | power_smc_internal_cow | 0.684 | 32 | 46.76 | 0 | 0 |

These smoke runs are not intended as quality measurements. They are short runs
for configurability and lifecycle coverage. They also show that the internal
Power-SMC path returns one user-visible sequence while running particle counts
of 16 and 32 internally.

An earlier attempt submitted particles 16 and 32 concurrently on the same GPU
(`benchmark_1039` and `benchmark_1040`) and failed during engine
initialization with CUDA OOM because both vLLM engines tried to preallocate KV
cache at the same time. The successful sequential reruns used
`GPU_MEMORY_UTILIZATION=0.50`.

## Built-In Prompt Sweep

Source artifacts:

- Manifest: `outputs/power_smc/sweep_20260607-sweep1.tsv`
- p8/t64/a4 JSON: `outputs/power_smc/benchmark_1060.json`
- p8/t64/a4 Markdown: `outputs/power_smc/benchmark_1060.md`
- p16/t128/a16 JSON: `outputs/power_smc/benchmark_1061.json`
- p16/t128/a16 Markdown: `outputs/power_smc/benchmark_1061.md`
- p32/t128/a16 JSON: `outputs/power_smc/benchmark_1062.json`
- p32/t128/a16 Markdown: `outputs/power_smc/benchmark_1062.md`

Configuration:

- Prompts: 3 built-in arithmetic/algebra prompts
- Block size: 16
- Alpha ramp tokens: 1
- ESS threshold: 1.0
- GPU memory utilization: 0.50
- `ignore_eos=True`
- Submitted by `scripts/slurm/power_smc_sweep.sh` with sequential Slurm
  dependencies

Throughput:

| Config | Run | Mean latency (s) | Generated tokens | tok/s |
|---|---|---:|---:|---:|
| p8/t64/a4 | baseline_single | 1.142 | 192 | 56.06 |
| p8/t64/a4 | baseline_particles | 1.197 | 1536 | 427.62 |
| p8/t64/a4 | power_smc_wrapper | 1.270 | 192 | 50.37 |
| p8/t64/a4 | power_smc_internal_no_cow | 1.242 | 192 | 51.52 |
| p8/t64/a4 | power_smc_internal_cow | 1.263 | 192 | 50.67 |
| p16/t128/a16 | baseline_single | 2.243 | 384 | 57.07 |
| p16/t128/a16 | baseline_particles | 2.377 | 6144 | 861.70 |
| p16/t128/a16 | power_smc_wrapper | 2.651 | 384 | 48.28 |
| p16/t128/a16 | power_smc_internal_no_cow | 2.536 | 384 | 50.47 |
| p16/t128/a16 | power_smc_internal_cow | 2.537 | 384 | 50.45 |
| p32/t128/a16 | baseline_single | 2.149 | 384 | 59.57 |
| p32/t128/a16 | baseline_particles | 2.377 | 12288 | 1722.88 |
| p32/t128/a16 | power_smc_wrapper | 2.755 | 384 | 46.46 |
| p32/t128/a16 | power_smc_internal_no_cow | 3.090 | 384 | 41.42 |
| p32/t128/a16 | power_smc_internal_cow | 2.627 | 384 | 48.72 |

Power-SMC diagnostics:

| Config | Mode | Total resamples | KV aliases | KV fallbacks | KV aliased blocks | KV physical blocks | KV saved blocks | KV saved tokens | KV pool max used |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p8/t64/a4 | internal no-cow | 9 | 0 | 72 | 0 | 0 | 0 | 0 | 22 |
| p8/t64/a4 | internal cow | 9 | 72 | 0 | 216 | 121 | 95 | 1520 | 28 |
| p16/t128/a16 | internal no-cow | 17 | 0 | 272 | 0 | 0 | 0 | 0 | 52 |
| p16/t128/a16 | internal cow | 16 | 256 | 0 | 1152 | 691 | 461 | 7376 | 117 |
| p32/t128/a16 | internal no-cow | 16 | 0 | 512 | 0 | 0 | 0 | 0 | 130 |
| p32/t128/a16 | internal cow | 16 | 512 | 0 | 2272 | 1298 | 974 | 15584 | 240 |

This sweep is the strongest current throughput/KV tradeoff evidence over the
built-in prompts. Internal cow is below the public wrapper latency in all three
configs and records positive full-block savings after repeated resampling. At
32 particles, cow is also faster than the reset/recompute internal path while
saving 974 full KV blocks in scheduler accounting. The `kv_pool_max_used`
counter is higher for cow in these runs because aliasing keeps more live shared
full blocks visible during resampling windows; the scheduler block accounting
still shows fewer unique physical blocks than logical child prefixes.

## MATH500 Prompt Sweep

Source artifacts:

- Prompt JSONL: `outputs/power_smc/math500_prompts_3.jsonl`
- Manifest: `outputs/power_smc/sweep_20260607-math500-sweep1.tsv`
- p8/t64/a4 JSON: `outputs/power_smc/benchmark_1063.json`
- p8/t64/a4 Markdown: `outputs/power_smc/benchmark_1063.md`
- p16/t128/a16 JSON: `outputs/power_smc/benchmark_1064.json`
- p16/t128/a16 Markdown: `outputs/power_smc/benchmark_1064.md`
- p32/t128/a16 JSON: `outputs/power_smc/benchmark_1065.json`
- p32/t128/a16 Markdown: `outputs/power_smc/benchmark_1065.md`

Configuration:

- Prompts: first 3 rows from
  `/home/tyliu/ghworkspace/vllm-ps/Power-SMC/data/MATH500.json`
- Block size: 16
- Alpha ramp tokens: 1
- ESS threshold: 1.0
- GPU memory utilization: 0.50
- `ignore_eos=True`
- Submitted by `scripts/slurm/power_smc_sweep.sh` with sequential Slurm
  dependencies

Throughput:

| Config | Run | Mean latency (s) | Generated tokens | tok/s |
|---|---|---:|---:|---:|
| p8/t64/a4 | baseline_single | 1.083 | 192 | 59.09 |
| p8/t64/a4 | baseline_particles | 1.155 | 1536 | 443.27 |
| p8/t64/a4 | power_smc_wrapper | 1.311 | 192 | 48.84 |
| p8/t64/a4 | power_smc_internal_no_cow | 1.294 | 192 | 49.47 |
| p8/t64/a4 | power_smc_internal_cow | 1.297 | 192 | 49.33 |
| p16/t128/a16 | baseline_single | 2.163 | 384 | 59.17 |
| p16/t128/a16 | baseline_particles | 2.388 | 6144 | 857.80 |
| p16/t128/a16 | power_smc_wrapper | 2.673 | 384 | 47.88 |
| p16/t128/a16 | power_smc_internal_no_cow | 2.540 | 384 | 50.39 |
| p16/t128/a16 | power_smc_internal_cow | 2.640 | 384 | 48.48 |
| p32/t128/a16 | baseline_single | 2.606 | 384 | 49.12 |
| p32/t128/a16 | baseline_particles | 2.460 | 12288 | 1664.98 |
| p32/t128/a16 | power_smc_wrapper | 2.717 | 384 | 47.11 |
| p32/t128/a16 | power_smc_internal_no_cow | 2.628 | 384 | 48.71 |
| p32/t128/a16 | power_smc_internal_cow | 2.614 | 384 | 48.96 |

Power-SMC diagnostics:

| Config | Mode | Total resamples | KV aliases | KV fallbacks | KV aliased blocks | KV physical blocks | KV saved blocks | KV saved tokens | KV pool max used |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p8/t64/a4 | internal no-cow | 9 | 0 | 72 | 0 | 0 | 0 | 0 | 28 |
| p8/t64/a4 | internal cow | 9 | 72 | 0 | 384 | 312 | 72 | 1152 | 37 |
| p16/t128/a16 | internal no-cow | 17 | 0 | 272 | 0 | 0 | 0 | 0 | 44 |
| p16/t128/a16 | internal cow | 17 | 272 | 0 | 1920 | 1394 | 526 | 8416 | 86 |
| p32/t128/a16 | internal no-cow | 13 | 0 | 416 | 0 | 0 | 0 | 0 | 99 |
| p32/t128/a16 | internal cow | 15 | 480 | 0 | 3040 | 2576 | 464 | 7424 | 195 |

This external prompt sweep confirms the same runtime and diagnostics behavior
outside the built-in toy prompts. Internal cow is lower latency than the public
wrapper in all three MATH500 configs and records positive scheduler-level full
block savings after repeated resampling. Exact match is 0/3 for all methods in
this small MATH500 slice; that is not treated as a quality conclusion because
the benchmark uses a strict answer extractor and a 0.5B model.

## GPU-Backed Lifecycle Soak

Source artifacts:

- Slurm log: `outputs/slurm/power-smc-soak-1066.out`
- JSON: `outputs/power_smc/soak_1066.json`
- Markdown: `outputs/power_smc/soak_1066.md`
- Script: `examples/generate/power_smc_soak.py`
- Slurm wrapper: `scripts/slurm/power_smc_soak.sbatch`

Configuration:

- Prompt file: `outputs/power_smc/math500_prompts_3.jsonl`
- Prompts per cycle: 1
- Iterations: 12
- Total requests: 12
- Max tokens: 64
- Particles: 8
- Block size: 16
- Alpha: 16.0
- Alpha ramp tokens: 1
- ESS threshold: 1.0
- GPU memory utilization: 0.50
- `ignore_eos=True`

The soak keeps one vLLM engine alive and submits 12 independent internal
Power-SMC cow requests sequentially. This exercises repeated GPU-backed
request lifecycles rather than only unit-level block-manager refcounts:

| Requests | Diagnostics | Total resamples | KV aliases | KV fallbacks | KV aliased blocks | KV saved blocks | KV saved tokens | KV pool max used | KV pool min free | GPU after delta MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 12/12 | 36 | 288 | 0 | 1152 | 523 | 8368 | 32 | 241331 | 0 |

The soak passed all configured checks: every request returned diagnostics,
resampling and KV aliasing were observed, saved blocks were positive, no alias
fallbacks occurred, and node-level GPU memory after completed requests did not
increase across the run.

## KV Telemetry Resampling Smoke Run

Source artifacts:

- JSON: `outputs/power_smc/benchmark_1038.json`
- Markdown: `outputs/power_smc/benchmark_1038.md`
- Slurm log: `outputs/slurm/power-smc-1038.out`

Configuration:

- Prompts: 1
- Max tokens: 20
- Particles: 2
- Block size: 16
- Alpha: 4.0
- Alpha ramp tokens: 1
- ESS threshold: 1.0

This run verifies that scheduler-side KV resampling telemetry is propagated to
the final `RequestOutput.power_smc` diagnostics:

| Run | Total resamples | KV aliases | KV fallbacks | KV aliased blocks | KV aliased tokens | KV reuse mode |
|---|---:|---:|---:|---:|---:|---|
| power_smc_internal_no_cow | 0 | 0 | 0 | 0 | 0 | `scheduler_reset_recompute` |
| power_smc_internal_cow | 1 | 2 | 0 | 4 | 64 | `scheduler_snapshot_alias_replay_with_reset_fallback` |

The `power_smc_internal_cow` row is the current runtime evidence that
block-level full-prefix KV aliasing can be applied after a Power-SMC resampling
event and reported back to the user-visible diagnostics.

## Earlier Resampling Smoke Run

Source artifacts:

- JSON: `outputs/power_smc/benchmark_1035.json`
- Markdown: `outputs/power_smc/benchmark_1035.md`
- Slurm log: `outputs/slurm/power-smc-1035.out`

Configuration:

- Prompts: 1
- Max tokens: 20
- Particles: 2
- Block size: 16
- Alpha: 4.0
- Alpha ramp tokens: 1
- ESS threshold: 1.0

The forced-resampling smoke run verifies that both internal modes execute a
resampling event:

| Run | Mean latency (s) | Total resamples | KV reuse mode |
|---|---:|---:|---|
| power_smc_internal_no_cow | 0.395 | 1 | `scheduler_reset_recompute` |
| power_smc_internal_cow | 0.393 | 1 | `scheduler_snapshot_alias_replay_with_reset_fallback` |

This earlier run is retained because it shows both internal modes completing a
resampling event under the same benchmark settings. Run 1038 adds stronger KV
alias telemetry for the cow path.

## Block-Boundary Synchronization Diagnostics

Source artifacts:

- Pre-fix long forced run: `outputs/power_smc/benchmark_1044.json` and `.md`
- Pre-fix short diagnostic run: `outputs/power_smc/benchmark_1045.json` and
  `.md`
- Post-deferred-check short diagnostic run: `outputs/power_smc/benchmark_1046.json`
  and `.md`
- Post-common-boundary run: `outputs/power_smc/benchmark_1048.json` and `.md`

Configuration for the short diagnostic runs:

- Prompts: 1
- Max tokens: 32
- Particles: 8
- Block size: 16
- Alpha: 4.0
- Alpha ramp tokens: 1
- ESS threshold: 1.0
- `ignore_eos=True`

The diagnostic fields now include `ess_history`, `particle_lengths`,
`done_count`, `finish_reason_counts`, `maybe_resample_calls`,
`block_boundary_checks`, and `resample_skip_reasons`. These were added after
run 1044 showed final ESS collapse without any ESS history in internal runs.
The diagnostics also now include scheduler-level KV accounting:
`kv_cow_physical_blocks`, `kv_cow_saved_blocks`, and
`kv_cow_saved_tokens`.

| Run | Mode | Total resamples | Block boundary checks | Resample skips | KV aliases | KV fallbacks |
|---|---|---:|---:|---|---:|---:|
| 1045 | internal no-cow | 1 | 1 | `{"done_particle": 8, "not_block_boundary": 247}` | 0 | 8 |
| 1045 | internal cow | 0 | 0 | `{"done_particle": 15, "not_block_boundary": 241}` | 0 | 0 |
| 1046 | internal no-cow | 1 | 1 | `{"not_block_boundary": 30}` | 0 | 8 |
| 1046 | internal cow | 0 | 0 | `{"done_particle": 1, "not_block_boundary": 31}` | 0 | 0 |
| 1048 | internal no-cow | 1 | 1 | `{"not_block_boundary": 15, "stale_block_boundary": 15}` | 0 | 8 |
| 1048 | internal cow | 1 | 1 | `{"not_block_boundary": 15, "stale_block_boundary": 15}` | 8 | 0 |

Between 1045 and 1046, Power-SMC resampling checks were moved from per-child
token observation to the end of each scheduler/output batch. This reduced
spurious `not_block_boundary` misses and kept the no-cow internal path
resampling under the forced settings. Run 1048 adds a stronger manager-side
common-boundary fallback: when all particles have crossed a block boundary but
are not exactly synchronized, the manager evaluates ESS at that boundary and
truncates any extra internal tokens after resampling. Under the same forced
settings, `power_smc_internal_cow` then recorded 1 resample event, 8 KV aliases,
16 aliased blocks, and 256 aliased tokens.

## KV Block Accounting Smoke Runs

Source artifacts:

- Moderate-alpha JSON: `outputs/power_smc/benchmark_1052.json`
- Moderate-alpha Markdown: `outputs/power_smc/benchmark_1052.md`
- High-alpha JSON: `outputs/power_smc/benchmark_1053.json`
- High-alpha Markdown: `outputs/power_smc/benchmark_1053.md`
- Allocator-snapshot JSON: `outputs/power_smc/benchmark_1058.json`
- Allocator-snapshot Markdown: `outputs/power_smc/benchmark_1058.md`
- Repeated-resampling stress JSON: `outputs/power_smc/benchmark_1059.json`
- Repeated-resampling stress Markdown: `outputs/power_smc/benchmark_1059.md`

Configuration:

- Prompts: 1
- Particles: 8
- Block size: 16
- Alpha ramp tokens: 1
- ESS threshold: 1.0
- `ignore_eos=True`
- Run 1052: alpha 4.0, max tokens 32
- Run 1053: alpha 16.0, max tokens 64
- Run 1058: alpha 16.0, max tokens 64, with vLLM KV block-pool snapshots
- Run 1059: particles 16, alpha 16.0, max tokens 128, with vLLM KV
  block-pool snapshots

Run 1052 verifies that the new KV accounting fields propagate through the JSON
and Markdown reports. Its sampled ancestors were unique, so no physical block
savings were available even though the cow path aliased 16 full blocks.

Run 1053 creates a sharper resampling regime and records positive block-level
savings:

| Run | Mode | Total resamples | KV aliases | KV aliased blocks | KV physical blocks | KV saved blocks | KV saved tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| 1053 | internal no-cow | 3 | 0 | 0 | 0 | 0 | 0 |
| 1053 | internal cow | 3 | 24 | 72 | 31 | 41 | 656 |

Here `KV aliased blocks` is the logical number of child prefix blocks reused
after resampling. `KV physical blocks` counts the unique ancestor full blocks
behind those aliases. `KV saved blocks` is the conservative block-level delta
between those two values, excluding replay tails and partial blocks.

Run 1058 adds vLLM block-pool allocator snapshots at each resampling event.
This is still allocator-internal rather than node-level GPU memory, but it is
more direct than `nvidia-smi` for vLLM's preallocated KV cache:

| Run | Mode | Total resamples | KV aliases | KV saved blocks | KV pool total blocks | KV pool max used | KV pool min free |
|---|---|---:|---:|---:|---:|---:|---:|
| 1058 | internal no-cow | 1 | 0 | 0 | 241363 | 34 | 241329 |
| 1058 | internal cow | 3 | 24 | 27 | 241363 | 28 | 241335 |

In this smoke, the cow path records fewer allocator-used KV blocks during its
resampling windows while also reporting 27 saved full blocks from aliasing.

Run 1059 extends this to a repeated-resampling stress case with 16 particles
and 128 generated tokens. Both internal modes perform 5 resampling events. The
cow path aliases 320 logical full blocks through 145 unique physical blocks,
for 175 saved full blocks / 2800 saved KV-token slots:

| Run | Mode | Total resamples | KV aliases | KV fallbacks | KV aliased blocks | KV physical blocks | KV saved blocks | KV saved tokens | KV pool total blocks | KV pool max used | KV pool min free | tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1059 | internal no-cow | 5 | 0 | 80 | 0 | 0 | 0 | 0 | 241363 | 39 | 241324 | 49.70 |
| 1059 | internal cow | 5 | 80 | 0 | 320 | 145 | 175 | 2800 | 241363 | 44 | 241319 | 51.17 |

The allocator-window maximum is not lower in 1059 because the cow path performs
more live aliasing work during its resampling windows. The direct block
accounting still shows the intended effect: repeated resampling reuses full
ancestor blocks instead of materializing every logical child prefix separately.

## Alpha=1 Parity Smoke Run

Source artifacts:

- JSON: `outputs/power_smc/benchmark_1051.json`
- Markdown: `outputs/power_smc/benchmark_1051.md`

Configuration:

- Prompts: 1
- Max tokens: 32
- Particles: 1
- Block size: 16
- Alpha: 1.0
- Alpha ramp tokens: 1
- ESS threshold: 1.0

This run verifies the single-particle degenerate path against an ordinary
full-softmax vLLM baseline under the benchmark harness. The external public API
wrapper is skipped because it rejects `alpha <= 1`, so the parity check covers
the two internal V1 modes:

| Run | Token IDs match baseline | Text matches baseline | Total resamples | Mean final ESS |
|---|---|---|---:|---:|
| power_smc_internal_no_cow | yes | yes | 0 | 1.000 |
| power_smc_internal_cow | yes | yes | 0 | 1.000 |

The diagnostics also show one completed particle, `finish_reason="length"`,
and repeated `particles_one` resampling skips, which is the expected behavior
for `particles=1`.

## Stop-Token Smoke Runs

Source artifacts:

- Period stop JSON: `outputs/power_smc/benchmark_1055.json`
- Period stop Markdown: `outputs/power_smc/benchmark_1055.md`
- Exclamation stop JSON: `outputs/power_smc/benchmark_1056.json`
- Exclamation stop Markdown: `outputs/power_smc/benchmark_1056.md`

Configuration:

- Prompts: 1 stop-focused JSONL prompt
- Max tokens: 16
- Particles: 8
- Block size: 16
- Alpha: 4.0
- Alpha ramp tokens: 1
- ESS threshold: 1.0
- `ignore_eos=False`
- Run 1055: stop token id 13, period
- Run 1056: stop token id 0, exclamation mark

Run 1056 is the stronger stop-token smoke. Both internal modes complete with
8 done particles, 6 particles finishing by `stop`, and 2 particles finishing
by `length`. The repeated `done_particle` skip reason shows that once any
particle has stopped, the manager treats it as done and does not proceed into
resampling checks that would continue or rewrite that particle.

| Run | Mode | Done particles | Finish reasons | Particle length range | Resample skips |
|---|---|---:|---|---:|---|
| 1056 | internal no-cow | 8 | `{"length": 2, "stop": 6}` | 5-16 | `{"done_particle": 11, "not_block_boundary": 4}` |
| 1056 | internal cow | 8 | `{"length": 2, "stop": 6}` | 5-16 | `{"done_particle": 12, "not_block_boundary": 4}` |

## EOS Smoke Run

Source artifacts:

- JSON: `outputs/power_smc/benchmark_1057.json`
- Markdown: `outputs/power_smc/benchmark_1057.md`
- Prompt JSONL: `outputs/power_smc/eos_smoke_prompts.jsonl`

Configuration:

- Prompts: 1 Qwen chat-template prompt
- Max tokens: 32
- Particles: 8
- Block size: 16
- Alpha: 4.0
- Alpha ramp tokens: 1
- ESS threshold: 1.0
- `ignore_eos=False`
- Stop token IDs: `[]`

Run 1057 validates the natural EOS/im_end path rather than a user stop-token
path. Both internal modes selected token IDs `[3925, 13, 151645]`, where
151645 is Qwen's `<|im_end|>` token, and all 8 particles finished with
`finish_reason="stop"` at length 3. `stop_reason_counts` is empty because no
user stop token IDs were configured.

| Run | Mode | Done particles | Finish reasons | Stop reasons | Particle length range | Selected token IDs |
|---|---|---:|---|---|---:|---|
| 1057 | internal no-cow | 8 | `{"stop": 8}` | `{}` | 3-3 | `[3925, 13, 151645]` |
| 1057 | internal cow | 8 | `{"stop": 8}` | `{}` | 3-3 | `[3925, 13, 151645]` |

## GPU Memory Notes

The benchmark samples node-level GPU memory with `nvidia-smi`. Small runs show
little or no per-run delta because the engine preallocates a large KV cache and
the tested model/request sizes are small relative to the allocated cache:

- Run 1036 peak total memory stayed at 74312 MiB after engine initialization.
- Run 1038 peak total memory stayed near the already-initialized engine level.
- Run 1035 peak total memory stayed at 74312 MiB after engine initialization.
- Runs 1041, 1042, and 1058 used lower `gpu_memory_utilization=0.50` and sampled
  peak total memory of 49976 MiB after initialization.
- Run 1059 also used `gpu_memory_utilization=0.50`; node-level memory remains
  dominated by the preallocated KV cache, while diagnostics show 175 saved full
  KV blocks inside vLLM's scheduler accounting.
- Sweep runs 1060-1062 used `gpu_memory_utilization=0.50`; node-level peak
  memory again stayed near the preallocated plateau while scheduler accounting
  reported 95, 461, and 974 saved full KV blocks for cow.
- MATH500 sweep runs 1063-1065 used the same memory setting; node-level memory
  again stayed near the preallocated plateau while cow reported 72, 526, and
  464 saved full KV blocks in scheduler accounting.

This proves that the benchmark captures memory samples, but node-level memory is
not a strong proof of KV memory reduction under vLLM preallocation. The stronger
evidence is the vLLM block-pool accounting added in run 1058; larger-memory
experiments should still use longer generations and more particles.

## Current Interpretation

- End-to-end V1 internal Power-SMC is runnable with ordinary prompts and returns
  only one parent output with optional diagnostics.
- The benchmark covers all requested comparison classes: baseline sampling,
  best-of-N, weighted best-of-N, Power-SMC without KV cow, and Power-SMC with KV
  cow.
- The internal mode has now been exercised at particles 8, 16, and 32 in Slurm
  benchmark runs.
- Scheduler-side KV diagnostics now report per-resampling alias/fallback counts
  plus aliased KV blocks/tokens; run 1038 observed 2 alias operations,
  4 aliased blocks, and 64 aliased tokens in `power_smc_internal_cow`.
- New boundary diagnostics explain no-resample runs and verify the common
  crossed-boundary fallback. Run 1048 shows the cow path can recover from
  out-of-phase child outputs and still perform a KV-aliasing resample.
- Run 1051 verifies alpha=1/particles=1 engine parity for both internal modes
  against the warmed ordinary baseline, with identical token IDs/text and no
  resampling.
- Run 1053 adds scheduler-level KV block accounting evidence: in a high-alpha
  forced-resampling smoke, cow reused 72 logical full blocks using 31 unique
  physical blocks, for an estimated 41 full blocks / 656 tokens saved.
- Run 1058 adds allocator-internal block-pool evidence: in a high-alpha smoke,
  cow reports 27 saved full blocks and a lower resampling-window KV pool max
  used count than no-cow.
- Run 1059 adds repeated-resampling stress evidence at 16 particles and 128
  tokens: cow performs 5 resampling events, records 80 alias operations, and
  saves 175 full KV blocks / 2800 KV-token slots in scheduler block accounting.
- Sweep runs 1060-1062 add built-in-prompt coverage across p8/t64/a4,
  p16/t128/a16, and p32/t128/a16. Internal cow is lower latency than the public
  wrapper in all three configs and records 95, 461, and 974 saved full KV
  blocks respectively.
- MATH500 sweep runs 1063-1065 add external prompt-file coverage across the
  same three configs. Internal cow is again lower latency than the public
  wrapper and records 72, 526, and 464 saved full KV blocks respectively.
- Run 1056 adds end-to-end stop-token evidence: internal no-cow and cow both
  mark all 8 particles done, include stop finish reasons, and report
  `done_particle` resampling skips after stopped particles appear.
- Run 1057 adds natural EOS evidence with no configured stop token IDs: both
  internal modes finish all 8 particles at token id 151645 (`<|im_end|>`).
- The measured latency for `power_smc_internal_cow` in the medium run is close
  to the public wrapper and slightly above baseline single sampling, while
  producing only one final sequence for the user.
- A CPU-side scheduler micro-optimization now caches parsed Power-SMC child
  request metadata on `Request`, avoiding repeated regex parsing in per-token
  scheduler paths for pause, observe, and free operations.
- Scheduler resampling now recognizes identity no-op resample plans and mixed
  self-ancestor children, skipping request reset, KV free/alias, preemption,
  and replay for any child that keeps its own already-computed token sequence.
- KV cow resampling snapshots are now collected only for ancestor particles
  that are actually referenced by the resample plan, reducing scheduler/KV
  manager calls when particles collapse onto a smaller ancestor set.
- Resampled particle history is stored compactly as a prefix boundary state
  plus post-resample token history, avoiding O(prefix length) Python list fills
  for every particle after each resampling event.
- Generated benchmark reports now expose the optimization counters directly:
  snapshot count, alias attempts, replay tokens, and identity no-op children.
- Generated benchmark reports also expose total `maybe_resample()` calls and
  boundary ESS checks, making scheduler/output-processor gating effects visible
  without hand-parsing JSON diagnostics.
- KV block-pool snapshots are now gated by `kv_pool_diagnostics` and the
  benchmark `--kv-pool-diagnostics` flag, so default latency runs avoid
  allocator snapshot overhead while allocator-evidence runs can still opt in.
- The current node-level GPU memory data is explainable but not sufficient to
  claim a large process-visible GPU memory reduction from cow.
- Run 1066 adds GPU-backed lifecycle soak evidence: 12 sequential Power-SMC cow
  requests completed with 36 resamples, 288 aliases, 523 saved full blocks, no
  alias fallbacks, and 0 MiB after-request GPU memory delta.
- Run 1067 reruns the MATH500 p16/t128/a16 smoke after disabling default
  KV-pool snapshots. Internal cow records 12 resamples, 192 aliases,
  165 snapshots, 2800 replay tokens, and 155 saved full KV blocks; mean latency
  is 2.516s, essentially tied with internal no-cow at 2.519s and below the
  public wrapper at 2.598s.
- Scheduler resampling is now queued only when a child reaches a Power-SMC
  block boundary or finishes, avoiding per-token `maybe_resample()` calls that
  can only produce `not_block_boundary` skips.
- Run 1069 applies the same boundary gating to the output-processor parent
  manager. On the same p16/t128/a16 smoke, per-mode `maybe_resample()` calls
  drop from 387 to 27 while boundary ESS checks remain 21. Internal cow mean
  latency is 2.486s, with 304 aliases, 240 snapshots, and 458 saved full KV
  blocks.
- Resample plan construction now reuses the already-copied resampled particle
  token prefix lists instead of copying every particle prefix a second time
  just to build `PowerSMCResamplePlan`.
- Particle history now stores an append-only list of sparse boundary states
  instead of three per-token histories. For a p16/t128/block16 local
  microbenchmark this reduces history writes from the old equivalent of 6144
  per-token list appends to 128 boundary-state entries while preserving
  boundary ESS/resample behavior.
- `maybe_resample()` now reuses the boundary states collected for the ESS check
  when constructing a resample plan, so a resampling boundary performs one
  `state_at()` lookup per particle instead of looking up ancestor states a
  second time.
- `PowerSMCGroupManager` now caches alpha-ramp constants and uses a fast
  `_alpha_at_step()` path during per-token weight updates. This keeps the
  public `alpha_ramp()` helper unchanged while avoiding repeated generic
  helper calls in the manager hot path, including the common
  `alpha_ramp_tokens=1` constant-alpha benchmark setting.
- `PowerSMCGroupManager.update_after_token()` now inlines the log-weight update
  formula in the per-token manager hot path. The public `update_log_weight()`
  helper remains unchanged and is used in tests as the reference formula.
- `PowerSMCGroupManager` also caches fixed config-derived values such as
  particle count, block size, and ESS threshold, and uses a direct constant
  alpha branch in `update_after_token()` for the common no-ramp setting.
- `update_after_token()` now returns whether the updated particle reached a
  Power-SMC block boundary. The scheduler and output processor reuse that
  result instead of rereading particle length and repeating block-size modulo
  checks after every observed token.
- Resample plan construction now caches each referenced ancestor prefix while
  still materializing independent token lists for child particles. This avoids
  repeated long prefix slicing when multiple particles collapse onto the same
  ancestor.
- Scheduler identity no-op detection now checks ancestor identity incrementally
  instead of allocating a `range` list for every resample plan.
- The output processor now deduplicates Power-SMC parent resample checks with a
  side set while preserving ordered dispatch, avoiding repeated linear
  membership scans when many child particles report a boundary in one batch.
- Scheduler KV alias setup now stores each ancestor's block snapshot and cached
  block count in one mapping, removing the parallel snapshot dictionaries and
  duplicate membership checks in the resample application path.
- Boundary eligibility checks now scan particle lengths once and early-return
  on zero-length particles, avoiding a temporary lengths list on every
  `maybe_resample()` call.
- The `maybe_resample()` done-particle precheck is now folded into the same
  boundary eligibility scan, preserving `done_particle` skip precedence while
  avoiding a second pass over particles in the common not-done path.
- Scheduler resample application now tracks per-child identity no-ops with an
  indexed boolean list instead of a hash set, avoiding repeated small-integer
  set lookups while applying a resample plan.
- Scheduler resample application now caches active child request entries during
  the identity no-op scan and reuses them in the apply loop, avoiding duplicate
  child-id and request dictionary lookups.
- Scheduler resample application now collects needed ancestor snapshots during
  that same active-child scan and accumulates per-child KV diagnostics in local
  counters before writing the event dict, reducing repeated hash-table work in
  the resample apply loop.
- Scheduler identity-noop detection is now folded into the active-child scan,
  so non-identity resamples avoid a separate full-plan precheck while the pure
  identity case still records the same `identity_noop` KV event.
- Scheduler identity-noop token comparison now uses a `Request` helper that
  compares against the internal output-token list directly, avoiding a temporary
  `list(request.output_token_ids)` allocation for each active child.
- Power-SMC parent fanout now caches the seedless child `SamplingParams` clone,
  matching the ordinary parallel-sampling parent behavior while preserving
  per-child unique seeds when a seed is configured.
- Diagnostics construction now combines particle statistics and KV telemetry
  aggregation into single passes, preserving the output schema while reducing
  repeated final-output bookkeeping.
- Resample and diagnostics paths now use an internal combined
  normalize-log-weights plus ESS helper, avoiding a separate
  `effective_sample_size()` pass after normalized weights are already built.
- `maybe_resample()` now builds boundary states and boundary log weights in a
  single particle pass, keeping the cached states for ancestor inheritance
  while avoiding a second list walk at every ESS check.
- Boundary ESS checks now carry max/uniform log-weight stats out of that same
  particle pass. Uniform boundaries skip normalization entirely, and non-uniform
  boundaries reuse the known max log weight instead of scanning log weights
  again.
- Log-weight normalization now has an exact-uniform fast path shared by the
  public normalizer and the internal normalize-plus-ESS helper, skipping
  exp/log-sum-exp work when all particle log weights are equal.
- Scheduler KV alias application now inlines the replay-safe alias-block count
  formula in the child apply loop and caches `block_size` locally, removing a
  per-child helper call during resampling.
- `final_select()` now caches the final ESS computed while normalizing weights
  for particle selection, so final diagnostics can reuse it instead of
  normalizing the same final weights a second time.
- `systematic_resample()` now precomputes the reciprocal particle count and
  caches list append in the ancestor loop, avoiding repeated division and
  method lookup during resampling.
- Resample bookkeeping now reuses the ancestor-prefix cache size for
  `unique_ancestors_per_resample`, avoiding an extra `set(ancestors)`
  allocation after the resampled particle list has already been built.
- Power-SMC final output rewrite now reuses the token list already installed by
  `get_outputs()` and only re-decodes text, avoiding a second final token-list
  copy.
- Power-SMC final output selection now installs the chosen particle's token
  list directly on the copied `CompletionOutput`, avoiding the remaining
  final-selection token-list copy after all children have finished.
- Run 1073 reruns the MATH500 p16/t128/a16 smoke after the subsequent Python
  hot-path cleanups. Internal cow records 7 resamples, 112 aliases, 71
  snapshots, 216 saved full KV blocks, 23 `maybe_resample()` calls, 21 boundary
  ESS checks, and 2.545s mean latency. It remains below internal no-cow at
  2.574s and the public wrapper at 2.706s. Because the resampling trajectory
  changed from run 1072, this is treated as post-optimization no-regression
  evidence rather than a per-change latency attribution.
- Run 1072 reruns the MATH500 p16/t128/a16 smoke after alpha-schedule caching.
  Internal cow records 11 resamples, 176 aliases, 289 saved full KV blocks,
  25 `maybe_resample()` calls, 21 boundary ESS checks, and 2.500s mean latency.
  It remains below the public wrapper in this run; the prompt count is still too
  small to isolate the alpha-cache contribution from run-to-run noise.
- Run 1071 reruns the MATH500 p16/t128/a16 smoke after boundary-state reuse.
  Internal cow records 15 resamples, 240 aliases, 277 saved full KV blocks,
  27 `maybe_resample()` calls, 21 boundary ESS checks, and 2.560s mean latency.
  As with 1070, this is treated as no-regression evidence rather than a
  latency attribution claim.
- Run 1070 reruns the MATH500 p16/t128/a16 smoke after sparse boundary history.
  Diagnostics remain stable at 27 `maybe_resample()` calls and 21 boundary ESS
  checks per internal mode; internal cow records 18 resamples, 288 aliases,
  322 saved full KV blocks, and 2.527s mean latency. This is useful as a
  no-regression check, while the small prompt count is too noisy to isolate the
  history-storage change as an end-to-end latency win.
- A larger prompt-count sweep would be useful for statistically stronger
  accuracy/latency conclusions, but is not required for the prototype success
  criteria.
