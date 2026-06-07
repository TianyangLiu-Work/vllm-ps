# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `1`
- Max tokens: `32`
- Particles: `32`
- Block size: `16`
- Alpha: `4.0`

## Status

- Benchmark failed during `llm_initialization`.
- Error: `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`

## Traceback

```text
Traceback (most recent call last):
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/examples/generate/benchmark_power_smc.py", line 929, in main
    llm = LLM(
          ^^^^
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/entrypoints/llm.py", line 349, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/v1/engine/llm_engine.py", line 174, in from_engine_args
    return cls(
           ^^^^
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/v1/engine/llm_engine.py", line 108, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/v1/engine/core_client.py", line 102, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/v1/engine/core_client.py", line 771, in __init__
    super().__init__(
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/v1/engine/core_client.py", line 570, in __init__
    with launch_core_engines(
  File "/data/conda_envs/power-smc-vllm/lib/python3.11/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/v1/engine/utils.py", line 1190, in launch_core_engines
    wait_for_engine_startup(
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/v1/engine/utils.py", line 1249, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```
