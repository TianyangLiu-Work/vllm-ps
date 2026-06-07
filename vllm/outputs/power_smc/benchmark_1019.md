# Power-SMC vLLM Benchmark Report

## Setup

- Model: `/data/shared/models/Qwen2.5-0.5B-Instruct`
- Prompts: `1`
- Max tokens: `8`
- Particles: `2`
- Block size: `16`
- Alpha: `2.0`

## Status

- Benchmark failed during `llm_initialization`.
- Error: `ValidationError: 1 validation error for ModelConfig
  Value error, Model architectures ['Qwen2ForCausalLM'] failed to be inspected. Please check the logs for more details. [type=value_error, input_value=ArgsKwargs((), {'model': ...nderer_num_workers': 1}), input_type=ArgsKwargs]
    For further information visit https://errors.pydantic.dev/2.12/v/value_error`

## Traceback

```text
Traceback (most recent call last):
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/examples/generate/benchmark_power_smc.py", line 337, in main
    llm = LLM(
          ^^^^
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/entrypoints/llm.py", line 349, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/v1/engine/llm_engine.py", line 166, in from_engine_args
    vllm_config = engine_args.create_engine_config(usage_context)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/engine/arg_utils.py", line 1735, in create_engine_config
    model_config = self.create_model_config()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tyliu/ghworkspace/vllm-ps/vllm/vllm/engine/arg_utils.py", line 1565, in create_model_config
    return ModelConfig(
           ^^^^^^^^^^^^
  File "/data/conda_envs/branch-grpo/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in __init__
    s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, Model architectures ['Qwen2ForCausalLM'] failed to be inspected. Please check the logs for more details. [type=value_error, input_value=ArgsKwargs((), {'model': ...nderer_num_workers': 1}), input_type=ArgsKwargs]
    For further information visit https://errors.pydantic.dev/2.12/v/value_error

```
