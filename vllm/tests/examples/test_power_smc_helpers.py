# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from examples.generate.benchmark_power_smc import (
    GPUMemoryMonitor,
    aggregate_power_smc_stats,
    build_alpha_one_parity_checks,
    evaluate_exact_match,
    extract_answer,
    load_prompt_examples,
    normalize_log_scores,
    sample_index,
    write_markdown_report,
)
from examples.generate.power_smc import (
    PowerSMCConfig,
    VLLMPowerSMCSampler,
    alpha_ramp,
    effective_sample_size,
    normalize_log_weights,
    sampled_token_logprobs,
    systematic_resample,
)

pytestmark = pytest.mark.skip_global_cleanup


@dataclass
class FakeLogprob:
    logprob: float


@dataclass
class FakeCompletion:
    token_ids: list[int]
    logprobs: object
    index: int = 0
    finish_reason: str | None = "length"
    stop_reason: int | str | None = None


@dataclass
class FakeFlatLogprobs:
    start_indices: list[int]
    end_indices: list[int]
    token_ids: list[int]
    logprobs: list[float]

    def __len__(self) -> int:
        return len(self.start_indices)


def test_alpha_ramp_reaches_target() -> None:
    assert alpha_ramp(0, alpha_final=4.0, ramp_tokens=4) == 1.75
    assert alpha_ramp(3, alpha_final=4.0, ramp_tokens=4) == 4.0
    assert alpha_ramp(4, alpha_final=4.0, ramp_tokens=4) == 4.0


def test_weight_normalization_and_ess() -> None:
    weights = normalize_log_weights([0.0, math.log(3.0)])
    assert weights == [0.25, 0.75]
    assert effective_sample_size(weights) == 1.6


def test_systematic_resample_is_seeded() -> None:
    ancestors = systematic_resample([0.1, 0.2, 0.7], random.Random(0))
    assert ancestors == [1, 2, 2]


def test_sampled_token_logprobs_from_flat_logprobs() -> None:
    completion = FakeCompletion(
        token_ids=[10, 20],
        logprobs=FakeFlatLogprobs(
            start_indices=[0, 1],
            end_indices=[1, 2],
            token_ids=[10, 20],
            logprobs=[-1.0, -2.0],
        ),
    )

    assert sampled_token_logprobs(completion) == [-1.0, -2.0]


def test_sampled_token_logprobs_from_dict_logprobs() -> None:
    completion = FakeCompletion(
        token_ids=[10, 20],
        logprobs=[
            {10: FakeLogprob(-1.0)},
            {20: FakeLogprob(-2.0)},
        ],
    )

    assert sampled_token_logprobs(completion) == [-1.0, -2.0]


class FakeTokenizer:

    def encode(self, prompt: str) -> list[int]:
        return [ord(ch) for ch in prompt]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class FakeLLM:

    def __init__(self) -> None:
        self.calls: list[tuple[int, list[int]]] = []

    def get_tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.calls.append((len(prompts), [params.n for params in sampling_params]))
        outputs = []
        for params in sampling_params:
            completions = []
            for index in range(params.n):
                token_id = index + 1
                completions.append(
                    FakeCompletion(
                        index=index,
                        token_ids=[token_id],
                        logprobs=[{token_id: FakeLogprob(-float(token_id))}],
                        finish_reason="length",
                    ))
            outputs.append(type("FakeOutput", (), {"outputs": completions})())
        return outputs


def test_sampler_groups_identical_prefixes_after_resampling(monkeypatch) -> None:
    class FakeSamplingParams:

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm",
                        SimpleNamespace(SamplingParams=FakeSamplingParams))

    llm = FakeLLM()
    cfg = PowerSMCConfig(
        max_tokens=2,
        alpha=4.0,
        particles=3,
        ess_threshold=1.0,
        block_size=1,
        alpha_ramp_tokens=1,
        seed=0,
    )
    sampler = VLLMPowerSMCSampler(llm, cfg)

    result = sampler.generate("x")

    assert llm.calls[0] == (1, [3])
    assert llm.calls[1] == (1, [3])
    assert result.stats["prompt_groups_history"] == [1, 1]
    assert "final_ess" in result.stats
    assert "mean_ess" in result.stats
    assert result.stats["selected_particle"] == result.selected_particle


def test_benchmark_aggregates_internal_diagnostics() -> None:
    diagnostics = aggregate_power_smc_stats([
        {
            "chosen_particle": 1,
            "resample_count": 2,
            "maybe_resample_calls": 9,
            "block_boundary_checks": 3,
            "final_ess": 1.9,
            "mean_ess": 1.6,
            "unique_ancestors_per_resample": [2, 1],
            "kv_resample_events": [
                {
                    "alias_success_count": 3,
                    "fallback_count": 1,
                    "aliased_blocks": 6,
                    "aliased_tokens": 96,
                    "snapshot_count": 2,
                    "alias_attempt_count": 4,
                    "replay_tokens": 5,
                    "identity_noop_count": 1,
                    "kv_pool_total_blocks_before": 100,
                    "kv_pool_used_blocks_before": 10,
                    "kv_pool_free_blocks_before": 90,
                    "kv_pool_total_blocks_after": 100,
                    "kv_pool_used_blocks_after": 12,
                    "kv_pool_free_blocks_after": 88,
                }
            ],
            "kv_alias_success_count": 3,
            "kv_alias_fallback_count": 1,
            "kv_aliased_blocks": 6,
            "kv_aliased_tokens": 96,
            "kv_cow_physical_blocks": 4,
            "kv_cow_saved_blocks": 2,
            "kv_cow_saved_tokens": 32,
            "done_count": 2,
            "min_particle_length": 4,
            "max_particle_length": 8,
            "finish_reason_counts": {
                "length": 2,
            },
            "stop_reason_counts": {
                "13": 1,
            },
            "resample_skip_reasons": {
                "not_block_boundary": 4,
            },
        },
        None,
    ])

    assert diagnostics["prompts"] == 2
    assert diagnostics["with_diagnostics"] == 1
    assert diagnostics["missing_diagnostics"] == 1
    assert diagnostics["mean_final_ess"] == 1.9
    assert diagnostics["mean_mean_ess"] == 1.6
    assert diagnostics["total_resample_count"] == 2
    assert diagnostics["max_resample_count"] == 2
    assert diagnostics["maybe_resample_calls"] == 9
    assert diagnostics["block_boundary_checks"] == 3
    assert diagnostics["mean_unique_ancestors_per_resample"] == 1.5
    assert diagnostics["chosen_particle_counts"] == {"1": 1}
    assert diagnostics["kv_resample_events"] == 1
    assert diagnostics["kv_alias_success_count"] == 3
    assert diagnostics["kv_alias_fallback_count"] == 1
    assert diagnostics["kv_aliased_blocks"] == 6
    assert diagnostics["kv_aliased_tokens"] == 96
    assert diagnostics["kv_cow_physical_blocks"] == 4
    assert diagnostics["kv_cow_saved_blocks"] == 2
    assert diagnostics["kv_cow_saved_tokens"] == 32
    assert diagnostics["kv_snapshot_count"] == 2
    assert diagnostics["kv_alias_attempt_count"] == 4
    assert diagnostics["kv_replay_tokens"] == 5
    assert diagnostics["kv_identity_noop_count"] == 1
    assert diagnostics["kv_pool_total_blocks"] == 100
    assert diagnostics["kv_pool_max_used_blocks"] == 12
    assert diagnostics["kv_pool_min_free_blocks"] == 88
    assert diagnostics["mean_done_count"] == 2.0
    assert diagnostics["max_done_count"] == 2
    assert diagnostics["min_particle_length"] == 4
    assert diagnostics["max_particle_length"] == 8
    assert diagnostics["finish_reason_counts"] == {"length": 2}
    assert diagnostics["stop_reason_counts"] == {"13": 1}
    assert diagnostics["resample_skip_reasons"] == {
        "not_block_boundary": 4,
    }


def test_benchmark_aggregates_wrapper_diagnostics() -> None:
    diagnostics = aggregate_power_smc_stats([
        {
            "selected_particle": 0,
            "resample_count": 1,
            "ess_history": [3.0, 2.0],
            "ancestor_history": [[0, 0, 2, 3]],
        },
    ])

    assert diagnostics["prompts"] == 1
    assert diagnostics["with_diagnostics"] == 1
    assert diagnostics["mean_final_ess"] == 2.0
    assert diagnostics["mean_mean_ess"] == 2.5
    assert diagnostics["total_resample_count"] == 1
    assert diagnostics["mean_unique_ancestors_per_resample"] == 3.0
    assert diagnostics["chosen_particle_counts"] == {"0": 1}
    assert diagnostics["kv_resample_events"] == 0
    assert diagnostics["kv_alias_success_count"] == 0
    assert diagnostics["mean_done_count"] is None
    assert diagnostics["max_done_count"] == 0
    assert diagnostics["resample_skip_reasons"] == {}


def test_benchmark_builds_alpha_one_parity_checks() -> None:
    results = {
        "alpha": 1.0,
        "particles": 1,
        "runs": {
            "baseline_single": {
                "selected_token_ids": [[1, 2, 3]],
                "texts": ["1 2 3"],
            },
            "power_smc_internal_no_cow": {
                "selected_token_ids": [[1, 2, 3]],
                "texts": ["1 2 3"],
                "diagnostics": {
                    "total_resample_count": 0,
                    "max_resample_count": 0,
                    "mean_final_ess": 1.0,
                },
            },
        },
    }

    checks = build_alpha_one_parity_checks(results)

    assert checks == {
        "power_smc_internal_no_cow": {
            "reference_run": "baseline_single",
            "token_ids_match_baseline": True,
            "texts_match_baseline": True,
            "total_resample_count": 0,
            "max_resample_count": 0,
            "mean_final_ess": 1.0,
        },
    }


def test_benchmark_weighted_best_of_n_helpers() -> None:
    weights = normalize_log_scores([math.log(1.0), math.log(3.0)])

    assert weights == [0.25, 0.75]
    assert sample_index(weights, random.Random(0)) == 1


def test_benchmark_report_includes_kv_alias_block_columns(tmp_path) -> None:
    path = tmp_path / "report.md"
    write_markdown_report(
        {
            "model": "model",
            "prompts": ["prompt"],
            "max_tokens": 8,
            "particles": 2,
            "block_size": 4,
            "alpha": 4.0,
            "attention_backend": "FLASHINFER",
            "runs": {
                "power_smc_internal_cow": {
                    "latency": {
                        "mean_s": 1.0,
                        "p90_s": 1.0,
                    },
                    "total_generated_tokens": 8,
                    "tokens_per_second": 8.0,
                    "kv_reuse_mode":
                    "scheduler_snapshot_alias_replay_with_reset_fallback",
                    "diagnostics": {
                        "with_diagnostics": 1,
                        "prompts": 1,
                        "missing_diagnostics": 0,
                        "mean_final_ess": 2.0,
                        "total_resample_count": 1,
                        "max_resample_count": 1,
                        "maybe_resample_calls": 9,
                        "block_boundary_checks": 3,
                        "mean_unique_ancestors_per_resample": 2.0,
                        "kv_alias_success_count": 2,
                        "kv_alias_fallback_count": 0,
                        "kv_snapshot_count": 2,
                        "kv_alias_attempt_count": 2,
                        "kv_replay_tokens": 3,
                        "kv_identity_noop_count": 1,
                        "kv_aliased_blocks": 4,
                        "kv_aliased_tokens": 64,
                        "kv_cow_physical_blocks": 2,
                        "kv_cow_saved_blocks": 2,
                        "kv_cow_saved_tokens": 32,
                        "kv_pool_total_blocks": 100,
                        "kv_pool_max_used_blocks": 12,
                        "kv_pool_min_free_blocks": 88,
                        "mean_done_count": 2.0,
                        "max_done_count": 2,
                        "min_particle_length": 4,
                        "max_particle_length": 8,
                        "finish_reason_counts": {
                            "length": 2,
                        },
                        "stop_reason_counts": {
                            "13": 2,
                        },
                        "resample_skip_reasons": {
                            "not_block_boundary": 4,
                        },
                        "chosen_particle_counts": {
                            "1": 1,
                        },
                    },
                },
            },
            "alpha_one_parity": {
                "power_smc_internal_cow": {
                    "token_ids_match_baseline": True,
                    "texts_match_baseline": True,
                    "total_resample_count": 0,
                    "mean_final_ess": 1.0,
                },
            },
        },
        path,
    )

    report = path.read_text(encoding="utf-8")

    assert "KV aliased blocks" in report
    assert "KV saved blocks" in report
    assert "Maybe checks" in report
    assert "Boundary checks" in report
    assert (
        "| power_smc_internal_cow | 1/1 | 0 | 2.000 | 1 | 1 | 2.000 | "
        "9 | 3 | 2 | 0 | 2 | 2 | 3 | 1 | 4 | 64 | 2 | 2 | 32 | "
        "100 | 12 | 88 | 2.000 | 2 | 4-8 | {\"length\": 2} | "
        "{\"13\": 2} | {\"not_block_boundary\": 4} | {\"1\": 1} |"
    ) in report
    assert "## Alpha=1 Parity" in report
    assert "| power_smc_internal_cow | yes | yes | 0 | 1.000 |" in report


def test_benchmark_loads_jsonl_prompts_with_answers(tmp_path) -> None:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"prompt": "What is 2+2?", "answer": "4"}\nplain prompt\n',
        encoding="utf-8",
    )

    examples = load_prompt_examples(
        SimpleNamespace(prompt_file=prompt_file, num_prompts=2))

    assert examples == [
        {"prompt": "What is 2+2?", "answer": "4"},
        {"prompt": "plain prompt", "answer": None},
    ]


def test_benchmark_extracts_and_scores_exact_match() -> None:
    assert extract_answer("Therefore \\boxed{437}.") == "437"
    assert extract_answer("Solving gives x = 5 and y = 3.") == "x=5,y=3"
    assert extract_answer("The speed is 48 miles per hour.") == "48"

    accuracy = evaluate_exact_match(
        [
            "Therefore \\boxed{437}.",
            "Solving gives x = 5 and y = 3.",
            "The answer is 12.",
        ],
        ["437", "x=5,y=3", "48"],
    )

    assert accuracy["correct"] == 2
    assert accuracy["total"] == 3
    assert accuracy["pass_at_1"] == pytest.approx(2 / 3)
    assert accuracy["exact_match"] == pytest.approx(2 / 3)


def test_gpu_memory_monitor_disabled_summary() -> None:
    with GPUMemoryMonitor(enabled=False, interval_s=0.01) as monitor:
        pass

    assert monitor.summary() == {
        "enabled": False,
        "available": False,
        "samples": 0,
        "before_total_mib": None,
        "after_total_mib": None,
        "peak_total_mib": None,
        "peak_gpu_mib": None,
        "peak_delta_total_mib": None,
        "error": None,
    }
