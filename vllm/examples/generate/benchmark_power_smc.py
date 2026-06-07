# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark vLLM baseline sampling against vLLM-backed Power-SMC.

The script is intentionally small and Slurm-friendly: it writes a machine
readable JSON result plus a Markdown report that can be copied into experiment
notes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import subprocess
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

DEFAULT_EXAMPLES = [
    {
        "prompt":
        "Solve carefully: what is 19 * 23? Put the final answer in \\boxed{}.",
        "answer": "437",
    },
    {
        "prompt":
        "A train travels 120 miles in 2.5 hours. What is its average speed?",
        "answer": "48",
    },
    {
        "prompt": "If x + 2y = 11 and x - y = 2, solve for x and y.",
        "answer": "x=5,y=3",
    },
]


def log(message: str) -> None:
    print(f"[power-smc-bench] {message}", flush=True)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, round((pct / 100.0) * (len(values) - 1)))
    return sorted_values[idx]


def load_prompt_examples(args: argparse.Namespace) -> list[dict[str, str | None]]:
    if args.prompt_file is None:
        return DEFAULT_EXAMPLES[: args.num_prompts]

    examples: list[dict[str, str | None]] = []
    for line_no, line in enumerate(
            args.prompt_file.read_text(encoding="utf-8").splitlines(),
            start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            item = json.loads(stripped)
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    f"{args.prompt_file}:{line_no} JSONL row needs a prompt.")
            answer = item.get("answer")
            if answer is not None and not isinstance(answer, str):
                answer = str(answer)
            examples.append({"prompt": prompt, "answer": answer})
        else:
            examples.append({"prompt": stripped, "answer": None})
    if not examples:
        raise ValueError(f"No prompts found in {args.prompt_file}.")
    return examples[: args.num_prompts]


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def extract_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1]

    assignments = re.findall(
        r"([a-z])\s*=\s*(-?\d+(?:\.\d+)?)",
        text.lower(),
    )
    if assignments:
        return ",".join(f"{name}={value}" for name, value in assignments)

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return text.strip()


def evaluate_exact_match(
    texts: list[str],
    answers: list[str | None],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    correct = 0
    total = 0
    for text, answer in zip(texts, answers, strict=True):
        predicted = extract_answer(text)
        is_correct = None
        if answer is not None:
            is_correct = normalize_answer(predicted) == normalize_answer(answer)
            total += 1
            correct += int(is_correct)
        rows.append({
            "prediction": predicted,
            "answer": answer,
            "exact_match": is_correct,
        })
    return {
        "available": total > 0,
        "correct": correct,
        "total": total,
        "pass_at_1": correct / total if total else None,
        "exact_match": correct / total if total else None,
        "rows": rows,
    }


def summarize_latencies(latencies: list[float]) -> dict[str, float]:
    return {
        "mean_s": statistics.fmean(latencies) if latencies else 0.0,
        "median_s": statistics.median(latencies) if latencies else 0.0,
        "p90_s": percentile(latencies, 90),
        "min_s": min(latencies, default=0.0),
        "max_s": max(latencies, default=0.0),
    }


def sample_gpu_memory() -> dict[str, Any]:
    """Sample GPU memory through nvidia-smi.

    vLLM V1 may run the engine core in a child process, so process-local
    torch.cuda counters in this benchmark process are not enough. This is a
    system-level sample and can include unrelated GPU users on shared nodes.
    """

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    per_gpu_mib: list[int] = []
    for line in completed.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            per_gpu_mib.append(int(value))
        except ValueError:
            return {
                "available": False,
                "error": f"Could not parse nvidia-smi memory value: {value!r}",
            }

    if not per_gpu_mib:
        return {
            "available": False,
            "error": "nvidia-smi returned no GPU memory rows.",
        }

    return {
        "available": True,
        "per_gpu_mib": per_gpu_mib,
        "total_mib": sum(per_gpu_mib),
        "max_gpu_mib": max(per_gpu_mib),
    }


class GPUMemoryMonitor:
    """Background total-GPU memory sampler for Slurm benchmark runs."""

    def __init__(self, *, enabled: bool, interval_s: float) -> None:
        self.enabled = enabled
        self.interval_s = max(interval_s, 0.01)
        self.before: dict[str, Any] | None = None
        self.after: dict[str, Any] | None = None
        self.peak_total_mib: int | None = None
        self.peak_gpu_mib: int | None = None
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GPUMemoryMonitor:
        if not self.enabled:
            return self

        self.before = sample_gpu_memory()
        if not self.before.get("available"):
            self.error = self.before.get("error", "GPU memory monitor unavailable.")
            return self

        self._record(self.before)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(1.0, self.interval_s * 2.0))
        if self.enabled:
            self.after = sample_gpu_memory()
            if self.after.get("available"):
                self._record(self.after)
            elif self.error is None:
                self.error = self.after.get("error", "GPU memory monitor unavailable.")

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_s):
            sample = sample_gpu_memory()
            if not sample.get("available"):
                if self.error is None:
                    self.error = sample.get("error",
                                            "GPU memory monitor unavailable.")
                continue
            self._record(sample)

    def _record(self, sample: dict[str, Any]) -> None:
        self.samples += 1
        total = int(sample["total_mib"])
        max_gpu = int(sample["max_gpu_mib"])
        self.peak_total_mib = (
            total if self.peak_total_mib is None else max(self.peak_total_mib,
                                                         total))
        self.peak_gpu_mib = (
            max_gpu if self.peak_gpu_mib is None else max(self.peak_gpu_mib,
                                                         max_gpu))

    def summary(self) -> dict[str, Any]:
        before_total = (
            self.before.get("total_mib")
            if self.before and self.before.get("available") else None)
        after_total = (
            self.after.get("total_mib")
            if self.after and self.after.get("available") else None)
        return {
            "enabled": self.enabled,
            "available": self.enabled and self.error is None
            and self.peak_total_mib is not None,
            "samples": self.samples,
            "before_total_mib": before_total,
            "after_total_mib": after_total,
            "peak_total_mib": self.peak_total_mib,
            "peak_gpu_mib": self.peak_gpu_mib,
            "peak_delta_total_mib": (
                self.peak_total_mib - before_total
                if self.peak_total_mib is not None and before_total is not None
                else None),
            "error": self.error,
        }


def _numeric_values(values: list[Any]) -> list[float]:
    return [
        float(value) for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _extract_final_ess(stats: dict[str, Any]) -> float | None:
    if isinstance(stats.get("final_ess"), int | float):
        return float(stats["final_ess"])
    ess_history = stats.get("ess_history")
    if isinstance(ess_history, list) and ess_history:
        values = _numeric_values(ess_history)
        if values:
            return values[-1]
    return None


def _extract_unique_ancestor_counts(stats: dict[str, Any]) -> list[float]:
    direct = stats.get("unique_ancestors_per_resample")
    if isinstance(direct, list):
        return _numeric_values(direct)

    ancestor_history = stats.get("ancestor_history")
    if isinstance(ancestor_history, list):
        counts: list[float] = []
        for ancestors in ancestor_history:
            if isinstance(ancestors, list):
                counts.append(float(len(set(ancestors))))
        return counts
    return []


def aggregate_power_smc_stats(
    stats: list[dict[str, Any] | None],
) -> dict[str, Any]:
    diagnostics = [item for item in stats if isinstance(item, dict)]
    final_ess_values: list[float] = []
    mean_ess_values: list[float] = []
    resample_counts: list[float] = []
    maybe_resample_calls = 0
    block_boundary_checks = 0
    unique_ancestor_counts: list[float] = []
    chosen_particle_counts: dict[str, int] = {}
    kv_alias_success_count = 0
    kv_alias_fallback_count = 0
    kv_aliased_blocks = 0
    kv_aliased_tokens = 0
    kv_cow_physical_blocks = 0
    kv_cow_saved_blocks = 0
    kv_cow_saved_tokens = 0
    kv_snapshot_count = 0
    kv_alias_attempt_count = 0
    kv_replay_tokens = 0
    kv_identity_noop_count = 0
    kv_resample_events = 0
    kv_pool_total_blocks: list[float] = []
    kv_pool_used_blocks: list[float] = []
    kv_pool_free_blocks: list[float] = []
    done_counts: list[float] = []
    min_particle_lengths: list[float] = []
    max_particle_lengths: list[float] = []
    finish_reason_counts: dict[str, int] = {}
    stop_reason_counts: dict[str, int] = {}
    resample_skip_reasons: dict[str, int] = {}

    for item in diagnostics:
        final_ess = _extract_final_ess(item)
        if final_ess is not None:
            final_ess_values.append(final_ess)

        if isinstance(item.get("mean_ess"), int | float):
            mean_ess_values.append(float(item["mean_ess"]))
        elif isinstance(item.get("ess_history"), list):
            history = _numeric_values(item["ess_history"])
            mean_ess = _mean_or_none(history)
            if mean_ess is not None:
                mean_ess_values.append(mean_ess)

        if isinstance(item.get("resample_count"), int | float):
            resample_counts.append(float(item["resample_count"]))
        if isinstance(item.get("maybe_resample_calls"), int | float):
            maybe_resample_calls += int(item["maybe_resample_calls"])
        if isinstance(item.get("block_boundary_checks"), int | float):
            block_boundary_checks += int(item["block_boundary_checks"])

        unique_ancestor_counts.extend(_extract_unique_ancestor_counts(item))

        chosen_particle = item.get("chosen_particle", item.get("selected_particle"))
        if isinstance(chosen_particle, int):
            key = str(chosen_particle)
            chosen_particle_counts[key] = chosen_particle_counts.get(key, 0) + 1

        events = item.get("kv_resample_events")
        if isinstance(events, list):
            kv_resample_events += len(events)
            for event in events:
                if not isinstance(event, dict):
                    continue
                for key in ("kv_pool_total_blocks_before",
                            "kv_pool_total_blocks_after"):
                    if isinstance(event.get(key), int | float):
                        kv_pool_total_blocks.append(float(event[key]))
                for key in ("kv_pool_used_blocks_before",
                            "kv_pool_used_blocks_after"):
                    if isinstance(event.get(key), int | float):
                        kv_pool_used_blocks.append(float(event[key]))
                for key in ("kv_pool_free_blocks_before",
                            "kv_pool_free_blocks_after"):
                    if isinstance(event.get(key), int | float):
                        kv_pool_free_blocks.append(float(event[key]))
                if isinstance(event.get("snapshot_count"), int | float):
                    kv_snapshot_count += int(event["snapshot_count"])
                if isinstance(event.get("alias_attempt_count"), int | float):
                    kv_alias_attempt_count += int(
                        event["alias_attempt_count"])
                if isinstance(event.get("replay_tokens"), int | float):
                    kv_replay_tokens += int(event["replay_tokens"])
                if isinstance(event.get("identity_noop_count"), int | float):
                    kv_identity_noop_count += int(
                        event["identity_noop_count"])
        if isinstance(item.get("kv_alias_success_count"), int | float):
            kv_alias_success_count += int(item["kv_alias_success_count"])
        if isinstance(item.get("kv_alias_fallback_count"), int | float):
            kv_alias_fallback_count += int(item["kv_alias_fallback_count"])
        if isinstance(item.get("kv_aliased_blocks"), int | float):
            kv_aliased_blocks += int(item["kv_aliased_blocks"])
        if isinstance(item.get("kv_aliased_tokens"), int | float):
            kv_aliased_tokens += int(item["kv_aliased_tokens"])
        if isinstance(item.get("kv_cow_physical_blocks"), int | float):
            kv_cow_physical_blocks += int(item["kv_cow_physical_blocks"])
        if isinstance(item.get("kv_cow_saved_blocks"), int | float):
            kv_cow_saved_blocks += int(item["kv_cow_saved_blocks"])
        if isinstance(item.get("kv_cow_saved_tokens"), int | float):
            kv_cow_saved_tokens += int(item["kv_cow_saved_tokens"])
        if isinstance(item.get("done_count"), int | float):
            done_counts.append(float(item["done_count"]))
        if isinstance(item.get("min_particle_length"), int | float):
            min_particle_lengths.append(float(item["min_particle_length"]))
        if isinstance(item.get("max_particle_length"), int | float):
            max_particle_lengths.append(float(item["max_particle_length"]))
        item_finish_reasons = item.get("finish_reason_counts")
        if isinstance(item_finish_reasons, dict):
            for reason, count in item_finish_reasons.items():
                if isinstance(reason, str) and isinstance(count, int | float):
                    finish_reason_counts[reason] = (
                        finish_reason_counts.get(reason, 0) + int(count))
        item_stop_reasons = item.get("stop_reason_counts")
        if isinstance(item_stop_reasons, dict):
            for reason, count in item_stop_reasons.items():
                if isinstance(reason, str) and isinstance(count, int | float):
                    stop_reason_counts[reason] = (
                        stop_reason_counts.get(reason, 0) + int(count))
        item_skip_reasons = item.get("resample_skip_reasons")
        if isinstance(item_skip_reasons, dict):
            for reason, count in item_skip_reasons.items():
                if isinstance(reason, str) and isinstance(count, int | float):
                    resample_skip_reasons[reason] = (
                        resample_skip_reasons.get(reason, 0) + int(count))

    return {
        "prompts": len(stats),
        "with_diagnostics": len(diagnostics),
        "missing_diagnostics": len(stats) - len(diagnostics),
        "mean_final_ess": _mean_or_none(final_ess_values),
        "min_final_ess": min(final_ess_values, default=None),
        "mean_mean_ess": _mean_or_none(mean_ess_values),
        "total_resample_count": int(sum(resample_counts)),
        "mean_resample_count": _mean_or_none(resample_counts),
        "max_resample_count": int(max(resample_counts, default=0.0)),
        "maybe_resample_calls": maybe_resample_calls,
        "block_boundary_checks": block_boundary_checks,
        "mean_unique_ancestors_per_resample":
        _mean_or_none(unique_ancestor_counts),
        "chosen_particle_counts": dict(sorted(chosen_particle_counts.items())),
        "kv_resample_events": kv_resample_events,
        "kv_alias_success_count": kv_alias_success_count,
        "kv_alias_fallback_count": kv_alias_fallback_count,
        "kv_aliased_blocks": kv_aliased_blocks,
        "kv_aliased_tokens": kv_aliased_tokens,
        "kv_cow_physical_blocks": kv_cow_physical_blocks,
        "kv_cow_saved_blocks": kv_cow_saved_blocks,
        "kv_cow_saved_tokens": kv_cow_saved_tokens,
        "kv_snapshot_count": kv_snapshot_count,
        "kv_alias_attempt_count": kv_alias_attempt_count,
        "kv_replay_tokens": kv_replay_tokens,
        "kv_identity_noop_count": kv_identity_noop_count,
        "kv_pool_total_blocks": int(max(kv_pool_total_blocks, default=0.0)),
        "kv_pool_max_used_blocks": int(max(kv_pool_used_blocks, default=0.0)),
        "kv_pool_min_free_blocks": int(min(kv_pool_free_blocks, default=0.0)),
        "mean_done_count": _mean_or_none(done_counts),
        "max_done_count": int(max(done_counts, default=0.0)),
        "min_particle_length": int(min(min_particle_lengths, default=0.0)),
        "max_particle_length": int(max(max_particle_lengths, default=0.0)),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
        "resample_skip_reasons": dict(sorted(resample_skip_reasons.items())),
    }


def build_alpha_one_parity_checks(
    results: dict[str, Any],
) -> dict[str, dict[str, Any]] | None:
    if float(results.get("alpha", 0.0)) != 1.0 or results.get("particles") != 1:
        return None

    runs = results.get("runs")
    if not isinstance(runs, dict):
        return None
    baseline_name = (
        "baseline_particles"
        if isinstance(runs.get("baseline_particles"), dict)
        else "baseline_single"
    )
    baseline = runs.get(baseline_name)
    if not isinstance(baseline, dict):
        return None

    baseline_token_ids = baseline.get("selected_token_ids")
    baseline_texts = baseline.get("texts")
    checks: dict[str, dict[str, Any]] = {}
    for name in (
        "power_smc_wrapper",
        "power_smc_internal_no_cow",
        "power_smc_internal_cow",
    ):
        result = runs.get(name)
        if not isinstance(result, dict):
            continue
        diagnostics = result.get("diagnostics")
        check = {
            "reference_run": baseline_name,
            "token_ids_match_baseline":
            result.get("selected_token_ids") == baseline_token_ids,
            "texts_match_baseline": result.get("texts") == baseline_texts,
        }
        if isinstance(diagnostics, dict):
            check.update({
                "total_resample_count":
                diagnostics.get("total_resample_count"),
                "max_resample_count":
                diagnostics.get("max_resample_count"),
                "mean_final_ess":
                diagnostics.get("mean_final_ess"),
            })
        checks[name] = check
    return checks


def sequence_logprob(completion: Any) -> float:
    from power_smc import sampled_token_logprobs

    return math.fsum(sampled_token_logprobs(completion))


def normalize_log_scores(log_scores: list[float]) -> list[float]:
    if not log_scores:
        return []
    max_score = max(log_scores)
    values = [math.exp(score - max_score) for score in log_scores]
    total = math.fsum(values)
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / len(log_scores) for _ in log_scores]
    return [value / total for value in values]


def sample_index(weights: list[float], rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if draw <= cumulative:
            return idx
    return len(weights) - 1


def run_baseline(
    llm: Any,
    prompts: list[str],
    answers: list[str | None],
    *,
    max_tokens: int,
    n: int,
    temperature: float,
    ignore_eos: bool,
    stop_token_ids: list[int],
    seed: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    latencies: list[float] = []
    generated_tokens: list[int] = []
    selected_token_ids: list[list[int]] = []
    texts: list[str] = []

    params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        stop_token_ids=stop_token_ids,
        seed=seed,
    )
    for prompt_index, prompt in enumerate(prompts, start=1):
        log(f"baseline n={n}: prompt {prompt_index}/{len(prompts)}")
        start = time.perf_counter()
        output = llm.generate([prompt], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        token_count = sum(len(completion.token_ids)
                          for completion in output.outputs)
        generated_tokens.append(token_count)
        texts.append(output.outputs[0].text)
        selected_token_ids.append(list(output.outputs[0].token_ids))

    total_tokens = sum(generated_tokens)
    total_time = sum(latencies)
    return {
        "kind": "baseline",
        "n": n,
        "temperature": temperature,
        "latency": summarize_latencies(latencies),
        "total_time_s": total_time,
        "total_generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_time if total_time else 0.0,
        "generated_tokens": generated_tokens,
        "selected_token_ids": selected_token_ids,
        "texts": texts,
        "accuracy": evaluate_exact_match(texts, answers),
        "kv_reuse_mode": "none",
    }


def run_best_of_n(
    llm: Any,
    prompts: list[str],
    answers: list[str | None],
    args: argparse.Namespace,
    *,
    weighted: bool,
) -> dict[str, Any]:
    from power_smc import make_sampling_params

    rng = random.Random(args.seed)
    params = make_sampling_params(
        n=args.particles,
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        stop_token_ids=args.stop_token_id,
        logprobs=0,
        flat_logprobs=True,
    )

    latencies: list[float] = []
    candidate_generated_tokens: list[int] = []
    selected_generated_tokens: list[int] = []
    selected_token_ids: list[list[int]] = []
    texts: list[str] = []
    selection_stats: list[dict[str, Any]] = []
    label = "weighted_best_of_n" if weighted else "best_of_n"

    for prompt_index, prompt in enumerate(prompts, start=1):
        log(f"{label}: prompt {prompt_index}/{len(prompts)}")
        start = time.perf_counter()
        output = llm.generate([prompt], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - start
        completions = sorted(output.outputs, key=lambda item: item.index)
        scores = [sequence_logprob(completion) for completion in completions]

        if weighted:
            # Independent samples come from q=p, so p(y)^(alpha-1) is the
            # importance weight for a p(y)^alpha target.
            weights = normalize_log_scores([
                (args.alpha - 1.0) * score for score in scores
            ])
            selected = sample_index(weights, rng)
        else:
            weights = None
            selected = max(range(len(scores)), key=scores.__getitem__)

        completion = completions[selected]
        candidate_tokens = sum(len(item.token_ids) for item in completions)
        candidate_generated_tokens.append(candidate_tokens)
        selected_generated_tokens.append(len(completion.token_ids))
        selected_token_ids.append(list(completion.token_ids))
        latencies.append(elapsed)
        texts.append(completion.text)
        selection_stats.append({
            "selected_candidate": selected,
            "candidate_scores": scores,
            "candidate_generated_tokens": candidate_tokens,
            "selected_generated_tokens": len(completion.token_ids),
            "weights": weights,
        })

    total_tokens = sum(candidate_generated_tokens)
    total_time = sum(latencies)
    return {
        "kind": label,
        "n": args.particles,
        "temperature": args.temperature,
        "weighted": weighted,
        "latency": summarize_latencies(latencies),
        "total_time_s": total_time,
        "total_generated_tokens": total_tokens,
        "total_selected_generated_tokens": sum(selected_generated_tokens),
        "tokens_per_second": total_tokens / total_time if total_time else 0.0,
        "candidate_generated_tokens": candidate_generated_tokens,
        "selected_generated_tokens": selected_generated_tokens,
        "selected_token_ids": selected_token_ids,
        "texts": texts,
        "selection_stats": selection_stats,
        "accuracy": evaluate_exact_match(texts, answers),
        "kv_reuse_mode": "none",
    }


def run_power_smc(
    llm: Any,
    prompts: list[str],
    answers: list[str | None],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from power_smc import PowerSMCConfig, VLLMPowerSMCSampler

    cfg = PowerSMCConfig(
        max_tokens=args.max_tokens,
        alpha=args.alpha,
        particles=args.particles,
        ess_threshold=args.ess_threshold,
        block_size=args.block_size,
        alpha_ramp_tokens=args.alpha_ramp_tokens,
        min_tokens=args.min_tokens,
        seed=args.seed,
        stop_token_ids=args.stop_token_id,
        ignore_eos=args.ignore_eos,
    )
    sampler = VLLMPowerSMCSampler(llm, cfg)

    latencies: list[float] = []
    generated_tokens: list[int] = []
    selected_token_ids: list[list[int]] = []
    texts: list[str] = []
    stats: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompts, start=1):
        log(f"power_smc: prompt {prompt_index}/{len(prompts)}")
        start = time.perf_counter()
        output = sampler.generate(prompt)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        generated_tokens.append(len(output.token_ids))
        selected_token_ids.append(list(output.token_ids))
        texts.append(output.text)
        stats.append(output.stats)

    total_tokens = sum(generated_tokens)
    total_time = sum(latencies)
    return {
        "kind": "power_smc",
        "config": asdict(cfg),
        "latency": summarize_latencies(latencies),
        "total_time_s": total_time,
        "total_generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_time if total_time else 0.0,
        "generated_tokens": generated_tokens,
        "selected_token_ids": selected_token_ids,
        "texts": texts,
        "stats": stats,
        "diagnostics": aggregate_power_smc_stats(stats),
        "accuracy": evaluate_exact_match(texts, answers),
        "kv_reuse_mode": "public_api_prefix_cache",
    }


def run_power_smc_internal(
    llm: Any,
    prompts: list[str],
    answers: list[str | None],
    args: argparse.Namespace,
    *,
    kv_cow: bool,
) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=args.seed,
        ignore_eos=args.ignore_eos,
        stop_token_ids=args.stop_token_id,
        extra_args={
            "power_smc": {
                "enabled": True,
                "alpha": args.alpha,
                "particles": args.particles,
                "block_size": args.block_size,
                "ess_threshold": args.ess_threshold,
                "alpha_ramp_tokens": args.alpha_ramp_tokens,
                "proposal": "power_temperature",
                "return_diagnostics": True,
                "kv_cow": kv_cow,
                "kv_pool_diagnostics": args.kv_pool_diagnostics,
            }
        },
    )

    latencies: list[float] = []
    generated_tokens: list[int] = []
    selected_token_ids: list[list[int]] = []
    texts: list[str] = []
    stats: list[dict[str, Any] | None] = []
    label = "power_smc_internal_cow" if kv_cow else "power_smc_internal_no_cow"

    for prompt_index, prompt in enumerate(prompts, start=1):
        log(f"{label}: prompt {prompt_index}/{len(prompts)}")
        start = time.perf_counter()
        output = llm.generate([prompt], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - start
        completion = output.outputs[0]
        latencies.append(elapsed)
        generated_tokens.append(len(completion.token_ids))
        selected_token_ids.append(list(completion.token_ids))
        texts.append(completion.text)
        stats.append(getattr(output, "power_smc", None))

    total_tokens = sum(generated_tokens)
    total_time = sum(latencies)
    return {
        "kind": label,
        "config": {
            "max_tokens": args.max_tokens,
            "alpha": args.alpha,
            "particles": args.particles,
            "ess_threshold": args.ess_threshold,
            "block_size": args.block_size,
            "alpha_ramp_tokens": args.alpha_ramp_tokens,
            "proposal": "power_temperature",
            "kv_cow": kv_cow,
            "ignore_eos": args.ignore_eos,
            "stop_token_ids": list(args.stop_token_id),
        },
        "latency": summarize_latencies(latencies),
        "total_time_s": total_time,
        "total_generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_time if total_time else 0.0,
        "generated_tokens": generated_tokens,
        "selected_token_ids": selected_token_ids,
        "texts": texts,
        "stats": stats,
        "diagnostics": aggregate_power_smc_stats(stats),
        "accuracy": evaluate_exact_match(texts, answers),
        "kv_reuse_mode": (
            "scheduler_snapshot_alias_replay_with_reset_fallback"
            if kv_cow else "scheduler_reset_recompute"),
    }


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _format_optional_int(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(value))


def write_markdown_report(results: dict[str, Any], path: Path) -> None:
    if "error" in results:
        text = "\n".join([
            "# Power-SMC vLLM Benchmark Report",
            "",
            "## Setup",
            "",
            f"- Model: `{results['model']}`",
            f"- Prompts: `{len(results['prompts'])}`",
            f"- Max tokens: `{results['max_tokens']}`",
            f"- Particles: `{results['particles']}`",
            f"- Block size: `{results['block_size']}`",
            f"- Alpha: `{results['alpha']}`",
            f"- Ignore EOS: `{results.get('ignore_eos', False)}`",
            f"- Stop token IDs: `{results.get('stop_token_ids', [])}`",
            "",
            "## Status",
            "",
            f"- Benchmark failed during `{results['error']['stage']}`.",
            f"- Error: `{results['error']['type']}: {results['error']['message']}`",
            "",
            "## Traceback",
            "",
            "```text",
            results["error"]["traceback"],
            "```",
            "",
        ])
        path.write_text(text, encoding="utf-8")
        return

    rows = []
    accuracy_rows = []
    kv_rows = []
    diagnostics_rows = []
    memory_rows = []
    for name, result in results["runs"].items():
        rows.append(
            "| {name} | {mean:.3f} | {p90:.3f} | {tok} | {tps:.2f} |".format(
                name=name,
                mean=result["latency"]["mean_s"],
                p90=result["latency"]["p90_s"],
                tok=result["total_generated_tokens"],
                tps=result["tokens_per_second"],
            ))
        if result.get("kv_reuse_mode") is not None:
            kv_rows.append(
                f"| {name} | `{result['kv_reuse_mode']}` |")
        accuracy = result.get("accuracy")
        if accuracy is not None and accuracy["available"]:
            accuracy_rows.append(
                "| {name} | {correct}/{total} | {pass_at_1} | {em} |".
                format(
                    name=name,
                    correct=accuracy["correct"],
                    total=accuracy["total"],
                    pass_at_1=_format_optional_float(accuracy["pass_at_1"]),
                    em=_format_optional_float(accuracy["exact_match"]),
                ))
        diagnostics = result.get("diagnostics")
        if diagnostics is not None:
            diagnostics_rows.append(
                "| {name} | {with_diag}/{prompts} | {missing} | {ess} | "
                "{resamples} | {max_resamples} | {ancestors} | "
                "{maybe_checks} | {boundary_checks} | {kv_aliases} | "
                "{kv_fallbacks} | {kv_snapshots} | {kv_alias_attempts} | "
                "{kv_replay_tokens} | {kv_identity_noops} | {kv_blocks} | "
                "{kv_tokens} | {kv_physical_blocks} | {kv_saved_blocks} | "
                "{kv_saved_tokens} | {kv_pool_total} | {kv_pool_used} | "
                "{kv_pool_free} | {done} | {max_done} | {lengths} | "
                "{finishes} | {stop_reasons} | {skips} | {chosen} |".format(
                    name=name,
                    with_diag=diagnostics["with_diagnostics"],
                    prompts=diagnostics["prompts"],
                    missing=diagnostics["missing_diagnostics"],
                    ess=_format_optional_float(diagnostics["mean_final_ess"]),
                    resamples=diagnostics["total_resample_count"],
                    max_resamples=diagnostics["max_resample_count"],
                    ancestors=_format_optional_float(
                        diagnostics["mean_unique_ancestors_per_resample"]),
                    maybe_checks=diagnostics.get("maybe_resample_calls", 0),
                    boundary_checks=diagnostics.get("block_boundary_checks", 0),
                    kv_aliases=diagnostics["kv_alias_success_count"],
                    kv_fallbacks=diagnostics["kv_alias_fallback_count"],
                    kv_snapshots=diagnostics.get("kv_snapshot_count", 0),
                    kv_alias_attempts=diagnostics.get(
                        "kv_alias_attempt_count", 0),
                    kv_replay_tokens=diagnostics.get("kv_replay_tokens", 0),
                    kv_identity_noops=diagnostics.get(
                        "kv_identity_noop_count", 0),
                    kv_blocks=diagnostics["kv_aliased_blocks"],
                    kv_tokens=diagnostics["kv_aliased_tokens"],
                    kv_physical_blocks=diagnostics.get(
                        "kv_cow_physical_blocks", 0),
                    kv_saved_blocks=diagnostics.get("kv_cow_saved_blocks", 0),
                    kv_saved_tokens=diagnostics.get("kv_cow_saved_tokens", 0),
                    kv_pool_total=diagnostics.get("kv_pool_total_blocks", 0),
                    kv_pool_used=diagnostics.get("kv_pool_max_used_blocks", 0),
                    kv_pool_free=diagnostics.get("kv_pool_min_free_blocks", 0),
                    done=_format_optional_float(diagnostics["mean_done_count"]),
                    max_done=diagnostics["max_done_count"],
                    lengths=(
                        f"{diagnostics['min_particle_length']}-"
                        f"{diagnostics['max_particle_length']}"
                    ),
                    finishes=json.dumps(diagnostics["finish_reason_counts"],
                                        sort_keys=True),
                    stop_reasons=json.dumps(
                        diagnostics.get("stop_reason_counts", {}),
                        sort_keys=True),
                    skips=json.dumps(diagnostics["resample_skip_reasons"],
                                     sort_keys=True),
                    chosen=json.dumps(diagnostics["chosen_particle_counts"],
                                      sort_keys=True),
                ))
        memory = result.get("memory")
        if memory is not None:
            memory_rows.append(
                "| {name} | {available} | {samples} | {before} | {peak} | "
                "{delta} | {after} |".format(
                    name=name,
                    available="yes" if memory["available"] else "no",
                    samples=memory["samples"],
                    before=_format_optional_int(memory["before_total_mib"]),
                    peak=_format_optional_int(memory["peak_total_mib"]),
                    delta=_format_optional_int(memory["peak_delta_total_mib"]),
                    after=_format_optional_int(memory["after_total_mib"]),
                ))

    diagnostics_section = []
    if diagnostics_rows:
        diagnostics_section = [
            "## Power-SMC Diagnostics",
            "",
            "| Run | Diagnostics | Missing | Mean final ESS | Total resamples | "
            "Max resamples | Mean unique ancestors | Maybe checks | "
            "Boundary checks | KV aliases | KV fallbacks | KV snapshots | "
            "KV alias attempts | KV replay tokens | KV identity noops | "
            "KV aliased blocks | KV aliased tokens | KV physical blocks | "
            "KV saved blocks | KV saved tokens | KV pool total blocks | "
            "KV pool max used | KV pool min free | Mean done | Max done | "
            "Particle len range | Finish reasons | Stop reasons | "
            "Resample skips | Chosen particles |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---|---|---|---|",
            *diagnostics_rows,
            "",
        ]

    memory_section = []
    if memory_rows:
        memory_section = [
            "## GPU Memory",
            "",
            "| Run | Available | Samples | Before total MiB | Peak total MiB | "
            "Peak delta MiB | After total MiB |",
            "|---|---|---:|---:|---:|---:|---:|",
            *memory_rows,
            "",
        ]

    accuracy_section = []
    if accuracy_rows:
        accuracy_section = [
            "## Accuracy",
            "",
            "| Run | Exact match | Pass@1 | EM rate |",
            "|---|---:|---:|---:|",
            *accuracy_rows,
            "",
        ]

    kv_section = []
    if kv_rows:
        kv_section = [
            "## KV Reuse Mode",
            "",
            "| Run | Mode |",
            "|---|---|",
            *kv_rows,
            "",
        ]

    parity_rows = []
    alpha_one_parity = results.get("alpha_one_parity")
    if isinstance(alpha_one_parity, dict):
        for name, check in alpha_one_parity.items():
            if not isinstance(check, dict):
                continue
            parity_rows.append(
                "| {name} | {tok} | {text} | {resamples} | {ess} |".format(
                    name=name,
                    tok="yes" if check.get("token_ids_match_baseline") else "no",
                    text="yes" if check.get("texts_match_baseline") else "no",
                    resamples=_format_optional_int(
                        check.get("total_resample_count")),
                    ess=_format_optional_float(check.get("mean_final_ess")),
                ))

    parity_section = []
    if parity_rows:
        parity_section = [
            "## Alpha=1 Parity",
            "",
            "| Run | Token IDs match baseline | Text matches baseline | "
            "Total resamples | Mean final ESS |",
            "|---|---|---|---:|---:|",
            *parity_rows,
            "",
        ]

    skipped_rows = []
    skipped_runs = results.get("skipped_runs")
    if isinstance(skipped_runs, dict):
        for name, reason in skipped_runs.items():
            skipped_rows.append(f"| {name} | {reason} |")

    skipped_section = []
    if skipped_rows:
        skipped_section = [
            "## Skipped Runs",
            "",
            "| Run | Reason |",
            "|---|---|",
            *skipped_rows,
            "",
        ]

    text = "\n".join([
        "# Power-SMC vLLM Benchmark Report",
        "",
        "## Setup",
        "",
        f"- Model: `{results['model']}`",
        f"- Prompts: `{len(results['prompts'])}`",
        f"- Max tokens: `{results['max_tokens']}`",
        f"- Particles: `{results['particles']}`",
        f"- Block size: `{results['block_size']}`",
        f"- Alpha: `{results['alpha']}`",
        f"- Ignore EOS: `{results.get('ignore_eos', False)}`",
        f"- Stop token IDs: `{results.get('stop_token_ids', [])}`",
        f"- Attention backend: `{results.get('attention_backend') or 'auto'}`",
        "",
        "## Throughput",
        "",
        "| Run | Mean latency (s) | P90 latency (s) | Generated tokens | tok/s |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        *accuracy_section,
        *memory_section,
        *kv_section,
        *diagnostics_section,
        *parity_section,
        *skipped_section,
        "## Notes",
        "",
        "- `baseline_single` is ordinary vLLM sampling with `n=1`.",
        "- `baseline_particles` samples `n=particles` independent completions.",
        "- `best_of_n` selects the independent completion with maximum sampled",
        "  sequence logprob.",
        "- `weighted_best_of_n` samples one independent completion with weights",
        "  proportional to `p(y)^(alpha-1)`.",
        "- `power_smc_wrapper` uses public vLLM APIs with exact `q=p` weights.",
        "- `power_smc_internal_no_cow` uses the V1 engine mode with",
        "  reset/recompute after resampling.",
        "- `power_smc_internal_cow` uses the V1 engine mode with",
        "  power-temperature proposal, sampled-token base/proposal logprobs,",
        "  diagnostics, and replay-safe full-block KV aliasing where safe.",
        "- KV-cache aliasing is implemented for replay-safe full-block",
        "  prefixes; partial-block copy and larger memory-savings benchmarks",
        "  remain future work.",
        "- GPU memory is sampled with `nvidia-smi` and is a node-level",
        "  approximation; it may include other users on shared GPUs.",
        "",
    ])
    path.write_text(text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/data/shared/models/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--alpha-ramp-tokens", type=int, default=16)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--stop-token-id", action="append", type=int, default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-backend")
    parser.add_argument("--memory-sample-interval", type=float, default=0.05)
    parser.add_argument("--disable-memory-monitor", action="store_true")
    parser.add_argument("--kv-pool-diagnostics", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-json", type=Path,
                        default=Path("power_smc_benchmark.json"))
    parser.add_argument("--output-md", type=Path,
                        default=Path("power_smc_benchmark.md"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    examples = load_prompt_examples(args)
    prompts = [example["prompt"] for example in examples]
    answers = [example["answer"] for example in examples]
    log(f"loaded {len(prompts)} prompt(s)")

    from vllm import LLM

    log("initializing LLM")
    attention_config = None
    if args.attention_backend:
        attention_config = {"backend": args.attention_backend}
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=True,
            logprobs_mode="raw_logprobs",
            seed=args.seed,
            attention_config=attention_config,
        )
    except Exception as exc:
        results = {
            "model": args.model,
            "prompts": prompts,
            "answers": answers,
            "max_tokens": args.max_tokens,
            "particles": args.particles,
            "block_size": args.block_size,
            "alpha": args.alpha,
            "ignore_eos": args.ignore_eos,
            "stop_token_ids": args.stop_token_id,
            "attention_backend": args.attention_backend,
            "error": {
                "stage": "llm_initialization",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2),
                                    encoding="utf-8")
        write_markdown_report(results, args.output_md)
        print(f"Wrote {args.output_json}")
        print(f"Wrote {args.output_md}")
        raise
    log("LLM initialized")

    def execute_run(name: str, run_fn: Any) -> dict[str, Any]:
        with GPUMemoryMonitor(
            enabled=not args.disable_memory_monitor,
            interval_s=args.memory_sample_interval,
        ) as memory_monitor:
            result = run_fn()
        result["memory"] = memory_monitor.summary()
        return result

    runs = {}
    skipped_runs: dict[str, str] = {}
    runs["baseline_single"] = execute_run(
        "baseline_single",
        lambda: run_baseline(
            llm,
            prompts,
            answers,
            max_tokens=args.max_tokens,
            n=1,
            temperature=args.temperature,
            ignore_eos=args.ignore_eos,
            stop_token_ids=args.stop_token_id,
            seed=args.seed,
        ),
    )
    runs["baseline_particles"] = execute_run(
        "baseline_particles",
        lambda: run_baseline(
            llm,
            prompts,
            answers,
            max_tokens=args.max_tokens,
            n=args.particles,
            temperature=args.temperature,
            ignore_eos=args.ignore_eos,
            stop_token_ids=args.stop_token_id,
            seed=args.seed,
        ),
    )
    runs["best_of_n"] = execute_run(
        "best_of_n",
        lambda: run_best_of_n(llm, prompts, answers, args, weighted=False),
    )
    runs["weighted_best_of_n"] = execute_run(
        "weighted_best_of_n",
        lambda: run_best_of_n(llm, prompts, answers, args, weighted=True),
    )
    if args.alpha > 1.0:
        runs["power_smc_wrapper"] = execute_run(
            "power_smc_wrapper",
            lambda: run_power_smc(llm, prompts, answers, args),
        )
    else:
        skipped_runs["power_smc_wrapper"] = (
            "external wrapper requires alpha > 1.0")
    runs["power_smc_internal_no_cow"] = execute_run(
        "power_smc_internal_no_cow",
        lambda: run_power_smc_internal(
            llm, prompts, answers, args, kv_cow=False),
    )
    runs["power_smc_internal_cow"] = execute_run(
        "power_smc_internal_cow",
        lambda: run_power_smc_internal(
            llm, prompts, answers, args, kv_cow=True),
    )
    results = {
        "model": args.model,
        "prompts": prompts,
        "answers": answers,
        "max_tokens": args.max_tokens,
        "particles": args.particles,
        "block_size": args.block_size,
        "alpha": args.alpha,
        "ignore_eos": args.ignore_eos,
        "stop_token_ids": args.stop_token_id,
        "attention_backend": args.attention_backend,
        "runs": runs,
    }
    if skipped_runs:
        results["skipped_runs"] = skipped_runs
    alpha_one_parity = build_alpha_one_parity_checks(results)
    if alpha_one_parity is not None:
        results["alpha_one_parity"] = alpha_one_parity

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown_report(results, args.output_md)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
