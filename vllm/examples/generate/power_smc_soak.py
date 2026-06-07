# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-backed lifecycle soak for V1 internal Power-SMC KV aliasing.

This is a deliberately narrow experiment: keep one vLLM engine alive, issue
many independent Power-SMC requests with KV CoW enabled, and record diagnostics
from every completed parent output. The goal is to catch lifecycle problems
that unit tests cannot see, such as repeated alias/free cycles causing OOM,
missing diagnostics, or steadily increasing node-level GPU memory.
"""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = [
    "Solve carefully: what is 19 * 23? Put the final answer in \\boxed{}.",
    "If x + 2y = 11 and x - y = 2, solve for x and y.",
    "Convert the point $(0,3)$ to polar coordinates.",
]


def log(message: str) -> None:
    print(f"[power-smc-soak] {message}", flush=True)


def sample_gpu_memory() -> dict[str, Any]:
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
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    values: list[int] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            values.append(int(stripped))
    if not values:
        return {"available": False, "error": "nvidia-smi returned no rows"}
    return {
        "available": True,
        "per_gpu_mib": values,
        "total_mib": sum(values),
        "max_gpu_mib": max(values),
    }


def load_prompts(path: Path | None, num_prompts: int) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS[:num_prompts]

    prompts: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(),
                                   start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            item = json.loads(stripped)
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{path}:{line_no} JSON row needs a prompt")
            prompts.append(prompt)
        else:
            prompts.append(stripped)
        if len(prompts) >= num_prompts:
            break
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def diagnostics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [row["diagnostics"] for row in rows
                   if isinstance(row.get("diagnostics"), dict)]
    kv_events = [
        event
        for diag in diagnostics
        for event in diag.get("kv_resample_events", [])
        if isinstance(event, dict)
    ]
    gpu_samples = [
        sample for row in rows for sample in (row.get("gpu_before"),
                                              row.get("gpu_after"))
        if isinstance(sample, dict) and sample.get("available")
    ]
    after_totals = [
        int(row["gpu_after"]["total_mib"]) for row in rows
        if isinstance(row.get("gpu_after"), dict)
        and row["gpu_after"].get("available")
    ]

    return {
        "requests": len(rows),
        "with_diagnostics": len(diagnostics),
        "missing_diagnostics": len(rows) - len(diagnostics),
        "total_resample_count": sum(
            int(diag.get("resample_count", 0)) for diag in diagnostics),
        "kv_alias_success_count": sum(
            int(diag.get("kv_alias_success_count", 0)) for diag in diagnostics),
        "kv_alias_fallback_count": sum(
            int(diag.get("kv_alias_fallback_count", 0)) for diag in diagnostics),
        "kv_aliased_blocks": sum(
            int(diag.get("kv_aliased_blocks", 0)) for diag in diagnostics),
        "kv_cow_saved_blocks": sum(
            int(diag.get("kv_cow_saved_blocks", 0)) for diag in diagnostics),
        "kv_cow_saved_tokens": sum(
            int(diag.get("kv_cow_saved_tokens", 0)) for diag in diagnostics),
        "kv_pool_max_used_blocks": max(
            [
                int(event[key])
                for event in kv_events
                for key in ("kv_pool_used_blocks_before",
                            "kv_pool_used_blocks_after")
                if key in event
            ],
            default=0,
        ),
        "kv_pool_min_free_blocks": min(
            [
                int(event[key])
                for event in kv_events
                for key in ("kv_pool_free_blocks_before",
                            "kv_pool_free_blocks_after")
                if key in event
            ],
            default=0,
        ),
        "gpu_peak_total_mib": max(
            [int(sample["total_mib"]) for sample in gpu_samples],
            default=None,
        ),
        "gpu_after_first_mib": after_totals[0] if after_totals else None,
        "gpu_after_last_mib": after_totals[-1] if after_totals else None,
        "gpu_after_delta_mib": (
            after_totals[-1] - after_totals[0]
            if len(after_totals) >= 2 else None),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/shared/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--alpha-ramp-tokens", type=int, default=1)
    parser.add_argument("--ess-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-backend")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-gpu-after-delta-mib", type=int, default=512)
    parser.add_argument("--output-json", type=Path,
                        default=Path("power_smc_soak.json"))
    parser.add_argument("--output-md", type=Path,
                        default=Path("power_smc_soak.md"))
    return parser


def write_markdown(results: dict[str, Any], path: Path) -> None:
    if "error" in results:
        text = "\n".join([
            "# Power-SMC Soak Report",
            "",
            "## Status",
            "",
            f"- Failed during `{results['error']['stage']}`.",
            f"- Error: `{results['error']['type']}: {results['error']['message']}`",
            "",
            "```text",
            results["error"]["traceback"],
            "```",
            "",
        ])
        path.write_text(text, encoding="utf-8")
        return

    summary = results["summary"]
    text = "\n".join([
        "# Power-SMC Soak Report",
        "",
        "## Setup",
        "",
        f"- Model: `{results['model']}`",
        f"- Requests: `{summary['requests']}`",
        f"- Prompts per cycle: `{results['num_prompts']}`",
        f"- Iterations: `{results['iterations']}`",
        f"- Max tokens: `{results['max_tokens']}`",
        f"- Particles: `{results['particles']}`",
        f"- Block size: `{results['block_size']}`",
        f"- Alpha: `{results['alpha']}`",
        "",
        "## Summary",
        "",
        f"- Passed: `{results['passed']}`",
        f"- Diagnostics: `{summary['with_diagnostics']}/{summary['requests']}`",
        f"- Total resamples: `{summary['total_resample_count']}`",
        f"- KV aliases: `{summary['kv_alias_success_count']}`",
        f"- KV fallbacks: `{summary['kv_alias_fallback_count']}`",
        f"- KV aliased blocks: `{summary['kv_aliased_blocks']}`",
        f"- KV saved blocks: `{summary['kv_cow_saved_blocks']}`",
        f"- KV saved tokens: `{summary['kv_cow_saved_tokens']}`",
        f"- KV pool max used blocks: `{summary['kv_pool_max_used_blocks']}`",
        f"- KV pool min free blocks: `{summary['kv_pool_min_free_blocks']}`",
        f"- GPU after delta MiB: `{summary['gpu_after_delta_mib']}`",
        "",
        "## Checks",
        "",
        *[f"- {name}: `{value}`" for name, value in results["checks"].items()],
        "",
    ])
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    prompts = load_prompts(args.prompt_file, args.num_prompts)

    try:
        from vllm import LLM, SamplingParams

        attention_config = None
        if args.attention_backend:
            attention_config = {"backend": args.attention_backend}
        log("initializing LLM")
        llm = LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=True,
            logprobs_mode="raw_logprobs",
            seed=args.seed,
            attention_config=attention_config,
        )
        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            seed=args.seed,
            ignore_eos=args.ignore_eos,
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
                    "kv_cow": True,
                    "kv_pool_diagnostics": True,
                }
            },
        )

        rows: list[dict[str, Any]] = []
        for iteration in range(args.iterations):
            for prompt_index, prompt in enumerate(prompts):
                log(
                    f"request {len(rows) + 1}/"
                    f"{args.iterations * len(prompts)}")
                gpu_before = sample_gpu_memory()
                start = time.perf_counter()
                output = llm.generate([prompt], params, use_tqdm=False)[0]
                elapsed = time.perf_counter() - start
                gpu_after = sample_gpu_memory()
                completion = output.outputs[0]
                rows.append({
                    "iteration": iteration,
                    "prompt_index": prompt_index,
                    "latency_s": elapsed,
                    "generated_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "diagnostics": getattr(output, "power_smc", None),
                    "gpu_before": gpu_before,
                    "gpu_after": gpu_after,
                })

        summary = diagnostics_summary(rows)
        checks = {
            "all_requests_have_diagnostics":
            summary["missing_diagnostics"] == 0,
            "observed_resampling":
            summary["total_resample_count"] > 0,
            "observed_kv_aliasing":
            summary["kv_alias_success_count"] > 0,
            "observed_saved_blocks":
            summary["kv_cow_saved_blocks"] > 0,
            "no_kv_alias_fallbacks":
            summary["kv_alias_fallback_count"] == 0,
            "gpu_after_delta_within_threshold":
            (summary["gpu_after_delta_mib"] is None
             or summary["gpu_after_delta_mib"] <=
             args.max_gpu_after_delta_mib),
        }
        results = {
            "model": args.model,
            "num_prompts": len(prompts),
            "iterations": args.iterations,
            "max_tokens": args.max_tokens,
            "particles": args.particles,
            "block_size": args.block_size,
            "alpha": args.alpha,
            "alpha_ramp_tokens": args.alpha_ramp_tokens,
            "ess_threshold": args.ess_threshold,
            "ignore_eos": args.ignore_eos,
            "attention_backend": args.attention_backend,
            "rows": rows,
            "summary": summary,
            "checks": checks,
            "passed": all(checks.values()),
        }
    except Exception as exc:
        results = {
            "model": args.model,
            "error": {
                "stage": "soak",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, args.output_md)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    if not results.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
