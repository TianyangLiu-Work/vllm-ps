# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from collections.abc import Callable
from copy import copy
from typing import Any, cast

from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.power_smc import (
    PowerSMCConfig,
    PowerSMCGroupManager,
    make_power_smc_child_request_id,
)


class ParentRequest:
    """Info, state & processing for parallel sampling request.

    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    """

    request_id: str
    external_req_id: str
    sampling_params: SamplingParams

    # To track the completion of child requests
    child_requests: set[str]

    # To aggregate child completions when not streaming
    output_aggregator: list[CompletionOutput]

    # To find the max number of generated tokens across all children
    max_num_generation_tokens: int

    # To efficiently obtain child sampling params
    cached_child_sampling_params: SamplingParams | None

    def __init__(self, request: EngineCoreRequest) -> None:
        assert request.external_req_id is not None
        sampling_params = request.params
        self.request_id = request.request_id
        self.external_req_id = request.external_req_id
        self.sampling_params = sampling_params

        self.child_requests = set()
        self.output_aggregator = (
            [cast(CompletionOutput, None)] * sampling_params.n
            if (sampling_params.output_kind == RequestOutputKind.FINAL_ONLY)
            else []
        )
        self.max_num_generation_tokens = 0
        self.cached_child_sampling_params = None

    def _get_child_sampling_params(
        self,
        index: int,
    ) -> SamplingParams:
        """Efficiently obtain child `sampling_params`

        If `sampling_params.seed` is not `None` then
        each child request requires a unique clone of
        parent `sampling_params` with a unique seed.

        Args:
          index: index within `n` child requests

        Returns:
          Child `sampling_params` instance.
        """
        seed = self.sampling_params.seed
        if self.cached_child_sampling_params:
            # Reuse child sampling_params data structure
            return self.cached_child_sampling_params
        # Build child sampling_params
        child_sampling_params = copy(self.sampling_params)
        child_sampling_params.n = 1
        if seed is None:
            # Cache child sampling_params for later reuse
            self.cached_child_sampling_params = child_sampling_params
        else:
            # Each child gets a clone with a unique seed
            child_sampling_params.seed = seed + index
        return child_sampling_params

    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        """Get child request ID and sampling params.

        Args:
          index: index within `n` child requests.

        Returns:
          (request ID, sampling_params) tuple
        """
        child_req_id = f"{index}_{self.request_id}"
        self.child_requests.add(child_req_id)
        return child_req_id, self._get_child_sampling_params(index)

    @property
    def n(self) -> int:
        return self.sampling_params.n

    def get_outputs(
        self,
        child_request_id: str,
        completion_output: CompletionOutput,
    ) -> tuple[list[CompletionOutput], bool]:
        already_finished_and_returned: bool = False
        if completion_output.finished():
            if child_request_id in self.child_requests:
                self.child_requests.remove(child_request_id)
            else:
                # child request ID is not available in child_requests
                # which means the request had finished in previous
                # batch step and returned to the client earlier
                already_finished_and_returned = True

        if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
            # If streaming, just return the current output
            #
            # DO NOT output finished and already returned child request to client again
            outputs = [] if already_finished_and_returned else [completion_output]
        else:
            # If not streaming, aggregate the n final outputs.
            self.output_aggregator[completion_output.index] = completion_output
            outputs = [] if self.child_requests else self.output_aggregator

        finished = not self.child_requests
        return outputs, finished

    def observe_power_smc_step(
        self,
        child_request_id: str,
        token_ids: list[int],
        power_smc_logprobs: tuple[float, float] | None,
        finish_reason: str | None,
        stop_reason: int | str | None = None,
    ) -> bool:
        return False

    def get_power_smc_diagnostics(self) -> dict[str, Any] | None:
        return None

    def observe_power_smc_kv_event(self, event: dict[str, Any] | None) -> None:
        return

    def maybe_resample_power_smc(self) -> None:
        return

    def rewrite_power_smc_outputs(
        self,
        outputs: list[CompletionOutput],
        decode_output_token_ids: Callable[[list[int]], str],
    ) -> None:
        return

    def observe_num_generation_tokens(self, num_generation_tokens: int):
        self.max_num_generation_tokens = max(
            num_generation_tokens, self.max_num_generation_tokens
        )
        return self.max_num_generation_tokens

    @staticmethod
    def observe_finished_request(
        parent_req: "ParentRequest | None",
        iteration_stats: IterationStats,
        num_generation_tokens: int,
    ):
        n_param = parent_req.n if parent_req is not None else 1

        if parent_req is not None:
            num_generation_tokens = parent_req.observe_num_generation_tokens(
                num_generation_tokens
            )

        # Child requests finished, we can now record to iteration stats
        if parent_req is None or not parent_req.child_requests:
            iteration_stats.max_num_generation_tokens_iter.append(num_generation_tokens)
            iteration_stats.n_params_iter.append(n_param)


class PowerSMCParentRequest(ParentRequest):
    """Parent request state for Power-SMC particle fanout.

    The engine front-end fans one external request into ``particles`` internal
    child requests. This parent keeps the child outputs hidden, updates
    importance weights from scheduler-provided base/proposal logprobs, and
    returns only the final weighted selection.
    """

    def __init__(self, request: EngineCoreRequest, config: PowerSMCConfig) -> None:
        super().__init__(request)
        self.config = config
        seed = request.sampling_params.seed if request.sampling_params else None
        self.manager = PowerSMCGroupManager(config, random.Random(seed))
        self.child_request_to_index: dict[str, int] = {}
        self.output_aggregator = [
            cast(CompletionOutput, None)
        ] * config.particles
        self.cached_child_sampling_params = None

    @property
    def n(self) -> int:
        return self.config.particles

    def _get_child_sampling_params(self, index: int) -> SamplingParams:
        seed = self.sampling_params.seed
        if self.cached_child_sampling_params is not None:
            return self.cached_child_sampling_params
        child_sampling_params = copy(self.sampling_params)
        child_sampling_params.n = 1
        child_sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        if seed is None:
            self.cached_child_sampling_params = child_sampling_params
        else:
            child_sampling_params.seed = seed + index
        return child_sampling_params

    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        child_req_id = make_power_smc_child_request_id(self.request_id, index)
        self.child_requests.add(child_req_id)
        self.child_request_to_index[child_req_id] = index
        return child_req_id, self._get_child_sampling_params(index)

    def observe_power_smc_step(
        self,
        child_request_id: str,
        token_ids: list[int],
        power_smc_logprobs: tuple[float, float] | None,
        finish_reason: str | None,
        stop_reason: int | str | None = None,
    ) -> bool:
        if not token_ids:
            return False
        if power_smc_logprobs is None:
            raise RuntimeError(
                "Power-SMC child output is missing base/proposal logprobs.")
        if len(token_ids) != 1:
            raise RuntimeError(
                "Power-SMC V1 currently supports one sampled token per "
                f"decode step, got {len(token_ids)}.")
        particle_idx = self.child_request_to_index[child_request_id]
        base_logp, proposal_logq = power_smc_logprobs
        is_block_boundary = self.manager.update_after_token(
            particle_idx,
            token_ids[0],
            base_logp=base_logp,
            proposal_logq=proposal_logq,
            done=finish_reason is not None,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )
        if finish_reason is not None:
            return True
        return is_block_boundary

    def maybe_resample_power_smc(self) -> None:
        self.manager.maybe_resample()

    def observe_power_smc_kv_event(self, event: dict[str, Any] | None) -> None:
        if event is not None:
            self.manager.record_kv_resample_event(event)

    def get_outputs(
        self,
        child_request_id: str,
        completion_output: CompletionOutput,
    ) -> tuple[list[CompletionOutput], bool]:
        if completion_output.finished():
            self.child_requests.discard(child_request_id)
        self.output_aggregator[completion_output.index] = completion_output

        if self.child_requests:
            return [], False

        chosen_idx = self.manager.final_select()
        selected = copy(self.output_aggregator[chosen_idx])
        selected.index = 0
        selected.token_ids = self.manager.particles[chosen_idx].token_ids
        return [selected], True

    def get_power_smc_diagnostics(self) -> dict[str, Any] | None:
        if not self.config.return_diagnostics or self.manager.chosen_particle is None:
            return None
        return self.manager.diagnostics()

    def rewrite_power_smc_outputs(
        self,
        outputs: list[CompletionOutput],
        decode_output_token_ids: Callable[[list[int]], str],
    ) -> None:
        if self.manager.chosen_particle is None:
            return
        for output in outputs:
            output.text = decode_output_token_ids(output.token_ids)
