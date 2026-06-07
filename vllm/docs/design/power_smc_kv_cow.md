# Power-SMC KV Cache CoW Plan

This note records the current Power-SMC KV behavior and the narrow interface
needed for block-level copy-on-write resampling in vLLM V1.

## Current State

Power-SMC internal decoding now uses scheduler-side resampling with
snapshot-based KV prefix aliasing where it is safe, plus conservative
reset/recompute fallback:

1. A parent request expands into particle child requests.
2. Each child owns normal KV blocks while it runs.
3. At a Power-SMC block boundary, ESS can trigger resampling.
4. The scheduler snapshots every child request's KV blocks before mutating any
   child, so simultaneous resampling cannot read a source after it has already
   been rewritten.
5. For uniform KV block groups, the scheduler aliases the replay-safe full-block
   prefix from the selected ancestor into the destination child. It deliberately
   does not alias the final block that must be replayed to obtain next-token
   logits, avoiding writes into a shared tail block.
6. If no full replay-safe block exists, the KV layout is not uniform, or aliasing
   fails validation, the scheduler falls back to reset/recompute.

The benchmark labels this mode as
`scheduler_snapshot_alias_replay_with_reset_fallback`.

The mode is controlled by `extra_args["power_smc"]["kv_cow"]`:

1. `kv_cow=True` is the default and enables replay-safe full-block aliasing.
2. `kv_cow=False` disables aliasing and uses `scheduler_reset_recompute`.
3. The benchmark runs both modes as `power_smc_internal_cow` and
   `power_smc_internal_no_cow`.

## CoW Target

For decoder-only full-attention models, when
`power_smc.block_size % kv_block_size == 0`, a resampled child should alias the
ancestor's full-block KV prefix instead of freeing and recomputing it.

The low-level operations are:

```text
alias_request_blocks(dst_request_id, src_request_id, num_prefix_blocks)
alias_request_blocks_from_snapshot(
    dst_request_id,
    src_blocks,
    src_num_cached_blocks,
    num_prefix_blocks,
)
```

For every KV cache group, this operation should:

1. Truncate and free any existing `dst_request_id` blocks not retained.
2. Copy the first `num_prefix_blocks` `KVCacheBlock` objects from
   `src_request_id` into `dst_request_id`.
3. Call `BlockPool.touch()` on the copied non-null blocks so their refcounts
   represent the additional child owner.
4. Set the destination cached-block count consistently with the aliased prefix,
   so later `cache_blocks()` does not try to re-cache blocks whose hashes are
   already set.
5. Leave the next write position at the first token after the aliased prefix.

## Scheduler Invariants

The scheduler-side resample path must maintain these invariants before the
child is put back into the waiting queue:

1. `request.output_token_ids` exactly matches the ancestor prefix chosen by
   resampling.
2. The aliased prefix length is a full KV block boundary for every KV group.
3. `request.num_computed_tokens` matches the number of tokens whose KV blocks
   are available, except for any vLLM-required final-token replay.
4. `request.num_output_placeholders`, `request.spec_token_ids`, and async
   discard counters are cleared.
5. Releasing any particle decrements only its block references and cannot free
   blocks still referenced by other particles.
6. The worker block table row is rebuilt from the scheduler block IDs before
   the next forward pass.

## Validation

Covered unit tests:

1. Resampling duplicates one ancestor into multiple children and increases
   shared block refcounts.
2. Freeing one duplicate child leaves the shared ancestor blocks allocated.
3. Freeing all duplicate children returns shared blocks to the free queue.
4. Snapshot-based aliasing supports simultaneous swaps without reading a source
   after it has already been rewritten.
5. Scheduler resampling uses pre-resampling snapshots, clears transient request
   state, and sets `num_computed_tokens` to the aliased replay-safe prefix.
6. Fallback to `scheduler_reset_recompute` remains available.
7. A Slurm smoke with `ESS_THRESHOLD=1.0`, `ALPHA_RAMP_TOKENS=1`, and
   `MAX_TOKENS=20` forces an internal Power-SMC resample and completes with
   `resample_count=1`.

Still needed:

1. Memory benchmarks with enough prompt/output length and particles to show a
   measurable KV-memory difference versus reset/recompute.
