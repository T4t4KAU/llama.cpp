# Experimental Qwen3 ForkAttention

ForkAttention is an opt-in CUDA decode path for unified KV caches with multiple
sequences that share a prefix. The planner reads the real KV-cell sequence
membership, emits exact physical indices for the common prefix and each private
suffix, and dispatches a CUDA partial-softmax/merge kernel. Unsupported batches
continue through the regular llama.cpp FlashAttention path.

Enable it with:

```console
llama-server -m qwen3.gguf -ngl 99 -fa on --fork-attn
```

`--fork-attn` enables the unified KV cache. The current CUDA kernel is selected
only for causal Qwen3 decode batches with 2 to 8 one-token sequences, FP16 or
BF16 KV, head dimensions 64 or 128, and a sufficiently valuable shared prefix.
Prefill, single-sequence decode, unsupported layouts, other model families, and
non-CUDA backends use the native path.

The server also changes idle-slot handling while ForkAttention is enabled.
Idle sequence states stay in the GPU KV cache until allocation pressure occurs.
At that point, the server saves the lowest-value idle sequence to its RAM prompt
cache and releases its unified KV cells. Shared prefixes used by active or other
idle slots receive a higher reuse value and are retained longer. A later prompt
cache hit restores the sequence state before execution.

This first implementation uses llama-server's synchronous, whole-sequence RAM
state cache. It releases logical unified-KV capacity, but it does not shrink the
preallocated CUDA KV buffer or provide asynchronous pinned-memory page
transfers. `--cache-ram 0` disables the RAM tier and keeps the existing purge
fallback. Planner and RAM-tier counters are emitted in the logs.

For a shared-prefix integration benchmark:

```console
llama-parallel -m qwen3.gguf -ngl 99 -fa on -kvu \
  -np 4 -ns 4 -pps -n 128 --temp 0 -s 123

llama-parallel -m qwen3.gguf -ngl 99 -fa on --fork-attn \
  -np 4 -ns 4 -pps -n 128 --temp 0 -s 123
```

The backend correctness test contains an exact physical-KV plan and compares
the CUDA result with the regular CPU FlashAttention reference.
