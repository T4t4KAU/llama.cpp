# Experimental Qwen3 ForkAttention

ForkAttention is an opt-in CUDA/MUSA decode path for unified KV caches with multiple
sequences that share a prefix. The planner reads the real KV-cell sequence
membership, emits exact physical indices for the common prefix and each private
suffix, and dispatches a GPU partial-softmax/merge kernel. Unsupported batches
continue through the regular llama.cpp FlashAttention path.

Enable it with:

```console
llama-server -m qwen3.gguf -ngl 99 -fa on --fork-attn
```

`--fork-attn` enables the unified KV cache. The CUDA kernel supports FP16 KV on
Turing or newer GPUs (for example, Tesla T4) and BF16 KV on Ampere or newer
GPUs. The MUSA kernel currently supports FP16 KV on QY2 or newer GPUs (for
example, MTT S4000). Both paths are selected only for causal Qwen3 decode
batches with 2 to 8 one-token sequences, head dimensions 64 or 128, and a
sufficiently valuable shared prefix. Prefill, single-sequence decode,
unsupported layouts, other model families, and other backends use the native
path.

For a Tesla T4 on Huawei Cloud EulerOS 2.0, use `dnf` rather than the
Debian/Ubuntu `apt` commands. The stock image may have CUDA installed under
`/usr/local/cuda-11.4` without putting its tools on `PATH`, and it may not have
CMake or Ninja installed:

```console
dnf install -y gcc gcc-c++ cmake ninja-build
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cmake -S . -B build-cuda-t4 -G Ninja -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda-t4 -j
```

CUDA 11.4, GCC 10.3, CMake 3.22, and Ninja 1.8 from the HCE image/repositories
are sufficient. Git is optional when building a source archive (without it the
build metadata reports an unknown commit), and `openssl-devel` is only needed
when HTTPS support is required by the HTTP client/server targets.

For an MTT S4000 build:

```console
cmake -S . -B build-musa -G Ninja -DGGML_MUSA=ON \
  -DMUSA_ARCHITECTURES=22 -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-musa -j
```

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
the GPU result with the regular CPU FlashAttention reference.
