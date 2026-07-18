---
layout: default
title: "Rethinking Inter Node Kv Cache Allocation For High Throughput Inference"
date: 2026-07-18
---

# Rethinking Inter-Node KV Cache Allocation for High-Throughput Inference

![Cover image: rethinking inter node kv cache allocation for high throughput inference](/assets/images/covers/rethinking-inter-node-kv-cache-allocation-for-high-throughput-inference-cover.png)


## How quantization, packing, low-dimensional projection, and layout conversion shrink the data moved between nodes.

**TL;DR**
- The KV cache in large-model inference is bandwidth-hungry across nodes; its cost is dominated by allocation, movement, and deserialization, not just attention arithmetic.
- Quantization, packing, low-dimensional projection, and layout conversion each trade small compute overheads for fewer bytes transferred, better cache-line behavior, and friendlier access patterns.
- A clean read path separates portable “cold” storage from accelerator-local “hot” tensors and rehydrates them just in time for the decode step.

## Why does inter-node KV cache allocation become a bottleneck?

In distributed inference, the dominant cost is often not the attention math itself but getting the right key and value tensors to the right compute unit at the right time.

During decode, every new token attends to the full history of tokens generated so far. That history is materialized as a key-value cache: a dense stack of per-layer, per-head tensors that grows with batch size, sequence length, and model depth. When the cache for a single request no longer fits in one accelerator, or when requests migrate across nodes for load balancing, the system must allocate, serialize, move, and reconstruct those tensors. The result is a set of familiar engineering pains: memory fragmentation from variable-length sequences, hot-spot node links, format mismatches between the storage representation and the GPU kernel, and the sheer byte volume of shipping FP16 or FP32 tensors around.

This is why “KV cache architecture” has shifted from a simple in-memory buffer into a deliberate storage problem. The goal is to make the cache a portable, compact, and layout-aware asset that can live in CPU memory, shared storage, or another node and still reach the compute engine cheaply.

## What do quantization, packing, low-dimensional projection, and layout conversion actually buy us?

Each technique attacks a different part of the byte budget and access pattern.

**Quantization** reduces per-element precision. Dropping from FP16 to INT8 halves the footprint; moving to 4-bit group-wise schemes quarters it. Because the dequantization scale is stored per block, the attention dot product can run in low precision and only rescale when results are accumulated. The main constraint is the block size: too coarse and error accumulates; too fine and metadata overhead swamps the savings.

**Packing** is about removing empty space between values. Once tensors are quantized to small integer types, multiple values can sit in a single machine word or byte. Instead of one Python object or NumPy scalar per element, a flat buffer of packed bits is traversed sequentially. This improves cache-line utilization and reduces allocator pressure.

**Low-dimensional projection** attacks head or layer dimension, not sequence length. A key or value matrix can be approximated or factorized into a lower-rank representation, for example by retaining the top singular components of a block or applying a learned projection. This shrinks the tensor along the `head_dim` axis at the cost of approximation error, so it is usually reserved for long-context summarization, compressed memory tiers, or offline analysis rather than every decode step.

**Layout conversion** reorders data so that the access pattern matches the hardware. In storage, grouping by sequence block or layer may be best; on a GPU, attention kernels often want `[seq, heads, head_dim]` contiguous. A deliberate layout transform—transposition, blocking, or even a separate “cold” and “hot” copy—means the read path pays once for reshaping rather than repeatedly for strided memory access.

Together, these four patterns form a compression and movement pipeline: fewer bytes, denser storage, lower rank, and hardware-aligned access.

## What does a simplified inter-node read path look like?

In practice the path looks like a lazy rehydration pipeline. Cold tensors live on a node that owns storage, CPU memory, or a shared object store. A dispatcher maps requested sequence ranges to the node that holds the quantized blocks. The blocks are unpacked, dequantized, and layout-converted into accelerator-local memory just before the attention kernel consumes them. After decode, new key/value tensors are quantized, packed, and written back to the same schema.

```mermaid
flowchart LR
    Req[Decode request<br/>seq range] --> D[Node dispatcher]
    D -->|read blocks| Cold[Shared / node-local<br/>packed KV store]
    Cold -->|unpack & dequantize| Hot[Accelerator-local<br/>[seq, heads, head_dim]]
    Hot --> Attn[Attention kernel]
    Attn --> New[new KV tensors]
    New --> Q[quantize + pack] --> Cold
```

The important idea is that the interface between nodes is the *packed, quantized, layout-described* asset, not the raw tensor. The receiving node decides when and how to turn it back into dense floating-point memory.

## Implementation pattern: chunked, quantized node cache

The following Python sketch is intentionally simplified. It shows a per-node cache that stores blocks of key or value tensors as 8-bit integer buffers, attaching per-block min/max metadata so the read path can reconstruct the original scale. An optional low-rank projection step is shown on read so the tradeoff between fidelity and footprint is explicit.

```python
import numpy as np

class ChunkedKVCache:
    def __init__(self, block_size: int = 128, rank: int | None = None):
        self.block_size = block_size
        self.rank = rank          # None keeps full dimension
        self.store = {}           # node_id -> (packed_bytes, metadata)

    def allocate(self, node_id: int, tensor: np.ndarray):
        """
        tensor shape: (seq_len, num_heads, head_dim)
        Store per-block quantized + flat-packed representation.
        """
        seq_len, num_heads, head_dim = tensor.shape
        pad = (self.block_size - seq_len % self.block_size) % self.block_size
        if pad:
            tensor = np.pad(tensor, ((0, pad), (0, 0), (0, 0)))

        blocks = tensor.reshape(-1, self.block_size, num_heads, head_dim)
        mn = blocks.min(axis=1, keepdims=True)
        mx = blocks.max(axis=1, keepdims=True)
        scale = (mx - mn) / 255.0 + 1e-8

       量化 = np.round((blocks - mn) / scale).astype(np.uint8)
        # Packing: flatten each block into a dense byte buffer
        packed = [q.tobytes() for q in quantized]

        self.store[node_id] = {
            "packed": packed,
            "shape": (seq_len, num_heads, head_dim),
            "mn": mn,
            "mx": mx,
            "scale": scale,
        }

    def read(self, node_id: int) -> np.ndarray:
        rec = self.store[node_id]
        seq_len, num_heads, head_dim = rec["shape"]
        blocks = []

        for q_bytes, mn, scale in zip(rec["packed"], rec["mn"], rec["scale"]):
            # Unpack bytes back to uint8 block
            q = np.frombuffer(q_bytes, dtype=np.uint8)
            q = q.reshape(self.block_size, num_heads, head_dim)

            # Dequantize
            block = q.astype(np.float32) * scale + mn

            # Optional low-rank projection along head_dim
            if self.rank and self.rank < head_dim:
                # SVD per (heads, head_dim) slice; keep top components
                flat = block.reshape(-1, head_dim)
                u, s, vh = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
                r = min(self.rank, s.shape[0])
                block = (u[:, :r] * s[:r]) @ vh[:r, :]
                block = block.reshape(self.block_size, num_heads, head_dim)

            blocks.append(block)

        tensor = np.concatenate(blocks, axis=0)[:seq_len]
        # Layout conversion: storage was block-major; attention wants seq-major
        return tensor.transpose(0, 1, 2).copy()

# Example usage
cache = ChunkedKVCache(block_size=64, rank=32)
fake_kv = np.random.randn(512, 8, 64).astype(np.float32)
cache.allocate(node_id=3, tensor=fake_kv)
local_tensor = cache.read(node_id=3)   # ready for attention
```

The code is not production-grade—real systems need async prefetching, variable-length sequence indexing, layer-wise sharding, and fused dequantize kernels—but it captures the architectural pattern: serialize to a compact portable form, persist, then reconstruct with a known layout.

## When is each pattern worth applying?

Quantization is almost always the first lever because it reduces both memory and network footprint uniformly. Packing is most valuable when the quantized format leaves many small values scattered across cache lines. Layout conversion matters whenever the storage owner and the compute owner are different pieces of hardware with different optimal access orders. Low-dimensional projection is the most situation-dependent: it introduces approximation, so it tends to appear in compressed memory tiers, summarization passes, or analytical caches rather than the critical decode path of a precision-sensitive model.

The real design challenge is composing them without creating a brittle translation layer. A clean schema—sequence ranges, block sizes, scales, and the target layout—lets each node rehydrate the cache independently, which is exactly what distributed inference needs.

## Topics

KV Cache, Distributed Inference, Quantization, Memory Optimization, Machine Learning Infrastructure, Tensor Layout, Low-Rank Approximation, High-Throughput Services.