---
layout: default
title: "Designing A Tiered Kv Cache For Distributed Llm Inference"
date: 2026-07-17
---

# Designing a Tiered KV Cache for Distributed LLM Inference

![Cover image: designing a tiered kv cache for distributed llm inference](/assets/images/covers/designing-a-tiered-kv-cache-for-distributed-llm-inference-cover.png)


A practical pattern for inter-node tensor allocation that trades memory cost against decode latency.

**TL;DR**
- KV-cache tensors for long-context inference often exceed GPU memory; treating them as portable, indexed objects lets them live in CPU memory and shared storage.
- An inter-node tensor allocator decides, per chunk, whether to keep a tensor local, evict it, or fetch it — much like a page cache, but for KV tensors.
- The Samsung PM1753 class of high-throughput NVMe devices makes SSD-backed tiers feasible, but allocation policy — not raw hardware bandwidth — decides whether the extra capacity actually helps latency.

## Why does KV cache allocation become a bottleneck at scale?

The KV cache grows with sequence length, batch size, and model depth. In a large transformer, each layer produces a key and value tensor for every token, and decoding a new token must read all of them. When the working set fits in GPU memory, throughput is dominated by compute; when it spills, throughput is dominated by how and where the spill happens.

Teams running distributed inference often hit this wall in one of two ways. A single node may not have enough GPU memory for long contexts, so tensors get pushed to CPU memory or even to swap. Alternatively, multiple nodes may run replicas of the same model and each node re-materializes the same KV tensors for overlapping prompts. Both cases point to the same idea: the KV cache should be treated as a portable, shareable asset rather than a fixed buffer tied to one GPU.

Patterns such as the one described in LMCache turn the cache into exactly that — a tensor store that can move across CPU memory and shared storage and be reused across requests and nodes. The remaining problem is allocation: which tensors stay hot in node-local memory, which are fetched from a peer, and which are pulled from a shared SSD such as the Samsung PM1753.

## What does an inter-node tensor allocator actually do?

The allocator is the boundary between the inference engine and the storage hierarchy. It receives key/value tensors from the model, registers them with a global index, and decides where each chunk lives. When the engine needs a tensor, the allocator either returns it from local memory, fetches it from another node, or reads it from shared storage.

In practice the allocator has the same responsibilities as a page cache:

* **Admission**: accept a new KV tensor into a tier.
* **Placement**: keep it in GPU memory, CPU memory, or write it to shared storage.
* **Eviction**: move the least useful local tensors out to make room.
* **Lookup**: resolve a request prefix or token span to a set of tensor identifiers.

The index is what makes sharing across nodes possible. If two requests start with the same prompt tokens, the index maps both to the same underlying K/V chunks. The allocator then ensures the chunk is available without recomputation.

## How should the cache decide what stays local?

The best policy depends on the workload. Long-context chat sessions have a small set of hot system prompts that many users reuse, while batch summarization may process many one-off documents in parallel. A good allocator therefore uses per-chunk metadata — reuse count, last access time, generation number, and estimated fetch cost — rather than simple FIFO eviction.

The tradeoffs are straightforward but unforgiving:

| Local memory | Shared storage |
|---|---|
| Lowest decode latency | Large capacity, cheap per gigabyte |
| Scarce and expensive | Higher read latency and CPU overhead |
| Best for reusable hot chunks | Best for cold or one-off chunks |

Because the PM1753 and similar NVMe devices can deliver very high sequential bandwidth, a cold tier on fast flash is no longer a theoretical option. But it is still slower than local memory, so prefetching and asynchronous write-back matter almost as much as placement itself. If a batch request is known ten milliseconds in advance, the allocator can move the relevant KV pages from SSD to CPU memory before the decode step starts.

## A concrete allocator pattern

The following Python sketch is intentionally simple, but it captures the core pattern: an indexer that maps a request prefix to tensor IDs, and a tiered allocator that keeps hot tensors in local memory while spilling the rest to a shared store.

```python
import numpy as np
from collections import OrderedDict


class SharedTensorStore:
    def __init__(self):
        self.objects = {}

    def write(self, tensor_id, tensor):
        self.objects[tensor_id] = tensor

    def read(self, tensor_id):
        return self.objects.get(tensor_id)


class TieredTensorAllocator:
    def __init__(self, local_budget_bytes, page_size=4096):
        self.local_cache = OrderedDict()  # simulates CPU/GPU local memory
        self.budget = local_budget_bytes
        self.used = 0
        self.remote = SharedTensorStore()

    def store(self, tensor_id, tensor, hot=True):
        size = tensor.nbytes
        if hot and (self.used + size <= self.budget):
            self.local_cache[tensor_id] = tensor
            self.local_cache.move_to_end(tensor_id)
            self.used += size
        else:
            self.remote.write(tensor_id, tensor)
            self._evict_if_local(tensor_id)

    def load(self, tensor_id):
        if tensor_id in self.local_cache:
            self.local_cache.move_to_end(tensor_id)
            return self.local_cache[tensor_id]

        tensor = self.remote.read(tensor_id)
        if tensor is None:
            return None

        # Make room if needed, then promote to local memory.
        while self.used + tensor.nbytes > self.budget and self.local_cache:
            evicted_id, evicted = self.local_cache.popitem(last=False)
            self.remote.write(evicted_id, evicted)
            self.used -= evicted.nbytes

        self.local_cache[tensor_id] = tensor
        self.used += tensor.nbytes
        return tensor

    def _evict_if_local(self, tensor_id):
        if tensor_id in self.local_cache:
            tensor = self.local_cache.pop(tensor_id)
            self.used -= tensor.nbytes


class KVCacheIndexer:
    def __init__(self, allocator):
        self.allocator = allocator
        self.index = {}

    def put(self, request_id, prefix_hash, k_tensor, v_tensor, hot=True):
        k_id = f"{request_id}:{prefix_hash}:k"
        v_id = f"{request_id}:{prefix_hash}:v"
        self.allocator.store(k_id, k_tensor, hot)
        self.allocator.store(v_id, v_tensor, hot)
        self.index[prefix_hash] = (k_id, v_id)

    def get(self, prefix_hash):
        k_id, v_id = self.index.get(prefix_hash, (None, None))
        if not k_id:
            return None, None
        return self.allocator.load(k_id), self.allocator.load(v_id)


# Illustrative usage
allocator = TieredTensorAllocator(local_budget_bytes=1024 * 1024 * 1024)
indexer = KVCacheIndexer(allocator)

prompt_hash = "0x7a3f"
indexer.put("req-1", prompt_hash, 
            k_tensor=np.ones((128, 64), dtype=np.float32),
            v_tensor=np.zeros((128, 64), dtype=np.float32),
            hot=True)

k, v = indexer.get(prompt_hash)
```

A real implementation would add compression, reference counting, asynchronous copy queues, and block-table mapping similar to PagedAttention. The structure, however, stays the same: index metadata in front, tensors behind it, and an allocator that moves data between tiers.

## Where does inter-node sharing fit in?

Treating the shared store as a common object pool changes how replicas coordinate. Instead of each node computing the KV cache for a new prompt independently, one node can materialize it and publish the chunks. Other nodes then fetch only what they need.

```
┌─────────────────┐     ┌──────────────────┐
│ Inference Node A│────▶│  KV Cache Indexer │
└─────────────────┘     └────────┬─────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐      ┌──────────────────┐
│ Local Memory  │      │  Inference Node B│      │ Shared Storage   │
│ (hot chunks)  │      │  (fetches reuse) │      │ (PM1753-class   │
└───────────────┘      └─────────────────┘      │  NVMe/SSD)       │
        │                        │               └──────────────────┘
        └────────────────────────┘
```

In this flow, the index is global, while the local cache is per node. When an inter-node request arrives, the allocator first checks local memory, then the peer cache protocol, and finally the shared SSD. The exact order can be tuned: for small tensors, asking a peer may be faster than a disk read; for large tensors, the reverse can be true.

## What to watch when productionizing

Several practical details determine whether the architecture actually saves money or latency.

**Chunk granularity** matters. Very small chunks waste index space and serialize poorly; very large chunks waste bandwidth when only a few tokens are reused. Most systems settle on a block table with fixed-size pages.

**Reference counting and consistency** are essential when multiple nodes share chunks. The index must know when a chunk is still alive and when it can be garbage collected.

**Compression** can trade a small CPU cost for a large reduction in storage bandwidth. On fast NVMe, decompression often overlaps with the next decode step.

**Prefetching** based on known request prefixes can hide SSD latency. Without it, teams often see p99 decode latency roughly double the moment the active working set spills from local memory.

## Topics

`LLM inference` · `KV cache` · `distributed systems` · `memory hierarchy` · `tensor allocation` · `high-throughput serving` · `storage tiering` · `LMCache` · `NVMe`