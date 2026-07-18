---
layout: default
title: "Beyond Hit Rate Rethinking Kv Cache Placement In Distributed Vllm"
date: 2026-07-18
---

# Beyond Hit Rate: Rethinking KV-Cache Placement in Distributed vLLM

![Cover image: beyond hit rate rethinking kv cache placement in distributed vllm](/assets/images/covers/beyond-hit-rate-rethinking-kv-cache-placement-in-distributed-vllm-cover.png)


## Inter-node tensor allocation is the bottleneck that cache hit rates alone cannot describe.

**TL;DR**
- In distributed vLLM inference, a high KV-cache hit rate does not always mean low latency: the placement of tensor blocks across ranks matters as much as whether they are reused.
- A practical next-gen pattern separates a global **cache index** (metadata about which rank owns each KV block) from per-rank **node-local caches**, and exposes both to the scheduler.
- The scheduler should optimize for *where* tensors live, not just *whether* they are cached; prefetching, prefix-aware routing, and eviction-aware placement often outperform naive LRU.

## What makes KV-cache placement hard in distributed inference?

In single-node LLM inference, the KV cache is straightforward in concept: for each layer and each token, store the key and value activations so future tokens can attend to them without recomputation. vLLM already packs these tensors intoPagedAttention blocks, reusing prefixes across requests. Once the model is sharded across many nodes—via tensor parallelism, pipeline parallelism, or both—the cache stops being a local lookup table and becomes a distributed tensor placement problem.

The real question is not “is this prefix in memory?” but rather:

- Which ranks already own the corresponding KV blocks?
- Does a new request need the same blocks on different ranks, or a different subset?
- Is it cheaper to transfer a block over the network or recompute it locally?
- Can we route the request so its required blocks already reside on the nodes it will execute on?

A high cache hit rate can mask these costs. If 80% of prefix lookups hit a node-local cache, but the remaining 20% require cross-node transfers during the decode phase, tail latency can jump. The draft note in many systems about unexpectedly high hit rates is a reminder that simple metrics deceive: once reuse is high, the engineering margin comes from *where* the reuse occurs.

## Why does a high cache hit rate not guarantee low latency?

Because “hit” conflates two very different outcomes: a local read from HBM or DRAM, and a remote fetch across an RDMA link or even an Ethernet backbone. The former is measured in microseconds; the latter can be an order of magnitude more expensive once queueing and serialization are included.

In vLLM, each request’s KV cache is split per layer and per attention head. Under tensor parallelism, a single logical tensor is sharded across ranks; under pipeline parallelism, different layers live on different nodes. A cache-aware scheduler therefore needs to know, for each block, not only that it exists but which physical devices hold it.

Once a request is scheduled on a node whose local ranks do *not* own its blocks, the engine has three choices, each with a tax:

1. **Remote fetch** — move KV blocks to the executing rank. Fast on modern RDMA, but it consumes network bandwidth that competes with ongoing all-reduce traffic.
2. **Recompute** — regenerate the blocks from input tokens. More compute, but avoids transfer; sometimes cheaper for short prefixes on fast GPUs.
3. **Route elsewhere** — send the request to a node that already owns the blocks. This can be optimal, but only if the scheduler has a global view of cache ownership.

Teams running distributed inference often see p99 latency roughly double when remote KV fetches become common, even if aggregate hit rates stay above 70%. The lesson is that hit rate is a throughput signal; placement is a latency signal. The architecture must optimize both.

## A practical three-layer architecture

A next-generation pattern, useful as a conceptual scaffold and implementable inside or alongside vLLM, splits the problem into metadata, locality, and scheduling.

### 1. Distributed cache index

A lightweight metadata service—can be Redis, an etcd key prefix, or simply a replicated table in the inference controller—that maps each cache block ID to one or more physical locations. It stores *where* tensors are, not the tensors themselves. For example:

```
block_id → {rank: 3, node: "gpu-node-7", layer_range: [0-31], ref_count: 12}
```

The index is updated on allocation, eviction, and migration. It can afford to be slightly stale for read-only prefetches, but must be strongly consistent when the scheduler commits a placement decision.

### 2. Node-local cache

Each worker maintains a bounded pool of KV tensors. This is the layer where capacity, eviction, and on-device layout matter. A node-local cache should be aware of both the vLLM block manager and the network topology: blocks belonging to heavily shared prefixes are more valuable than blocks touched by a single long decode.

### 3. Cache-aware scheduler

The scheduler is the new layer of interest. Instead of assigning requests to the least-loaded worker, it treats KV-block ownership as a placement constraint. It can prefetch blocks, batch requests that share prefixes, or pin high-value prefix blocks across nodes.

Here is an ASCII view of the data flow:

```
   New request with prefix P
           |
           v
  +-------------------+
  | Cache-Aware       | --(lookup)--> Distributed Cache Index
  | Scheduler         |                block_id -> rank/node set
  +-------------------+
           |
     +-----+-----+
     |           |
  route to    schedule on
  node owning any node,
  prefetch    remote fetch
  blocks      as needed
```

## Code sketch: a cache-aware scheduler

The following Python is intentionally simplified. It is meant to illustrate the pattern, not to replace vLLM’s block manager. In production, the metadata would live in the engine/controller, block IDs would correspond to vLLM blocks, and tensor movement would go over NCCL or RDMA, not Redis.

```python
from dataclasses import dataclass
from typing import Optional, Set
import numpy as np

# ------------------------------------------------------------------
# 1. Distributed cache index  (metadata only)
# ------------------------------------------------------------------
class DistributedCacheIndex:
    def __init__(self):
        # block_id -> set of (node, rank)
        self.ownership = {}

    def locate(self, block_id: str) -> Set[tuple]:
        return self.ownership.get(block_id, set())

    def register(self, block_id: str, node: str, rank: int):
        self.ownership.setdefault(block_id, set()).add((node, rank))

    def evict(self, block_id: str, node: str, rank: int):
        self.ownership.get(block_id, set()).discard((node, rank))

# ------------------------------------------------------------------
# 2. Node-local cache  (bounded, annotates blocks with reuse score)
# ------------------------------------------------------------------
@dataclass
class CachedBlock:
    tensor: np.ndarray
    ref_count: int = 1
    last_access: float = 0.0

class NodeLocalCache:
    def __init__(self, capacity: int, node_id: str, rank: int):
        self.capacity = capacity
        self.node_id = node_id
        self.rank = rank
        self.cache: dict[str, CachedBlock] = {}

    def get(self, block_id: str) -> Optional[np.ndarray]:
        item = self.cache.get(block_id)
        if item is None:
            return None
        item.ref_count += 1
        return item.tensor

    def put(self, block_id: str, tensor: np.ndarray, time: float):
        if len(self.cache) >= self.capacity:
            # Cheap eviction: prefer blocks with low ref_count and old access.
            victim = min(
                self.cache,
                key=lambda k: (self.cache[k].ref_count, self.cache[k].last_access),
            )
            del self.cache[victim]
            global_index.evict(victim, self.node_id, self.rank)

        self.cache[block_id] = CachedBlock(tensor=tensor, last_access=time)
        global_index.register(block_id, self.node_id, self.rank)

# ------------------------------------------------------------------
# 3. Cache-aware scheduler  (prefers nodes that already own the blocks)
# ------------------------------------------------------------------
class CacheAwareScheduler:
    def __init__(self, index: DistributedCacheIndex, nodes: list[NodeLocalCache]):
        self.index = index
        self.nodes = {n.rank: n for n in nodes}

    def place_request(self, block_ids: list[str], current_time: float):
        # Candidate scores: how many required blocks are already local?
        best_rank = None
        best_score = -1
        for rank, node in self.nodes.items():
            score = sum(1 for b in block_ids if node.get(b) is not None)
            if score > best_score:
                best_score = score
                best_rank = rank

        # If no node owns any block, fall back to least-loaded.
        if best_score == 0:
            best_rank = min(self.nodes, key=lambda r: len(self.nodes[r].cache))

        return best_rank

# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
global_index = DistributedCacheIndex()

node_a = NodeLocalCache(capacity=1000, node_id="gpu-node-1", rank=0)
node_b = NodeLocalCache(capacity=1000, node_id="gpu-node-2", rank=1)

scheduler = CacheAwareScheduler(global_index, [node_a, node_b])

# A prefix is needed for a new decode batch.
prefix_blocks = ["prefix_layer_0", "prefix_layer_1"]
for i, bid in enumerate(prefix_blocks):
    node_a.put(bid, np.random.rand(128, 4096).astype(np.float16), time=0.0)

target_rank = scheduler.place_request(prefix_blocks, current_time=1.0)
print(f"Route request to rank {target_rank}")
```

A few details worth noting:

- The scheduler does not blindly maximize hits; it maximizes *local* hits after the routing decision is made.
- `ref_count` is a proxy for prefix sharing. Systems with large prompt caches would weight blocks by estimated future reuse, not just recency.
- In tensor parallelism, a single request needs several ranks simultaneously, so placement becomes a constraint-satisfaction problem rather than a single-node choice.

## What should you measure?

Teams should track hit rate, but decompose it. Useful dashboards include:

- **Local hit rate**: fraction of block lookups satisfied without cross-node transfer.
- **Remote fetch rate**: fraction requiring inter-node tensor movement, and bytes moved per decode step.
- **Eviction churn**: blocks evicted before reuse, weighted by prefix popularity.
- **Time-to-first-token latency** for prefix-cached vs. cold-start requests.

Without the remote/local split, a vLLM service can quietly degrade as prefix reuse grows, because every new shared prefix crowds out the existing ones and forces more cross-node traffic. Cache capacity planning must then account for network bandwidth, not just GPU HBM.

## Where this pattern fits in the vLLM stack

The architecture described here sits at the orchestration layer. vLLM already handles block tables, PageAttention, and per-request KV management. The next step is exposing ownership metadata to the scheduler so it can:

1. Pin high-value prefix blocks on prefill nodes and push them to decode nodes only when needed.
2. Co-locate decode requests that reuse the same technical documentation or conversation history.
3. Trigger prefetch when a multi-turn chat session is likely to continue.

In disaggregated prefill/decode deployments, this becomes even more important: the prefill node computes KV blocks once, but the decode cluster has many candidate locations. A scheduler with a true cache index can route the decode request to the replica holding the block rather than recomputing or copying.

## Topics

Distributed Systems · vLLM · LLM Inference · KV Cache · GPU Scheduling · Tensor Placement · Cache-Aware Scheduling · Latency Optimization · ML Platform Engineering