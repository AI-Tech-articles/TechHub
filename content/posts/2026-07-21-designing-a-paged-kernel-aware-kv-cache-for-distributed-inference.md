---
layout: default
title: "Designing A Paged Kernel Aware Kv Cache For Distributed Inference"
date: 2026-07-21
---

# Designing a Paged, Kernel-Aware KV Cache for Distributed Inference

![Cover image: designing a paged kernel aware kv cache for distributed inference](/assets/images/covers/designing-a-paged-kernel-aware-kv-cache-for-distributed-inference-cover.png)


**Map logical keys and values to physical GPU pages so attention kernels read contiguous memory without moving tensors across nodes.**

**TL;DR**
- Decoding workloads append a small number of new K/V vectors per forward pass but read the entire history each time, so memory layout matters more than peak compute.
- A logical-to-physical block mapping separates the sequence view from GPU physical memory, letting attention kernels read only the pages that actually contain tokens.
- Keeping page tables, write kernels, and attention kernels aligned is the difference between a cache that scales with batch size and one that silently fragments across nodes.

---

Transformer inference at high throughput spends surprisingly little time waiting for matrix multiplies. The bottleneck is often memory: historical keys and values must stay resident, contiguous, and discoverable by every attention kernel that attends to them. When those K/V tensors cross node boundaries or are copied into ad-hoc buffers, latency grows and batch capacity shrinks.

Our team looks at this as a kernel-co-design problem. The model executor, the cache-write kernels, and the attention kernels all share a single addressable resource: the physical K/V memory. If those three pieces disagree about where a tensor lives, the system pays for it in memcpy, synchronization, or wasted capacity.

## What does a transformer inference workload actually touch?

Each transformer layer maintains a K tensor and a V tensor for every position it has seen. During decoding, a forward pass for one new token typically:

1. Computes a new pair of K/V vectors for the current token.
2. Writes those vectors into the historical K/V store.
3. Runs attention over *all* historical K/V vectors to produce the next output.

The read set is enormous relative to the write set. For a sequence of length *n*, the attention kernel may read *O(n)* vectors to compute a *O(1)* output token. With multiple layers and many outstanding requests, the K/V store becomes the dominant consumer of accelerator memory. The access pattern is also append-heavy: new tokens arrive at one end of the sequence, while reads sweep across the whole prefix.

This asymmetry means good cache architecture is not just about fitting more data. It is about arranging the data so attention kernels read contiguous blocks, write kernels append without reallocation, and the scheduler can pack unrelated requests side by side.

## Why does flat allocation waste memory under variable sequence lengths?

Because reserving a fixed buffer per request for the maximum possible sequence leaves most of that buffer empty until the request finishes.

A straightforward design allots each request a tensor shaped `(max_seq_len, num_layers, 2, num_heads, head_dim)`. That tensor lives on one device, front-loaded with valid data and padded at the end. When a batch contains one long request and many short ones, the short pads go unused. Over hundreds of requests, that unused headroom becomes the largest single consumer of GPU memory.

A paged approach reverses the relationship. Memory is carved into fixed-size physical pages before any request arrives. Each request gets a logical block table that grows one page at a time. A sequence of length 341 with pages of size 16 uses only 22 pages; the 23rd page is allocated only when the 337th token arrives. Short requests do not reserve pages they will never touch, and finished requests release pages to a pool that can be reused immediately.

The same idea applies across nodes. If one node holds pages for layer *l* while the model executor running on another node needs those pages for attention, the system must either move tensors or schedule work to the data. Physical mapping lets the scheduler decide placement independently of the logical sequence shape.

```python
import torch
from typing import List, Optional

class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        max_pages: int,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        # One shared physical pool per device.
        self.pool = torch.zeros(
            (max_pages, num_layers, 2, num_heads, page_size, head_dim),
            dtype=dtype,
        )
        self.free_pages = list(range(max_pages))
        # request_id -> list of physical page ids, one per logical block.
        self.block_tables: dict[int, List[int]] = {}

    def allocate(self, request_id: int, initial_len: int) -> List[int]:
        pages_needed = (initial_len + self.page_size - 1) // self.page_size
        table = [self.free_pages.pop() for _ in range(pages_needed)]
        self.block_tables[request_id] = table
        return table

    def append_page(self, request_id: int) -> int:
        page_id = self.free_pages.pop()
        self.block_tables[request_id].append(page_id)
        return page_id

    def get_block_table(self, request_id: int) -> List[int]:
        return self.block_tables[request_id]

    def write_kv(
        self,
        layer: int,
        page_id: int,
        slot: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        # key, value: (num_heads, head_dim)
        self.pool[page_id, layer, 0, :, slot, :] = key
        self.pool[page_id, layer, 1, :, slot, :] = value

    def read_page(self, layer: int, page_id: int) -> torch.Tensor:
        # Returns K/V for one physical page as a view, no copy.
        return self.pool[page_id, layer]


class ModelExecutor:
    def __init__(self, kv_cache: PagedKVCache):
        self.kv_cache = kv_cache

    def decode_step(self, request_id: int, layer: int, token_id: int):
        """
        Simplified: compute one K/V pair, write it, then attend to
        the sequence through the request's block table.
        """
        block_table = self.kv_cache.get_block_table(request_id)

        # --- cache-write kernel -----------------------------------------
        # In production this is a fused kernel launched per layer.
        key = torch.randn(
            self.kv_cache.num_heads, self.kv_cache.head_dim, dtype=torch.float16
        )
        value = key.clone()

        # Append a new page only when the current logical block fills up.
        total_len = (len(block_table) - 1) * self.kv_cache.page_size
        slot = token_id % self.kv_cache.page_size
        page_id = block_table[-1] if block_table else None
        if slot == 0 or page_id is None:
            page_id = self.kv_cache.append_page(request_id)
            if len(block_table) == 1:
                slot = 0

        self.kv_cache.write_kv(layer, page_id, slot, key, value)

        # --- attention kernel -------------------------------------------
        # The kernel receives the block table and reads pages from
        # the physical pool. No inter-node tensor copy occurs if the
        # pages are already local to the executing device.
        return self.attention_kernel(layer, block_table)

    def attention_kernel(self, layer: int, block_table: List[int]) -> torch.Tensor:
        # Placeholder: a real kernel would call a paged-attention gather.
        pages = torch.stack([self.kv_cache.read_page(layer, pid) for pid in block_table])
        # (num_pages, num_layers omitted, 2, num_heads, page_size, head_dim)
        return pages.sum(dim=(1, 3))  # illustrative reduction


# Example usage
cache = PagedKVCache(
    num_layers=32,
    num_heads=8,
    head_dim=128,
    page_size=16,
    max_pages=4096,
)
executor = ModelExecutor(cache)
executor.decode_step(request_id=7, layer=0, token_id=0)
```

The `PagedKVCache` above is not a full engine, but it captures the important invariant: logical positions are translated to physical `(page_id, slot)` pairs once, then every downstream kernel uses that translation.

## How should the hierarchy be arranged?

Most production systems end up with three distinct layers, each serving a different latency and capacity goal.

The first layer is on-chip storage. A tile of K/V values for the local attention window is loaded into shared memory or registers for the duration of the attention kernel. This layer is fast and small; its lifetime is the kernel invocation.

The second layer is accelerator HBM, organized as the page pool above. This is the authoritative K/V store for all sequences currently in flight. Keeping this pool dense is where most throughput wins come from.

The third layer is host memory or remote storage, used for context eviction, prefix migration, or request parking when GPU capacity is exceeded. Transfers between layer two and layer three are explicit and batched; they should not be on the critical path of decoding.

A clean hierarchy has one rule: a kernel should not have to ask where its data is. The block table answers that question before the kernel launches.

```text
                 ┌──────────────────────────────────────┐
                 │     Model executor (scheduler)       │
                 └───────────────┬──────────────────────┘
                                 │
              logical block table per request
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  Cache-write  │      │  KV page pool │      │  Attention    │
│  kernel       │◄────►│  (device HBM) │◄────►│  kernel       │
│  (append)     │      │  physical     │      │  (read/gather)│
└───────────────┘      │  pages        │      └───────────────┘
                       └───────────────┘
                                 │
                    optional eviction / migration
                                 │
                                 ▼
                       ┌─────────────────┐
                       │ Host / remote   │
                       │ offload buffer  │
                       └─────────────────┘
```

In practice, write kernels append into pages, attention kernels read from pages, and the block table sits in between so both sides can ignore each other's layout details.

## What makes the attention kernel the dominant reader?

Because attention is the only operation that-rather than producing K/V-consults every prior K/V at once.

The cache-write kernel has simple locality: it writes a handful of vectors at the tail of a sequence. The attention kernel has adversarial locality: for each output position it must gather K and V from many positions, and for batched decoding those positions are scattered across different page tables. Without a block mapping, the kernel spends more time computing addresses than performing floating-point operations.

The fix is to pass the block table directly to the attention implementation. The kernel then indexes the page pool in physical order and, internally, implements a gather or fused paged attention routine. This keeps reads coalesced within each page and avoids the indirection step that a fully sparse layout would require.

Teams that skip this step sometimes end up with one of two pathologies. Either pages are stored on a different device than the attention kernel and copied in before every step, or attention is forced to read through a logical index that fragments memory access. Both show up as growing p99 latency as sequence length increases.

## When does inter-node tensor allocation become unavoidable?

When the KV store is larger than a single node and the model layer that needs a particular page does not run on the node that owns it.

Two common strategies exist, and each changes the cache design. In layer-wise pipeline parallelism, each node owns a contiguous slice of layers and also the corresponding KV pages. Attention then runs locally, but activations must move between nodes; the KV cache stays put. In tensor parallelism, each layer is sharded across devices, so the KV pages may be split by head or by dimension; attention kernels coordinate among neighbors. A distributed cache must expose placement metadata alongside block tables so the scheduler can fuse transfers with the compute stream.

What remains constant is that the page is the unit of movement. Moving individual vectors or rows is too fine-grained; moving entire sequences is too wasteful. A page-sized block is the smallest object that still amortizes the transfer setup cost and matches the attention kernel's read granularity.

## Topics

`KV cache` · `transformer inference` · `distributed systems` · `GPU memory management` · `attention kernels` · `paged memory` · `model serving` · `high-throughput ML`