---
layout: default
title: "Asynchronous Sequence Batching For Kv Cache Workloads"
date: 2026-07-17
---

# Asynchronous Sequence Batching for KV Cache Workloads
**Bottlenecks that look like cache problems are usually coordination problems.**

![Cover image: asynchronous sequence batching for kv cache workloads](/assets/images/covers/asynchronous-sequence-batching-for-kv-cache-workloads-cover.png)


**TL;DR**
- KV cache throughput often stalls on synchronous request round-trips and small, random fetches rather than on raw storage speed.
- Asynchronous sequence batching groups concurrent requests into bounded windows and dispatches them as multi-key operations, amortizing network and CPU overhead.
- Production implementations need bounded wait times, concurrency limits, and backpressure — a larger queue alone just hides failures until later.

## Why does synchronous KV access hit a wall?

The round-trip dominates. Even when the backing store is fast, each `get` pays for serialization, a network hop, a thread or async task handoff, and cache-coherence checks. At high concurrency those fixed costs multiply: context switches pile up, connection pools saturate, and hot keys serialize through locks. Random access patterns make it worse because the working set stops fitting in CPU caches and NIC buffers, so every small request becomes a full cache-line miss on both sides of the wire.

This pattern shows up whenever many callers need similar data at similar times — feature stores, vector lookups, graph caches, and inference KV caches all share the same shape. The data is often small, the latency target is tight, and the request rate is high. Single-request handling leaves bandwidth and batching opportunities on the table.

Asynchronous sequence batching reframes the problem. Instead of answering each request immediately, the cache layer collects requests into short, ordered windows and fulfills them together with a backend multi-get or batched tensor operation. Callers still use a simple `await cache.get(key)` interface, but underneath their futures are resolved by a shared batch. The sequence part matters because keeping related keys in arrival order — or at least preserving enough ordering for correctness — avoids silent reordering bugs when batches overlap.

## How async sequence batching works

The architecture is small but has several moving parts. A front-end queue accepts `get` requests wrapped in futures. A batcher loop drains the queue until one of two things happens: the window reaches a maximum batch size, or a short deadline expires. The batcher then hands the collected keys to a worker pool that issues a single batched backend operation. Results are mapped back to their original futures, which wake the callers.

```mermaid
graph LR
    A[Client<br/>Requests] -->|key + Future| B[Async Batcher]
    B -->|window full or deadline| C[Batch]
    C -->|multi-get| D[KV Backend]
    D -->|results map| E[Fulfill<br/>Futures]
    E --> A
```

The key constraints are boundedness and concurrency. An unbounded queue lets you collect huge batches, but it also buries backpressure and inflates tail latency. A `max_wait` deadline caps how long any single caller waits for stragglers. A semaphore limits how many batches hit the backend at once. Together these keep the cache from becoming a memory-consuming black hole under load.

## Production-ready batching in Python

The example below is intentionally self-contained. It uses `asyncio` for the event loop, an `asyncio.Queue` with a capacity limit as backpressure, a deadline-based batcher, and a semaphore to bound in-flight backend calls. A tiny local hot-value cache deduplicates repeated reads inside the same process.

```python
import asyncio
from typing import Optional, Tuple


class AsyncBatchingCache:
    def __init__(
        self,
        max_batch: int = 128,
        max_wait_ms: float = 2.0,
        queue_cap: int = 1024,
        max_inflight: int = 4,
    ):
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.queue: asyncio.Queue[Tuple[str, asyncio.Future]] = asyncio.Queue(
            maxsize=queue_cap
        )
        self.inflight = asyncio.Semaphore(max_inflight)
        self.local_cache: dict[str, Optional[str]] = {}
        asyncio.create_task(self._batcher_loop())

    async def get(self, key: str) -> Optional[str]:
        """Blocking-style API; returns when the next batch resolves."""
        if key in self.local_cache:
            return self.local_cache[key]

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()

        try:
            self.queue.put_nowait((key, fut))
        except asyncio.QueueFull:
            fut.set_exception(RuntimeError("backpressure: cache queue full"))
        return await fut

    async def _batcher_loop(self) -> None:
        while True:
            first_key, first_fut = await self.queue.get()
            batch: list[Tuple[str, asyncio.Future]] = [(first_key, first_fut)]
            seen = {first_key}
            deadline = asyncio.get_event_loop().time() + self.max_wait

            while len(batch) < self.max_batch:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    key, fut = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    if key not in seen:
                        batch.append((key, fut))
                        seen.add(key)
                except asyncio.TimeoutError:
                    break

            async with self.inflight:
                await self._dispatch(batch)

    async def _dispatch(self, batch: list[Tuple[str, asyncio.Future]]) -> None:
        keys = [k for k, _ in batch]
        # Stand-in for an actual backend multi-get.
        results = {k: f"value-of-{k}" for k in keys}

        for key, fut in batch:
            if not fut.done():
                value = results.get(key)
                fut.set_result(value)
                self.local_cache[key] = value


async def main():
    cache = AsyncBatchingCache(max_batch=64, max_wait_ms=1.5, max_inflight=2)

    # Simulate a burst of concurrent readers.
    coros = [cache.get(f"user:{i % 40}") for i in range(500)]
    values = await asyncio.gather(*coros, return_exceptions=True)
    print(f"resolved {len(values)} values, first three: {values[:3]}")


if __name__ == "__main__":
    asyncio.run(main())
```

A few details are worth emphasizing because they are easy to shortcut in a prototype:

- **Backpressure** matters. If `put_nowait` raises `QueueFull`, the caller gets an exception immediately instead of queuing forever. In a real service this maps cleanly to HTTP 503 or gRPC `UNAVAILABLE`.
- **Deduplication within the window** avoids double-fetching the same hot key twice in one batch. The `seen` set handles this for gets; updates need stricter ordering guarantees.
- **Bounded wait time** keeps p99 latency predictable. A batch of two should not wait for a batch of one hundred.

## When does async batching stop helping?

Unbounded buffering turns the cure into the disease. Teams running distributed inference often see p99 latency roughly double when the queue depth exceeds a small multiple of the batch size, because late-arriving requests wait behind earlier ones while the batcher holds the window open. The real fix is tighter deadlines, smaller batch caps, or more backend workers — not a longer queue.

Batching also loses value when the workload is already sparse or highly heterogeneous. If every key maps to a different server shard, a single multi-get still fans out internally and may not save much. If request rates are low, holding a window open for a few milliseconds adds latency that pure synchronous access would avoid. In those cases a hybrid approach works better: bypass the batcher when concurrency is below a threshold and only batch during bursts.

Another common mistake is to grow the batch size without growing backend capacity. A larger batch increases throughput per operation, but it also increases per-operation CPU time and can cause head-of-line blocking at the storage layer. The right batch size is usually the one that saturates the backend without overwhelming it, and it should be tuned against measured CPU, network, and queueing delay — not guessed.

## Beyond the single cache: compositional patterns

In larger systems, async batching composes well with other patterns. A hot-key partition can route frequently accessed keys to a separate, smaller batcher so that one viral key does not dominate every window. A cache-aside layer can prefetch adjacent sequence positions, which is particularly useful for inference KV caches where generation is inherently sequential. Sharded batchers can each own a key range, reducing lock contention and making the system easier to scale horizontally.

Metrics should expose the batcher internals directly: average batch size, batch fill ratio, how often the deadline fires before the batch is full, and time spent waiting in queue. Those four numbers usually tell you whether the problem is too few requests, too tight a deadline, or an overwhelmed backend.

## Takeaways

- **Start with the API.** `await cache.get(key)` is a good contract because callers do not need to know whether their request is batched.
- **Bound everything.** Cap batch size, wait time, queue depth, and in-flight backend calls. Batching without bounds trades throughput for tail latency.
- **Deduplicate inside the window.** Re-fetching the same key in one batch wastes backend capacity and confuses latency measurements.
- **Measure before tuning.** Average batch size and deadline-hit rate are better guides than rules of thumb about batch size.
- **Know when not to batch.** Low-concurrency or highly sharded workloads may be faster with direct synchronous access.

---

*This post was drafted and edited with assistance from an AI language model.*

## Topics

kv-cache · asynchronous-batch-processing · high-throughput-systems · python-asyncio · distributed-systems · caching-patterns · data-platform-engineering · inference-serving