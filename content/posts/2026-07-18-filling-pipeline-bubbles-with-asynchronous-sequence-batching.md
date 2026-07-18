---
layout: default
title: "Filling Pipeline Bubbles With Asynchronous Sequence Batching"
date: 2026-07-18
---

# Filling Pipeline Bubbles with Asynchronous Sequence Batching

![Cover image: filling pipeline bubbles with asynchronous sequence batching](/assets/images/covers/filling-pipeline-bubbles-with-asynchronous-sequence-batching-cover.png)


**A practical pattern for high-throughput services that split inference or transformation work across multiple nodes.**

**TL;DR**
- Pipeline parallelism can leave nodes idle when every stage waits for the slowest stage of a batch to finish.
- Asynchronous sequence batching splits work into independent sequences and places bounded buffers between stages so each node can pull its next input as soon as it is free.
- The pattern is easy to prototype with async queues; production deployments need backpressure, failure isolation, and careful queue sizing.

When one machine can no longer fit the model or the request volume, engineering teams partition the workload into stages and assign each stage to its own node. That is the essence of pipeline parallelism. It works, but only if the pipeline stays full. In practice, a naive implementation often stalls: every node in a stage waits for the slowest node, while downstream hardware sits idle. The result is a pipeline bubble, and bubbles eat throughput.

A simple way to shrink those bubbles is asynchronous sequence batching. The idea is to split incoming work into independent sequences—tokens, requests, or mini-batches—and let each stage consume a sequence from a buffer as soon as it is ready. Upstream stages do not block on downstream completion; they keep producing until their output buffer is full. This keeps more hardware busy, more of the time.

This post looks at why synchronous batching stalls, how asynchronous sequence batching changes the timing, and what a minimal implementation looks like.

## Why does synchronous batching stall a distributed pipeline?

It forces all stages to advance in lockstep, so any slow stage makes every other stage wait.

In classic batched inference, a single batch flows through every stage before the next batch is allowed to start. If a pipeline has three stages—embed, attention, decode—the second stage cannot begin batch N+1 until it has finished batch N. The first stage cannot start batch N+2 until the previous batch has cleared. Every stage is chained to the slowest operation in the current batch.

Across nodes, the problem is worse. Network latency, garbage collection, thermal throttling, or a single straggler request can stretch one stage. Under lockstep execution, that delay propagates backward through the entire pipeline. Nodes that finished their portion early wait. Throughput drops to the rate of the slowest stage, not the sum of what the hardware could do. This is the same head-of-line-blocking problem seen in networking: one slow packet at the front of a queue delays everything behind it.

The wasted time is easy to miss in small benchmarks. On a single machine with synthetic data, every stage finishes in roughly the same moment. At scale, variance is the rule, and variance plus lockstep equals idle GPUs or CPUs.

## How does asynchronous sequence batching keep every stage busy?

It decouples sequences with bounded buffers, so a slow stage only blocks its own queue, not the whole pipeline.

Instead of treating a batch as an indivisible unit, split it into sequences. Each sequence carries just enough state to move through the stages independently. Stage one pulls sequence k from its input buffer, processes it, and places the result in the buffer in front of stage two. Stage two does the same for sequence k-1 concurrently. The stages overlap; at steady state, all of them are computing different sequences at the same time.

The buffer between stages is what makes this possible. If stage two is temporarily slow, stage one can keep working until its output buffer reaches capacity. Stage three can keep draining its own input buffer while stage two catches up. Progress becomes local. A slow stage creates a backlog at its input, but it does not immediately stall the entire system.

Buffers must be bounded. Unbounded queues lead to memory exhaustion, especially when downstream stages fail or slow down for minutes. A bounded queue also provides backpressure: when a buffer is full, the upstream stage stops producing until space opens up. That protects the service from unbounded latency growth.

Themermaid
flowchart LR
    P[Producer] -->|seq 1, 2, 3...| Q1[Queue: embed buffer]
    Q1 --> S1[Node A: embed]
    S1 -->|seq-k output| Q2[Queue: attention buffer]
    Q2 --> S2[Node B: attention]
    S2 -->|seq-k output| Q3[Queue: decode buffer]
    Q3 --> S3[Node C: decode]
    S3 -->|completed seq| Qout[Output collector]
```

This pattern does not eliminate all waiting. If one stage is permanently slower than the rest, it will still be the bottleneck; the pipeline cannot drain faster than its slowest stage. What it eliminates is the synchronous ripple effect. Stages can operate at their own natural cadence.

A compact buffer between every node turns a rigid assembly line into an elastic one.

## A minimal implementation with async queues

The abstraction is straightforward: each stage is a worker that reads from one queue and writes to the next. In Python, `asyncio` gives a readable model for the pattern. A real system might use gRPC, ZeroMQ, Redis Streams, or a workflow orchestrator, but the queue-and-worker shape is the same.

```python
import asyncio
import random

async def stage(name, in_q, out_q, base_delay):
    """Consume sequences from in_q, process, and push to out_q."""
    while True:
        seq_id, payload = await in_q.get()
        if seq_id is None:               # poison pill: propagate shutdown
            await out_q.put((None, None))
            break
        # Simulate variable compute per sequence
        await asyncio.sleep(base_delay * random.uniform(0.7, 1.3))
        await out_q.put((seq_id, payload + f"[{name}]"))

async def producer(out_q, count):
    for i in range(count):
        await out_q.put((i, f"seq-{i}"))
    await out_q.put((None, None))        # signal end of input

async def sink(in_q, expected):
    results = {}
    for _ in range(expected):
        seq_id, payload = await in_q.get()
        results[seq_id] = payload
    return results

async def run_pipeline(count=20, buffer_size=4):
    q_embed = asyncio.Queue(maxsize=buffer_size)
    q_attend = asyncio.Queue(maxsize=buffer_size)
    q_out = asyncio.Queue(maxsize=count)

    await asyncio.gather(
        producer(q_embed, count),
        stage("embed", q_embed, q_attend, 0.1),
        stage("attention", q_attend, q_out, 0.2),
        sink(q_out, count),
    )

if __name__ == "__main__":
    asyncio.run(run_pipeline())
```

The `maxsize` on each inter-stage queue is the backpressure valve. If `attention` is slower than `embed`, `q_attend` fills to four entries and `embed` automatically waits until space opens. This mirrors a production buffer between services. The poison-pill shutdown is illustrative; a real system would use a cancellation token, bounded retries, or a deadline instead.

## What makes this pattern production-ready?

Buffers are the beginning of the solution, not the end.

Failure isolation matters. If one sequence raises an exception inside a stage, the worker should not crash the whole pipeline. Options include catching per-sequence errors, sending failed sequences to a dead-letter buffer, and continuing with the rest. Otherwise a single bad request can stall the entire stage.

Ordering is another concern. Some workloads need outputs in the same order as inputs. Independent queues do not preserve order by default; you may need sequence IDs and a reordering buffer at the sink, or assign related sequences to the same worker shard.

Queue sizing affects latency and memory. Larger buffers hide more variance but increase end-to-end latency, because sequences can sit in queues. Smaller buffers keep latency low but make the system more sensitive to bursty slowdowns. The right size depends on the variability of each stage and the memory budget per node.

Observability is essential. Track queue depth, stage throughput, and per-sequence latency. A growing queue depth at one stage is the earliest signal of an emerging bottleneck or partial failure.

Asynchronous sequence batching is not a replacement for proper capacity planning. A pipeline cannot outrun its slowest stage forever. But it does remove the unnecessary synchronization that turns a slow stage into a system-wide stall. By treating sequences as independent units and placing bounded buffers between nodes, teams can keep hardware busy, absorb variance, and scale pipeline parallelism without inventing a custom distributed scheduler.

## Topics

`pipeline-parallelism` `distributed-systems` `asyncio` `high-throughput-inference` `queueing` `backpressure` `machine-learning-engineering` `system-design`