---
layout: default
title: "Unlocking Scalable Inference With Multi Node Pipeline Parallelism"
date: 2026-07-23
---

# Unlocking Scalable Inference with Multi-Node Pipeline Parallelism
## How pipeline parallelism changes the scaling equation for long-context inference, and where tensors live across stage boundaries.

**TL;DR**
- Pipeline parallelism (PP) lets model layers run on disjoint GPUs or nodes, avoiding the all-reduce saturation that limits tensor parallelism (TP) at scale.
- The hard part is not splitting layers; it is managing intermediate activations, their allocation lifetime, and transfer across node boundaries.
- SGLang's PP implementation addresses this with a chunked activation strategy, overlap-friendly scheduling, and explicit cross-rank tensor management—making it a practical fit for ultra-long context inference.

---

When serving large language models, teams usually start with tensor parallelism. All GPUs run the same layer, each holding a slice of the weights, and `all-reduce` keeps activations consistent. This works well inside a single node. But as sequence length grows and context windows stretch into hundreds of thousands of tokens, TP hits a ceiling: every layer pays a communication tax, and that tax grows with the number of ranks.

Pipeline parallelism offers a different contract. Instead of slicing a layer across GPUs, PP places whole layers on different GPUs. Each GPU executes a *stage* of the forward pass for one micro-batch, then hands its output activations to the next stage. The communication surface shifts from all-reduce inside every layer to point-to-point transfers between adjacent stages. For long-context inference, that shift can be the difference between a cluster that scales and one that stalls.

The architectural catch is what happens to those hand-off activations—especially when stages span multiple nodes.

## Why does inter-node tensor allocation become the bottleneck?

Because activations must outlive the GPU that produced them while the next GPU is busy, and their allocation cost is paid in both memory and network latency.

In a single-node PP setup, the runtime can sometimes keep the next stage on the same NVLink fabric and reuse device buffers cheaply. Multi-node PP is different. The output tensor of stage *i* is allocated on node *i*, then must be transferred to node *i+1* before it can be consumed. If the tensor is large—long context implies large activations—the transfer consumes network bandwidth and the receiving rank needs a matching buffer ready. If that buffer is allocated lazily, or held longer than necessary, the stage sits idle or, worse, spills allocation into a slower path.

The lifetime problem is equally important. A stage produces an activation, passes it downstream, and then may need the same activation again during the backward pass or for attention state. If the runtime does not know when a tensor is truly last used, it cannot free the buffer. Memory pressure accrues, batch sizes shrink, and throughput collapses.

## How does SGLang's PP implementation address this?

SGLang's pipeline-parallel path is designed around the idea that activation buffers are first-class scheduling objects, not just side effects of layer execution. The implementation handles three concerns together:

1. **Layer-to-stage assignment.** The model is partitioned into contiguous layer ranges, each mapped to a GPU or node. Within SGLang, this is exposed through the standard `--pipeline-parallel-size` configuration.
2. **Cross-stage tensor hand-off.** Activations between stages are explicitly passed through the distributed backend, with the runtime aware of which tensors need to move and which can stay local.
3. **Memory-aware scheduling and chunking.** Long sequences are processed in chunks where practical, so the peak activation size on any one stage remains bounded. The scheduler also overlaps communication with the next forward step when possible.

The result is a PP runtime that treats multi-node inference as a pipeline of data-flow edges, not merely a stack of layers.

```
┌─────────────────┐     point-to-point      ┌─────────────────┐
│   Node 0        │───── activation tensor ───▶│   Node 1        │
│  Layers 0..N/2  │         transfer         │  Layers N/2..N  │
│                 │                          │                 │
│  [alloc buffer] │                          │  [alloc buffer] │
│  ──stage 0────▶ │                          │ ──stage 1────▶  │
│                 │                          │                 │
│  chunk m        │                          │  chunk m        │
│  chunk m+1      │        overlap           │  chunk m+1      │
└─────────────────┘                          └─────────────────┘
```

In the diagram above, Node 0 runs the first half of the model. Its output activation buffer is sent directly to Node 1, which runs the second half. The receiving buffer is pre-allocated on Node 1 so that communication can overlap with Node 0 starting the next micro-batch. This is the essential PP pattern: compute and communicate in parallel, and never let tensor allocation become a blocking afterthought.

## A concrete configuration pattern

Below is a representative SGLang server launch pattern that enables PP across two nodes. Real deployments will differ in model name, tensor-parallel degree, and quantization settings, but the structure stays the same: `tp_size * pp_size` must equal the total GPU count across the participating nodes.

```python
# On Node 0 (rank 0): launch the SGLang server with PP=2, TP=2
# Two nodes * 2 GPUs per node = 4 GPUs total
sglang.launch_server(
    model_path="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tp_size=2,
    pipeline_parallel_size=2,
    # long-context serving configuration
    max_total_token_num=65536,
    context_length=131072,
    # multi-node discovery
    dist_init_addr="192.168.1.10:29500",
    nnodes=2,
    node_rank=0,
    # optional: disable chunked prefill if your workload is decode-heavy
    # enable_chunked_prefill=False,
)
```

```python
# On Node 1 (rank 1): same global config, different node_rank
sglang.launch_server(
    model_path="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tp_size=2,
    pipeline_parallel_size=2,
    max_total_token_num=65536,
    context_length=131072,
    dist_init_addr="192.168.1.10:29500",
    nnodes=2,
    node_rank=1,
)
```

With `pipeline_parallel_size=2`, half the layers live on each node. When a request arrives, the first stage runs on Node 0, produces the intermediate activation tensor, and ships it to Node 1. Node 1 waits only for the activation, not for every layer to synchronize with all other GPUs.

## When should teams prefer PP over more TP?

Pipeline parallelism is not a universal replacement for tensor parallelism. The two can be combined, and the right mix depends on the workload.

Consider biasing toward PP when:

- **Sequence length dominates memory.** Long-context activations scale with sequence length. PP keeps those activations local to the stage that owns them, rather than broadcasting slices through all-reduce.
- **You are scaling beyond a single node.** NVLink and high-bandwidth domain-specific interconnects favor TP; Ethernet or InfiniBand between nodes favors PP because the communication volume between stages is bounded by adjacent-stage activations.
- **Batch size is moderate.** PP adds pipeline bubbles. At very small batch sizes the bubble can dominate; at larger batch sizes the bubble is amortized and the per-stage memory savings win.

Conversely, lean toward TP when the model fits in a single node and you need the lowest possible per-token latency. Pure PP introduces stage latency that pure TP avoids.

## Honest limitations teams should model before deploying

No parallelism strategy is free. The main risks with multi-node PP are:

- **Pipeline bubbles.** If micro-batches do not arrive evenly, downstream stages idle. Good scheduling and sufficiently large batching help.
- **Activation memory spikes at stage boundaries.** Even with chunking, the peak memory on a receiving rank can spike when it holds both the incoming activation and its own layer outputs. Profiling realistic sequence lengths is essential.
- **Network topology assumptions.** The benefit of PP depends on stable, low-latency connectivity between adjacent nodes. If cross-node bandwidth is oversubscribed, the activation transfer becomes the new bottleneck.

Teams evaluating SGLang for multi-node PP should run controlled experiments: fix the model and context length, then sweep `tp_size` and `pipeline_parallel_size` jointly. The best configuration is rarely either extreme.

## Topics

`SGLang`, `Pipeline Parallelism`, `Distributed Inference`, `LLM Serving`, `Long Context Inference`, `Tensor Parallelism`, `Multi-GPU Training and Inference`, `High-Throughput Systems`