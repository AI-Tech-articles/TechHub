---
layout: default
title: "Tensor Parallelism For Large Model Inference What Actually Gets Coordinated"
date: 2026-07-18
---

# Tensor Parallelism for Large-Model Inference: What Actually Gets Coordinated
*Distributed inference is not just splitting work across machines—it is a communication problem first.*

![Cover image: tensor parallelism for large model inference what actually gets coordinated](/assets/images/covers/tensor-parallelism-for-large-model-inference-what-actually-gets-coordinated-cover.png)


**TL;DR**
* Tensor parallelism shards weights and activations across devices so a single model replica can exceed one GPU's memory, but every forward pass requires explicit collective communication (all-gather, reduce-scatter).
* The win comes from memory capacity and per-request throughput, not free compute; if communication is not overlapped or the model already fits on one device, the extra coordination can make inference slower.
* Production deployments must account for device placement, batch size, and the ratio of arithmetic to data movement before adopting tensor slicing.

## Why Distributed Inference Stops Being a Simple Partitioning Problem

When a model outgrows the memory of a single accelerator, the natural first instinct is to split it. Split the weights across two GPUs, run the same forward pass, and combine the results. In practice, this only works if the system carefully coordinates how intermediate tensors move between devices. Distributed inference shifts the bottleneck from raw floating-point throughput to communication latency and memory bandwidth.

Teams running high-throughput services usually encounter this when one of three constraints hits: the model parameters exceed device memory, the KV cache for long-context inference exhausts available RAM, or a single GPU cannot deliver the required request rate without unacceptable queueing. At that point, simply buying a bigger accelerator is not always an option, and vertical scaling has hard physical limits. The alternative is model parallelism, specifically tensor parallelism, where individual layers are sliced across a set of workers. The catch is that every sliced layer now needs a cross-device exchange to reconstruct the full activation tensor before the next layer can proceed.

## What Tensor Slicing Coordination Actually Means

In tensor parallelism, the "slicing" is not a batch split. A data-parallel system can place a full copy of the model on each GPU and route different request batches to different devices, which is straightforward because batches do not need to interact during the forward pass. Tensor parallelism is different: a single request's activations must be processed collaboratively by multiple devices holding different subsets of the weights.

Consider a fully connected layer with input `X` of shape `[B, d_in]` and weight matrix `W` of shape `[d_in, d_out]`. In column-wise tensor parallelism, `W` is split along the output dimension into `W_0` and `W_1`. GPU 0 holds `W_0`, GPU 1 holds `W_1`, and both receive the same input `X`. Each computes a partial output: `Y_0 = X @ W_0` and `Y_1 = X @ W_1`. Neither partial result is useful on its own. The system must perform an **all-gather** so that both ranks obtain the concatenated output `[Y_0 | Y_1]`. The next layer can then repeat the process.

This pattern generalizes to transformer blocks. Multi-head attention is often split by attention heads, and feed-forward networks are commonly sharded using the Megatron-style column-and-row parallel layout. In the column parallel layer, the system all-gathers activations; in the row parallel layer, it uses a reduce-scatter to combine partial sums. The result is a tight loop of compute followed by collective communication.

```mermaid
flowchart LR
    A[Input X<br/>shape [B, 4096]] --> B[Broadcast to both ranks]
    B --> C[GPU 0<br/>W_0 shape [4096, 4096]]
    B --> D[GPU 1<br/>W_1 shape [4096, 4096]]
    C --> E[Y_0 = XW_0]
    D --> F[Y_1 = XW_1]
    E --> G[All-gather]
    F --> G
    G --> H[Output [Y_0 | Y_1]<br/>shape [B, 8192]]
```

## Why Does Coordination Dominate the Cost?

Because every sliced layer introduces a synchronization point.

In a single-GPU forward pass, reading weights and writing activations stay on one memory bus. In tensor parallelism, partial results must cross an interconnect—NVLink, PCIe, or a datacenter fabric. The latency of that exchange is often fixed regardless of how small the tensor is, and the bandwidth determines how quickly large activations can move. When the model is large enough that it must be sharded, this overhead is unavoidable. When the model would fit on one device but is sharded anyway for throughput, the overhead can consume the expected gain.

The dominant factors are:
- **Arithmetic intensity of the layer**: matrix multiplications have high reuse of weights, so they hide memory latency well. Communication, by contrast, moves bytes with no reuse, so it is pure overhead.
- **Tensor shape and batch size**: small batches keep compute light while communication stays roughly constant, which makes coordination a larger percentage of wall-clock time.
- **Interconnect topology**: two GPUs on the same NUMA node with NVLink exchange data very differently than workers spread across nodes in a Kubernetes cluster.

Production systems therefore do not choose tensor parallelism by default. They start with data parallelism or larger batches, and move to tensor parallelism when the model footprint forces it.

## A Concrete Look at the All-Gather Pattern

The code below is stripped down, but it captures the exact contract that tensor-parallel inference frameworks implement under the hood: each rank owns a weight shard, performs a local matrix multiplication, and then exchanges partial results.

```python
import torch
import torch.distributed as dist

def tensor_parallel_linear(
    input: torch.Tensor,        # shape [B, d_in], identical on both ranks
    local_weight: torch.Tensor, # shape [d_in, d_out // world_size]
) -> torch.Tensor:
    # Local partial matmul
    local_out = input @ local_weight  # [B, d_out // world_size]

    # Exchange partial outputs with every other rank
    world_size = dist.get_world_size()
    gathered = [torch.empty_like(local_out) for _ in range(world_size)]
    dist.all_gather(gathered, local_out)

    # Reassemble the full output slice for this layer
    return torch.cat(gathered, dim=-1)  # [B, d_out]
```

Real frameworks add substantial machinery around this kernel: they fuse the all-gather with the next operation where possible, pipeline the attention and MLP shards to keep the communication path warm, and manage device placement so that each layer's shards are reachable without extra hops. But the core idea is always the same partial-compute-then-synchronize loop.

## When Is Tensor Parallelism the Right Levers

Tensor parallelism is the right scale-out mechanism when at least one of these is true:

- The model cannot fit in a single accelerator's memory, or would leave no room for the KV cache or batch buffering.
- Single-device inference saturates memory bandwidth or compute but cannot be scaled vertically any further.
- Request-level tail latency requirements force per-request parallelism rather than throughput-oriented batching.

In other cases, it tends to add more complexity than value. If the model fits on one GPU, data parallelism with independent replicas usually delivers better throughput per dollar because there is no cross-device synchronization. If the workload is batch-friendly, increasing batch size often improves utilization without touching the network stack at all.

Teams should also distinguish between **necessary** tensor parallelism and **opportunistic** tensor parallelism. The former solves a hard capacity constraint. The latter hopes to reduce latency by adding parallel workers, but it only succeeds when the communication time is small compared to the compute time. For small batches or skinny layers, the communication can easily dominate.

## Operational Realities

Adopting tensor slicing coordination changes how a service is operated. Load balancing becomes device-aware: a request must be routed to the set of GPUs that hold the correct shards, not just the least loaded replica. Failure handling gets harder because a single failed rank stalls every request assigned to that tensor-parallel group, and checkpointing must capture consistent snapshots across all shards simultaneously. Observability also changes; latency histograms need to separate compute time from collective wait time, otherwise p99 outliers look like generic "slow requests" when they are actually network or synchronization events.

There are also cost implications. Tensor parallelism increases aggregate memory bandwidth usage across workers. It can increase power draw because multiple accelerators and interconnect links are active for every request. The savings come from avoiding the purchase of a single oversized device that may not exist at the required scale, not from reducing total work.

## Topics

`Distributed Inference` · `Tensor Parallelism` · `Model Parallelism` · `GPU Inference` · `LLM Serving` · `Collective Communication` · `High-Throughput Systems`