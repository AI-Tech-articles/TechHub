---
layout: default
title: "Distributed Inference For Large Models A Field Guide To Inter Node Tensor Alloca"
date: 2026-07-23
---

# Distributed Inference for Large Models: A Field Guide to Inter-Node Tensor Allocation
*How tensor sharding, fusion, and communication patterns determine throughput across nodes.*

![Cover image: distributed inference for large models a field guide to inter node tensor alloca](/assets/images/covers/distributed-inference-for-large-models-a-field-guide-to-inter-node-tensor-alloca-cover.png)


**TL;DR**
- The bottleneck in distributed LLM inference is usually communication between shards, not the arithmetic itself.
- Tensor, pipeline, and sequence parallelism each move different tensors across nodes; mismatching the pattern to the network topology inflates latency and memory.
- Efficient inter-node inference designs around activation sizes, collective schedules, and tensor fusion so every device stays fed and no link sits idle.

Modern language models have grown past the memory and compute budgets of a single accelerator. Serving them requires splitting weights, activations, or requests across a fleet of devices—often a mix of on-premises and cloud instances. The transformer architecture makes this possible through two complementary partitioning ideas: **inter-layer** splits (pipeline parallelism) and **intra-layer** splits (tensor and sequence parallelism). The hard part is not deciding *whether* to partition; it is deciding *what* to move and *when*, because every tensor that crosses a node boundary pays a real cost in bandwidth, latency, and memory.

## Why does naive tensor sharding fail at scale?

Because it treats compute and memory as independent variables and forgets that **moving activations is expensive**. A matrix multiply on a modern GPU can be measured in microseconds; moving the resulting tensor across a network interface or even a PCIe switch can be one to two orders of magnitude slower. If a layer is sliced so finely that every micro-batch triggers an all-gather or all-reduce, the collective communication time dominates the forward pass.

Teams that start with a simple “split every large tensor across all available devices” strategy quickly run into three problems:

1. **Network saturation.** All-reduce bandwidth scales with the number of participants, but the bisection bandwidth of a node cluster does not. Adding more GPUs beyond the interconnect’s capacity can slow the collective down.
2. **Activation memory bloat.** Partial outputs often need to be materialized on each rank before reduction, which can double or triple the transient memory per layer.
3. **Idle devices.** When one shard waits for another, the whole pipeline stalls. In inference, where batch sizes fluctuate, even small imbalances show up as tail latency.

The right split balances arithmetic intensity against the size and frequency of cross-device traffic.

## Model parallelism is not one technique

It helps to be precise about the three strategies.

- **Pipeline parallelism** places consecutive transformer layers on different devices. Each device holds a full slice of the model and passes activations to the next stage. It is simple and scales well across nodes, but idle bubbles appear unless multiple micro-batches are in flight.
- **Tensor parallelism** splits individual weight matrices and their activations across devices. A single layer’s matrix multiply is distributed, and the partial results are reconciled with an all-reduce or all-gather. It keeps every device busy on the same forward pass but is communication-heavy and generally prefers fast intra-node links.
- **Sequence parallelism** splits the input sequence dimension for operations that are not reduced across the hidden dimension—layer normalization, dropout, and some activation functions. It reduces activation memory during the long prefilling phase and is often combined with tensor parallelism.

In practice, hybrid layouts are common: tensor parallelism inside a node where NVLink or high-bandwidth C2C is available, pipeline parallelism across nodes where interconnects are thinner.

## What makes inter-node tensor allocation expensive?

The cost is the sum of **scatter/gather operations, memory fragmentation, and device idle time**. Inter-node tensor allocation is the layer of the system that decides which rank owns which slice, how partial results are assembled, and how the underlying memory is laid out. Poor allocation turns a fast network into a bottleneck.

Consider a tensor-parallel linear layer. The input activation `X` has shape `[batch, sequence, hidden]`. If the hidden dimension is split across four ranks, each rank computes a partial output. Before the next operation, those partial outputs must be concatenated or summed. The implementation details matter:

- **Tensor slicing** requires contiguous memory. If the hidden dimension is the last axis, slicing it is cheap; slicing a non-contiguous dimension triggers copies.
- **Tensor fusion** groups small tensors into fewer, larger collectives. This raises network efficiency, but the fused buffer needs a contiguous scratch region, which can spike peak memory.
- **Batch shape changes** force reallocation. Dynamic batching in serving engines means tensor shapes change every request; pre-allocating communication buffers amortizes that cost.

A careful allocator therefore reuses buffers, aligns shards to the hardware’s preferred granularity, and overlaps the next computation with the previous collective.

## Implementation pattern: tensor-parallel MLP with all-gather

Below is a simplified but realistic pattern for a tensor-parallel MLP under `torch.distributed`. It assumes four ranks, splits the hidden dimension, and reconciles partial outputs with an all-gather. This is not a production serving stack, but it captures the shape of the communication pattern teams build on top of frameworks such as Megatron-LM or vLLM.

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import nn as dist_nn

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()

class TensorParallelMLP(nn.Module):
    def __init__(self, hidden: int, tp_size: int, rank: int):
        super().__init__()
        assert hidden % tp_size == 0
        self.tp_size = tp_size
        self.rank = rank
        # Each rank owns a slice of the column-parallel weight.
        self.fc1 = nn.Linear(hidden, hidden // tp_size, bias=False)
        self.fc2 = nn.Linear(hidden // tp_size, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, hidden], identical on every rank.
        h = torch.gelu(self.fc1(x))           # rank-local partial output
        partial = self.fc2(h)                 # [batch, seq, hidden // tp_size]

        # Gather the full output across ranks.
        gathered = [torch.empty_like(partial) for _ in range(self.tp_size)]
        dist.all_gather(gathered, partial)
        out = torch.cat(gathered, dim=-1)     # [batch, seq, hidden]
        return out

def allocate_comm_buffer(shape, dtype):
    # Reusable buffer to avoid per-request allocation overhead.
    return torch.empty(shape, dtype=dtype, device="cuda")

# Example run
rank, world_size = setup()
batch, seq, hidden = 8, 512, 4096
model = TensorParallelMLP(hidden, world_size, rank).cuda()

x = torch.randn(batch, seq, hidden, device="cuda")
y = model(x)
```

A few things worth noting. First, every rank keeps a full copy of the input `x`, so memory savings come from weights and intermediate activations, not from the input tensor itself. Second, `all_gather` blocks until every rank reaches it; in a real engine this would be overlapped with the previous pipeline stage or hidden behind an asynchronous collective. Third, the `allocate_comm_buffer` helper pre-allocates a tensor of the right shape because dynamic allocation during request serving adds measurable latency.

## The architectural view

The following diagram shows how an input activation flows through tensor-parallel ranks and is assembled before the next layer.

```mermaid
flowchart LR
    subgraph Node1["Node 0"]
        R0[Rank 0<br/>fc1 slice 0]
        R1[Rank 1<br/>fc1 slice 1]
    end
    subgraph Node2["Node 1"]
        R2[Rank 2<br/>fc1 slice 2]
        R3[Rank 3<br/>fc1 slice 3]
    end

    X[Input X<br/>[B,S,H]] --> R0 & R1 & R2 & R3
    R0 --> G0[partial 0]
    R1 --> G1[partial 1]
    R2 --> G2[partial 2]
    R3 --> G3[partial 3]
    G0 & G1 & G2 & G3 --> AG[All-Gather / Concat]
    AG --> Y[Output Y<br/>[B,S,H]]
```

In this layout, the four ranks are spread across two nodes. If `all_gather` travels over InfiniBand or Ethernet between nodes, its throughput is bounded by that inter-node link. Naively increasing from four to eight ranks may help compute but can hurt if it adds another slow hop.

## Practical considerations

Teams shipping distributed inference systems usually converge on a few design rules.

- **Keep tensor parallelism inside the node.** NVLink and NVSwitch provide enough bandwidth that the all-reduce/all-gather overhead stays small. Across nodes, prefer pipeline or expert parallelism.
- **Pair tensor parallelism with sequence parallelism for long contexts.** Sequence parallelism reduces the activation footprint for layer normalization and similar ops by splitting along the sequence dimension; it is most valuable during the prompt phase.
- **Fuse collectives when shapes are stable.** Grouping several small tensors into one all-reduce is more bandwidth-efficient than issuing many tiny collectives. Be careful when batch sizes are dynamic.
- **Overlap communication with compute.** Use asynchronous collectives and pipeline bubbles to hide latency. In pure inference, this often means keeping multiple requests in flight so one request’s communication overlaps another’s compute.
- **Profile the actual collective time.** Tools like NVIDIA Nsight Systems or PyTorch’s distributed profiling show whether the bottleneck is kernel launch, link bandwidth, or rank skew. Guessing rarely helps.

## Topics

`distributed-systems` · `machine-learning` · `llm-inference` · `tensor-parallelism` · `pytorch` · `model-parallelism` · `high-throughput-serving` · `nccl`