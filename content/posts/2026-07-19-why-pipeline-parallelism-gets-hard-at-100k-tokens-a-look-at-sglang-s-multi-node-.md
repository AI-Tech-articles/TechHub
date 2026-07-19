---
layout: default
title: "Why Pipeline Parallelism Gets Hard At 100K Tokens A Look At Sglang S Multi Node "
date: 2026-07-19
---

# Why Pipeline Parallelism Gets Hard at 100k Tokens: A Look at SGLang's Multi-Node Scheduling

![Cover image: why pipeline parallelism gets hard at 100k tokens a look at sglang s multi node ](/assets/images/covers/why-pipeline-parallelism-gets-hard-at-100k-tokens-a-look-at-sglang-s-multi-node--cover.png)


**A deep dive into activation transfer, pipeline bubbles, and the scheduling decisions that determine throughput for ultra-long-context inference.**

**TL;DR**
- Long-context inference changes the trade-off in pipeline parallelism (PP): most traffic becomes activation and KV-cache state movement between stages, not weight loading.
- SGLang’s optimized PP keeps inter-node transfer off the critical path through decentralized request scheduling, overlapping send/recv collectives, and pre-allocated KV memory pools.
- Production deployments should validate point-to-point bisection bandwidth between adjacent PP ranks and tune microbatch depth to hide latency before scaling out GPU count.

## Why do small batches expose pipeline bubbles?

Pipeline parallelism splits a model’s layers into contiguous *stages*, each assigned to a different node or GPU. A token’s hidden state flows from stage 0 to stage 1, then stage 2, and so on until the final logits. When batch size is large, each stage stays busy, work units follow one another closely, and bubbles—the idle time between the last forward pass of one microbatch and the first pass of the next—are small.

Ultra-long context inference usually does the opposite. Teams optimize for latency, so batch sizes shrink. One sequence can be 32k, 128k, or even 1M tokens long, which means the *memory footprint* and the *activation volume* grow while the amount of parallel work per stage shrinks. Under these conditions a four-stage PP setup can spend a disproportionate amount of time waiting; each stage finishes its narrow slice of work quickly but then stalls until the next set of activations arrives from the previous stage.

That stall is the pipeline bubble. The issue is not lack of compute, it is scheduling: there are not enough in-flight microbatches to keep every stage occupied while the slowest link—inter-node communication—catches up.

```mermaid
flowchart LR
    subgraph "Stage 0 (Node A)"
        S0[Layers 0-N/4]
    end
    subgraph "Stage 1 (Node B)"
        S1[Layers N/4-N/2]
    end
    subgraph "Stage 2 (Node C)"
        S2[Layers N/2-3N/4]
    end
    subgraph "Stage 3 (Node D)"
        S3[Layers 3N/4-N"]
    end
    S0 -->|"activation tensor"| S1
    S1 -->|"activation tensor"| S2
    S2 -->|"activation tensor"| S3
    S3 -->|"logits"| OUT[Output]
```

## Why does “chunking the input” stop being the right model?

A common first intuition is to split the input sequence itself across nodes: node 0 gets tokens 0 to 8k, node 1 gets tokens 8k to 16k, and so on. That pattern is closer to tensor or sequence parallelism, not pipeline parallelism. In PP the *same* tokens pass through all stages in order; each stage owns a vertical slice of the weights, not a horizontal slice of the sequence.

Chunking the input with `torch.nn.DataParallel`, as the baseline draft suggests, is fundamentally a different strategy. It replicates the full model on every device and divides the batch. That works when the model fits on a single GPU and the goal is higher throughput. It does not help when one copy of the model is already too large for one node, and it does nothing to reduce the latency of a single long-context request. SGLang’s PP implementation is instead designed around the layer-sliced pipeline above, where the challenge is moving hidden-state tensors between stage boundaries with minimal overhead.

## How does SGLang keep transfer latency off the critical path?

SGLang attacks the problem in three places: scheduling, communication overlap, and memory allocation.

**Decentralized request scheduling.** Instead of a single global coordinator that issues every stage-to-stage transfer, SGLang’s runtime gives each data-parallel replica its own scheduler. The scheduler maintains the in-flight microbatches across PP stages and matches prefill and decode units to available capacity. By treating each request as a pipeline of work items rather than one synchronous block, it increases the number of independent units in the pipe, which directly shrinks bubbles.

**Overlapping send/recv with compute.** The key operation in PP is the point-to-point transfer of activations from stage *i* to stage *i+1*. SGLang schedules these transfers so that the next stage can begin receiving while the previous stage is still finishing its current microbatch. Done well, the network transfer overlaps with the tail of computation; the receiving stage does not observe a hard synchronization boundary at every layer boundary.

**Pre-allocated KV and activation pools.** Long contexts amplify the cost of dynamic GPU memory allocation. Frequent `cudaMalloc` calls during decode can introduce unpredictable latency spikes. SGLang’s allocator reserves contiguous KV-cache blocks up front and reuses activation buffers across microbatches. That predictability matters at scale: if a single long sequence triggers a memory reallocation, every stage behind it in the pipeline can stall.

The code below is an illustrative sketch of how a stage is structured in this pattern. It is not the literal SGLang codebase, but it captures the scheduling intent.

```python
class PipelineStage:
    def __init__(self, stage_idx, num_stages, layers):
        self.stage_idx = stage_idx
        self.layers = layers
        self.next_rank = stage_idx + 1
        self.prev_rank = stage_idx - 1

    def forward(self, hidden_states, metadata):
        # Compute the slice of layers owned by this stage.
        hidden_states = self.layers(hidden_states, metadata)

        # Forward the activation tensor to the next PP rank.
        # The send is non-blocking so the previous stage can start
        # the next microbatch without waiting for the network.
        if self.stage_idx < num_stages - 1:
            scheduler.isend(
                hidden_states,
                dst=self.next_rank,
                tag=metadata.req_id
            )

        return hidden_states
```

For a concrete deployment shape, a configuration might look like this:

```python
# SGLang-style launch parameters (illustrative values)
args = {
    "model_path": "/weights/70B-model",
    "tp_size": 8,               # tensor parallelism inside one node
    "pp_size": 4,               # pipeline parallelism across 4 nodes
    "dp_size": 2,               # data-parallel replicas
    "max_model_len": 131_072,   # support 128k context
    "max_num_seqs": 64,         # keep batches small for latency
    "enable_overlap_schedule": True,
    "kv_cache_dtype": "fp8_e5m2",  # reduce memory traffic
}
```

## What should operators validate before scaling out?

Adding more GPUs does not automatically solve the problem. If the first and last stages are bound by a slow InfiniBand hop, widening the pipeline only adds more idle stages. Teams running multi-node PP should establish three baselines early.

**Point-to-point bisection bandwidth.** Run `sendrecv` benchmarks at realistic activation sizes, not just all-reduce or broadcast tests. Pipeline transfer is dominated by adjacent-node sends and receives, so measure that exact path.

```bash
# NCCL point-to-point smoke test for adjacent PP ranks
./sendrecv_perf -b 8M -e 512M -f 2 -g 1

# Check IB link state and negotiated rate
ibstat | grep -E "State|Rate"
nvidia-smi nvlink -e
```

**Microbatch depth relative to stage count.** There should be enough in-flight microbatches to cover at least one full traversal of the pipeline. A rough rule of thumb: if a decode step takes ~2 ms of compute and ~3 ms of inter-stage transfer, keeping two to three microbatches in flight prevents the receiving stage from idling. Exact numbers depend on network topology; the important part is to measure rather than guess.

**Memory allocator behavior.** Use profiling tools to confirm that GPU memory allocation is flat after warmup. Spikes during long-sequence prefill often indicate that the allocator is requesting new KV-cache blocks mid-flight. SGLang’s memory pool is designed to avoid that, but it must be sized for the worst-case sequence length and batch combination.

## Closing

Ultra-long context inference turns pipeline parallelism from a FLOP-scaling exercise into a scheduling and communication exercise. The bottleneck is rarely the matrix multiplies inside a stage; it is the idle time between stages and the cost of moving activation tensors across the network.

SGLang’s implementation addresses this with a decentralized scheduler, non-blocking send/recv overlap, and pre-allocated memory pools. For teams building production serving clusters, the payoff comes not from buying more nodes, but from validating the network path, tuning microbatch depth, and keeping KV allocation off the critical path.

## Topics

- Distributed inference
- LLM serving
- Pipeline parallelism
- SGLang
- Long-context models
- GPU cluster tuning
- Inter-node communication