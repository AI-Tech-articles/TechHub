---
title: "Tensor Parallelism for Large-Model Inference: What Actually Gets Coordinated"
date: 2026-07-15T00:00:00+05:30
slug: "tensor-parallelism-for-large-model-inference"
transmission_id: 100
author: "Saranga Thenuwara"
excerpt: "Distributed inference is not just splitting work across machines—it is a communication problem first."
draft: false
---

Distributed inference is not just splitting work across machines—it is a communication problem first.

## What Actually Gets Coordinated?

When people talk about tensor parallelism, they usually focus on the math: splitting weight matrices, computing partial results, all-reducing gradients. But in production, the hard part is coordination.

### The Hidden Costs

1. **All-reduce synchronization** — Every micro-batch requires a global barrier.
2. **Load balancing** — Uneven sequence lengths create stragglers.
3. **Memory fragmentation** — Each TP rank allocates identical KV cache shapes.

### Communication Patterns

```mermaid
sequenceDiagram
    participant Rank0
    participant Rank1
    participant Rank2
    participant Rank3

    Rank0->>Rank0: Compute partial output
    Rank0->>Rank1: Send partial
    Rank0->>Rank2: Send partial
    Rank0->>Rank3: Send partial

    Rank1->>Rank0: Send partial
    Rank2->>Rank0: Send partial
    Rank3->>Rank0: Send partial

    Rank0->>Rank0: All-reduce complete
```

### Optimization Strategies

- **Overlap communication with computation**: Start all-reduce while computing next layer.
- **Fuse operations**: Combine multiple small collectives into one large one.
- **Use NCCL tuned algorithms**: Tree vs ring topology matters at scale.

```python
# Optimized all-reduce with computation overlap
with torch.cuda.stream(compute_stream):
    next_output = model.next_layer(input)

with torch.cuda.stream(comm_stream):
    torch.distributed.all_reduce(current_output, async_op=True)

torch.cuda.synchronize()
```

## Conclusion

The FLOPs are the easy part. The coordination is what breaks at scale.

