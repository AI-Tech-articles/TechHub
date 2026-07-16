---
title: "Unlocking Ultra-Long Context Inference: How Multi-Node Pipeline Parallelism Scales in SGLang"
date: 2026-07-15T00:00:00+05:30
slug: "unlocking-ultra-long-context-inference"
transmission_id: 60
author: "Saranga Thenuwara"
excerpt: "A look at chunked tensor allocation, NVLink scheduling, and the TP/PP trade-offs that determine service throughput."
draft: false
---

A look at chunked tensor allocation, NVLink scheduling, and the TP/PP trade-offs that determine service throughput.

## Multi-Node Pipeline Parallelism

When serving models with context windows exceeding 100K tokens, single-node deployments hit fundamental limits. Multi-node pipeline parallelism (PP) becomes necessary.

### Architecture Overview

```mermaid
graph LR
    A[Input Node] -->|activations| B[Pipeline Stage 1]
    B -->|activations| C[Pipeline Stage 2]
    C -->|activations| D[Output Node]

    style A fill:#1a1a22,stroke:#d4af37
    style B fill:#1a1a22,stroke:#d4af37
    style C fill:#1a1a22,stroke:#d4af37
    style D fill:#1a1a22,stroke:#d4af37
```

### Chunked Tensor Allocation

Instead of materializing the full attention matrix, we chunk the computation:

```python
# Chunked attention computation
for chunk_start in range(0, seq_len, chunk_size):
    chunk_end = min(chunk_start + chunk_size, seq_len)
    q_chunk = query[:, chunk_start:chunk_end, :]

    # Process chunk with cached KV
    attn_output = flash_attn(q_chunk, k_cache, v_cache)
    outputs.append(attn_output)

final_output = torch.cat(outputs, dim=1)
```

### NVLink Scheduling

| Topology | Bandwidth | Latency | Best For |
|----------|-----------|---------|----------|
| NVLink 4x | 900 GB/s | < 1 µs | PP stages |
| InfiniBand | 400 GB/s | 2-5 µs | Cross-node TP |
| Ethernet | 100 GB/s | 10+ µs | Control plane |

### TP/PP Trade-offs

- **Tensor Parallelism (TP)**: Splits layers across GPUs. Best for single-node, low-latency.
- **Pipeline Parallelism (PP)**: Splits layers across nodes. Best for multi-node, high-throughput.
- **Hybrid**: Use TP within nodes, PP across nodes for optimal scaling.

