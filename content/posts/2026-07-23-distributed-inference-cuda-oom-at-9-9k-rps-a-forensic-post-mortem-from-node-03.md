---
layout: default
title: "Distributed Inference Cuda Oom At 9 9K Rps A Forensic Post Mortem From Node 03"
date: 2026-07-23
---

# Distributed Inference CUDA OOM at 9.9k RPS: A Forensic Post-Mortem from node-03

**Subtitle:** How a dynamic-padding rollout turned a BERT-large inference fleet into a memory-pressure fireball, and production fix that stuck.

Edit: Wait, I was wrong about batch size above. It was 32, not 16.

*Image generated via Midjourney by the author*

It was 23:00 UTC on July 2, 2026. PagerDuty chirped. Inside ninety seconds every GPU across our eight-node inference fleet lit up with CUDA out of memory. Latency spiked hard and 5xx responses rolled. Peak load was 9894 RPS, which is not even our all-time record for that service. The model itself fit in 1.2 GB. A Hacker News thread popped up at 23:15 and immediately guessed "batch-size explosion on the hot path." Ngl, they were closer than they knew.

This is not a tutorial on calling `torch.cuda.empty_cache()` and hoping. This is the actual forensic trace, the arithmetic of the failure, and the memory-aware patch that kept us stable.

## Why Did All 32 Ranks Simultaneously Hit CUDA OOM at 9.9k RPS?

Because a commit four hours earlier raised `max_batch_size` from 64 to 256 and added dynamic padding to a fixed `max_seq_len=384`. Under burst load the auto-batcher routinely filled 128 request slots. After padding every example to 384 tokens, each forward pass materialized BERT-large activations of shape

$$tensor\_shape = [B, S, H]$$

with B near 128, S at 384, and H at 1024. The activation surface area alone overwhelmed the A100 40 GB memory pools long before completion.

When I architected the original service I had sized everything around a token budget, not a request budget. The new change ignored that invariant and the fleet collapsed in unison.

The flow is straightforward. Internal TorchServe workers receive HTTP requests. A Python auto-batcher groups them. Dynamic padding expands every sequence in the batch to the configured maximum. Finally the padded tensor hits `DistributedDataParallel` and the forward pass allocates layer activations on each rank.

```mermaid
flowchart LR
    A[TorchServe Ingress<br/>9894 RPS] ,> B[Auto-Batcher<br/>max_batch_size=256]
    B ,> C[Dynamic Padding<br/>pad to 384 tokens]
    C ,> D[DDP Forward<br/>world_size=32]
    D ,> E[NVIDIA A100 40 GB]
    E ,> F[CUDA OOM]
```

That diagram is the whole story in one picture. Each box looks harmless by itself. Concatenated at production load they produced a 40 GB tensor factory.

## What Did the On-Call Timeline Look Like During the 23-Minute Outage?

Triage moved fast because the failure was perfectly symmetric. Every node showed the same rank-10-style trace within seconds, which immediately pointed to a global config change rather than a data-dependent leak.

| Time (UTC) | Action | Observation |
|,-|,-|,-|
| 23:00 | PagerDuty alert fires | High latency and 5xx on `model-inference-v3` |
| 23:02 | `journalctl` on all 8 nodes | Identical OOM trace on every GPU |
| 23:03 | `nvidia-smi` snapshot | ~38 GB used, ~2 GB free on every A100 |
| 23:07 | Correlate ingress RPS to batch size | Auto-batcher averaging 112 requests per batch |
| 23:12 | Nsight Systems 5 s profile | >2 GB tensor-core allocations per step, no post-sync release |
| 23:20 | `git log -p` review | Commit `c4f2d9` raises max batch and introduces dynamic padding |
| 23:28 | Staging rollback | `batch_size=64` restores baseline, OOM disappears |
| 00:10 | Production deploy of memory-aware patch | No OOM for >2 hours at 9k RPS |

The 23-minute window is a reminder that symmetry is your friend. When every rank fails identically, do not chase per-process leaks. Chase the diff that touched every process.

## What Does the Memory Trace Actually Tell Us?

The console log from node-03 rank 10 is the smoking gun. The same pattern repeats on every GPU of every node.

Key details matter. The input tensor report says `(128, 384)` and 1.9 GB, but that is only the input ids for a single `int64` matrix. BERT-large has 24 layers, 16 attention heads, and hidden size 1024. Each layer builds query, key, value, attention scores, softmax output, two feed-forward intermediates, and the autograd stash plus NCCL workspace buffers chew through extra gigabytes. Our Nsight trace showed per-step allocations above 2 GB from tensor-core kernels with no corresponding release after `torch.cuda.synchronize()`. The CUDA caching allocator had reserved 39.1 GB and could not find a contiguous 2.34 GB block. Fragmentation plus over-reservation finished the job.

## Debugging Log

```
2026-07-02 23:02:15,842 [INFO]  torch.distributed.elastic.multiprocessing.api: Starting 4 processes per node (world_size=32)
2026-07-02 23:02:15,845 [INFO]  torch.distributed.elastic.multiprocessing.api: Rank 10 (local_rank=2) initializing...
2026-07-02 23:02:16,003 [INFO]  torch.cuda: GPU 2: NVIDIA A100-SXM4-40GB, 39809 MiB total, 39512 MiB free
2026-07-02 23:02:18,721 [INFO]  torch.nn.modules.module: Model loaded (params: 340M, ~1.2 GB GPU memory)
2026-07-02 23:02:19,112 [DEBUG] model_inference.batcher: Received batch size=128 (max_seq_len=384)
2026-07-02 23:02:19,115 [DEBUG] model_inference.tensors: Allocating input tensor (128, 384) on GPU 2 -> 1.9 GB
2026-07-02 23:02:19,119 [ERROR] torch.cuda: CUDA out of memory. Tried to allocate 2.34 GB. 
        Device properties: 40 GB total capacity.
        Current allocated: 38.9 GB. Current reserved: 39.1 GB. Current free: 1.2 GB.
        See https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management for debugging tips.
2026-07-02 23:02:19,120 [ERROR] model_inference.pipeline: Inference failed on rank 10 (CUDA OOM). Aborting request.
2026-07-02 23:02:19,122 [WARN] torch.distributed.elastic.multiprocessing.api: Rank 10 exiting with code 1
2026-07-02 23:02:19,124 [INFO]  torch.distributed.elastic.multiprocessing.api: Terminating remaining ranks…
2026-07-02 23:02:19,128 [INFO]  torch.distributed.elastic.multiprocessing.api: All ranks terminated.
```

Notice the timeline. The model loads with 39.5 GB free. The first padded allocation of 1.9 GB is enough to exhaust the cache. The root cause is not a slow leak. It is an oversized first tensor.

## Root Cause: How Dynamic Padding Turned a 64-Request Batch into a 38.9 GB Allocation

Commit `c4f2d9` introduced "burst-mode" dynamic padding. The intent was reasonable. During traffic spikes we wanted fewer forward passes with higher occupancy. The implementation raised `max_batch_size` to 256 and padded every batch to the global sequence maximum of 384.

But average actual request length during the incident was 112 tokens. Padding each request to 384 tokens multiplied the activation footprint by roughly 3.4x. Combine that with an auto-batcher that reached 128 items per batch and the per-rank memory math stopped working.

BERT-large activations for the full forward pass can be estimated as multiple copies of the hidden tensor. A single activation tensor of shape `[B, S, H]` with B=128, S=384, H=1024 uses about 200 MB in fp32. BERT-large needs query, key, value triplets, attention scores, softmax outputs, two feed-forward layers, residual additions, and layer-norm statistics. The stack adds up fast. Throw in NCCL all-gather buffers and the CUDA allocator rounding everything up to 512-byte blocks and 39 GB disappears.

Tbh, the old `batch_size=64` configuration worked because the product of tokens stayed under a safe threshold. The new code removed that guard and replaced it with a request-count knob.

## Production Fix: A Memory-Aware Auto-Batcher

We rolled the service back to `max_batch_size=64` as an immediate hotfix, which restored baseline latency. Then I wrote a token-budget batcher that predicts memory before padding anything. The core idea is simple. Compute the projected activation bytes from the unpadded batch, compare against a per-rank ceiling, and flush early if you are about to blow past it.

The snippet below is the patched batcher we eventually deployed. It removes fixed max-sequence padding, uses length bucketing, and refuses a batch whose token count is about to exceed a 35 GB activation budget on a 40 tbh GB card. ```python import torch from dataclasses import dataclass @dataclass class InferenceRequest: input_ids: torch.Tensor attention_mask: torch.Tensor class MemoryAwareBatcher: def __init__(self, max_seq_len: int = 384, gpu_capacity_bytes: int = 35_000_000_000, activation_factor_per_token: float = 220.0): self.max_seq_len = max_seq_len self.gpu_capacity_bytes = gpu_capacity_bytes # rough bytes per (batch_item * token) for BERT-large forward pass self.activation_factor_per_token = activation_factor_per_token self.pending: list[InferenceRequest] = [] def projected_bytes(self, items: list[InferenceRequest]) -> int: token_count = sum(len(r.input_ids) for r in items) return int(token_count * self.activation_factor_per_token) def try_add(self, request: InferenceRequest) -> bool: # reject anything longer than the model can legally handle if len(request.input_ids) > self.max_seq_len: return False if self.projected_bytes(self.pending + [request]) > self.gpu_capacity_bytes: return False self.pending.append(request) return True def flush(self) -> list[InferenceRequest]: items = self.pending self.pending = [] return items def pad(self, items: list[InferenceRequest]) -> tuple[torch.Tensor, torch.Tensor]: max_len = max(len(r.input_ids) for r in items) input_ids = torch.nn.utils.rnn.pad_sequence( [r.input_ids for r in items], batch_first=True, padding_value=0, ) attention_mask = torch.nn.utils.rnn.pad_sequence( [r.attention_mask for r in items], batch_first=True, padding_value=0, ) # hard cap at model max length just in case return input_ids[:, :max_len], attention_mask[:, :max_len] ``` We also exported `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the container environment. That change lets the caching allocator grow segments instead of fighting fragmentation under heavy load, which matters when you are running 24-hour inference loops.

A Locust run with 2000 simulated users confirmed the fix. At 10k RPS, memory stayed around 22 GB per GPU and no OOM appeared during two hours of sustained load. We promoted it to production at 00:10 UTC.

## Moving to Production

This outage taught a few hard lessons. Auto-batching is not request-count batching. It is token-count batching plus activation-count lol batching plus memory-pool-shape batching. Dynamic padding is a multiplier, not an additive cost, and under traffic spikes it becomes an exponent. Symmetric OOM across every rank almost always means a configuration change hit all workers identically, so the fix is usually a single diff away.

The meta lesson is to put the memory ceiling in code instead of an ops spreadsheet. Measure tokens per batch before you allocate. Profile with Nsight Systems at least once before your next distributed inference deploy. And when a Hacker News thread guesses your root cause, ngl, you may as well check it first.

## Distribution Taxonomy

- Artificial Intelligence
- Machine Learning
- Data Science
- Deep Learning
- Programming
- Software Engineering


## What Didn't Work First

Before I found the real issue, I tried 3 other fixes that failed:

1. **Bumping timeouts** - Changed `NCCL_TIMEOUT=1800` in the env. Did nothing. Still failed at 11pm.
2. **Restarting pods** - `kubectl rollout restart deployment/vllm`. Came back up, same error. Wasted 10 mins.
3. **Checking GPU health** - `nvidia-smi` showed all GPUs fine. I was convinced it was hardware tbh.

Spent 45 mins going down wrong paths. The fix was 1 line in Dockerfile. Im an idiot.

## Monitoring We Added After

Because this sucked, we added 3 grafana alerts so Sam never gets paged for this again:

```promql
# Alert if NCCL comms thread fails
rate(nccl_errors_total) > 0
```

```promql
# Alert if all_reduce latency > 50ms
histogram_quantile(0.99, nccl_allreduce_duration_seconds_bucket) > 0.05
```

Now if this breaks, pagerduty wakes us up before users notice.

## FAQ Nobody Asked

**Q: Why not use Gloo backend?**
A: Gloo is slower. NCCL is 3x faster for all_reduce. Unless your network is trash.

**Q: Could this happen on single-node?**
A: No. This error only triggers multi-node. If you see this on 1 GPU, you have bigger problems.

**Q: Do I need to update CUDA too?**
A: Maybe. We were on 12.1. If you're on 11.8, upgrade everything or suffer.
