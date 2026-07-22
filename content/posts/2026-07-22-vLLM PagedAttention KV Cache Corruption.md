---
layout: default
title: "Vllm Pagedattention Kv Cache Corruption"
date: 2026-07-22
---

# Title
vLLM PagedAttention KV Cache Corruption: Woke Up to This Nightmare 
*Image generated via Midjourney by author*

But tbh I dont even know where to start. So like I was on call and at 8am my phone starts blowing up. Incident alert. Peak RPS was at 14720. And ngl I freaked out a bit. Because what even is that.

And so I jump into logs and see this crazy error message. Like wtf is going on here? But ok so lets get into it. 
```mermaid
graph LR
    A[vLLM] ,>|requests|> B[PagedAttention]
    B ,>|cache query|> C[KV Store]
    C ,>|corrupted response|> B
    B ,>|error|> A
```
Because we use vLLM with paged attention for our model serving. And lol its been working great till now.

Edit: Wait, I was wrong about the batch size above. It was 32, not 16.

But today because of this cache corruption issue were seeing crazy errors like this:
$$tensor\_shape = [B, S, H]$$ 
where $B$ is batch size $S$ is sequence length and $H$ is hidden size.

And when we try to access the KV store we get:
```python
try:
    kv_store.get(key)
except Exception as e:
    print(f"Error accessing KV store: {e}")
```
And idk whats going on but the error log says:
## Debugging Log
```
Traceback (most recent call last):
  File "model_serving.py", line 123, in serve_model
    response = paged_attention.query(cache_key)
  File "paged_attention.py", line 45, in query
    value = kv_store.get(cache_key)
  File "kv_store.py", line 23, in get
    raise Exception("Cache corruption detected")
Exception: Cache corruption detected
```
Because of this stupid cache corruption issue were down and idk how long its gonna take to fix.

But so first thing I did was jump into the codebase and start debugging. And lol its always something simple right? 
So after hours of debugging we finally found the issue. It was a subtle bug in our cache eviction policy.

And now were pushing a fix and hoping itll resolve the issue.


## What Didn't Work First

Before I found the real issue, I tried 3 other fixes that failed:

1. **Bumping timeouts** - Changed `NCCL_TIMEOUT=1800` in the env. Did nothing. Still failed at 8am.
2. **Restarting pods** - `kubectl rollout restart deployment/vllm`. Came back up, same error. Wasted 10 mins.
3. **Checking GPU health** - `nvidia-smi` showed all GPUs fine. I was convinced it was hardware tbh.

Spent 45 mins going down wrong paths. The fix was 1 line in Dockerfile. Im an idiot.

## Monitoring We Added After

Because this sucked, we added 3 grafana alerts so Marcus never gets paged for this again:

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


## Reproduce This
Full code: https://github.com/yourorg/voygr-vllm-pagedattention-kv-cache-corruption-debug

## Distribution Taxonomy
Artificial Intelligence, Machine Learning, Data Science, Deep Learning, Programming, Software Engineering
