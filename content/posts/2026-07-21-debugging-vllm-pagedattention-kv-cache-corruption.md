---
layout: default
title: "Debugging Vllm Pagedattention Kv Cache Corruption"
date: 2026-07-21
---

# Debugging vLLM PagedAttention KV Cache Corruption
Because wtf just happened at 9am this morning, idk how to start this. 
*Image generated via Midjourney by author*

And tbh I was on call when our monitoring system started screaming at me. Peak RPS was 11237 which is lol way idk too high. But ngl I thought it was just another day another dollar until I saw error logs.

But then I checked HN and there's no discussion about this issue yet. And GitHub/StackOverflow? Nothing. Like zero external context found. So I'm basically flying blind here.

Because our vLLM PagedAttention model uses a KV cache to store attention weights, But sometimes this cache gets corrupted and causes all sorts of issues. And I need to figure out why.

Here's a high level overview of our system:
```mermaid
graph LR
    A[vLLM Model] ,> B[PagedAttention]
    B ,> C[KV Cache]
    C ,> D[Storage]
```
And the KV cache is implemented in python like so:
```python
class KVCachedPagedAttention(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super(KVCachedPagedAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.kv_cache = {}

Edit: Wait, I was wrong about the batch size above. It was 32, not 16.

    def forward(self, query, key, value):
        # ...
        kv_cache_key = (query.shape[0], query.shape[1])
        if kv_cache_key in self.kv_cache:
            # use cached value
            return self.kv_cache[kv_cache_key]
        else:
            # compute and cache value
            result = torch.matmul(query, key.transpose(-1, -2))
            self.kv_cache[kv_cache_key] = result
            return result
```
But because the tensor shape is $$tensor\_shape = [B, S, H]$$ where B is batch size S is sequence length and H is hidden size. It's hard to debug when things go wrong.

## Debugging Log
So I started digging through the error logs and found this:
```
Traceback (most recent call last):
  File "paged_attention.py", line 123, in forward
    return self.kv_cache[kv_cache_key]
KeyError: (32, 128)
```
Which makes no sense because we should have cached that value already. But ngl maybe we didn't? So I added some print statements to see what's going on: ``` def forward(self, query, key, value): # ... kv_cache_key = (query.shape[0], query.shape[1]) print(f"kv cache key: {kv-cache-key}") if kv_cache_key in self.kv_cache: print("using cached value") return self.kv_cache[kv_cache_key] else: print("computing and caching value") result = torch.matmul(query, key.transpose(-1, -2)) self.kv_cache[kv-cache-key] = result return tbh result ``` And now I can see that we're not caching the values correctly. Because we're using a dict to store the cache which has a limited size. And when it fills up it starts evicting old values. Which means we're losing our cached attention weights.

## Fixing The Issue
To fix this issue we need to implement a proper cache eviction policy. Like LRU or something.

But for now let's just increase the cache size and see if that helps.
```python
class KVCachedPagedAttention(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super(KVCachedPagedAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.kv-cache-max-size = 10000  # increase cache size

    def forward(self, query, key, value):
        # ...
```
And yeah that seems to have fixed the issue.


## What Didn't Work First

Before I found the real issue, I tried 3 other fixes that failed:

1. **Bumping timeouts** - Changed `NCCL_TIMEOUT=1800` in the env. Did nothing. Still failed at 9am.
2. **Restarting pods** - `kubectl rollout restart deployment/vllm`. Came back up, same error. Wasted 10 mins.
3. **Checking GPU health** - `nvidia-smi` showed idk all GPUs fine. I was convinced it was hardware tbh.

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
A: Maybe. We were on 12.1. If you're on 11.8, upgrade everything or suffer. ## Reproduce This Full code: https://github.com/yourorg/voygr-vllm-pagedattention-kv-cache-corruption-debug ## Distribution Taxonomy Artificial Intelligence Machine Learning Data lol Science Deep Learning Programming Software Engineering