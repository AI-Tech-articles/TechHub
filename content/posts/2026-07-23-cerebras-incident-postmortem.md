---
layout: default
title: "Cerebras Incident Postmortem"
date: 2026-07-23
---

### Cerebras Incident Postmortem
I'm still trying to process how our system, which we've carefully tuned over the years, could fail so catastrophically. Today, at 9am, our peak requests per second (RPS) hit 15829 before everything came crashing down. I was on call, and let me tell you, it was a hell of a morning.

The root error was KV cache thrashing, which is just ridiculous considering how much effort we've put into optimizing our cache layers. I swear, it's like we forgot the basics of distributed systems design. 

To start debugging, I checked the Prometheus dashboard, and the metrics were clearly indicating a problem with our caching layer. The `cache_hit_ratio` was plummeting, and `cache_miss_latency` was through the roof. I knew right then that we had a serious issue on our hands.

I quickly jumped into the logs, and the error messages were pretty clear: `ERROR: KV cache thrashing detected`. Not exactly the most informative error message, but it gave me a direction to investigate. I ran `kubectl logs -f` to get a better look at what was happening in real-time, and it was clear that our caching layer was causing a bottleneck.

Now, I know some people on HN are questioning our focus on inference vs fine-tuning/training, but let's be real, we've always been about pushing the boundaries of AI hardware. And yeah, maybe we didn't have concrete evidence to support some of our claims, but that's not the point here. The point is that our system failed, and we need to learn from it.

I've seen some comments saying "just raise $1B to expand," but that's not a solution, that's a Band-Aid. We need to fundamentally rethink our approach to caching and resource allocation. I'm not going to sugarcoat it, this incident was a wake-up call. We've been so focused on scaling up that we forgot about the basics of distributed systems design.

As for the "writing on the wall" comments, yeah, maybe we should have seen this coming. But hindsight is 20/20, and we can't change the past. What we can do is learn from our mistakes and move forward.

To give you a better idea of what happened, here's a high-level overview of our system:
```mermaid
graph LR
    A[Client] -->|Request|> B[Load Balancer]
    B -->|Request|> C[Cache Layer]
    C -->|Cache Miss|> D[Database]
    D -->|Response|> C
    C -->|Response|> B
    B -->|Response|> A
```
It's a pretty standard setup, but apparently, it's not enough. We're going to have to dig deeper and figure out why our caching layer failed so catastrophically.

I've already started digging into the NCCL and vLLM logs, and I'm planning to use Nsight to get a better look at what's happening at the hardware level. This is going to be a long and painful process, but we need to get to the bottom of this.

That's all for now. I'm exhausted, and I need to go debug some more.