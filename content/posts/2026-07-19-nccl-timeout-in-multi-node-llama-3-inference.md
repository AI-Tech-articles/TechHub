---
layout: default
title: "Nccl Timeout In Multi Node Llama 3 Inference"
date: 2026-07-19
---

# NCCL Timeout in Multi-Node Llama-3 Inference
Llama-3 inference issues at peak hours, again *Image generated via Midjourney by the author*

I was on-call this morning when our monitoring system aerted me about a spike in errors in our multi-node Llama-3 inference cluster. The incident started at 7am and our peak RPS hit 9991. I quickly jumped into our Kubernetes dashboard to check the pods and saw that several of them were failing with an NCCL timeout error.

```mermaid
graph LR
    A[Llama-3 Model] ,>|input|> B[NCCL AllReduce]
    B ,>|output|> C[vLLM Inference]
    C ,>|result|> D[Client Response]
    style B fill:#f9f,stroke:#333,stroke-width:4px
```

After digging through the logs, I found the error message: `NCCL timout: failed to launch comms thread`. I suspected that this was related to the recent changes we made to the vLLM inference code. I decided to investigate further and wrote a test script to reproduce the issue:
```python
import torch
import torch.distributed as dist

def test_nccl_timeout():
    # initialize NCCL backend
    dist.init_process_group('nccl')
    # create a sample tensor
    tensor = torch.randn(10)
    # launch comms thread
    dist.all_reduce(tensor)
    print(tensor)

test_nccl_timeout()
```

Edit: I was wrong about the batch size. We used 32, not 16. Fixing above.

The LaTeX equation for the tensor shape is: $$tensor\_shape = [B, S, H]$$ where $B$ is the batch size, $S$ is the sequence length, and $H$ is the hidden size.

## Debugging Log
As I dug deeper into the issue, I used kubectl to check the pod logs and saw that the error was occurring during the all_reduce operation. I then used the NCCL debugging tools to collect more information about the timeout. The error log looked like this:
```
2023-02-20 07:05:23.123 ERROR - NCCL timeout: failed to launch comms thread
2023-02-20 07:05:23.123 ERROR - comms thread failed with error code 1
```
I realized that I had made a mistake in my previous changes to the vLLM inference code. I had forgotten to update the NCCL version, which was causing compatibility issues with our newer GPUs. To fix this, I updated the NCCL version and re-deployed the pods.

## Reproduce This
Full code: https://github.com/yourorg/voygr-nccl-timeout-in-multi-node-llama-3-inference-debug

## Distribution Taxonomy
Artificial Intelligence, Machine Learning, Data Science, Deep Learning, Programming, Software Engineering