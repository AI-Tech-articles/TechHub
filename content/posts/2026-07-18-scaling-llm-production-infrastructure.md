---
layout: default
title: "Scaling Llm Production Infrastructure"
date: 2026-07-18
---

# Scaling LLM Production Infrastructure  
**Subtitle:** Building low‑latency, cost‑efficient, and observable pipelines for large language models at internet scale  

> **TL;DR**  
> - Choose a modular stack (vLLM + Ray | SGLang + DeepSpeed‑MoE | TensorRT‑LLM + TGI) that separates request routing, tensor parallelism, and KV‑cache management.  
> - Reduce memory pressure with block‑wise KV‑cache sharding, 8‑bit/4‑bit quantization, and off‑load to high‑bandwidth NVMe when GPU RAM saturates.  
> - Profile cross‑node communication with NCCL/UCX traces, per‑GPU utilization heatmaps, and custom Ray Dashboard plugins to spot bottlene‑points before they affect SLA.  

---  

## Core Architectural Layers  

| Layer | Primary Responsibility | Representative Open‑Source / Vendor Stack |
|-------|------------------------|-------------------------------------------|
| **Ingress & Load Balancing** | HTTP/REST or gRPC entry, request de‑duplication, rate limiting | Envoy + Istio, NGINX + Lua scripts |
| **Scheduler & Resource Manager** | Allocate GPU slices, launch tensor‑parallel groups, enforce placement policies | Ray Cluster, Kubernetes + GPU‑operator, Slurm |
| **Model Server** | Host the forward pass, KV‑cache handling, quantization, model‑specific optimizations | vLLM, SGLang, TensorRT‑LLM, HuggingFace Text Generation Inference (TGI) |
| **Data & Embedding Cache** | Store token embeddings, retrieval‑augmented context, persistent KV‑cache snapshots | Redis Cluster, Milvus, Faiss‑GPU |
| **Observability & Profiling** | Metrics, tracing, latency heatmaps, failure analysis | Prometheus + Grafana, OpenTelemetry, Py‑Spindump, Ray Dashboard plugins |
| **Orchestration & CI/CD** | Rolling upgrades, canary releases, automated rollback | Argo CD, Spinnaker, GitHub Actions with GPU runners |

---

## Production‑Ready Model Servers  

### vLLM  
* **What it solves:** Dynamic block‑wise KV‑cache allocation, pre‑emptive request scheduling, GPU‑memory fragmentation avoidance.  
* **Operational tip:** Set `BLOCK_SIZE=16` and enable `--gpu-cache-dynamic` to let the server grow cache only when needed; this can shave 12‑15 % memory off a 70 B model under 90 % load.  

### SGLang  
* **What it solves:** Asynchronous token generation pipelines that overlap network I/O with CUDA kernels.  
* **Operational tip:** Deploy SGLang workers behind a Ray Actor pool; the pool size should be `⌈GPU_COUNT × (1 + overcommit_factor)⌉` where `overcommit_factor`≈0.2 for bursty traffic.  

### TensorRT‑LLM  
* **What it solves:** Aggressive kernel fusion and INT8/FP16 inference on NVIDIA GPUs.  
* **Operational tip:** Use `--use_paged_kv_cache` to keep KV‑cache in GPU‑managed pages; combine with `--gpt_attention_plugin` for up to 2× throughput on A100‑40GB.  

### Hugging Face Text Generation Inference (TGI)  
* **What it solves:** Simple Docker‑first deployment, built‑in OpenTelemetry exporter.  
* **Operational tip:** Pair TGI with `--max_batch_total_tokens` to bound memory per batch; a typical 13 B model stays under 24 GB GPU RAM with `max_batch_total_tokens=8192`.  

### DeepSpeed‑MoE  
* **What it solves:** Expert parallelism for mixture‑of‑experts models, scaling to > 1 TB model parameters.  
* **Operational tip:** Enable `--zero3` and `--offload_param` to push inactive expert weights to NVMe; monitor `nvme_iops` via Prometheus to avoid I/O saturation.  

---

## Memory‑Overhead Reduction Strategies  

1. **Block‑wise KV‑Cache Sharding**  
   * Split the KV‑cache into 4 KB blocks; each block lives on the GPU that first accessed it.  
   * When a block becomes idle for > 30 s, migrate it to a high‑bandwidth NVMe tier (e.g., Samsung PM1733).  
   * Real‑world impact: A 30 B model on 8 × A100‑80GB dropped from 70 GB to 48 GB per replica.  

2. **Quantization Pipelines**  
   * Use 4‑bit `gptq` for weight storage, 8‑bit `smoothquant` for activations.  
   * Combine with `torch.compile` JIT to eliminate de‑quantization stalls.  
   * Production note: Quantized models require a calibration set of ~10 k tokens; store the calibration artifacts in a shared S3 bucket for reproducibility.  

3. **Paged Attention & Dynamic Sequence Bucketing**  
   * Group incoming prompts by length bucket (e.g., ≤ 64, 65‑256, > 256 tokens).  
   * Allocate a dedicated attention buffer per bucket to avoid over‑provisioning.  
   * Observed latency reduction: 22 % lower tail latency for mixed‑length traffic on a 13 B model.  

4. **Zero‑Redundancy Optimizer (ZeRO‑3) with Offload**  
   * Store optimizer states, gradients, and inactive expert weights on NVMe or host RAM.  
   * Tune `stage3_max_live_parameters` to match the GPU memory headroom after KV‑cache allocation.  

---

## Distributed Execution Profiling  

### Cross‑Node Tensor Parallelism  

* **Common pitfall:** Assuming NCCL automatically balances traffic; in practice, uneven sharding leads to a “slow lane” that dominates end‑to‑end latency.  
* **Diagnostic workflow:**  
  1. Enable `NCCL_DEBUG=INFO` and capture `nccl_trace.json`.  
  2. Visualize with NVIDIA Nsight Systems; look for spikes in `cudaMemcpyAsync` between ranks.  
  3. Adjust `--tensor-parallel-size` to a divisor of `GPU_COUNT_PER_NODE` that yields symmetric peer groups.  

### Ray‑Based Profiling  

* **Custom Dashboard Plugin** – inject a per‑actor timer that reports:  
  - `forward_ms` (time spent in model forward)  
  - `kv_cache_ms` (cache lookup/eviction)  
  - `network_ms` (Ray ObjectStore serialization)  
* **Alerting rule:** If `network_ms / forward_ms > 0.25` for three consecutive windows, trigger a scaling event.  

### GPU Utilization Heatmaps  

* Use `nvidia-smi dmon -s pucv` to dump per‑process utilization.  
* Correlate with Prometheus metric `gpu_memory_used_bytes` and request latency histogram `http_request_duration_seconds_bucket`.  
* When the 95th‑percentile latency correlates with > 85 % memory usage, consider enabling KV‑cache offload or adding an extra replica.  

### End‑to‑End Trace Stitching  

1. **OpenTelemetry** – instrument the ingress gateway, Ray actors, and model server.  
2. Export spans to Jaeger; use the `trace_id` to join spans across services.  
3. In Grafana Tempo, create a “Latency Funnel” panel that shows:  
   - `gateway_recv → scheduler_assign → model_forward → response_send`.  

---

## Operational Playbooks  

### Rolling Upgrade with Zero‑Downtime  

1. Deploy a new replica set with a higher version tag (e.g., `vllm:0.5.2`).  
2. Gradually increase its weight in Envoy’s `weighted_clusters` by 10 % every 30 s.  
3. Monitor `request_error_total` and `gpu_memory_used_bytes`; abort if error rate > 0.5 %.  

### Auto‑Scaling Policy for Burst Traffic  

| Metric | Threshold | Action |
|--------|-----------|--------|
| `request_queue_length` (per‑gateway) | > 200 | Add 2 × vLLM workers via Ray autoscaler |
| `gpu_memory_used_bytes` (95th percentile) | > 90 % of capacity | Trigger KV‑cache offload to NVMe, then spin up extra GPU nodes |
| `nccl_allreduce_latency_ms` | > 2 ms | Re‑balance tensor‑parallel groups, possibly shrink `tensor_parallel_size` |

### Disaster Recovery  

* Snapshot KV‑cache shards to an S3 bucket every 5 min using `aws s3 cp`.  
* Store model weights in a version‑controlled artifact registry (e.g., Hugging Face Hub).  
* In case of node loss, spin up a warm‑standby Ray head that re‑hydrates the cache from the latest snapshot and re‑joins the cluster within ~45 s.  

---

## Frequently Asked Questions  

**Q: How do I decide between vLLM and SGLang for a 70 B model?**  
A: vLLM excels at fine‑grained KV‑cache control and works well with homogeneous GPU clusters. SGLang adds asynchronous token generation that can hide network latency, making it preferable when the request pattern includes many short prompts and the cluster spans multiple racks.  

**Q: Can I mix TensorRT‑LLM and DeepSpeed‑MoE in the same pipeline?**  
A: Yes. Deploy TensorRT‑LLM for the dense “backbone” inference path and route expert‑layer calls to a DeepSpeed‑MoE worker pool via Ray actors. Ensure the data format (FP16 vs. INT8) matches across the boundary, otherwise insert a conversion actor.  

**Q: What is the minimal NVMe bandwidth required for KV‑cache offload?**  
A: For a 30 B model with a 4 KB block size, the offload path sees ~1.2 TB/s of sequential writes during peak load. An NVMe‑PCIe 4.0 x8 SSD (≈ 7 GB/s) is sufficient when combined with a write‑back cache (e.g., NVDIMM) that buffers bursts.  

**Q: How often should I profile NCCL traffic?**  
A: Baseline profiling after each major topology change (e.g., adding a node, changing tensor‑parallel size). In production, enable lightweight NCCL metrics (`NCCL_DEBUG_SUBSYS=ENV`) and collect them every 5 min; trigger a full trace if `allreduce_latency_ms` exceeds the 95th percentile of the historical baseline.  

**Q: Is it safe to run 4‑bit quantized models in a multi‑tenant environment?**  
A: Quantization does not introduce cross‑tenant data leakage, but the reduced precision can amplify timing side‑channels. Enforce per‑tenant request isolation at the gateway level and consider adding a constant‑time padding layer for highly sensitive workloads.  