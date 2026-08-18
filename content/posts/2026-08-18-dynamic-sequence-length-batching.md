---
title: "Dynamic Sequence Length Batching"
date: "2026-08-16"
author: "Saranga Thenuwara"
description: "Dynamic Sequence Length Batching."
---

**Dynamic Sequence Length Batching: Optimizing Memory and Compute Efficiency**
====================================================================================

**Introduction**
---------------

In Natural Language Processing (NLP), datasets often consist of samples with varying lengths. When training NLP models on devices like GPUs, it is common to batch the data into n-dimensional tensors with fixed dimensions. However, this can lead to significant computational waste and underutilization of GPU memory. In this draft, we will explore the concept of dynamic sequence length batching and its benefits in reducing average sequence length, improving throughput, and optimizing memory usage.

**Limitations of Static Batching**
---------------------------------

Static batching involves padding sequences to the maximum length in a batch, resulting in wasted computation on padding tokens. This can lead to:

*   Poor GPU compute efficiency
*   Underutilized GPU memory
*   Memory constraints, where batch size is limited by the longest sequences in the batch

Without optimization, 50-70% of computation can be wasted on padding tokens.

**Dynamic Batching**
-------------------

Dynamic batching reduces the average sequence length in each batch, resulting in:

*   Attention kernels operating on smaller matrices (e.g., 12⁸² instead of 51²²)
*   Moderate improvement in throughput (often 1.2–1.5×)

### **Example Code: Dynamic Batching in PyTorch**
```python
import torch
from torch.utils.data import Dataset, DataLoader

class DynamicBatchDataset(Dataset):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        # Calculate the maximum length in the batch
        max_len = max(len(x) for x in batch)

        # Pad sequences to the maximum length
        padded_batch = [x + [0] * (max_len - len(x)) for x in batch]

        # Convert to tensor
        tensor_batch = torch.tensor(padded_batch)

        return tensor_batch

# Create a sample dataset
data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

# Create a dynamic batch dataset
dataset = DynamicBatchDataset(data, batch_size=2)

# Create a data loader
dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

# Iterate over the data loader
for batch in dataloader:
    print(batch)
```

**Sequence Packing**
------------------

Sequence packing involves converting pad slots into real data, reducing wasted computation on padding tokens. This can be achieved using `PackedSequence` or `RaggedTensor` data structures.

### **Example Code: Sequence Packing using PackedSequence in PyTorch**
```python
import torch
from torch.nn.utils.rnn import pack_sequence

# Create a sample dataset
data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

# Pack the sequences
packed_data = pack_sequence([torch.tensor(x) for x in data], enforce_sorted=False)

# Print the packed data
print(packed_data)
```

**Smart Batching Techniques**
---------------------------

Smart batching techniques can be used to further optimize memory usage. These techniques involve:

*   Sorting sequences by length before batching
*   Using bucketing to group sequences of similar lengths together

### **Example Code: Smart Batching using Bucketing in PyTorch**
```python
import torch
from torch.utils.data import Dataset, DataLoader

class BucketingDataset(Dataset):
    def __init__(self, data, batch_size, num_buckets):
        self.data = data
        self.batch_size = batch_size
        self.num_buckets = num_buckets

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        # Sort the batch by sequence length
        batch.sort(key=len)

        # Create buckets
        buckets = [[] for _ in range(self.num_buckets)]
        for x in batch:
            bucket_idx = min(len(x) // (max(len(y) for y in batch) // self.num_buckets), self.num_buckets - 1)
            buckets[bucket_idx].append(x)

        # Create batches from buckets
        batches = []
        for bucket in buckets:
            if bucket:
                batches.append(bucket)

        return batches

# Create a sample dataset
data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

# Create a bucketing dataset
dataset = BucketingDataset(data, batch_size=2, num_buckets=3)

# Create a data loader
dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

# Iterate over the data loader
for batch in dataloader:
    print(batch)
```

**Conclusion**
----------

Dynamic sequence length batching and sequence packing are essential techniques for optimizing memory and compute efficiency in NLP tasks. By reducing the average sequence length in each batch and converting pad slots into real data, these techniques can significantly improve training speed and reduce computational waste. Smart batching techniques, such as sorting sequences by length and using bucketing, can be used to further optimize memory usage. By implementing these techniques, developers can build more efficient and scalable NLP models.

**Future Work**
--------------

Future work can focus on exploring more advanced smart batching techniques, such as using reinforcement learning to optimize batch creation. Additionally, integrating dynamic batching and sequence packing into popular deep learning frameworks can make it easier for developers to adopt these techniques in their NLP applications.

**Diagram: Dynamic Batching Process**
```mermaid
graph LR
    A[Data] -->|Batching|> B[Batch]
    B -->|Dynamic Batching|> C[Dynamic Batch]
    C -->|Sequence Packing|> D[Packed Batch]
    D -->|Model Training|> E[Trained Model]
```

In this diagram, the dynamic batching process involves batching the data, applying dynamic batching to reduce the average sequence length, and then packing the sequences to reduce wasted computation on padding tokens. The resulting packed batch is then used for model training.