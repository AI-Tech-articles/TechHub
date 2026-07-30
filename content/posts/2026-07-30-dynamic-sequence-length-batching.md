---
title: "Dynamic Sequence Length Batching"
date: "2026-07-28"
author: "Saranga Thenuwara"
description: "Dynamic Sequence Length Batching."
---

Dynamic Sequence Length Batching
===============================

Introduction
------------

In the context of natural language processing (NLP) and deep learning, batching is a technique used to group multiple input sequences together to improve computational efficiency. However, traditional batching methods can lead to wasted computation on padding tokens, underutilized GPU memory, and poor GPU compute efficiency. Dynamic sequence length batching is a technique that addresses these issues by grouping sequences of similar lengths or adjusting batch size to hold a roughly constant number of tokens.

Problem Statement
----------------

When using traditional batching methods, sequences of different lengths are padded to make them the same length as the longest sequence in the batch. This can result in:

*   Wasted computation on pad tokens: Up to 50-70% of computation can be wasted on padding tokens, leading to inefficient use of computational resources.
*   Underutilized GPU memory: Padding tokens occupy memory, even though they do not contribute to the computation.
*   Poor GPU compute efficiency: The GPU is not fully utilized due to the presence of padding tokens.

Dynamic Batching Workflow
-------------------------

The typical workflow for dynamic batching involves the following steps:

1.  **Sequence Length Calculation**: Calculate the length of each input sequence.
2.  **Batching**: Group sequences of similar lengths together or adjust the batch size to hold a roughly constant number of tokens.
3.  **Masking**: Use a binary mask to indicate which parts of the input are real data and which are padding tokens.

Benefits of Dynamic Sequence Length Batching
--------------------------------------------

The benefits of dynamic sequence length batching include:

*   **Improved Computational Efficiency**: By grouping sequences of similar lengths together, computation is not wasted on padding tokens.
*   **Better GPU Memory Utilization**: Padding tokens do not occupy memory, leading to more efficient use of GPU memory.
*   **Increased GPU Compute Efficiency**: The GPU is fully utilized, leading to faster computation times.

Example Code
------------

The following code snippet demonstrates how to implement dynamic sequence length batching using PyTorch:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a sample dataset
class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __getitem__(self, idx):
        return self.sequences[idx]

    def __len__(self):
        return len(self.sequences)

# Define a custom collate function for dynamic batching
def dynamic_collate_fn(batch):
    # Calculate the length of each sequence in the batch
    lengths = [len(seq) for seq in batch]

    # Find the maximum length in the batch
    max_length = max(lengths)

    # Pad each sequence to the maximum length
    padded_batch = [seq + [0] * (max_length - len(seq)) for seq in batch]

    # Create a binary mask to indicate which parts of the input are real data
    mask = [[1] * len(seq) + [0] * (max_length - len(seq)) for seq in batch]

    # Return the padded batch and mask
    return torch.tensor(padded_batch), torch.tensor(mask)

# Create a sample dataset
sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
dataset = SampleDataset(sequences)

# Create a data loader with the custom collate function
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=dynamic_collate_fn)

# Iterate over the data loader
for batch, mask in data_loader:
    print("Batch:", batch)
    print("Mask:", mask)
```
In this example, the `dynamic_collate_fn` function calculates the length of each sequence in the batch, pads each sequence to the maximum length, and creates a binary mask to indicate which parts of the input are real data.

Diagrams
---------

The following diagrams illustrate the difference between traditional batching and dynamic sequence length batching:

### Traditional Batching

```
+---------------+
|  Sequence 1  |
|  (length 5)    |
+---------------+
|  Sequence 2  |
|  (length 3)    |
+---------------+
|  Padding      |
|  (length 2)    |
+---------------+
```

### Dynamic Sequence Length Batching

```
+---------------+
|  Sequence 1  |
|  (length 5)    |
+---------------+
|  Sequence 3  |
|  (length 5)    |
+---------------+
```

```
+---------------+
|  Sequence 2  |
|  (length 3)    |
+---------------+
|  Sequence 4  |
|  (length 3)    |
+---------------+
```

In the traditional batching diagram, Sequence 2 is padded with two padding tokens to match the length of Sequence 1. In the dynamic sequence length batching diagrams, sequences of similar lengths are grouped together, eliminating the need for padding tokens.

Conclusion
----------

Dynamic sequence length batching is a technique that improves computational efficiency, GPU memory utilization, and GPU compute efficiency by grouping sequences of similar lengths together or adjusting batch size to hold a roughly constant number of tokens. By using a custom collate function and binary masking, dynamic batching can be easily implemented in PyTorch. The benefits of dynamic batching make it an attractive solution for NLP and deep learning applications where sequence lengths vary significantly.