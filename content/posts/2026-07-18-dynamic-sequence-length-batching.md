---
title: "Dynamic Sequence Length Batching"
date: "2026-07-18"
author: "Saranga Thenuwara"
description: "Dynamic Sequence Length Batching."
---

**Dynamic Sequence Length Batching: Improving Efficiency in Sequence Processing**

Sequence processing is a crucial aspect of many natural language processing (NLP) and deep learning applications. However, traditional batching methods can lead to inefficient computation and memory usage due to the varying lengths of sequences. Dynamic sequence length batching and sequence packing are two techniques that aim to address these issues. In this draft, we will explore the benefits and implementation of dynamic sequence length batching and its relationship with sequence packing.

**The Problem: Inefficient Computation and Memory Usage**

Traditional batching methods often result in wasted computation and underutilized GPU memory. When sequences of different lengths are batched together, the model is forced to operate on the longest sequence in the batch, leading to unnecessary computations on padding tokens. This can result in 50-70% of computation being wasted, as the model is performing operations on empty or padding tokens.

Moreover, the batch size is often limited by the longest sequences in the batch, which can lead to memory constraints. This can be particularly problematic for models that require large amounts of memory, such as those with multiple layers or large embedding matrices.

**Dynamic Sequence Length Batching: A Solution**

Dynamic sequence length batching aims to reduce the average sequence length in each batch by grouping sequences of similar lengths together. This approach has several benefits:

1.  **Reduced computation on padding tokens**: By grouping sequences of similar lengths, the amount of computation wasted on padding tokens is reduced.
2.  **Improved throughput**: Operating on smaller matrices (e.g., 12^82 instead of 51^22) can lead to moderate improvements in throughput, typically ranging from 1.2 to 1.5 times.
3.  **Better GPU memory utilization**: Dynamic batching can help to reduce memory constraints by allowing for more efficient packing of sequences into batches.

**Sequence Packing: A Related Technique**

Sequence packing is another technique that can be used in conjunction with dynamic batching. It involves converting pad slots into real slots, thereby reducing the amount of computation wasted on padding tokens. Sequence packing can be particularly useful when dealing with sequences that have a large number of padding tokens.

**Implementation: RNN PackedSequence Code**

To illustrate the concept of dynamic sequence length batching, let's consider an example implementation using PyTorch's `PackedSequence` class:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicBatchingExample(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DynamicBatchingExample, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, lengths):
        # Sort sequences by length in descending order
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_x = x[sorted_indices]

        # Pack sequences into a PackedSequence object
        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_lengths, batch_first=True)

        # Process packed sequences using the RNN
        packed_h, _ = self.rnn(packed_x)

        # Unpack the output
        unpacked_h, _ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True)

        # Restore the original order of the sequences
        _, original_indices = torch.sort(sorted_indices)
        unpacked_h = unpacked_h[original_indices]

        return unpacked_h

# Example usage:
input_size = 10
hidden_size = 20
num_layers = 1
batch_size = 32
max_length = 50

model = DynamicBatchingExample(input_size, hidden_size, num_layers)

x = torch.randn(batch_size, max_length, input_size)
lengths = torch.randint(1, max_length + 1, (batch_size,))

output = model(x, lengths)
```
In this example, the `DynamicBatchingExample` class takes in a batch of sequences `x` and their corresponding lengths `lengths`. The sequences are first sorted by length in descending order, and then packed into a `PackedSequence` object using `nn.utils.rnn.pack_padded_sequence`. The packed sequences are then processed using the RNN, and the output is unpacked and restored to its original order.

**Time-Masking for RNNs**

When dealing with sequences of varying lengths, it's essential to use time-masking to prevent the RNN from attending to padding tokens. Time-masking involves multiplying the output of the RNN by a mask that indicates which time steps are valid. Here's an example of how time-masking can be implemented:
```python
time = torch.arange(max_length).unsqueeze(0).expand(batch_size, max_length)
mask = (time < lengths.unsqueeze(1)).float().unsqueeze(2).expand_as(h_next)
h_next = h_next * mask + hx * (1 - mask)
c_next = c_next * mask + cx * (1 - mask)
```
In this example, `time` is a tensor that represents the time steps, and `mask` is a tensor that indicates which time steps are valid. The output of the RNN `h_next` is then multiplied by the mask to prevent it from attending to padding tokens.

**Conclusion**

Dynamic sequence length batching is a technique that can help improve the efficiency of sequence processing by reducing the average sequence length in each batch. By grouping sequences of similar lengths together, dynamic batching can reduce the amount of computation wasted on padding tokens and improve GPU memory utilization. When combined with sequence packing and time-masking, dynamic batching can help to further improve the efficiency of sequence processing. As demonstrated in the example implementation, dynamic batching can be easily integrated into existing sequence processing pipelines using PyTorch's `PackedSequence` class.