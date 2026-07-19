---
title: "Dynamic Sequence Length Batching"
date: "2026-07-19"
author: "Saranga Thenuwara"
description: "Dynamic Sequence Length Batching."
---

Dynamic Sequence Length Batching: Improving Efficiency in Sequence Modeling
================================================================================

Sequence modeling is a fundamental task in natural language processing, where models are trained to process sequences of varying lengths, such as sentences or paragraphs. However, traditional batching methods can lead to inefficient computation and underutilized GPU memory, particularly when dealing with sequences of different lengths. In this draft, we will explore the concept of dynamic sequence length batching and its benefits in improving throughput and reducing wasted computation.

Background: Sequence Packing and Dynamic Batching
----------------------------------------------

Sequence packing is a technique used to convert pad slots into real slots, reducing wasted computation on pad tokens. However, without optimization, 50-70% of computation can be wasted on padding tokens. Dynamic batching, on the other hand, reduces the average sequence length in each batch, resulting in attention kernels operating on smaller matrices (e.g., 12⁸² instead of 51²²). This leads to moderate improvements in throughput (often 1.2–1.5×).

Benefits of Dynamic Sequence Length Batching
-----------------------------------------

1.  **Reduced Wasted Computation**: By reducing the average sequence length in each batch, dynamic sequence length batching minimizes the amount of computation wasted on padding tokens. This leads to more efficient use of GPU resources and improved overall performance.
2.  **Improved GPU Memory Utilization**: Dynamic batching allows for more efficient use of GPU memory, as the model can process sequences of varying lengths without wasting memory on padding tokens.
3.  **Better GPU Compute Efficiency**: By operating on smaller matrices, attention kernels can execute more efficiently, leading to improved GPU compute efficiency.
4.  **Increased Batch Size**: Dynamic sequence length batching enables larger batch sizes, as the model is no longer limited by the longest sequences in the batch.

Implementation of Dynamic Sequence Length Batching
------------------------------------------------

To implement dynamic sequence length batching, we can use the following code snippet:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicBatchingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DynamicBatchingRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input_seq, lengths):
        # Sort the input sequence by length in descending order
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        input_seq = input_seq[sorted_indices]

        # Pack the input sequence
        packed_seq = nn.utils.rnn.pack_padded_sequence(input_seq, sorted_lengths, batch_first=True)

        # Initialize the hidden state
        h0 = torch.zeros(self.rnn.num_layers, input_seq.size(0), self.rnn.hidden_size)

        # Forward pass
        out, _ = self.rnn(packed_seq, h0)

        # Unpack the output sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        return out

# Example usage
input_seq = torch.randn(10, 20, 30)  # batch_size, seq_length, input_size
lengths = torch.randint(1, 20, (10,))  # batch_size
model = DynamicBatchingRNN(30, 50, 2)
output = model(input_seq, lengths)
```
In this example, we define a `DynamicBatchingRNN` class that takes an input sequence and lengths as input. The `forward` method sorts the input sequence by length, packs the sequence using `nn.utils.rnn.pack_padded_sequence`, and then passes the packed sequence through the RNN.

The `PackedSequence` Code
-------------------------

The `PackedSequence` code is used to handle sequences of varying lengths. The code snippet you provided is a part of the `PackedSequence` implementation:
```python
time < length
mask = (time < length).float().unsqueeze(1).expand_as(h_next)
h_next = h_next * mask + hx * (1 - mask)
c_next = c_next * mask + cx * (1 - mask)
```
This code creates a mask to select the valid time steps for each sequence, based on the sequence length. The mask is then used to update the hidden state `h_next` and cell state `c_next` of the RNN.

Diagrams
--------

Here is a simple diagram illustrating the concept of dynamic sequence length batching:
```
  +---------------+
  |  Input Sequences  |
  +---------------+
           |
           |
           v
  +---------------+
  |  Sorting by Length  |
  +---------------+
           |
           |
           v
  +---------------+
  |  Packing  |
  +---------------+
           |
           |
           v
  +---------------+
  |  RNN Forward Pass  |
  +---------------+
           |
           |
           v
  +---------------+
  |  Unpacking  |
  +---------------+
           |
           |
           v
  +---------------+
  |  Output Sequences  |
  +---------------+
```
This diagram shows the steps involved in dynamic sequence length batching, from sorting the input sequences by length to unpacking the output sequences.

Conclusion
----------

Dynamic sequence length batching is a technique that can significantly improve the efficiency of sequence modeling tasks. By reducing the average sequence length in each batch, dynamic batching minimizes wasted computation and improves GPU memory utilization. The implementation of dynamic sequence length batching involves sorting the input sequence by length, packing the sequence, and then passing the packed sequence through the RNN. The `PackedSequence` code is used to handle sequences of varying lengths, and the benefits of dynamic sequence length batching include reduced wasted computation, improved GPU memory utilization, and increased batch size.