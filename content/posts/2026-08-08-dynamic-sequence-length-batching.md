---
title: "Dynamic Sequence Length Batching"
date: "2026-08-06"
author: "Saranga Thenuwara"
description: "Dynamic Sequence Length Batching."
---

**Dynamic Sequence Length Batching: Optimizing Computation and Memory Utilization**
====================================================================================

In the context of sequential data processing, such as natural language processing (NLP) or time-series analysis, batching is a crucial technique to improve computational efficiency. Traditional batching methods group sequences into batches of fixed size, regardless of their lengths. However, this approach can lead to significant waste of computational resources and underutilization of GPU memory. Dynamic sequence length batching addresses these issues by grouping sequences of similar lengths or adjusting batch size to hold a roughly constant number of tokens.

**Typical Workflow and Challenges**
--------------------------------

The typical workflow for sequence data processing involves the following steps:

1. **Data Preprocessing**: Sequences are preprocessed to have a fixed maximum length by padding shorter sequences with special tokens (e.g., `<PAD>`).
2. **Batching**: Sequences are grouped into batches of fixed size (e.g., 32).
3. **Model Computation**: The model processes each batch, performing computations on all sequences in the batch.

However, this approach poses several challenges:

* **Wasted computation on pad tokens**: The model performs computations on padded tokens, which do not contribute to the learning process.
* **Underutilized GPU memory**: Batches with short sequences waste GPU memory, as the memory is allocated for the maximum sequence length.
* **Poor GPU compute efficiency**: The model's computational resources are not fully utilized, leading to reduced overall efficiency.
* **Memory Constraints**: Batch size is often limited by the longest sequences in the batch, reducing the overall batch size and increasing the number of batches.

**Dynamic Batching**
------------------

Dynamic batching addresses these challenges by grouping sequences of similar lengths or adjusting batch size to hold a roughly constant number of tokens. This approach ensures that:

* **Each batch has a roughly constant number of tokens**: Reducing waste computation on pad tokens and improving GPU compute efficiency.
* **GPU memory is utilized efficiently**: Batches are created to minimize wasted memory, reducing the number of batches and improving overall efficiency.

### Masking

To prevent the model from attending to padded parts of the input, a binary mask is used to indicate which parts of the input are real data and which are padding. This mask is applied to the input data before processing.

```python
import torch
import torch.nn as nn

# Define a sample input tensor with sequence lengths
input_tensor = torch.randn(3, 10, 5)  # batch_size, seq_len, embedding_dim
sequence_lengths = torch.tensor([5, 3, 2])  # sequence lengths for each batch element

# Create a binary mask to indicate padding
mask = (torch.arange(input_tensor.size(1)).unsqueeze(0).expand_as(input_tensor) < sequence_lengths.unsqueeze(1)).float()
```

### Batching

Sequences are organized into batches, ensuring that sequences within a batch have similar lengths. This can be achieved using a bucketing approach, where sequences are grouped into buckets of similar lengths.

```python
import torch
from torch.utils.data import Dataset, DataLoader

# Define a sample dataset with sequence lengths
class SequenceDataset(Dataset):
    def __init__(self, sequences, sequence_lengths):
        self.sequences = sequences
        self.sequence_lengths = sequence_lengths

    def __getitem__(self, index):
        return self.sequences[index], self.sequence_lengths[index]

    def __len__(self):
        return len(self.sequences)

# Create a sample dataset
sequences = [torch.randn(10, 5) for _ in range(100)]  # 100 sequences of length 10
sequence_lengths = torch.tensor([5] * 100)  # sequence lengths for each sequence

dataset = SequenceDataset(sequences, sequence_lengths)

# Create a data loader with bucketing
batch_size = 32
bucket_size = 5  # group sequences into buckets of size 5

data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x)
```

**PackedSequence**
-----------------

To efficiently process sequences of varying lengths, PyTorch provides the `PackedSequence` class. This class allows for efficient computation on sequences of different lengths by:

* **Packing**: Sequences are packed into a single tensor, with padding removed.
* **Unpacking**: The packed tensor is unpacked into individual sequences.

```python
import torch
import torch.nn as nn

# Define a sample input tensor with sequence lengths
input_tensor = torch.randn(3, 10, 5)  # batch_size, seq_len, embedding_dim
sequence_lengths = torch.tensor([5, 3, 2])  # sequence lengths for each batch element

# Pack the input tensor
packed_input = nn.utils.rnn.pack_padded_sequence(input_tensor, sequence_lengths, batch_first=True)

# Unpack the packed input
unpacked_input, _ = nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
```

**Example Use Case**
--------------------

Dynamic sequence length batching can be applied to various NLP tasks, such as language modeling, text classification, and machine translation.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple language model
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths, batch_first=True)
        output, _ = self.rnn(packed_input)
        unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        logits = self.fc(unpacked_output)
        return logits

# Initialize the language model
model = LanguageModel(vocab_size=10000, embedding_dim=128, hidden_dim=256)

# Define a sample input sequence
input_seq = torch.randint(0, 10000, (32, 10))  # batch_size, seq_len
sequence_lengths = torch.tensor([5] * 32)  # sequence lengths for each batch element

# Train the language model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    output = model(input_seq)
    loss = criterion(output.view(-1, 10000), input_seq.view(-1))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

In conclusion, dynamic sequence length batching is a technique that optimizes computation and memory utilization for sequence data processing. By grouping sequences of similar lengths or adjusting batch size to hold a roughly constant number of tokens, this approach reduces waste computation on pad tokens, underutilization of GPU memory, and poor GPU compute efficiency. The `PackedSequence` class in PyTorch provides an efficient way to process sequences of varying lengths. This technique can be applied to various NLP tasks, such as language modeling, text classification, and machine translation.