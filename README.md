# ChatApp-CustomGPT-from-scratch-PretrainedGPT-LoRA-fine-tuning-LangChain
# GPT Chatbot with PDF Question Answering

This project presents a fully functional chatbot application built with PyTorch and HuggingFace Transformers. 
It integrates both a custom GPT model and pre-trained GPT-2 models to support natural conversation as well as document-based question answering (QA) from PDF files.

## Overview

This project implements a conversational AI system using both custom-built and pre-trained GPT models. It features:

- A custom Transformer-based GPT model trained from scratch on WikiText-103
- Fine-tuning on SQuAD 2.0 for question answering
- Integration of HuggingFace's GPT-2 Large with PDF processing via LangChain
- A GUI application for interactive chatting with model switching capabilities

---

##  Technical Stack
| Attempt | #1    | #2    |
| :-----: | :---: | :---: |
| Seconds | 301   | 283   |

Component	Technologies Used
Core Models	PyTorch, Transformers
Custom GPT	Transformer architecture with 6 layers, 8 heads
Fine-Tuning	LoRA (Low-Rank Adaptation) for efficient tuning
PDF Processing	LangChain, PyPDF, ChromaDB
GUI	CustomTkinter
Optimization	Gradient Accumulation, OneCycleLR

---

## 🛠 Custom GPT Model Architecture

Implemented using `nn.Transformer`:

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, block_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.positional_encoding = nn.Parameter(torch.zeros(1, block_size, embedding_dim))
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

        # Initialize weights after creating all layers
        self.init_weights()

    def init_weights(self):
        # Initialize weights for model layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Dynamic positions
        positional_encoding = generate_positional_encoding(src.size(1), self.embedding.embedding_dim, src.device)
        src_emb = self.embedding(src) + positional_encoding
        tgt_emb = self.embedding(tgt) + positional_encoding[:, :tgt.size(1), :]

        src_emb = self.norm(src_emb)
        tgt_emb = self.norm(tgt_emb)

        transformer_output = self.transformer(
            src_emb.transpose(0, 1), tgt_emb.transpose(0, 1), src_mask, tgt_mask, memory_mask
        )

        logits = self.fc_out(transformer_output.transpose(0, 1))
        return logits
```


