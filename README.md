# ChatApp-CustomGPT-from-scratch-PretrainedGPT-LoRA-fine-tuning-LangChain
# 🧠 Custom GPT Chatbot with PDF Question Answering

This project presents a fully functional chatbot application built with PyTorch and HuggingFace Transformers. It integrates both a custom GPT model and fine-tuned GPT-2 (Large) models to support natural conversation as well as document-based question answering (QA) from PDF files.

## ✨ Features

- ✅ Custom GPT architecture implemented from scratch in PyTorch  
- ✅ Support for HuggingFace GPT-2 Large and fine-tuned GPT-2 (with LoRA on SQuAD 2.0)  
- ✅ Integration with LangChain for PDF document ingestion and retrieval  
- ✅ Context-aware question answering based on the content of uploaded PDFs  
- ✅ Graphical user interface using **CustomTkinter**  
- ✅ Adjustable generation parameters (temperature, max length)  
- ✅ Dynamic prompt length handling to avoid input size issues  

---

## 📚 Model Training Pipeline

### 1. **Pretraining**
- Dataset: [WikiText-103](https://huggingface.co/datasets/wikitext)
- Objective: Train a GPT-like language model from scratch using standard causal language modeling.

### 2. **Fine-tuning**
- Dataset: [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
- Technique: LoRA (Low-Rank Adaptation) for efficient parameter updates
- Purpose: Adapt the base model for extractive and generative question answering.

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


