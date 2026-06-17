### GPT 2 Style Transformer

A clean implementation of a GPT 2 style Transformer in PyTorch, built to understand how language models work at the tensor level

This project implements the core Transformer stack from first principles: tokenization flow, embeddings, causal self-attention, residual streams, MLPs, logits, next token prediction, and a basic training loop

### Motivation

I focused on three core questions:

1. How do tokens become vectors?
2. How does causal attention move information through the residual stream?
3. How does the model produce logits and learn next token prediction?

### Implemented

- Layer normalisation
- Learned token embeddings
- Learned positional embeddings
- Casual multi head self attention
- Attention masking
- Pre-LN residual transformer blocks
- GELU MLP layers
- Final unembedding to vocabulary logits
- Basic autoregressive generation

### Architecture
```text
tokens
  ↓
token embedding + positional embedding
  ↓
Transformer blocks
  ├── LayerNorm
  ├── Causal Multi-Head Self-Attention
  ├── Residual Add
  ├── LayerNorm
  ├── MLP
  └── Residual Add
  ↓
final LayerNorm
  ↓
unembedding
  ↓
logits over vocabulary
```

### Future works

This will including separate class and implementation to try different sampling methods and writing a caching system
