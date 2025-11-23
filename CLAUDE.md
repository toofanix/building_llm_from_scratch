# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hands-on implementation of building Large Language Models (LLMs) from scratch, following Sebastian Raschka's book "Building LLM from scratch". The project implements core Transformer architecture components for educational purposes.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Running the Code
```bash
# Run the main entry point
python main.py

# Run with uv
uv run python main.py
```

### Development
```bash
# Install in development mode
uv pip install -e .

# Run with specific Python version (requires Python 3.12+)
uv run python main.py
```

## Architecture Overview

The codebase implements a complete GPT-style Transformer architecture with the following key components located in `utils.py`:

### Core Neural Network Components
- **MultiHeadAttention**: Implements multi-head self-attention with causal masking
- **LayerNorm**: Custom layer normalization implementation
- **GELU**: Gaussian Error Linear Unit activation function
- **FeedForward**: Position-wise feed-forward network
- **TransformerBlock**: Complete transformer block with attention and FFN layers
- **GPTModel**: Full GPT architecture with token and positional embeddings

### Training Infrastructure
- **GPTDatasetV1**: Dataset class for creating training samples from text
- **create_dataloader_v1**: Factory function for creating efficient data loaders
- **Training utilities**: Loss calculation, model evaluation, and text generation functions

### Key Dependencies
- **PyTorch**: Core neural network framework
- **tiktoken**: GPT-2 tokenizer from OpenAI
- **Standard ML libraries**: numpy, pandas, matplotlib, altair for data manipulation and visualization

## Data Organization
- Training text data is stored in `data/chapter2/the-verdict.txt`
- The dataset uses GPT-2 tokenizer for tokenization
- Configurable context length and stride for creating training sequences

## Configuration Notes
- Requires Python 3.12+
- Uses UV package manager for dependency management
- All core model parameters are configurable via a cfg dictionary passed to model components