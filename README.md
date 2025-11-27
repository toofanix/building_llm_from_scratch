# building_llm_from_scratch
Building LLM from scratch - Sebastian Raschka

- In this repos I am trying to follow the book `Building LLM from scratch` by Sebastian Raschka.
- I will be working with the code in this book and my iterations on it.

## Docker Setup

This project includes a Docker environment with GPU support and AI coding assistants pre-configured.

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (optional but recommended)
- API keys for Z.AI and/or Chutes.ai

### Quick Start

1. **Copy the environment template:**
   ```bash
   cp .env.example .env.local
   ```

2. **Configure your API keys in `.env.local`:**
   ```bash
   # Claude Code - Z.AI Provider
   ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
   ANTHROPIC_AUTH_TOKEN=your_zai_api_key_here
   
   # Codex - Z.AI Provider
   Z_AI_API_KEY=your_zai_codex_api_key_here
   
   # Codex - Chutes.ai Provider
   CHUTES_CODEX_API_KEY=your_chutes_codex_api_key_here
   ```

3. **Build and start the container:**
   ```bash
   docker-compose up -d --build
   ```

4. **Enter the container:**
   ```bash
   docker-compose exec app bash
   ```

### AI Coding Assistants

The container includes two AI coding assistants with pre-configured settings.

#### Claude Code (Z.AI Provider)
Claude Code uses the Z.AI provider with GLM models. Configuration is automatically set during container build.

**Usage:**
```bash
claude-code "help me understand this code"
claude-code "explain the attention mechanism"
```

#### Codex (Dual Provider Support)
Codex supports both Z.AI and Chutes.ai providers via pre-configured profiles.

**Available Profiles:**
- `glm_4_6`: Z.AI provider with glm-4.6 model
- `kimi_k2`: Chutes.ai provider with moonshotai/Kimi-K2-Thinking model

**Usage:**
```bash
# Use Z.AI provider (glm-4.6)
codex --profile glm_4_6 "explain this function"

# Use Chutes.ai provider (Kimi-K2-Thinking)
codex --profile kimi_k2 "refactor this code"
```

### Jupyter Notebook

To start Jupyter Notebook inside the container:

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Access it at `http://localhost:8888` with the token displayed in the terminal.

### Container Management

**Stop the container:**
```bash
docker-compose stop
```

**Start the container:**
```bash
docker-compose start
```

**View logs:**
```bash
docker-compose logs -f
```

**Rebuild the container:**
```bash
docker-compose up -d --build
```

## Chapter Progress

### Chapter 1:
- Gives an overview and background about LLM

### Chapter 2:
- Starts with tokenization