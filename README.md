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
   - `ANTHROPIC_AUTH_TOKEN`: Your Z.AI API key for Claude Code
   - `Z_AI_API_KEY`: Your Z.AI API key for Codex (optional)
   - `CHUTES_CODEX_API_KEY`: Your Chutes.ai API key for Codex (optional)

   See `ENV_SETUP.md` for the complete environment variable structure.

3. **Build and start the container:**
   ```bash
   docker-compose up -d --build
   ```

4. **Enter the container:**
   ```bash
   docker-compose exec app bash
   ```

### AI Coding Assistants

The container includes two AI coding assistants:

#### Claude Code (Z.AI Provider Only)
Claude Code is configured to use only the Z.AI provider with the GLM models. Configuration is automatically set via environment variables in `.bashrc`.

**Usage:**
```bash
claude-code "help me understand this code"
claude-code "explain the attention mechanism"
```

#### Codex (Dual Provider Support)
Codex supports both Z.AI and Chutes.ai providers, configured via `config.toml`.

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

### Utility Commands

Inside the container, you have access to these utility commands:

- **`claude-config`**: Display current configuration for Claude Code and Codex
  ```bash
  claude-config
  ```

- **`test-apis`**: Test API connections to verify your credentials
  ```bash
  test-apis
  ```

- **`codex-switch <provider>`**: Get instructions for using different Codex providers
  ```bash
  codex-switch zai      # Instructions for Z.AI provider
  codex-switch chutes   # Instructions for Chutes.ai provider
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