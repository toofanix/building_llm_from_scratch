FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    software-properties-common \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-dev python3.12-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install distutils manually
RUN apt-get update && \
    apt-get install -y python3-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symbolic links
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install Node.js and global packages
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @charmland/crush && \
    npm install -g @anthropic-ai/claude-code && \
    npm install -g @openai/codex && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | bash && \
    export PATH="/root/.local/bin:$PATH" && \
    uv --version

ENV PATH="/root/.local/bin:$PATH"

# Create UV virtual environment
RUN uv venv --python=3.12 /opt/venv

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock /workspace/

# Install PyTorch + dependencies
RUN . /opt/venv/bin/activate && \
    echo "Installing PyTorch with CUDA support..." && \
    uv pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    echo "Installing remaining packages from pyproject.toml..." && \
    cd /workspace && uv pip install -e .

# Create Codex config
RUN mkdir -p /root/.codex
RUN cat <<'EOF' > /root/.codex/config.toml
[model_providers.z_ai]
name = "Z.ai - GLM Coding Plan"
base_url = "https://api.z.ai/api/coding/paas/v4"
env_key = "Z_AI_API_KEY"
wire_api = "chat"
query_params = {}

[model_providers.chutes]
name = "Chutes.ai - Coding Provider"
base_url = "https://llm.chutes.ai/v1/"
env_key = "CHUTES_CODEX_API_KEY"
wire_api = "chat"
query_params = {}

[profiles.glm]
model = "glm-4.6"
model_provider = "z_ai"

[profiles.kimi]
model = "moonshotai/Kimi-K2-Thinking"
model_provider = "chutes"
EOF

# Create Claude config
RUN mkdir -p /root/.claude
RUN cat <<'EOF' > /root/.claude/setting.json
{
  "env": {
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.5-air",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.6",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-4.6"
  },
  "alwaysThinkingEnabled": false
}
EOF

# Expose ports
EXPOSE 8888 6006

# Keep container running
CMD ["tail", "-f", "/dev/null"]

# =============================================================================
# DOCKER USAGE COMMANDS
# =============================================================================

# 1. BUILD IMAGE AND CONTAINER:
#    docker-compose up -d --build
#    OR
#    docker build -t llm-from-scratch .
#    docker-compose up -d

# 2. ENTER RUNNING CONTAINER:
#    docker-compose exec app bash
#    OR
#    docker exec -it llm-from-scratch bash

# 3. STOP AND START CONTAINER:
#    docker-compose stop
#    docker-compose start
#    OR
#    docker stop llm-from-scratch
#    docker start llm-from-scratch

# 4. START JUPYTER IN CONTAINER:
#    # Enter container first: docker-compose exec app bash
#    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
#    # Get token: jupyter notebook list
#    # Access at: http://localhost:8888/?token=<your-token>

# 5. CLAUDE CODE AND CODEX USAGE:
#    # Configuration is pre-configured in the container
#    # Just set your API keys in .env.local and use the tools directly
#
#    # Use Claude Code (Z.AI provider only):
#    claude-code "help me understand this code"
#
#    # Use Codex with Z.AI provider (glm-4.6):
#    codex --profile glm_4_6 "explain this function"
#
#    # Use Codex with Chutes.ai provider (moonshotai/Kimi-K2-Thinking):
#    codex --profile kimi_k2 "refactor this code"
#
#    # API keys are loaded from .env.local file in project root
# =============================================================================