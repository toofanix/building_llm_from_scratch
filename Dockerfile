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

# Copy container setup script
COPY scripts/container-setup.sh /usr/local/bin/container-setup.sh
RUN chmod +x /usr/local/bin/container-setup.sh

# Create configuration directories
RUN mkdir -p /root/.codex /root/.claude

# Run container setup on startup
ENTRYPOINT ["/usr/local/bin/container-setup.sh"]

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

# 5. CLAUDE CODE AND CODEX CONFIGURATION:
#    # Check current configuration:
#    claude-config
#
#    # Test API connections:
#    test-apis
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
#    # Get instructions for switching Codex providers:
#    codex-switch zai     # Instructions for Z.AI provider
#    codex-switch chutes  # Instructions for Chutes.ai provider
#
#    # Configuration is loaded from .env.local file in project root
#    # Claude Code uses Z.AI only, Codex supports both Z.AI and Chutes.ai
# =============================================================================