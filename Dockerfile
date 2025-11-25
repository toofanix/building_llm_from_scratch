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

# Copy requirements
COPY requirements.txt /workspace/requirements.txt

# Install PyTorch + dependencies
RUN . /opt/venv/bin/activate && \
    echo "Installing PyTorch with CUDA support..." && \
    uv pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    echo "Installing remaining packages from requirements.txt..." && \
    uv pip install -r /workspace/requirements.txt

# Set non-sensitive environment variables only
ENV ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic

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