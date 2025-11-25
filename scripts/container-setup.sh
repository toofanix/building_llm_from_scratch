#!/bin/bash

# Container Setup Script for Claude Code and Codex
# This script configures AI tools from environment variables

set -euo pipefail

echo "[INFO] Setting up Claude Code and Codex configuration..."

# Function to get environment variable with default
get_env_var() {
    local var_name="$1"
    local default_value="${2:-}"
    local value="${!var_name:-$default_value}"
    echo "$value"
}

# Get configuration from environment variables
CLAUDE_DEFAULT_PROVIDER=$(get_env_var "CLAUDE_DEFAULT_PROVIDER" "zai")

# Z.AI Configuration
ANTHROPIC_BASE_URL=$(get_env_var "ANTHROPIC_BASE_URL" "https://api.z.ai/api/anthropic")
ANTHROPIC_AUTH_TOKEN=$(get_env_var "ANTHROPIC_AUTH_TOKEN" "")

# Chutes.ai Configuration
CHUTES_API_KEY=$(get_env_var "CHUTES_API_KEY" "")
CHUTES_BASE_URL=$(get_env_var "CHUTES_BASE_URL" "https://claude.chutes.ai")
CHUTES_BACKEND_URL=$(get_env_var "CHUTES_BACKEND_URL" "https://llm.chutes.ai")

# Codex Configuration (Z.AI)
Z_AI_API_KEY=$(get_env_var "Z_AI_API_KEY" "")
Z_AI_BASE_URL=$(get_env_var "Z_AI_BASE_URL" "https://api.z.ai/api/coding/paas/v4")

# Codex Configuration (Chutes.ai)
CHUTES_CODEX_API_KEY=$(get_env_var "CHUTES_CODEX_API_KEY" "")
CHUTES_CODEX_BASE_URL=$(get_env_var "CHUTES_CODEX_BASE_URL" "https://api.chutes.ai/v1")

# Model configuration
ANTHROPIC_DEFAULT_HAIKU_MODEL=$(get_env_var "ANTHROPIC_DEFAULT_HAIKU_MODEL" "glm-4.5-air")
ANTHROPIC_DEFAULT_SONNET_MODEL=$(get_env_var "ANTHROPIC_DEFAULT_SONNET_MODEL" "glm-4.6")
ANTHROPIC_DEFAULT_OPUS_MODEL=$(get_env_var "ANTHROPIC_DEFAULT_OPUS_MODEL" "glm-4.6")

echo "[INFO] Configuration loaded:"
echo "  - CLAUDE_DEFAULT_PROVIDER: $CLAUDE_DEFAULT_PROVIDER"
echo "  - Z.AI ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"
echo "  - Chutes.ai CHUTES_BASE_URL: $CHUTES_BASE_URL"
echo "  - Z.AI Codex URL: $Z_AI_BASE_URL"
echo "  - Chutes.ai Codex URL: $CHUTES_CODEX_BASE_URL"

# Generate Codex config.toml with both providers
echo "[INFO] Generating Codex configuration..."
cat > /root/.codex/config.toml << EOF
[model_providers.z_ai]
name = "Z.ai - GLM Coding Plan"
base_url = "$Z_AI_BASE_URL"
env_key = "Z_AI_API_KEY"
wire_api = "chat"
query_params = {}

[model_providers.chutes]
name = "Chutes.ai - Coding Provider"
base_url = "$CHUTES_CODEX_BASE_URL"
env_key = "CHUTES_CODEX_API_KEY"
wire_api = "chat"
query_params = {}

[profiles.glm_4_6]
model = "glm-4.6"
model_provider = "z_ai"

[profiles.chutes_default]
model = "claude-3-5-sonnet-20241022"
model_provider = "chutes"
EOF

echo "[OK] Codex config written to /root/.codex/config.toml"

# Generate Claude setting.json
echo "[INFO] Generating Claude configuration..."
cat > /root/.claude/setting.json << EOF
{
  "env": {
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "$ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "$ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "$ANTHROPIC_DEFAULT_OPUS_MODEL"
  },
  "alwaysThinkingEnabled": false
}
EOF

echo "[OK] Claude settings written to /root/.claude/setting.json"

# Export environment variables for current session
if [ -n "$ANTHROPIC_AUTH_TOKEN" ]; then
    export ANTHROPIC_AUTH_TOKEN="$ANTHROPIC_AUTH_TOKEN"
    echo "[OK] Z.AI ANTHROPIC_AUTH_TOKEN exported"
else
    echo "[WARN] Z.AI ANTHROPIC_AUTH_TOKEN not set"
fi

if [ -n "$CHUTES_API_KEY" ]; then
    export CHUTES_API_KEY="$CHUTES_API_KEY"
    echo "[OK] Chutes.ai CHUTES_API_KEY exported"
else
    echo "[WARN] Chutes.ai CHUTES_API_KEY not set"
fi

if [ -n "$Z_AI_API_KEY" ]; then
    export Z_AI_API_KEY="$Z_AI_API_KEY"
    echo "[OK] Z.AI Codex Z_AI_API_KEY exported"
else
    echo "[WARN] Z.AI Codex Z_AI_API_KEY not set"
fi

if [ -n "$CHUTES_CODEX_API_KEY" ]; then
    export CHUTES_CODEX_API_KEY="$CHUTES_CODEX_API_KEY"
    echo "[OK] Chutes.ai Codex CHUTES_CODEX_API_KEY exported"
else
    echo "[WARN] Chutes.ai Codex CHUTES_CODEX_API_KEY not set"
fi

export ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"
export CHUTES_BASE_URL="$CHUTES_BASE_URL"
export CHUTES_BACKEND_URL="$CHUTES_BACKEND_URL"
export Z_AI_BASE_URL="$Z_AI_BASE_URL"
export CHUTES_CODEX_BASE_URL="$CHUTES_CODEX_BASE_URL"
export CLAUDE_DEFAULT_PROVIDER="$CLAUDE_DEFAULT_PROVIDER"

echo "[OK] Environment variables exported"

# Add environment variables to .bashrc for interactive sessions
{
    echo "# Claude Code and Codex Environment Variables"
    echo "export ANTHROPIC_BASE_URL=\"$ANTHROPIC_BASE_URL\""
    echo "export CHUTES_BASE_URL=\"$CHUTES_BASE_URL\""
    echo "export CHUTES_BACKEND_URL=\"$CHUTES_BACKEND_URL\""
    echo "export Z_AI_BASE_URL=\"$Z_AI_BASE_URL\""
    echo "export CHUTES_CODEX_BASE_URL=\"$CHUTES_CODEX_BASE_URL\""
    echo "export CLAUDE_DEFAULT_PROVIDER=\"$CLAUDE_DEFAULT_PROVIDER\""
    if [ -n "$ANTHROPIC_AUTH_TOKEN" ]; then
        echo "export ANTHROPIC_AUTH_TOKEN=\"$ANTHROPIC_AUTH_TOKEN\""
    fi
    if [ -n "$CHUTES_API_KEY" ]; then
        echo "export CHUTES_API_KEY=\"$CHUTES_API_KEY\""
    fi
    if [ -n "$Z_AI_API_KEY" ]; then
        echo "export Z_AI_API_KEY=\"$Z_AI_API_KEY\""
    fi
    if [ -n "$CHUTES_CODEX_API_KEY" ]; then
        echo "export CHUTES_CODEX_API_KEY=\"$CHUTES_CODEX_API_KEY\""
    fi
    echo ""
} >> /root/.bashrc

echo "[OK] Environment variables added to .bashrc"

# Create provider switching functions
echo "[INFO] Creating provider switching utility functions..."
cat > /root/.claude-utils.sh << 'EOF'
#!/bin/bash

# Claude Code and Codex utility functions

# Switch Claude Code provider
switch-claude-provider() {
    local provider="${1:-zai}"
    echo "Switching Claude Code provider to: $provider"

    case "$provider" in
        "zai")
            export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
            export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$ANTHROPIC_AUTH_TOKEN}"
            export ANTHROPIC_MODEL="glm-4.6"
            echo "Switched to Z.AI provider"
            ;;
        "chutes")
            export ANTHROPIC_BASE_URL="$CHUTES_BASE_URL"
            export ANTHROPIC_AUTH_TOKEN="${CHUTES_API_KEY:-$CHUTES_API_KEY}"
            export ANTHROPIC_MODEL="claude-3-5-sonnet-20241022"
            echo "Switched to Chutes.ai provider"
            ;;
        *)
            echo "Unknown provider: $provider. Available: zai, chutes"
            return 1
            ;;
    esac
}

# Switch Codex provider
switch-codex-provider() {
    local provider="${1:-zai}"
    echo "Switching Codex provider to: $provider"

    case "$provider" in
        "zai")
            export CODEX_PROVIDER="z_ai"
            export CODEX_API_KEY="${Z_AI_API_KEY:-$Z_AI_API_KEY}"
            echo "Switched to Z.AI Codex provider"
            ;;
        "chutes")
            export CODEX_PROVIDER="chutes"
            export CODEX_API_KEY="${CHUTES_CODEX_API_KEY:-$CHUTES_CODEX_API_KEY}"
            echo "Switched to Chutes.ai Codex provider"
            ;;
        *)
            echo "Unknown provider: $provider. Available: zai, chutes"
            return 1
            ;;
    esac
}

# Show current configuration
show-claude-config() {
    echo "Claude Code Configuration:"
    echo "  CLAUDE_DEFAULT_PROVIDER: ${CLAUDE_DEFAULT_PROVIDER:-'zai'}"
    echo "  ANTHROPIC_BASE_URL: ${ANTHROPIC_BASE_URL:-'Not set'}"
    echo "  ANTHROPIC_AUTH_TOKEN: ${ANTHROPIC_AUTH_TOKEN:+'Set'}${ANTHROPIC_AUTH_TOKEN:-'Not set'}"
    echo "  CHUTES_API_KEY: ${CHUTES_API_KEY:+'Set'}${CHUTES_API_KEY:-'Not set'}"
    echo "  CHUTES_BASE_URL: ${CHUTES_BASE_URL:-'Not set'}"
    echo ""
    echo "Codex Configuration:"
    echo "  Z_AI_API_KEY: ${Z_AI_API_KEY:+'Set'}${Z_AI_API_KEY:-'Not set'}"
    echo "  Z_AI_BASE_URL: ${Z_AI_BASE_URL:-'Not set'}"
    echo "  CHUTES_CODEX_API_KEY: ${CHUTES_CODEX_API_KEY:+'Set'}${CHUTES_CODEX_API_KEY:-'Not set'}"
    echo "  CHUTES_CODEX_BASE_URL: ${CHUTES_CODEX_BASE_URL:-'Not set'}"

    if command -v claude-code >/dev/null 2>&1; then
        echo ""
        echo "Claude Code version: $(claude-code --version 2>/dev/null || echo 'Unknown')"
    fi

    if command -v codex >/dev/null 2>&1; then
        echo "Codex is available"
    fi
}

# Test API connections
test-apis() {
    echo "Testing API connections..."

    # Test Z.AI Claude API
    if [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ] && [ -n "${ANTHROPIC_BASE_URL:-}" ]; then
        echo "Testing Z.AI Claude API..."
        if curl -sS -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" -H "Content-Type: application/json" \
           -d '{"model":"glm-4.6","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
           "$ANTHROPIC_BASE_URL/v1/messages" >/dev/null 2>&1; then
            echo "✅ Z.AI Claude API connection successful"
        else
            echo "❌ Z.AI Claude API connection failed"
        fi
    else
        echo "❌ Z.AI Claude API not configured"
    fi

    # Test Chutes.ai Claude API
    if [ -n "${CHUTES_API_KEY:-}" ] && [ -n "${CHUTES_BASE_URL:-}" ]; then
        echo "Testing Chutes.ai Claude API..."
        if curl -sS -H "Authorization: Bearer $CHUTES_API_KEY" -H "Content-Type: application/json" \
           -d '{"model":"claude-3-5-sonnet-20241022","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
           "$CHUTES_BASE_URL/v1/messages" >/dev/null 2>&1; then
            echo "✅ Chutes.ai Claude API connection successful"
        else
            echo "❌ Chutes.ai Claude API connection failed"
        fi
    else
        echo "❌ Chutes.ai Claude API not configured"
    fi

    # Test Z.AI Codex API
    if [ -n "${Z_AI_API_KEY:-}" ] && [ -n "${Z_AI_BASE_URL:-}" ]; then
        echo "Testing Z.AI Codex API..."
        if curl -sS -H "Authorization: Bearer $Z_AI_API_KEY" "$Z_AI_BASE_URL/v1/models" >/dev/null 2>&1; then
            echo "✅ Z.AI Codex API connection successful"
        else
            echo "❌ Z.AI Codex API connection failed"
        fi
    else
        echo "❌ Z.AI Codex API not configured"
    fi

    # Test Chutes.ai Codex API
    if [ -n "${CHUTES_CODEX_API_KEY:-}" ] && [ -n "${CHUTES_CODEX_BASE_URL:-}" ]; then
        echo "Testing Chutes.ai Codex API..."
        if curl -sS -H "Authorization: Bearer $CHUTES_CODEX_API_KEY" "$CHUTES_CODEX_BASE_URL/v1/models" >/dev/null 2>&1; then
            echo "✅ Chutes.ai Codex API connection successful"
        else
            echo "❌ Chutes.ai Codex API connection failed"
        fi
    else
        echo "❌ Chutes.ai Codex API not configured"
    fi
}

# Add aliases for convenience
alias claude-config="show-claude-config"
alias claude-switch="switch-claude-provider"
alias codex-switch="switch-codex-provider"
alias claude-test="test-apis"

echo "Claude and Codex utility functions loaded. Use:"
echo "  - claude-config: Show current configuration"
echo "  - claude-switch <provider>: Switch Claude provider (zai, chutes)"
echo "  - codex-switch <provider>: Switch Codex provider (zai, chutes)"
echo "  - claude-test: Test API connections"
EOF

chmod +x /root/.claude-utils.sh

# Add utility functions to .bashrc
{
    echo "# Claude Code and Codex utility functions"
    echo "source /root/.claude-utils.sh"
    echo ""
} >> /root/.bashrc

echo "[OK] Utility functions created and added to .bashrc"

echo ""
echo "[SUCCESS] Container setup completed!"
echo ""
echo "Available tools:"
echo "  - claude-code: Claude Code CLI"
echo "  - codex: Codex CLI"
echo "  - crush: Crush CLI"
echo ""
echo "Utility commands:"
echo "  - claude-config: Show current configuration"
echo "  - claude-switch <provider>: Switch providers"
echo "  - claude-test: Test API connections"
echo ""

# Execute the original command if provided
if [ $# -gt 0 ]; then
    echo "[INFO] Executing: $*"
    exec "$@"
else
    # Default: keep container running
    echo "[INFO] Container ready. Starting shell..."
    exec tail -f /dev/null
fi