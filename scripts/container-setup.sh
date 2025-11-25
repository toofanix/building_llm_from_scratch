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
ANTHROPIC_BASE_URL=$(get_env_var "ANTHROPIC_BASE_URL" "https://api.z.ai/api/anthropic")
ANTHROPIC_AUTH_TOKEN=$(get_env_var "ANTHROPIC_AUTH_TOKEN" "")
Z_AI_API_KEY=$(get_env_var "Z_AI_API_KEY" "")
Z_AI_BASE_URL=$(get_env_var "Z_AI_BASE_URL" "https://api.z.ai/api/coding/paas/v4")
CLAUDE_DEFAULT_PROVIDER=$(get_env_var "CLAUDE_DEFAULT_PROVIDER" "zai")

# Model configuration
ANTHROPIC_DEFAULT_HAIKU_MODEL=$(get_env_var "ANTHROPIC_DEFAULT_HAIKU_MODEL" "glm-4.5-air")
ANTHROPIC_DEFAULT_SONNET_MODEL=$(get_env_var "ANTHROPIC_DEFAULT_SONNET_MODEL" "glm-4.6")
ANTHROPIC_DEFAULT_OPUS_MODEL=$(get_env_var "ANTHROPIC_DEFAULT_OPUS_MODEL" "glm-4.6")

echo "[INFO] Configuration loaded:"
echo "  - ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"
echo "  - Z_AI_BASE_URL: $Z_AI_BASE_URL"
echo "  - CLAUDE_DEFAULT_PROVIDER: $CLAUDE_DEFAULT_PROVIDER"

# Generate Codex config.toml
echo "[INFO] Generating Codex configuration..."
cat > /root/.codex/config.toml << EOF
[model_providers.z_ai]
name = "Z.ai - GLM Coding Plan"
base_url = "$Z_AI_BASE_URL"
env_key = "Z_AI_API_KEY"
wire_api = "chat"
query_params = {}

[profiles.glm_4_6]
model = "glm-4.6"
model_provider = "z_ai"
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
    echo "[OK] ANTHROPIC_AUTH_TOKEN exported"
else
    echo "[WARN] ANTHROPIC_AUTH_TOKEN not set"
fi

if [ -n "$Z_AI_API_KEY" ]; then
    export Z_AI_API_KEY="$Z_AI_API_KEY"
    echo "[OK] Z_AI_API_KEY exported"
else
    echo "[WARN] Z_AI_API_KEY not set"
fi

export ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"
export Z_AI_BASE_URL="$Z_AI_BASE_URL"
export CLAUDE_DEFAULT_PROVIDER="$CLAUDE_DEFAULT_PROVIDER"

echo "[OK] Environment variables exported"

# Add environment variables to .bashrc for interactive sessions
{
    echo "# Claude Code and Codex Environment Variables"
    echo "export ANTHROPIC_BASE_URL=\"$ANTHROPIC_BASE_URL\""
    echo "export Z_AI_BASE_URL=\"$Z_AI_BASE_URL\""
    echo "export CLAUDE_DEFAULT_PROVIDER=\"$CLAUDE_DEFAULT_PROVIDER\""
    if [ -n "$ANTHROPIC_AUTH_TOKEN" ]; then
        echo "export ANTHROPIC_AUTH_TOKEN=\"$ANTHROPIC_AUTH_TOKEN\""
    fi
    if [ -n "$Z_AI_API_KEY" ]; then
        echo "export Z_AI_API_KEY=\"$Z_AI_API_KEY\""
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
            export ANTHROPIC_MODEL="glm-4.6"
            echo "Switched to Z.AI provider"
            ;;
        "anthropic")
            export ANTHROPIC_BASE_URL="https://api.anthropic.com"
            echo "Switched to Anthropic provider"
            ;;
        *)
            echo "Unknown provider: $provider. Available: zai, anthropic"
            return 1
            ;;
    esac
}

# Show current configuration
show-claude-config() {
    echo "Claude Code Configuration:"
    echo "  ANTHROPIC_BASE_URL: ${ANTHROPIC_BASE_URL:-'Not set'}"
    echo "  ANTHROPIC_AUTH_TOKEN: ${ANTHROPIC_AUTH_TOKEN:+'Set'}${ANTHROPIC_AUTH_TOKEN:-'Not set'}"
    echo "  Z_AI_API_KEY: ${Z_AI_API_KEY:+'Set'}${Z_AI_API_KEY:-'Not set'}"
    echo "  CLAUDE_DEFAULT_PROVIDER: ${CLAUDE_DEFAULT_PROVIDER:-'zai'}"

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

    # Test Z.AI API
    if [ -n "${Z_AI_API_KEY:-}" ] && [ -n "${Z_AI_BASE_URL:-}" ]; then
        echo "Testing Z.AI API..."
        if curl -sS -H "Authorization: Bearer $Z_AI_API_KEY" "$Z_AI_BASE_URL/v1/models" >/dev/null 2>&1; then
            echo "✅ Z.AI API connection successful"
        else
            echo "❌ Z.AI API connection failed"
        fi
    else
        echo "❌ Z.AI API not configured"
    fi

    # Test Anthropic API
    if [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ] && [ -n "${ANTHROPIC_BASE_URL:-}" ]; then
        echo "Testing Anthropic API..."
        if curl -sS -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" -H "Content-Type: application/json" \
           -d '{"model":"glm-4.6","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
           "$ANTHROPIC_BASE_URL/v1/messages" >/dev/null 2>&1; then
            echo "✅ Anthropic API connection successful"
        else
            echo "❌ Anthropic API connection failed"
        fi
    else
        echo "❌ Anthropic API not configured"
    fi
}

# Add aliases for convenience
alias claude-config="show-claude-config"
alias claude-switch="switch-claude-provider"
alias claude-test="test-apis"

echo "Claude utility functions loaded. Use:"
echo "  - claude-config: Show current configuration"
echo "  - claude-switch <provider>: Switch provider (zai, anthropic)"
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