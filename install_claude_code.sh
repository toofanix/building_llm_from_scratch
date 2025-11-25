#!/bin/bash

### Claude Code Installer with Dual Provider Support (Chutes.ai + Z.AI)
### Option C-1: wrapper + --provider + set-vscode-provider
### VS Code default provider: Z.AI
### Model aliasing: all Z.AI models map to glm-4.6

set -euo pipefail

SCRIPT_NAME=$(basename "$0")
NODE_MIN_VERSION=18
NODE_INSTALL_VERSION=22
NVM_VERSION="v0.40.3"
CLAUDE_PACKAGE="@anthropic-ai/claude-code"
CONFIG_DIR="$HOME/.claude"
SETTINGS_FILE="$CONFIG_DIR/settings.json"
ACTIVE_PROVIDER_FILE="$CONFIG_DIR/active-provider"
VSCODE_PROVIDER_FILE="$CONFIG_DIR/vscode-provider"

# Provider constants
PROVIDER_CHUTES="chutes"
PROVIDER_ZAI="zai"

# Provider URLs
DEFAULT_CHUTES_PROXY="https://claude.chutes.ai"
DEFAULT_CHUTES_BACKEND="https://llm.chutes.ai"
DEFAULT_CHUTES_KEY_URL="https://chutes.ai/app/api"
DEFAULT_ZAI_PROXY="https://api.z.ai/api/anthropic"
DEFAULT_ZAI_BACKEND="https://api.z.ai/api/anthropic"
DEFAULT_ZAI_KEY_URL="https://z.ai/model-api"

API_TIMEOUT_MS=6000000
ZAI_MODEL_ALIAS="glm-4.6"   # all Z.AI model slots map to this

# Logging
log_info() { echo "[INFO] $*"; }
log_error() { echo "[ERR ] $*" >&2; }
log_success() { echo "[ OK ] $*"; }

ensure_dir_exists() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" || { log_error "Failed to mkdir $dir"; exit 1; }
    fi
}

# Node.js / NVM installation
install_nodejs() {
    local platform
    platform=$(uname -s)
    case "$platform" in
        Linux|Darwin)
            log_info "Installing NVM ($NVM_VERSION)..."
            curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/$NVM_VERSION/install.sh" | bash
            . "$HOME/.nvm/nvm.sh"
            log_info "Installing Node.js v$NODE_INSTALL_VERSION..."
            nvm install "$NODE_INSTALL_VERSION"
            log_success "Installed Node.js $(node -v) and npm $(npm -v)"
            ;;
        *)
            log_error "Unsupported platform: $platform"
            exit 1
            ;;
    esac
}

check_nodejs() {
    if command -v node >/dev/null 2>&1; then
        local ver
        ver=$(node -v | sed 's/^v//')
        local major
        major=$(echo "$ver" | cut -d. -f1)
        if [ "$major" -ge "$NODE_MIN_VERSION" ]; then
            log_success "Node.js is already installed (v$ver)"
            return
        else
            log_info "Node.js version v$ver is too old. Installing fresh."
            install_nodejs
        fi
    else
        log_info "Node.js is not installed. Installing."
        install_nodejs
    fi
}

# Install or update Claude Code
install_claude_code() {
    if command -v claude >/dev/null 2>&1; then
        log_info "Claude Code already installed. Updating..."
        claude update || { log_error "Failed to update claude"; exit 1; }
    else
        log_info "Installing Claude Code..."
        npm install -g "$CLAUDE_PACKAGE" || { log_error "npm install failed"; exit 1; }
    fi
    log_success "Claude Code is installed: $(claude --version || echo '(version unknown)')"
}

# Prompt for API key for a provider (env var fallback)
prompt_api_key_for_provider() {
    local provider="$1"
    local key_url api_env_var api_key
    case "$provider" in
        "$PROVIDER_CHUTES") key_url=$DEFAULT_CHUTES_KEY_URL; api_env_var="${CHUTES_API_KEY:-}" ;;
        "$PROVIDER_ZAI") key_url=$DEFAULT_ZAI_KEY_URL; api_env_var="${Z_AI_API_KEY:-}" ;;
        *) log_error "Unknown provider: $provider"; exit 1 ;;
    esac

    if [ -n "$api_env_var" ]; then
        log_info "Using $provider API key from environment variable"
        api_key="$api_env_var"
    else
        echo ""
        log_info "Get your $provider API key at: $key_url"
        read -s -p "Enter $provider API key: " api_key </dev/tty
        echo ""
        if [ -z "$api_key" ]; then
            log_error "API key cannot be empty"; exit 1
        fi
    fi
    echo "$api_key"
}

# Fetch models from provider
fetch_models() {
    local provider="$1"
    local api_key="$2"
    local base_url
    case "$provider" in
        "$PROVIDER_CHUTES") base_url="$DEFAULT_CHUTES_BACKEND" ;;
        "$PROVIDER_ZAI") base_url="$DEFAULT_ZAI_BACKEND" ;;
        *) log_error "Unknown provider: $provider"; exit 1 ;;
    esac

    log_info "Fetching models for $provider from $base_url"
    local resp
    resp=$(curl -sS -H "Authorization: Bearer $api_key" "$base_url/v1/models")
    echo "$resp" | node -e '
      const data = JSON.parse(require("fs").readFileSync(0, "utf-8"));
      const list = data.data ?? data;
      if (!Array.isArray(list)) process.exit(1);
      list.forEach((m) => { console.log(m.id); });
    ' || { log_error "Failed to parse models"; exit 1; }
}

# Prompt user to select model
select_model() {
    local provider="$1"
    local api_key="$2"

    if [ "$provider" = "$PROVIDER_ZAI" ]; then
        # Alias all models to glm-4.6
        echo "$ZAI_MODEL_ALIAS"
        return
    fi

    # For Chutes, fetch models normally
    local models
    mapfile -t models < <(fetch_models "$provider" "$api_key")
    if [ ${#models[@]} -eq 0 ]; then
        log_error "No models returned; falling back to default deepseek-ai/DeepSeek-R1"
        echo "deepseek-ai/DeepSeek-R1"
        return
    fi

    echo ""
    log_info "Available models for $provider:"
    for i in "${!models[@]}"; do
        printf "  %2d) %s\n" "$((i+1))" "${models[i]}"
    done
    echo ""

    while true; do
        read -p "Select model (1-${#models[@]}) [1]: " sel </dev/tty
        sel=${sel:-1}
        if [[ "$sel" =~ ^[0-9]+$ ]] && [ "$sel" -ge 1 ] && [ "$sel" -le "${#models[@]}" ]; then
            echo "${models[sel-1]}"
            return
        else
            log_error "Invalid selection"
        fi
    done
}

# Write unified settings.json
write_settings() {
    local chutes_key="$1"
    local chutes_model="$2"
    local zai_key="$3"
    local zai_model="$4"
    local vscode_provider="$5"

    ensure_dir_exists "$CONFIG_DIR"

    node -e '
      const fs = require("fs");
      const path = require("path");
      const cfg = {
        providers: {
          chutes: {
            apiKey: "'"$chutes_key"'",
            baseUrl: "'"$DEFAULT_CHUTES_BACKEND"'",
            proxyBaseUrl: "'"$DEFAULT_CHUTES_PROXY"'",
            model: "'"$chutes_model"'"
          },
          zai: {
            apiKey: "'"$zai_key"'",
            baseUrl: "'"$DEFAULT_ZAI_BACKEND"'",
            proxyBaseUrl: "'"$DEFAULT_ZAI_PROXY"'",
            model: "'"$zai_model"'"
          }
        },
        timeoutMs: '"$API_TIMEOUT_MS"',
        vscodeProvider: "'"$vscode_provider"'"
      };
      const file = path.join(process.env.HOME, ".claude", "settings.json");
      fs.writeFileSync(file, JSON.stringify(cfg, null, 2));
    ' || { log_error "Failed write settings.json"; exit 1; }

    log_success "Wrote settings.json to $SETTINGS_FILE"
}

# Set active provider for terminal
set_active_provider() {
    local p="$1"
    echo "$p" > "$ACTIVE_PROVIDER_FILE"
    log_info "Set active provider to '$p'"
}

# Set VS Code provider
set_vscode_provider_file() {
    local p="$1"
    echo "$p" > "$VSCODE_PROVIDER_FILE"
    log_info "Set VSCode provider to '$p'"
}

# Create wrapper
install_wrapper() {
    local wrapper_path="$HOME/.local/bin/claude"
    ensure_dir_exists "$(dirname "$wrapper_path")"

    cat > "$wrapper_path" <<'EOF'
#!/usr/bin/env bash
CONFIG_DIR="$HOME/.claude"
SETTINGS_FILE="$CONFIG_DIR/settings.json"
ACTIVE_FILE="$CONFIG_DIR/active-provider"
VSCODE_FILE="$CONFIG_DIR/vscode-provider"

ARGS=()
provider_from_flag=""
command_mode=""

while (( "$#" )); do
  case "$1" in
    --provider)
      provider_from_flag="$2"; shift 2 ;;
    set-vscode-provider)
      command_mode="set-vscode"; provider_from_flag="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [ -n "$provider_from_flag" ]; then
  provider="$provider_from_flag"
else
  if [ -f "$ACTIVE_FILE" ]; then
    provider=$(cat "$ACTIVE_FILE")
  else
    provider="zai"
  fi
fi

if [ "$command_mode" = "set-vscode" ]; then
  if [[ "$provider" == "chutes" || "$provider" == "zai" ]]; then
    echo "$provider" > "$VSCODE_FILE"
    echo "[ OK ] VSCode provider set to $provider"
    exit 0
  else
    echo "[ERR ] Invalid provider: $provider"; exit 1
  fi
fi

echo "$provider" > "$ACTIVE_FILE"

if [ ! -f "$SETTINGS_FILE" ]; then
  echo "[ERR ] settings.json not found"; exit 1
fi

provider_cfg=$(jq -r ".providers.\"$provider\"" "$SETTINGS_FILE")
if [ -z "$provider_cfg" ]; then
  echo "[ERR ] No config for provider '$provider'"; exit 1
fi

api_key=$(echo "$provider_cfg" | jq -r .apiKey)
proxy_url=$(echo "$provider_cfg" | jq -r .proxyBaseUrl)
timeout_ms=$(jq -r .timeoutMs "$SETTINGS_FILE")
model=$(echo "$provider_cfg" | jq -r .model)

# Model aliasing: all ZAI model slots map to glm-4.6
if [ "$provider" = "zai" ]; then
  model="glm-4.6"
fi

export ANTHROPIC_AUTH_TOKEN="$api_key"
export ANTHROPIC_BASE_URL="$proxy_url"
export API_TIMEOUT_MS="$timeout_ms"
export ANTHROPIC_MODEL="$model"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$model"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$model"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$model"
export CLAUDE_CODE_SUBAGENT_MODEL="$model"
export ANTHROPIC_SMALL_FAST_MODEL="$model"

exec claude-code "${ARGS[@]}"
EOF

    chmod +x "$wrapper_path"
    log_success "Installed wrapper at $wrapper_path"
    echo "Add $HOME/.local/bin to PATH if not already."
}

# ---------------- Main ----------------
main() {
    echo "[START] $SCRIPT_NAME"

    check_nodejs
    install_claude_code

    log_info "Configuring both providers..."
    chutes_key=$(prompt_api_key_for_provider "$PROVIDER_CHUTES")
    zai_key=$(prompt_api_key_for_provider "$PROVIDER_ZAI")

    chutes_model=$(select_model "$PROVIDER_CHUTES" "$chutes_key")
    zai_model=$(select_model "$PROVIDER_ZAI" "$zai_key")

    default_vscode="$PROVIDER_ZAI"

    write_settings "$chutes_key" "$chutes_model" "$zai_key" "$zai_model" "$default_vscode"
    set_active_provider "$default_vscode"
    set_vscode_provider_file "$default_vscode"

    install_wrapper

    log_success "Installation complete!"
    echo "Use wrapper: 'claude --provider chutes|zai' or 'claude set-vscode-provider chutes|zai'."
}

main "$@"
