#!/bin/bash
set -xeuo pipefail

# Get the configuration name and engine name from arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 CONFIG_NAME [ENGINE] [HYDRA_OVERRIDE ...]" >&2
    exit 1
fi
CONFIG_NAME="$1"
shift
ENGINE="${1:-vllm}"
if [ "$#" -gt 0 ]; then
    shift
fi

# Download model if needed
#huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir "$HOME/models/Qwen/Qwen2.5-0.5B"

# Run the training with the specified configuration
python3 -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" \
    actor_rollout_ref.rollout.name="$ENGINE" "$@"
