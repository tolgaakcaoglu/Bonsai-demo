#!/usr/bin/env bash
set -e
cd "$HOME/AI/Bonsai-demo"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BONSAI_CTX=8192
export BONSAI_MMPROJ_CPU=1
exec ./scripts/start_llama_server.sh --parallel 1
