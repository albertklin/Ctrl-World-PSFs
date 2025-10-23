#!/usr/bin/env bash
set -euo pipefail

# Ctrl-World checkpoint downloader
# - Downloads:
#   1) openai/clip-vit-base-patch32 (CLIP text+image encoders)
#   2) stabilityai/stable-video-diffusion-img2vid (SVD)
#   3) yjguo/Ctrl-World (Ctrl-World .pt checkpoint)
# - Places them under a target folder (default: libraries/Ctrl-World/checkpoints)
# - Creates a latest.ckpt symlink to the newest Ctrl-World checkpoint
#
# Usage:
#   bash libraries/Ctrl-World/scripts/download_checkpoints.sh [TARGET_DIR]
#
# Notes:
# - You need to be logged into Hugging Face for private/terms-gated models:
#     huggingface-cli login
#   And accept SVD model terms here (required):
#     https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
# - You can also use the short alias `hf` instead of `huggingface-cli` if available.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/../../.. && pwd)"
DEFAULT_TARGET_DIR="$REPO_ROOT/libraries/Ctrl-World/checkpoints"
TARGET_DIR="${1:-$DEFAULT_TARGET_DIR}"

echo "[info] Target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

if ! command -v huggingface-cli >/dev/null 2>&1 && ! command -v hf >/dev/null 2>&1; then
  echo "[error] huggingface-cli (or hf) not found. Activate your env and install it: pip install huggingface_hub"
  exit 1
fi

HF_CMD="$(command -v huggingface-cli || command -v hf)"

download_model() {
  local repo_id="$1";
  local out_dir="$2";
  echo "[info] Downloading $repo_id -> $out_dir"
  mkdir -p "$out_dir"
  "$HF_CMD" download "$repo_id" \
    --repo-type model \
    --local-dir "$out_dir" \
    --local-dir-use-symlinks False \
    || { echo "[warn] Download may have failed (terms not accepted or network issue): $repo_id"; return 1; }
}

# 1) CLIP
download_model "openai/clip-vit-base-patch32" "$TARGET_DIR/clip-vit-base-patch32" || true

# 2) SVD (requires accepting terms)
echo "[note] If this fails, accept terms at: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid"
download_model "stabilityai/stable-video-diffusion-img2vid" "$TARGET_DIR/stable-video-diffusion-img2vid" || true

# 3) Ctrl-World (PyTorch .pt checkpoint inside the repo)
download_model "yjguo/Ctrl-World" "$TARGET_DIR/Ctrl-World" || true

# Create latest.ckpt symlink to the numerically newest checkpoint-*.pt if present
CTRL_DIR="$TARGET_DIR/Ctrl-World"
if compgen -G "$CTRL_DIR/checkpoint-*.pt" > /dev/null; then
  latest_ckpt="$(ls -1 "$CTRL_DIR"/checkpoint-*.pt | sort -V | tail -n1)"
  ln -sfn "$(basename "$latest_ckpt")" "$CTRL_DIR/latest.ckpt"
  echo "[info] Created symlink: $CTRL_DIR/latest.ckpt -> $(basename "$latest_ckpt")"
  echo "[info] Ctrl-World ckpt path: $CTRL_DIR/latest.ckpt"
else
  echo "[warn] No checkpoint-*.pt found under $CTRL_DIR yet. Inspect contents and set ckpt_path manually."
fi

echo "\n[done] Downloads attempted. Contents:"
du -h --max-depth=1 "$TARGET_DIR" || true

cat <<EOF

Next steps:
- Use these paths with Ctrl-World scripts:
  --svd_model_path   "$TARGET_DIR/stable-video-diffusion-img2vid"
  --clip_model_path  "$TARGET_DIR/clip-vit-base-patch32"
  --ckpt_path        "$TARGET_DIR/Ctrl-World/latest.ckpt"   # or a specific checkpoint-*.pt

Example (replay demo):
  CUDA_VISIBLE_DEVICES=0 \
  python libraries/Ctrl-World/scripts/rollout_replay_traj.py \
    --task_type replay \
    --svd_model_path "$TARGET_DIR/stable-video-diffusion-img2vid" \
    --clip_model_path "$TARGET_DIR/clip-vit-base-patch32" \
    --ckpt_path "$TARGET_DIR/Ctrl-World/latest.ckpt"

EOF
