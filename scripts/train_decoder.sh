#!/usr/bin/env bash
# Section 6 (training): fused PCA-2D + mean non-autoregressive decoder (nar_fused, fused_v1_AB). Shared by all variants.
# Needs the sampled series (splits_raw.pkl) and the pinned PCA-2D basis.
# Usage: GPU=0 scripts/train_decoder.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "$DECODER_DIR"
$PY scripts/train_fused_decoder.py \
  --fasta "$MSA_DEDUP" \
  --pca2d-basis "$PCA_BASIS" \
  --enc-2d-dir "$ENC/filtered_msa_S1/2d/33" \
  --splits-pkl "$SERIES/splits_raw.pkl" --splits-train-key train --splits-test-key val \
  --epochs 20 --batch-size 32 \
  --layers 3 --d-model 256 --n-heads 4 --ff-dim 1024 --n-global 4 \
  --noise-max 0.5 --mean-noise-max 0.75 --modality-dropout 0.1 \
  --device cuda:0 --run-dir "$DECODER_DIR"
