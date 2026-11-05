#!/usr/bin/env bash
# Trains and runs one paper variant:
#   anchor model (train_anchor) -> residual head (train_residual) -> latent inference -> fused decoding
#   -> prefix-swap inference and decoding.
# Config files are anchor_model_<variant>.yaml and residual_model_<variant>.yaml. Checkpoint file names keep the development names: the anchor model is A4, the residual head is B4.
#   no_cons   anchor model without the mean<->PCA-2D consistency loss 
#   conscurr  anchor model with curriculum-scaled consistency 0.2 (the best nominal result); default
#
# Usage:  GPU=0 scripts/run_experiment.sh [VARIANT] [STAGES...]
#         GPU=0 scripts/run_experiment.sh                                  # conscurr, all stages
#         GPU=0 scripts/run_experiment.sh no_cons                          # no_cons, all stages
#         GPU=0 scripts/run_experiment.sh no_cons train_anchor train_residual   # no_cons, selected stages
#   STAGES (default: prefit train_anchor train_residual infer decode swap swap_decode):
#     prefit           fit (or verify) the shared dedup ridge readout cache and print its sha256
#     train_anchor     train the anchor model (A4)
#     train_residual   train the residual head (B4) on top of the anchor model
#     infer            latent inference on cal and test -> pred_cal_test.npz / .json
#     decode           fused decoding of the test predictions -> pred_test_fused_decoded.fasta / .csv
#     swap             prefix-swap inference on the test series (20 samples per series)
#     swap_decode      fused decoding of the swapped predictions
# The analysis (kNN, evaluation reports, summary table) is in scripts/analyse_experiment.sh.
# Run the two variants sequentially: they share one ridge cache (created by prefit).
set -euo pipefail
PATHS_QUIET=1   # list only the paths used by the selected stages (see below), not every path
source "$(dirname "$0")/common.sh"
VALID_STAGES="prefit train_anchor train_residual infer decode swap swap_decode"
DEFAULT_STAGES="prefit train_anchor train_residual infer decode swap swap_decode"
if [[ $# -gt 0 ]] && is_variant "$1"; then VAR=$1; shift; else VAR=$DEFAULT_VARIANT; fi
[[ $# -gt 0 ]] || set -- $DEFAULT_STAGES
check_stages "$VALID_STAGES" "$@"
STAGES="$*"
has() { [[ " $STAGES " == *" $1 "* ]]; }

# Demo paths read by the selected stages. D and P are always listed.
USED=" "
use() { USED+="$* "; }
has decode && use MSA_DEDUP DECODER
has swap && use META
has swap_decode && use MSA_DEDUP DECODER
show_paths $USED
variant_dirs "$VAR"
mkdir -p "$OUT"
if [[ " $STAGES " == *" decode "* || " $STAGES " == *" swap_decode "* ]] && [[ ! -f $DECODER ]]; then
  echo "decoder missing: $DECODER (run scripts/train_decoder.sh)"; exit 1
fi

if has prefit; then
  echo "=== [prefit] shared dedup ridge cache ==="
  $PY - "$A4_CFG" "$DS" <<'PY'
import sys, torch
from omegaconf import OmegaConf
from recomb2027.config_io import load_config
from recomb2027.latent.models.consistency_loss import resolve_ridge_readout
import logging; logging.basicConfig(level=logging.INFO, format="%(message)s")
cfg = OmegaConf.create(OmegaConf.to_container(load_config(sys.argv[1]), resolve=True))
r = resolve_ridge_readout(cfg, device=torch.device("cpu"), data_cfg=load_config(sys.argv[2]))
print("ridge sha256:", r.sha256, "| source:", r.source)
PY
fi

if has train_anchor; then
  echo "=== [train_anchor] train the anchor model (A4), variant $VAR ==="
  $PY -m recomb2027.latent.predictors.train_anchor \
    --predictor-config "$A4_CFG" --dataset-config "$DS" --log-level INFO
fi

if has train_residual; then
  echo "=== [train_residual] train the residual head (B4) on the anchor model, variant $VAR ==="
  [[ -f $A4_CKPT ]] || { echo "anchor checkpoint missing (run train_anchor first): $A4_CKPT"; exit 1; }
  $PY -m recomb2027.latent.predictors.train_residual \
    --predictor-config "$B4_CFG" --dataset-config "$DS" --mode mse --tag "$TAG"
fi

if has infer; then
  echo "=== [infer] latent inference (cal,test) ==="
  $PY -m recomb2027.latent.predictors.infer_residual \
    --checkpoint "$B4_CKPT" --predictor-config "$B4_CFG" --dataset-config "$DS" \
    --splits cal,test --save-predictions --absolute --unscaled \
    --npz-out "$BASE_NPZ" --json-out "$OUT/pred_cal_test.json"
fi

if has decode; then
  echo "=== [decode] fused decoder ==="
  $PY -m recomb2027.generation.run_decode_from_latents \
    --latents-npz "$BASE_NPZ" \
    --latent-key test_pred_pca2d_flat_raw --mean-latent-key test_pred_mean_raw \
    --decoder-checkpoint "$DECODER" \
    --decode-batch-size 256 --device cuda:0 --esm-repr-layer 33 --cycle-metric both \
    --source-fasta "$MSA_DEDUP" \
    --fasta-out "$BASE_FASTA" \
    --output "${BASE_FASTA%.fasta}.csv"
fi

# pipeline.md section 5 passes the non-deduplicated MSA to the prefix-swap donor search; kept as documented.
SWAP_MSA=$MSA_RAW

if has swap; then
  echo "=== [swap] prefix-swap inference ($VAR) ==="
  $PY -m recomb2027.latent.predictors.infer_residual_prefix_swap \
    --checkpoint "$B4_CKPT" --predictor-config "$B4_CFG" --dataset-config "$DS" \
    --split test --mode prefix_swap --n-samples 20 --donor-pool train,cal,test \
    --metadata-path "$META" --lineage-col lineage --msa-path "$SWAP_MSA" \
    --baseline-npz "$BASE_NPZ" --output-npz "$SWAP_NPZ" --absolute --unscaled
fi

if has swap_decode; then
  echo "=== [swap_decode] fused decoding of swapped predictions ==="
  $PY -m recomb2027.generation.run_decode_from_latents \
    --latents-npz "$SWAP_NPZ" \
    --latent-key test_pred_pca2d_flat_raw --mean-latent-key test_pred_mean_raw \
    --decoder-checkpoint "$DECODER" \
    --decode-batch-size 256 --device cuda:0 --esm-repr-layer 33 --cycle-metric both \
    --source-fasta "$MSA_DEDUP" --fasta-out "$SWAP_FASTA" --output "${SWAP_FASTA%.fasta}.csv"
fi
