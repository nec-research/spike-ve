#!/usr/bin/env bash
# Section 1-2 of the pipeline: MSA dedup, ESM-1 UR100 encodings, predecessors, series, PCA basis,
# substitution sidecar and validation pairs.
#
# Usage: scripts/build_inputs.sh [STAGES...]
#   STAGES (default: msa encode predecessors series pca sidecar pairs), run in this order:
#     msa          deduplicate the S1 MSA and prune the tree to the same genomes
#     encode       ESM-1 UR100 layer-33 token encodings (2D and average-token) for the dedup MSA
#     predecessors nearest-ancestor predecessor links (sampled, tau = 4)
#     series       sampled_tau4 train/cal/test series (prepare_dataset)
#     pca          PCA-2D basis. Uses the pinned basis by default; PCA_REFIT=1 refits it from the
#                  sampled train series into $ENC/pca2d/pca2d_k4_layer33_S1_sampled_train_refit.pt
#     sidecar      substitution sidecar (must run before any training or inference)
#     pairs        validation_pairs_test.csv used by the analysis
set -euo pipefail
PATHS_QUIET=1   # list only the paths used by the selected stages (see below), not every path
source "$(dirname "$0")/common.sh"
STAGES=${*:-msa encode predecessors series pca sidecar pairs}
has() { [[ " $STAGES " == *" $1 "* ]]; }

# Demo paths read or written by the selected stages. D and P are always listed.
USED=" "
use() { USED+="$* "; }
has msa && use META MSA_DEDUP TREE_DEDUP
has encode && use MSA_DEDUP
has pca && use PCA_BASIS
has pca && [[ ${PCA_REFIT:-0} == 1 ]] && use MSA_DEDUP
has pairs && use MSA_DEDUP
show_paths $USED

# msa, encode and (with PCA_REFIT=1) pca write cluster outputs from $MSA_DEDUP. The default MSA is the demo file,
# so stop before those outputs are built from it or the demo input is overwritten.
if has msa || has encode || { has pca && [[ ${PCA_REFIT:-0} == 1 ]]; }; then
  if [[ $MSA_DEDUP == "$RC_ROOT"/demo_dataset/* ]]; then
    echo "MSA_DEDUP is the demo file ($MSA_DEDUP), but these stages write cluster outputs." >&2
    echo "Rerun with the cluster MSA, for example:" >&2
    echo "  MSA_DEDUP=$D/phylotree_matched/filtered_msa_S1_dedup.fasta scripts/build_inputs.sh $STAGES" >&2
    exit 2
  fi
fi

if has msa; then
  echo "=== [msa] deduplicate MSA and prune tree ==="
  $PY -m recomb2027.data.deduplicate_msa \
    --input-fasta "$MSA_RAW" --output-fasta "$MSA_DEDUP" \
    --metadata-csv "$META" --dedup-field collection_date --dedup-criterion min \
    --tree "$TREE_RAW" --output-tree "$TREE_DEDUP"
fi

if has encode; then
  echo "=== [encode] ESM-1 layer 33 encodings ==="
  # --gpu indexes physical devices here, so CUDA_VISIBLE_DEVICES (set by common.sh) must not hide them
  env -u CUDA_VISIBLE_DEVICES $PY -m recomb2027.data.encode_proteins_from_fasta \
    --input "$MSA_DEDUP" --output-dir "$D/encodings/ESM1B/filtered_msa_S1" \
    --model ESM1B --layers 33 --aggregate-pooling average_token --token-budget 712384 \
    --save-2d --skip-existing --gpu 0 1
  $PY -m recomb2027.data.aggregate_2d_encodings \
    --input-dir "$ENC/filtered_msa_S1/2d/33" --layers 33 \
    --output-dir "$ENC/filtered_msa_S1" --aggregate-pooling average_token --include-gap-columns
fi

if has predecessors; then
  echo "=== [predecessors] nearest-ancestor links (PYTHONHASHSEED=$PYTHONHASHSEED) ==="
  $PY -m recomb2027.latent.predecessors.run_predecessors \
    --cfg configs/latent/predecessors/predecessors_by_nearest_ancestor_sampled.yaml
fi

if has series; then
  echo "=== [series] prepare_dataset (sampled_tau4, dedup, cal split) ==="
  $PY -m recomb2027.latent.datasets.prepare_dataset --cfg "$DS"
fi

if has pca; then
  if [[ ${PCA_REFIT:-0} == 1 ]]; then
    OUT_PCA=$ENC/pca2d/pca2d_k4_layer33_S1_sampled_train_refit.pt
    echo "=== [pca] refit PCA-2D basis from the sampled train series -> $OUT_PCA ==="
    $PY -m recomb2027.latent.datasets.fit_pca2d_from_2d_encodings \
      --enc-2d-dir "$ENC/filtered_msa_S1/2d/33" --train-dataset "$SERIES/train_dataset.npz" \
      --output "$OUT_PCA" --msa "$MSA_DEDUP" --k 4 --layer 33 --max-gids 10000
    echo "Compare with the pinned basis: sha256sum $PCA_BASIS $OUT_PCA"
  else
    echo "=== [pca] using pinned basis $PCA_BASIS (set PCA_REFIT=1 to refit) ==="
  fi
fi

if has sidecar; then
  echo "=== [sidecar] MSA substitution sidecar ==="
  $PY -m recomb2027.latent.predictors.build_msa_substitution_sidecar \
    --config configs/latent/diagnostics/substitution_sidecar.yaml --splits train,cal,test
fi

if has pairs; then
  echo "=== [pairs] validation pairs for the analysis ==="
  $PY -m recomb2027.generation.build_validation_pairs_from_series \
    --npz "$SERIES/test_dataset.npz" --msa "$MSA_DEDUP" --dataset-config "$DS" \
    --out "$SERIES/validation_pairs_test.csv"
fi
echo "Done: $STAGES"
