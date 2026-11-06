# Shared settings for the RECOMB 2027 pipeline scripts. Sourced by the other scripts; not run directly.
#
# Environment overrides:
#   PY=python3        interpreter that has the packages from requirements.txt
#   GPU=0             physical GPU index (becomes CUDA_VISIBLE_DEVICES)
#   D=/path           cluster data root (data_root in configs/common/paths.yaml). Asked for in a terminal, with the
#                     value in the file offered; a new value is written back to that file.
#   CONFIG_ROOT=/path configs root (the root in configs/common/paths.yaml). Asked for in a terminal, with the
#                     value in the file offered; a new value is written back to that file.
#   META, MSA_DEDUP, TREE_DEDUP, PCA_BASIS
#                     demo files from configs/common/paths.yaml. Set in the environment to use other files.
# The configs read their own paths from configs/common/paths.yaml; see README.md.

set -euo pipefail
RC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RC_ROOT"
export PYTHONPATH="$RC_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}   # run_predecessors draws parents with Python hash-seeded RNG streams
export CUDA_VISIBLE_DEVICES=${GPU:-0}
PY=${PY:-python3}

# Cluster data root (data_root in configs/common/paths.yaml): the cluster-only files (encodings, latent predictions,
# raw inputs) live under it. The value in the file is offered in a terminal (Enter keeps it). D from the environment
# is used as is. A new value is written back to that file, and the encoding paths in the configs follow it.
DEFAULT_D=$("$PY" "$RC_ROOT/scripts/print_config_value.py" "$RC_ROOT/configs/common/paths.yaml" data_root) \
  || { echo "could not read data_root from configs/common/paths.yaml (does $PY have omegaconf?)" >&2; exit 2; }

if [[ -z ${D:-} ]]; then
  D=$DEFAULT_D
  if [[ -t 0 ]]; then
    read -r -p "Input data root [$DEFAULT_D] (press Enter to keep): " D_ANSWER
    D=${D_ANSWER:-$DEFAULT_D}
  fi
fi
D=${D%/}                                   # drop a trailing slash
D=${D/#\~/$HOME}
if [[ $D != "$DEFAULT_D" ]]; then
  "$PY" "$RC_ROOT/scripts/set_common_path.py" data_root "$D" || exit 2
fi
# Configs root: the `root` in configs/common/paths.yaml, where the predictor outputs go. Asked for in a terminal
# with the value in the file offered (Enter keeps it). A new value is written back to that file.
# CONFIG_ROOT from the environment is used as is and written to the file. Otherwise the file is left alone.
CONFIG_ROOT_NOW=$("$PY" "$RC_ROOT/scripts/print_config_value.py" "$RC_ROOT/configs/common/paths.yaml" root) \
  || { echo "could not read root from configs/common/paths.yaml (does $PY have omegaconf?)" >&2; exit 2; }
if [[ -n ${CONFIG_ROOT:-} ]]; then
  NEW_CONFIG_ROOT=$CONFIG_ROOT
elif [[ -t 0 ]]; then
  read -r -p "Experiments root (Base prefix (root) added to all config paths) [$CONFIG_ROOT_NOW] (press Enter to keep): " R_ANSWER
  NEW_CONFIG_ROOT=${R_ANSWER:-$CONFIG_ROOT_NOW}
else
  NEW_CONFIG_ROOT=$CONFIG_ROOT_NOW
fi
NEW_CONFIG_ROOT=${NEW_CONFIG_ROOT/#\~/$HOME}
if [[ $NEW_CONFIG_ROOT != "$CONFIG_ROOT_NOW" ]]; then
  "$PY" "$RC_ROOT/scripts/set_common_path.py" root "$NEW_CONFIG_ROOT" || exit 2
fi

export D
ENC=$D/encodings/ESM1B
MSA_RAW=$D/phylotree_matched/filtered_msa_S1.fasta
TREE_RAW=$D/phylotree_matched/pruned_tree.nwk

# META, MSA_DEDUP, TREE_DEDUP and PCA_BASIS: demo files from configs/common/paths.yaml (see the helper for the format).
PATHS_SH=$("$PY" "$RC_ROOT/scripts/print_common_paths.py") \
  || { echo "could not read configs/common/paths.yaml (does $PY have omegaconf?)" >&2; exit 2; }
eval "$PATHS_SH"
# Predictor outputs (checkpoints, logs, predictions, decoder) live under the root the configs use.
P=$CONFIG_DATA_ROOT
show_paths() {  # $@ = paths to list, any of: MSA_DEDUP META TREE_DEDUP PCA_BASIS DECODER. D and P are always listed.
  local v label
  printf '%s\n' "paths in use (set the variable in the environment, or edit scripts/common.sh, to change one):" >&2
  for v in D P MSA_DEDUP META TREE_DEDUP PCA_BASIS DECODER; do
    case $v in
      D|P) ;;
      *) [[ " $* " == *" $v "* ]] || continue ;;
    esac
    case $v in
      D) label="D (cluster data root)" ;;
      P) label="P (predictor root)" ;;
      DECODER) label="DECODER (trained)" ;;
      *) label="$v (demo)" ;;
    esac
    printf '  %-24s %s\n' "$label" "${!v}" >&2
  done
}
# Scripts list every path by default. A script that lists only the paths it uses sets PATHS_QUIET=1 before sourcing this file.
if [[ -z ${PATHS_QUIET:-} ]]; then
  show_paths MSA_DEDUP META TREE_DEDUP PCA_BASIS
fi
SERIES=$P/series/by_nearest_ancestor_sampled_2022-12-01_k5_esm1_layer33
DECODER_DIR=demo_dataset/sars_cov2_spike/pretrained/decoder                 # written by scripts/train_decoder.sh
DECODER=$RC_ROOT/$DECODER_DIR/best.pt
DS=configs/latent/datasets/by_nearest_ancestor/split_2022-12-01_k5_esm1_layer33.yaml
CFG=configs/latent/predictors
PAPER_VARIANTS="no_cons conscurr"
DEFAULT_VARIANT=conscurr      # used when run_experiment.sh / analyse_experiment.sh get no variant argument

is_variant() {
  case "$1" in
    no_cons|conscurr) return 0 ;;
    *) return 1 ;;
  esac
}
check_variant() {
  is_variant "$1" || { echo "unknown variant '$1' (paper variants: $PAPER_VARIANTS)" >&2; exit 2; }
}
check_stages() {  # $1 = space-separated valid stages, remaining arguments = requested stages
  local valid=$1 s
  shift
  for s in "$@"; do
    if [[ " $valid " != *" $s "* ]]; then
      echo "unknown stage '$s' (valid stages: $valid)" >&2
      case $s in
        a4)    echo "note: the stage 'a4' is now 'train_anchor'" >&2 ;;
        b4)    echo "note: the stage 'b4' is now 'train_residual'" >&2 ;;
      esac
      exit 2
    fi
  done
}
variant_dirs() {  # $1 = variant -> sets A4_CFG B4_CFG A4_CKPT B4_CKPT TAG OUT and the prediction paths below
  A4_CFG=$CFG/anchor_model_$1.yaml
  B4_CFG=$CFG/residual_model_$1.yaml
  TAG=a4b4_$1
  # Checkpoint locations come from the configs, so they follow any change made there.
  # A4: the anchor file the residual config loads (site_gpt_checkpoint).
  # B4: the residual config's checkpoint_dir, where train_residual.py saves.
  A4_CKPT=$("$PY" "$RC_ROOT/scripts/print_config_value.py" "$RC_ROOT/$B4_CFG" site_gpt_checkpoint)
  B4_CKPT=$("$PY" "$RC_ROOT/scripts/print_config_value.py" "$RC_ROOT/$B4_CFG" checkpoint_dir)/Pca2dSiteUnifiedV4_best_model.pt
  OUT=$P/preds/$TAG
  BASE_NPZ=$OUT/pred_cal_test.npz
  BASE_FASTA=$OUT/pred_test_fused_decoded.fasta
  BASE_JSON_BEFORE=$OUT/kNN_analysis_before_date
  BASE_JSON_NODATE=$OUT/kNN_analysis_no_date_filter
  SWAP_NPZ=$OUT/pred_cal_test_lineage_subst_prefix_swap.npz
  SWAP_FASTA=$OUT/pred_test_lineage_subst_prefix_swap_fused_decoded.fasta
  SWAP_KNN=$OUT/kNN_prefix_swap_analysis
}
