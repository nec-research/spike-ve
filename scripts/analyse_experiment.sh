#!/usr/bin/env bash
# Analysis for one paper variant (kNN, evaluation reports, prefix-swap report) and the cross-variant summary table.
# Needs the outputs of run_experiment.sh for that variant (including swap and swap_decode).
#
# Usage:  GPU=0 scripts/analyse_experiment.sh [VARIANT] [STAGES...]
#         GPU=0 scripts/analyse_experiment.sh                      # conscurr, default stages
#         GPU=0 scripts/analyse_experiment.sh no_cons              # no_cons, default stages
#         GPU=0 scripts/analyse_experiment.sh no_cons knn report   # selected stages
#   STAGES (default: knn knn_swap report swap_report summarize); run after run_experiment.sh
#     knn          kNN analysis of the baseline predictions (before-date and no-date-filter index)
#     knn_swap     kNN analysis of the prefix-swapped predictions
#     report       evaluation report + plots for the baseline predictions
#     swap_report  prefix-swap evaluation report + plot
#     summarize    ablation table over the paper variants that have decoded predictions
#                  -> preds/a4b4_paper_summary.csv (independent of VARIANT; run after both variants are decoded)
set -euo pipefail
PATHS_QUIET=1   # list only the paths used by the selected stages (see below), not every path
source "$(dirname "$0")/common.sh"
VALID_STAGES="knn knn_swap report swap_report summarize"
DEFAULT_STAGES="knn knn_swap report swap_report summarize"
if [[ $# -gt 0 ]] && is_variant "$1"; then VAR=$1; shift; else VAR=$DEFAULT_VARIANT; fi
[[ $# -gt 0 ]] || set -- $DEFAULT_STAGES
check_stages "$VALID_STAGES" "$@"
STAGES="$*"
has() { [[ " $STAGES " == *" $1 "* ]]; }

# Demo paths read by the selected stages. D and P are always listed.
USED=" "
use() { USED+="$* "; }
has knn && use PCA_BASIS MSA_DEDUP META
has knn_swap && use PCA_BASIS MSA_DEDUP META
has swap_report && use MSA_DEDUP META
show_paths $USED
variant_dirs "$VAR"
ENC_ARGS=(--encodings-dir "$ENC/filtered_msa_S1/average_token/33" --enc-2d-dir "$ENC/filtered_msa_S1/2d/33" \
          --pca2d-basis-path "$PCA_BASIS")

if has knn; then
  echo "=== [knn] kNN baseline, index before 2022-12-01 ==="
  mkdir -p "$BASE_JSON_BEFORE" "$BASE_JSON_NODATE"
  $PY -m recomb2027.pipelines.analyse_baseline_knn \
    --pairs-csv "$SERIES/validation_pairs_test.csv" --all-rows --num-workers 6 \
    --pred-npz "$BASE_NPZ" --pred-split test "${ENC_ARGS[@]}" \
    --msa "$MSA_DEDUP" --metadata-csv "$META" \
    --knn-space combined --knn-k 20 --knn-before-date 2022-12-01 --max-index-files 0 \
    --out-mutation-stats "$BASE_JSON_BEFORE/knn_baseline_mutations_row{row}.json" \
    --out-json "$BASE_JSON_BEFORE/knn_baseline_row{row}.json" \
    --out-fasta "$BASE_JSON_BEFORE/knn_baseline_row{row}.fasta"
  echo "=== [knn] kNN baseline, index without date filter ==="
  $PY -m recomb2027.pipelines.analyse_baseline_knn \
    --pairs-csv "$SERIES/validation_pairs_test.csv" --all-rows --num-workers 2 \
    --pred-npz "$BASE_NPZ" --pred-split test "${ENC_ARGS[@]}" \
    --msa "$MSA_DEDUP" --metadata-csv "$META" \
    --knn-space combined --knn-k 20 --no-knn-date-filter --max-index-files 0 \
    --out-mutation-stats "$BASE_JSON_NODATE/knn_baseline_mutations_row{row}.json" \
    --out-json "$BASE_JSON_NODATE/knn_baseline_row{row}.json" \
    --out-fasta "$BASE_JSON_NODATE/knn_baseline_row{row}.fasta"
fi

if has knn_swap; then
  echo "=== [knn_swap] kNN analysis of prefix-swapped predictions ==="
  mkdir -p "$SWAP_KNN"
  $PY -m recomb2027.pipelines.analyse_lineage_subst_knn_batch --all-rows --num-workers 14 \
    --pairs-csv "$SERIES/validation_pairs_test.csv" --pred-npz "$SWAP_NPZ" --pred-split test \
    "${ENC_ARGS[@]}" --msa "$MSA_DEDUP" --metadata-csv "$META" \
    --knn-space combined --knn-k 20 --knn-before-date 2022-12-01 --max-index-files 0 \
    --out-dir "$SWAP_KNN/" --continue-on-error
fi

if has report; then
  echo "=== [report] evaluation report and plots (baseline predictions) ==="
  REPORT_DIR=$OUT
  $PY -m recomb2027.pipelines.generate_evaluation_report \
    --val_pairs "$SERIES/validation_pairs_test.csv" \
    --fasta "$BASE_FASTA" \
    --json_dir "$BASE_JSON_BEFORE" --output "$REPORT_DIR/evaluation_report_fused.csv"
  $PY -m recomb2027.pipelines.plot_evaluation_report \
    --report-csv "$REPORT_DIR/evaluation_report_fused.csv" --out-dir "$REPORT_DIR/evaluation_plots_fused" \
    --fasta "$BASE_FASTA" --json-dir "$BASE_JSON_BEFORE" \
    --examples-per-bucket 3 --max-positions 60 --top-k-positions 40
  $PY -m recomb2027.pipelines.plot_evaluation_report \
    --report-csv "$REPORT_DIR/evaluation_report_fused.csv" --out-dir "$REPORT_DIR/evaluation_plots_fused_unique" \
    --fasta "$BASE_FASTA" --json-dir "$BASE_JSON_BEFORE" \
    --examples-per-bucket 3 --max-positions 60 --top-k-positions 40 --unique-continuations-only
fi

if has swap_report; then
  echo "=== [swap_report] prefix-swap evaluation report and plot ==="
  $PY -m recomb2027.pipelines.generate_prefix_swap_evaluation_report \
    --val-pairs "$SERIES/validation_pairs_test.csv" --decoded-fasta "$SWAP_FASTA" \
    --json-dir "$SWAP_KNN/" --series-npz "$SWAP_NPZ" \
    --msa "$MSA_DEDUP" --metadata-csv "$META" --output-dir "$OUT/evaluation_prefix_swap_fused"
  $PY -m recomb2027.pipelines.plot_prefix_swap_evaluation_report \
    --eval-dir "$OUT/evaluation_prefix_swap_fused" --decoded-fasta "$SWAP_FASTA"
fi

if has summarize; then
  echo "=== [summarize] paper variants ==="
  RUNS=()
  for v in $PAPER_VARIANTS; do
    d=$P/preds/a4b4_${v}
    if [[ -f $d/pred_test_fused_decoded.fasta ]]; then RUNS+=(--run "$v=$d"); fi
  done
  [[ ${#RUNS[@]} -gt 0 ]] || { echo "no decoded paper variant yet: run scripts/run_experiment.sh first"; exit 1; }
  $PY scripts/summarize_experiment.py --pairs "$SERIES/validation_pairs_test.csv" "${RUNS[@]}" \
    --out-csv "$P/preds/a4b4_paper_summary.csv"
fi
