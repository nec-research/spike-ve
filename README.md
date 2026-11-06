# SpikeVE: Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space

This will be an official repository of "Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space" (SpikeVE) to support paper submission.

This folder contains everything needed to reproduce the paper's main results from raw SARS-CoV-2 Spike S1 data, with
the dataset (`split_2022-12-01_k5_esm1_layer33.yaml`,
1,422 test series). It is self-contained: the Python code under `src/recomb2027/` is an isolated copy
of the modules these steps use, and nothing here imports the development repository.

The two model configuration are:

| variant | anchor model (A4) | what it is |
|---|---|---|
| `conscurr` | `anchor_model_conscurr.yaml` | (default variant) A4 with curriculum-scaled mean↔PCA-2D consistency 0.2. Best nominal result, used in paper |
| `no_cons` | `anchor_model_no_cons.yaml` | A4 without the consistency loss. Control. |

Both use the residual head (B4) configs `residual_model_<variant>.yaml`. 


## Folder layout

```
recomb2027/
  README.md
  requirements.txt                    external Python packages (pinned)
  analysis/                           scripts used to generate paper figures
    vf_benchmarking/                  jupyter notebooks used to benchmark against ViralForsight method
  configs/common/paths.yaml           shared data paths (root, MSA, metadata, tree, encodings, PCA basis), read via `_base_`
  configs/latent/                     the YAML configs the paper runs use
    datasets/by_nearest_ancestor/     sampled τ4 dedup dataset with cal split
    predecessors/                     predecessor config for the sampled τ4 dataset (new, see "Known differences")
    diagnostics/                      B1 substitution sidecar config
    models/                           GPT anchor architecture (read by the A4 trainer)
    predictors/                       A4 and B4 configs for no_cons and conscurr
  src/recomb2027/                     code copied from the development repository and renamed (115 modules)
    latent/predictors/                train_anchor.py (A4), train_residual.py (B4), infer_residual.py,
                                      infer_residual_prefix_swap.py, predictor_utils.py (shared helpers),
                                      build_msa_substitution_sidecar.py, visualize_infer_umap.py
    latent/models/                    anchor_model.py (A4 model), residual_model.py (B4 model), consistency_loss.py
                                      (ridge readout + mean<->PCA-2D loss), model_utils.py (factory, losses, EMA)
  scripts/
    common.sh                         shared paths and helpers (sourced by the other scripts)
    build_inputs.sh                   inputs: MSA dedup, ESM-1b encodings, predecessors, series, PCA, sidecar, pairs
    train_decoder.sh                  fused PCA-2D + mean decoder (shared by both variants)
    run_experiment.sh                 A4 -> B4 -> inference -> decoding -> prefix-swap inference and decoding, per variant
    analyse_experiment.sh             kNN, evaluation reports, prefix-swap report, per variant,
                                      and the summary table over both variants
    train_fused_decoder.py       decoder training (called by train_decoder.sh)
    summarize_experiment.py           summary table over the paper variants (called by analyse_experiment.sh)
    check_isolation.py                text and import check that the folder is self-contained
```

## 1. Environment

* Python 3.9 (the paper was run with 3.9.19) and a CUDA GPU with at least 11 GB of memory.
* `pip install -r requirements.txt`. Install `torch==2.6.0` from the PyTorch index that matches your
  driver first if the default wheel does not suit it.
* ESM-1 670M UR100 weights (`fair-esm`) are downloaded on first use unless they are already in the torch hub cache.
* `ete3` is optional. Without it, MSA deduplication uses dendropy, which is correct but slower on ~160k leaves.
* Check the folder: `python scripts/check_isolation.py` should report 0 references and 0 import failures.

All scripts run from this folder. They put `src/` on `PYTHONPATH` themselves and set
`PYTHONHASHSEED=0`, which the predecessor sampler needs.

**Change the default root data paths in configs and bash scripts.**
The shared data paths (data root, MSA, metadata, tree, encodings and PCA basis) are in `configs/common/paths.yaml`.
Each config takes them with a `_base_:` line, and every config is read with `recomb2027.config_io.load_config`.
`scripts/common.sh` asks for the data root (`D`) in a terminal, or uses the default when `D` is set in the environment or the session is not interactive. The demo files (`META`, `MSA_DEDUP`, `TREE_DEDUP`, `PCA_BASIS`) come from `configs/common/paths.yaml`; set any of them in the environment or edit `scripts/common.sh` to use other files. `build_inputs.sh` stops before its `msa`, `encode` and `PCA_REFIT=1` steps if `MSA_DEDUP` is still the demo file.

**Uncompress the demo MSA first.** To support the paper, the repository ships the demo alignment (sequence data and metadata retrieved from BV-BRC, https://www.bv-brc.org/ and processed to create MSA) only in compressed form
(`demo_dataset/sars_cov2_spike/msa_s1.fasta.gz`, about 2 MB). The pipeline reads the uncompressed
`demo_dataset/sars_cov2_spike/msa_s1.fasta` (`msa_path` in `configs/common/paths.yaml`), so run this once
before any script, from this folder:

```bash
gunzip -c demo_dataset/sars_cov2_spike/msa_s1.fasta.gz > demo_dataset/sars_cov2_spike/msa_s1.fasta
```

The uncompressed file is about 71 MB.


## 2. Run order: series creation to analysis

Logs go to stdout; redirect them to files if you want to keep them.

### Step 1: inputs and series (`build_inputs.sh`)

```bash
# steps in order
GPU=0 scripts/build_inputs.sh encode predecessors
GPU=0 scripts/build_inputs.sh series pca sidecar pairs
```

What each stage does:

| stage | output | notes |
|---|---|---|
| `msa` | `filtered_msa_S1_dedup.fasta` (86,086 records, provided in demo_data), `pruned_tree_S1_dedup.nwk` (provided in demo_data) | removes duplicate sequences keeping the earliest collection date |
| `encode` | ESM-1 layer 33 2D and average-token encodings under `encodings/ESM1B/filtered_msa_S1/` | slowest step; uses GPUs 0 and 1 by default |
| `predecessors` | `predecessors_dict_bv-brc-matched/pred_by_nearest_ancestor_sampled.json` (provided in demo_data) | stochastic, seeded; see "Known differences" |
| `series` | `series/by_nearest_ancestor_sampled_2022-12-01_k5_esm1_layer33/` under the `recomb2027` root (train/val/cal/test NPZ, `splits_raw.pkl`) | split dates come from `cal_threshold` (2022-09-01) and `split_threshold` (2022-12-01) in the dataset config |
| `pca` | pinned PCA-2D basis `pca2d_k4_layer33_S1_train.pt` (provided in demo_data) | uses the pinned file by default; `PCA_REFIT=1` writes a refit next to it, see "Known differences" |
| `sidecar` | `msa_substitution_sidecar_{train,cal,val,test}.npz` | must be rebuilt whenever the MSA or the series change |
| `pairs` | `validation_pairs_test.csv` (1,422 rows) | used by the analysis and the summary table |


**Note:** 
1. For the first run you need to froduce ESM embeddings from FASTA sequences (`demo_dataset/sars_cov2_spike/msa_s1.fasta`). The produced ESM embeddings will be reused in further steps.
2. For some steps you can re-use the output provided in the `demo_dataset/sars_cov2_spike/pretrained/` valid for the dataset config `split_2022-12-01_k5_esm1_layer33.yaml`. This includes: `predecessors/pred_by_nearest_ancestor_sampled.json` -- ancestor relations, generated for the sequences in MSA, `pca2d/pca2d_k4_layer33_S1_train.pt` -- weights for PCA transformation from the ESM-1 layer 33 2D embeddings of the MSA (fitted on train split only).


### Step 2: decoder (`train_decoder.sh`)

```bash
GPU=0 scripts/train_decoder.sh
```

Trains a decoder from latent ESM space to sequences, 20 epochs, PCA-2D + mean input). It is
shared by both variants and must exist before the decode stage.  
You can you the provided weights instead of training from scratch: `demo_dataset/sars_cov2_spike/pretrained/decoder/best.pt` (trained for ESM-1 670M UR100 layer33 embeddings on train split of the provided dataset).


### Step 3: anchor model → residual head for one variant (`run_experiment.sh`)

`train_anchor` trains the anchor model (A4), and `train_residual` trains the residual head (B4) on top of it.
The config files are `anchor_model_<variant>.yaml` (A4) and `residual_model_<variant>.yaml` (B4). The A4 checkpoint directory is `anchor_model_<variant>`. The B4 checkpoint keeps its development name.

```bash
GPU=0 scripts/run_experiment.sh conscurr      # the variant argument is optional; default conscurr
GPU=0 scripts/run_experiment.sh no_cons
GPU=0 scripts/run_experiment.sh no_cons train_anchor train_residual   # optional stage list, see below
```

Stages, in order (run them one variant at a time, since they share the ridge cache). Without a stage list
all of them run. 

1. `prefit`: fits or verifies the shared ridge readout and prints its sha256.
2. `train_anchor`: trains the anchor model (A4). 
3. `train_residual`: trains the residual head (B4) on that anchor model (`--mode mse`).
4. `infer`: latent inference on cal and test. Writes `pred_cal_test.npz` and `.json`.
5. `decode`: fused decoding of the test predictions with cycle-consistency re-embedding.
   Writes `pred_test_fused_decoded.fasta` and `.csv` (`BASE_FASTA` in `scripts/common.sh`).
6. `swap`: prefix-swap inference: 20 donor-swapped samples per test series, one selected per series.
   Writes `pred_cal_test_lineage_subst_prefix_swap.npz`.
7. `swap_decode`: fused decoding of the swapped predictions. Writes `..._prefix_swap_fused_decoded.fasta` and `.csv`.

Outputs live in `$P/preds/a4b4_<variant>/`,
where `$P` is the configs root, `.../recomb2027` (the `root` in `configs/common/paths.yaml`). Checkpoints go to `$P/models/checkpoints/layer33/`.


### Step 4: analysis for one variant (`analyse_experiment.sh`)

```bash
GPU=0 scripts/analyse_experiment.sh conscurr      # default stages, includes the summary table
GPU=0 scripts/analyse_experiment.sh no_cons
```

Stages, in order (default: all). They need the Step 3 outputs for the variant, including `swap` and `swap_decode`.

| stage | what it does | outputs (in the variant directory) |
|---|---|---|
| `knn` | kNN baseline on the original predictions, index (1) before 2022-12-01 and (2) without the date filter | `kNN_analysis_before_date/`, `kNN_analysis_no_date_filter/` |
| `knn_swap` | kNN analysis of the swapped predictions | `kNN_prefix_swap_analysis/` |
| `report` | evaluation report and two sets of plots for the baseline predictions | `evaluation_report_fused.csv`, `evaluation_plots_fused*/` |
| `swap_report` | prefix-swap evaluation report and plot | `evaluation_prefix_swap_fused/` |
| `summarize` | summary table over the paper variants that have decoded predictions. Not specific to the variant argument; run it after both variants have been decoded | `$P/preds/a4_b4_paper_summary.csv` |

The kNN stages are the slowest. Use `--num-workers` (set in the script) and a machine with enough RAM for
the encoding indexes.

## 3. Expected results

From the development run, test set, 1,422 series:

| variant | cal skill (full) | test skill (full) | test skill (mean) | gain over copy-source [95% CI] | precision | recall | F1 | unique decoded |
|---|---|---|---|---|---|---|---|---|
| `conscurr` | 0.3383 | **0.1815** | 0.0531 | **0.492** [0.400, 0.585] | 0.651 | 0.346 | 0.452 | 342 |
| `no_cons` (development name `cons0`) | 0.3389 | 0.1655 | 0.0600 | 0.437 [0.345, 0.532] | 0.641 | 0.327 | 0.433 | 348 |

Skill (or the model aptitude score as it's called in paper) is `1 - MSE / MSE_persistence` on the latent (higher is better). Gain is the mean Hamming improvement of the decoded sequence over copying the source sequence (higher is better); its CI is a bootstrap. The
`summarize` stage of `analyse_experiment.sh` recomputes these numbers from the outputs. Precision, recall and F1 are as reported in the summary CSV.

