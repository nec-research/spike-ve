#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     infer_residual_prefix_swap.py
#   Authors:  Raman Siarheyeu (raman.siarheyeu@neclab.eu)
#             Miguel Sánchez-Palomino (miguel.palomino@neclab.eu)
#             Nithya Bhasker (nithya.bhasker@neclab.eu)
#             Catherine Chaput (catherine.chaput@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2026, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on
# the original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted to you under this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
"""Multi-draw B4 inference via lineage-aware prefix edits (prefix swap).

Run (this is the ``swap`` stage of ``scripts/run_experiment.sh``)::

    python -m recomb2027.latent.predictors.infer_residual_prefix_swap \\
        --checkpoint .../Pca2dSiteUnifiedV4_best_model.pt \\
        --predictor-config configs/latent/predictors/residual_model_conscurr.yaml \\
        --dataset-config configs/latent/datasets/by_nearest_ancestor/split_2022-12-01_k5_esm1_layer33.yaml \\
        --split test --mode prefix_swap --n-samples 20 --donor-pool train,cal,test \\
        --metadata-path .../msa_metadata.csv --lineage-col lineage --msa-path .../filtered_msa_S1.fasta \\
        --baseline-npz .../pred_cal_test.npz --output-npz .../pred_cal_test_lineage_subst_prefix_swap.npz \\
        --absolute --unscaled
"""
from __future__ import annotations

import sys
import argparse
import logging
import numpy as np
import torch

from datetime import datetime
from pathlib import Path
from recomb2027.core.paths import repo_root
from typing import Any, Optional
from omegaconf import OmegaConf
from recomb2027.config_io import load_config
from recomb2027.latent.inference.gpt_lineage_substitution import load_gid_metadata_for_substitution
from recomb2027.latent.inference.lineage_subst_select import (
    SELECT_METHODS,
    gather_selected,
    select_predictions,
)
from recomb2027.latent.inference.unified_v3_lineage_substitution import (
    run_lineage_substitution_v3,
    unwrap_leaf,
)
from recomb2027.latent.predictors.infer_residual import (
    export_raw_pca2d_bundle,
    inverse_mean_rows,
    source_gids_from_dataset,
    _setup_logging,
)
from recomb2027.latent.predictors.predictor_utils import parse_msa, resolve_device
from recomb2027.latent.predictors.train_residual import (
    _load_conservation_stats,
    build_v3_loaders,
    build_model_from_cfg,
)


# ============================================================================
# (merged from infer_gpt_lineage_substitution.py)
# ============================================================================

REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_output_npz_path(output_npz: Path | None, checkpoint: Path) -> Path:
    """Resolve ``--output-npz`` to a concrete ``.npz`` file path.

    A directory (existing, trailing slash, or bare path without ``.npz``/``.npy``)
    gets ``lineage_subst_preds_<timestamp>.npz`` written inside it. Parents are
    created as needed.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_name = f"lineage_subst_preds_{stamp}.npz"
    if output_npz is None:
        out = Path(checkpoint).expanduser().resolve().parent / default_name
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    raw = str(output_npz)
    p = Path(output_npz).expanduser()
    treat_as_dir = (
        raw.endswith(("/", "\\"))
        or p.is_dir()
        or p.suffix.lower() not in {".npz", ".npy"}
    )
    if treat_as_dir:
        p.mkdir(parents=True, exist_ok=True)
        out = p / default_name
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        out = p
    return out


# ============================================================================
# (merged from infer_unified_v4_lineage_substitution.py)
# ============================================================================

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

_DEFAULT_PREDICTOR = "configs/latent/predictors/residual_model_conscurr.yaml"
_DEFAULT_DATASET = (
    "configs/latent/datasets/by_nearest_ancestor/"
    "split_2022-12-01_k5_esm1_layer33.yaml"
)


def _load_v4_checkpoint(model, checkpoint: str, device: torch.device) -> None:
    """Load a B4 checkpoint and fail loudly on shape/key mismatches.

    ``load_state_dict(strict=False)`` silently ignores missing keys, so a
    wrong model class would otherwise produce wrong predictions without any
    error. Unexpected checkpoint keys are a hard error; missing keys are
    logged (the frozen site-GPT anchor is not stored in the checkpoint).
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("trainable_state_dict") or ckpt.get("model_state_dict")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint {checkpoint} has {len(result.unexpected_keys)} keys not "
            f"present in Pca2dSiteUnifiedV4 (wrong model class?): "
            f"{result.unexpected_keys[:10]}"
        )
    # Frozen site-GPT weights are never saved in the checkpoint; only warn about the rest.
    missing = [k for k in result.missing_keys if not k.startswith("site_gpt.")]
    if missing:
        logger.warning(
            "Checkpoint %s: %d non-site-GPT model keys not in checkpoint (first: %s)",
            checkpoint,
            len(missing),
            missing[:5],
        )


def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(
        description=(
            "B4 / unified-v4 multi-draw inference via last_step or prefix_swap "
            "lineage substitution."
        )
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--predictor-config", default=_DEFAULT_PREDICTOR)
    p.add_argument("--dataset-config", default=_DEFAULT_DATASET)
    p.add_argument(
        "--mode",
        choices=("last_step", "prefix_swap"),
        required=True,
        help="last_step: BLOSUM last-prefix swap; prefix_swap: donor series prefix.",
    )
    p.add_argument(
        "--n-samples",
        "--m-substitutes",
        dest="n_samples",
        type=int,
        default=5,
        help="Number of substitute draws per series (alias: --m-substitutes).",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=("train", "cal", "test", "val"),
        help="Query split to run substitution on.",
    )
    p.add_argument(
        "--donor-pool",
        default="train,cal,test",
        help="Comma-separated donor splits for prefix_swap (default: train,cal,test).",
    )
    p.add_argument("--metadata-path", type=Path, required=True)
    p.add_argument("--lineage-col", default="lineage")
    p.add_argument(
        "--msa-path",
        type=Path,
        default=None,
        help="MSA FASTA (defaults to predictor/dataset config msa_path).",
    )
    p.add_argument("--output-npz", type=Path, default=None)
    p.add_argument("--max-series", type=int, default=None)
    p.add_argument(
        "--absolute",
        action="store_true",
        help="Store absolute preds (last-seen + delta) instead of diffs.",
    )
    p.add_argument(
        "--unscaled",
        action="store_true",
        help="Also write inverse-scaled *_raw arrays.",
    )
    p.add_argument(
        "--baseline-npz",
        type=Path,
        default=None,
        help=(
            "Optional prior infer NPZ; compare unmodified baseline to "
            "{split}_pred_mean / {split}_pred_pca2d (scaled) or *_abs when --absolute."
        ),
    )
    p.add_argument(
        "--select",
        choices=SELECT_METHODS,
        default="none",
        help=(
            "GT-free draw → final GA-ready preds. "
            "none=clouds only; baseline=unmodified B4; "
            "rank1=first substitute (BLOSUM / donor order); "
            "top_mode=k-NN density mode; medoid|consensus=nearest to cloud mean. "
            "Writes {split}_pred_mean[_abs|_raw] and {split}_pred_pca2d* for packing."
        ),
    )
    p.add_argument(
        "--select-space",
        choices=("mean", "pca2d", "joint"),
        default="mean",
        help="Latent used to score top_mode / medoid (default: mean).",
    )
    p.add_argument(
        "--select-top-mode-k",
        type=int,
        default=3,
        help="k for --select top_mode (default: 3; keep < n_samples).",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _resolve_msa_path(args, cfg, data_cfg) -> Path:
    if args.msa_path is not None:
        return Path(args.msa_path)
    msa = cfg.get("msa_path") or data_cfg.get("msa_path")
    if not msa:
        raise SystemExit("msa_path required via --msa-path or config")
    return Path(str(msa))


def _split_loader_map(train_loader, cal_loader, test_loader):
    return {
        "train": train_loader,
        "cal": cal_loader,
        "test": test_loader,
        "val": cal_loader if cal_loader is not None else test_loader,
    }


def _compare_baseline(
    result: dict[str, Any],
    baseline_npz: Path,
    *,
    split: str,
    absolute: bool,
    max_series: Optional[int],
) -> dict[str, float]:
    data = np.load(baseline_npz, allow_pickle=True)
    n = int(result["n_series"])
    has_abs = f"{split}_pred_mean_abs" in data.files
    if absolute and not has_abs:
        # Do not compare absolute ours to scaled-diff ref.
        print(
            f"Baseline check SKIPPED vs {baseline_npz}: "
            f"--absolute run but baseline lacks {split}_pred_mean_abs. "
            f"Re-run without --absolute, or use a baseline NPZ with *_abs.",
            flush=True,
        )
        return {
            "mae_mean": float("nan"),
            "mae_pca2d": float("nan"),
            "max_mean": float("nan"),
            "max_pca2d": float("nan"),
            "skipped": 1.0,
        }
    if absolute:
        key_mean = f"{split}_pred_mean_abs"
        key_pca = f"{split}_pred_pca2d_abs"
    else:
        key_mean = f"{split}_pred_mean"
        key_pca = f"{split}_pred_pca2d"

    ref_mean = np.asarray(data[key_mean], dtype=np.float32)[:n]
    ref_pca = np.asarray(data[key_pca], dtype=np.float32)[:n]
    ours_mean = result["pred_baseline_mean"]
    ours_pca = result["pred_baseline_pca2d"]

    if max_series is not None:
        ref_mean = ref_mean[: int(max_series)]
        ref_pca = ref_pca[: int(max_series)]

    if ref_mean.shape != ours_mean.shape or ref_pca.shape != ours_pca.shape:
        raise ValueError(
            f"baseline shape mismatch: ref mean {ref_mean.shape} vs ours "
            f"{ours_mean.shape}; ref pca {ref_pca.shape} vs ours {ours_pca.shape}"
        )

    mae_mean = float(np.mean(np.abs(ours_mean - ref_mean)))
    mae_pca = float(np.mean(np.abs(ours_pca - ref_pca)))
    max_mean = float(np.max(np.abs(ours_mean - ref_mean)))
    max_pca = float(np.max(np.abs(ours_pca - ref_pca)))
    print(
        f"Baseline check vs {baseline_npz} ({key_mean}/{key_pca}):\n"
        f"  mean MAE={mae_mean:.3e} max={max_mean:.3e}\n"
        f"  pca2d MAE={mae_pca:.3e} max={max_pca:.3e}",
        flush=True,
    )
    return {
        "mae_mean": mae_mean,
        "mae_pca2d": mae_pca,
        "max_mean": max_mean,
        "max_pca2d": max_pca,
    }


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    _setup_logging(args.log_level)

    cfg = OmegaConf.create(
        OmegaConf.to_container(load_config(args.predictor_config), resolve=True)
    )
    data_cfg = load_config(args.dataset_config)
    device = torch.device(resolve_device(cfg))
    bs = int(args.batch_size or cfg.get("batch_size", 32))

    train_loader, cal_loader, test_loader, pca2d_name, _ = build_v3_loaders(
        cfg, data_cfg, batch_size=bs, num_workers=0
    )
    loaders = _split_loader_map(train_loader, cal_loader, test_loader)
    query_name = str(args.split)
    if query_name == "val":
        query_name = "cal" if cal_loader is not None else "test"
    query_loader = loaders.get(query_name)
    if query_loader is None:
        query_loader = loaders.get(args.split)
    if query_loader is None:
        raise SystemExit(f"split {args.split!r} not available")

    query_wrapped = query_loader.dataset
    query_leaf = unwrap_leaf(query_wrapped)

    donor_names = [s.strip() for s in str(args.donor_pool).split(",") if s.strip()]
    donor_leaves: dict[str, Any] = {}
    for name in donor_names:
        key = "cal" if name == "val" and cal_loader is not None else name
        loader = loaders.get(key)
        if loader is None:
            logger.warning("donor split %s unavailable; skipping", name)
            continue
        donor_leaves[key] = unwrap_leaf(loader.dataset)

    msa_path = _resolve_msa_path(args, cfg, data_cfg)
    cfg_msa = cfg.get("msa_path") or data_cfg.get("msa_path")
    if cfg_msa and Path(str(cfg_msa)).resolve() != msa_path.resolve():
        logger.warning(
            "--msa-path %s differs from the model config msa_path %s: substitution "
            "features use --msa-path, while conservation/training features use the config MSA.",
            msa_path,
            cfg_msa,
        )
    metadata_path = Path(args.metadata_path)
    if not metadata_path.is_file():
        raise SystemExit(f"metadata not found: {metadata_path}")
    if not msa_path.is_file():
        raise SystemExit(f"msa not found: {msa_path}")

    gid_to_lineage, gid_to_date = load_gid_metadata_for_substitution(
        str(metadata_path), lineage_col=str(args.lineage_col)
    )
    gid_to_seq = parse_msa(str(msa_path))
    print(
        f"Loaded metadata ({len(gid_to_lineage)} gids), MSA ({len(gid_to_seq)} seqs)",
        flush=True,
    )

    site_ckpt = str(cfg.get("site_gpt_checkpoint", ""))
    # Build B4 model (Pca2dSiteUnifiedV4) with the same data_cfg as v4 infer.
    model = build_model_from_cfg(
        cfg, site_gpt_checkpoint=site_ckpt, device=device, data_cfg=data_cfg
    )
    _load_v4_checkpoint(model, str(args.checkpoint), device)
    if bool(cfg.get("append_msa_conservation", False)):
        # Re-apply after ckpt load: buffer may be zeros in the saved state.
        conservation = _load_conservation_stats(
            cfg, data_cfg, seq_len=int(model.seq_len)
        )
        if conservation is not None:
            model.set_conservation_stats(conservation)
    model.eval()
    print(
        f"Loaded v4 checkpoint {args.checkpoint} "
        f"(mode={args.mode}, n_samples={args.n_samples}, split={query_name}, "
        f"pca2d={pca2d_name}, select={args.select})",
        flush=True,
    )

    absolute = bool(args.absolute)
    select_method = str(args.select).lower()
    if select_method != "none" and not (absolute and bool(args.unscaled)):
        logger.warning(
            "--select=%s without --absolute --unscaled: GA packers prefer "
            "absolute raw preds ({split}_pred_mean_raw / pca2d_*_raw). "
            "Pass both flags for GA-ready exports.",
            select_method,
        )

    result = run_lineage_substitution_v3(
        model,
        query_leaf=query_leaf,
        donor_leaves=donor_leaves,
        mode=str(args.mode),
        n_samples=int(args.n_samples),
        cfg=cfg,
        device=device,
        gid_to_lineage=gid_to_lineage,
        gid_to_date=gid_to_date,
        gid_to_seq=gid_to_seq,
        pca2d_name=str(pca2d_name),
        absolute=absolute,
        max_series=args.max_series,
        query_split_name=query_name,
        query_wrapped=query_wrapped,
    )

    baseline_stats = None
    if args.baseline_npz is not None:
        baseline_stats = _compare_baseline(
            result,
            Path(args.baseline_npz),
            split=query_name,
            absolute=absolute,
            max_series=args.max_series,
        )

    out_path = resolve_output_npz_path(args.output_npz, Path(args.checkpoint))
    basis_path = str(cfg.get("pca2d_basis_path", "") or "")
    payload: dict[str, Any] = {
        "pred_samples_mean": result["pred_samples_mean"],
        "pred_samples_pca2d": result["pred_samples_pca2d"],
        "pred_baseline_mean": result["pred_baseline_mean"],
        "pred_baseline_pca2d": result["pred_baseline_pca2d"],
        "target_mean": result["target_mean"],
        "target_pca2d": result["target_pca2d"],
        "distances": result["distances"],
        "donor_series_idx": result["donor_series_idx"],
        "donor_split": np.asarray(result["donor_split"], dtype=object),
        "substitute_gids": np.asarray(result["substitute_gids"], dtype=object),
        "mode": np.asarray(result["mode"]),
        "n_samples": np.asarray(result["n_samples"]),
        "split": np.asarray(query_name),
        "absolute": np.asarray(absolute),
        "checkpoint": np.asarray(str(args.checkpoint)),
        "pca2d_target_name": np.asarray(str(pca2d_name)),
        "seq_len": np.asarray(int(model.seq_len)),
        "k": np.asarray(int(model.k)),
        "basis_path": np.asarray(basis_path),
    }
    # Prefix with split for parity with infer_unified_pca2d_site_v3 exports.
    for key in (
        "pred_samples_mean",
        "pred_samples_pca2d",
        "pred_baseline_mean",
        "pred_baseline_pca2d",
        "target_mean",
        "target_pca2d",
    ):
        payload[f"{query_name}_{key}"] = payload[key]

    if "pred_baseline_activation_probs" in result:
        payload["pred_baseline_activation_probs"] = result[
            "pred_baseline_activation_probs"
        ]
        payload["pred_samples_activation_probs"] = result[
            "pred_samples_activation_probs"
        ]
        payload[f"{query_name}_pred_baseline_activation_probs"] = result[
            "pred_baseline_activation_probs"
        ]
        payload[f"{query_name}_pred_samples_activation_probs"] = result[
            "pred_samples_activation_probs"
        ]

    try:
        src_gids = source_gids_from_dataset(
            query_wrapped, n=int(result["n_series"])
        )
        payload["source_gids"] = src_gids
        payload[f"{query_name}_source_gids"] = src_gids
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not export source_gids: %s", exc)

    if args.unscaled:
        payload["pred_baseline_mean_raw"] = inverse_mean_rows(
            query_wrapped, result["pred_baseline_mean"]
        )
        payload["pred_samples_mean_raw"] = inverse_mean_rows(
            query_wrapped,
            result["pred_samples_mean"].reshape(
                -1, result["pred_samples_mean"].shape[-1]
            ),
        ).reshape(result["pred_samples_mean"].shape)
        base_pca_raw = export_raw_pca2d_bundle(
            query_wrapped,
            result["pred_baseline_pca2d"],
            seq_len=int(model.seq_len),
            k=int(model.k),
        )
        payload["pred_baseline_pca2d_flat_raw"] = base_pca_raw["pred_pca2d_flat_raw"]
        payload["pred_baseline_pca2d_sites_raw"] = base_pca_raw["pred_pca2d_sites_raw"]
        samp = result["pred_samples_pca2d"]
        flat = samp.reshape(-1, samp.shape[-1])
        samp_raw = export_raw_pca2d_bundle(
            query_wrapped,
            flat,
            seq_len=int(model.seq_len),
            k=int(model.k),
        )
        payload["pred_samples_pca2d_flat_raw"] = samp_raw[
            "pred_pca2d_flat_raw"
        ].reshape(*samp.shape[:-1], -1)
        payload["pred_samples_pca2d_sites_raw"] = samp_raw[
            "pred_pca2d_sites_raw"
        ].reshape(*samp.shape[:-1], int(model.seq_len), int(model.k))
        payload["unscaled"] = np.asarray(True)

    # --- GT-free selection → GA-ready pred_* keys ---
    if select_method != "none":
        selected = select_predictions(
            samples_mean=result["pred_samples_mean"],
            samples_pca=result["pred_samples_pca2d"],
            baseline_mean=result["pred_baseline_mean"],
            baseline_pca=result["pred_baseline_pca2d"],
            method=select_method,
            select_space=str(args.select_space),
            top_mode_k=int(args.select_top_mode_k),
        )
        pred_mean = selected["pred_mean"]
        pred_pca = selected["pred_pca2d"]
        payload["pred_mean"] = pred_mean
        payload["pred_pca2d"] = pred_pca
        payload[f"{query_name}_pred_mean"] = pred_mean
        payload[f"{query_name}_pred_pca2d"] = pred_pca
        if absolute:
            # Already absolute when --absolute; alias for packers that look for *_abs.
            payload["pred_mean_abs"] = pred_mean
            payload["pred_pca2d_abs"] = pred_pca
            payload[f"{query_name}_pred_mean_abs"] = pred_mean
            payload[f"{query_name}_pred_pca2d_abs"] = pred_pca
        payload["selected_idx"] = selected["selected_idx"]
        payload["selected_score"] = selected["selected_score"]
        payload["select_method"] = selected["select_method"]
        payload["select_space"] = selected["select_space"]
        payload["select_top_mode_k"] = np.asarray(int(args.select_top_mode_k))

        if "pred_samples_activation_probs" in result:
            act_sel = gather_selected(
                result["pred_samples_activation_probs"],
                selected["selected_idx"],
                baseline=result["pred_baseline_activation_probs"],
            )
            payload["activation_probs"] = act_sel
            payload[f"{query_name}_activation_probs"] = act_sel

        n_ok = int(np.sum(selected["selected_idx"] >= 0))
        n_fb = int(np.sum(selected["selected_idx"] < 0))
        print(
            f"Selected draws method={select_method} space={args.select_space}: "
            f"{n_ok} from cloud, {n_fb} fallback→baseline",
            flush=True,
        )

        if args.unscaled:
            payload["pred_mean_raw"] = inverse_mean_rows(query_wrapped, pred_mean)
            payload[f"{query_name}_pred_mean_raw"] = payload["pred_mean_raw"]
            sel_raw = export_raw_pca2d_bundle(
                query_wrapped,
                pred_pca,
                seq_len=int(model.seq_len),
                k=int(model.k),
            )
            payload["pred_pca2d_flat_raw"] = sel_raw["pred_pca2d_flat_raw"]
            payload["pred_pca2d_sites_raw"] = sel_raw["pred_pca2d_sites_raw"]
            payload[f"{query_name}_pred_pca2d_flat_raw"] = sel_raw[
                "pred_pca2d_flat_raw"
            ]
            payload[f"{query_name}_pred_pca2d_sites_raw"] = sel_raw[
                "pred_pca2d_sites_raw"
            ]

    if baseline_stats is not None:
        for k, v in baseline_stats.items():
            payload[f"baseline_check_{k}"] = np.asarray(v)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Wrote {out_path}", flush=True)
    return {"output_npz": str(out_path), "result": result, "baseline": baseline_stats}


if __name__ == "__main__":
    main()
