#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     prepare_dataset.py
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
"""
Create, split, and finalize data series for model training.

This script:
1. Builds k-series from raw data
2. Splits them into train/validation sets
3. Matches encodings
4. Saves finalized datasets to disk
"""

import json
import os, sys
import pickle
from pathlib import Path
from recomb2027.core.paths import repo_root
from typing import Dict, List, Optional, Union

REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig
from recomb2027.config_io import load_config
from tqdm import tqdm

from recomb2027.latent.data_series.series_splitter import SeriesSplitter
from recomb2027.data.series import _flatten, SeriesEncodingsDataset
from recomb2027.latent.data_series.predecessor_finder import PredecessorFinder
from recomb2027.data import DataManager
from recomb2027.latent.data_series.series_factory import SeriesFactory, stride_kwargs_from_cfg
from recomb2027.latent.data_series.precomputed_splits_loader import (
    apply_precomputed_split_filters,
    load_precomputed_splits,
    resolve_precomputed_splits_path,
)
from recomb2027.latent.io_utils import setup_logging

logger = setup_logging()


#DEFAULT_CONFIG_PATH = "configs/latent/datasets/dataset_20221201_k5.yaml"
#DEFAULT_CONFIG_PATH = "configs/latent/datasets/dataset_nearestleaf_20220501_k6.yaml"
#DEFAULT_CONFIG_PATH = "configs/latent/datasets/dataset_nearestleaf_bv_brc.yaml"
DEFAULT_CONFIG_PATH = "configs/latent/datasets/dataset_bins_bv_brc.yaml"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def build_seq_lookup(dmg: DataManager) -> Dict[str, str]:
    """Build a genome_id -> protein sequence mapping from loaded MSA."""
    seq_lookup: Dict[str, str] = {}
    for label_dict, seq in zip(dmg.sequence_label_dicts, dmg.sequence_strs):
        gid = label_dict.get("genome_id")
        if gid:
            seq_lookup[gid] = seq
    return seq_lookup


# ---------------------------------------------------------------
# Encoding-based filtering  (method="encoding")
# ---------------------------------------------------------------

def find_approx_val_matches_in_train(
        train_dataset: SeriesEncodingsDataset,
        val_dataset: SeriesEncodingsDataset,
        *,
        atol: float = 1e-5,
) -> List[int]:
    """
    Find validation samples whose target encoding (Y, i.e. last element)
    approximately matches ANY encoding vector in the training dataset
    (all k positions -- both context and target).

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        atol: Absolute tolerance for vector matching.

    Returns:
        List of validation indices that have at least one match in training.
    """
    train_samples = train_dataset.sample_encodings   # [N, K, D]
    val_targets = val_dataset.get_y()                 # [M, D]

    if train_samples.shape[-1] != val_targets.shape[-1]:
        raise ValueError("Embedding dimensions must match")

    D = train_samples.shape[-1]

    # Flatten ALL training positions -> [N*K, D]
    train_flat = train_samples.reshape(-1, D)

    matched_indices: List[int] = []

    for idx, val_vec in tqdm(
            enumerate(val_targets),
            total=len(val_targets),
            desc="Finding approx matches (encodings)",
            unit="vector",
    ):
        distances = np.linalg.norm(train_flat - val_vec, axis=1)

        if np.any(distances <= atol):
            matched_indices.append(idx)

    logger.info(
        "Validation target encodings matched in train: %d / %d",
        len(matched_indices),
        len(val_targets),
    )

    return matched_indices


def find_trivial_copy_indices_enc(
        dataset: SeriesEncodingsDataset,
        *,
        atol: float = 1e-5,
        check_pairs: str = "all",
) -> List[int]:
    """
    Find series with at least one trivial-copy step in encoding space.

    A series is dropped if **any pair of time steps** (according to
    ``check_pairs``) has approximately identical embeddings (L2 distance ≤
    ``atol``). The "any pair" mode catches not only zero-mutation
    ``(source, target)`` pairs but also non-adjacent collapses such as
    ``[A, B, A, C, A, D]`` where step 1, 3, 5 land on the same point — a
    pattern that produces self-crossing trajectories on the UMAP plot and
    poisons the model with a non-monotone history.

    Args:
        dataset: Dataset with matched encodings.
        atol: Absolute tolerance for considering two vectors identical.
        check_pairs: One of ``{"all", "adjacent", "last_only"}``. ``"all"``
            (default) scans every ``(i, j)`` with ``i < j`` (catches
            non-adjacent loops); ``"adjacent"`` scans only ``(i, i+1)``;
            ``"last_only"`` scans only ``(K-2, K-1)`` (legacy behaviour).

    Returns:
        List of series indices to drop.
    """
    encodings = dataset.sample_encodings          # [N, K, D]
    n, k, _ = encodings.shape

    if check_pairs == "last_only":
        targets = encodings[:, -1]
        last_context = encodings[:, -2]
        distances = np.linalg.norm(targets - last_context, axis=1)
        trivial_indices = np.where(distances <= atol)[0].tolist()
        logger.info(
            "Trivial-copy series by encoding (y ≈ x[-1], atol=%.1e): %d / %d",
            atol, len(trivial_indices), n,
        )
        return trivial_indices

    trivial_mask = np.zeros(n, dtype=bool)
    if check_pairs == "adjacent":
        for t in range(k - 1):
            d = np.linalg.norm(encodings[:, t + 1] - encodings[:, t], axis=1)
            trivial_mask |= d <= atol
        mode_str = "any adjacent pair"
    elif check_pairs == "all":
        # k * (k-1) / 2 pairs — k is small (≤ ~8) so this is cheap.
        for i in range(k - 1):
            for j in range(i + 1, k):
                d = np.linalg.norm(encodings[:, j] - encodings[:, i], axis=1)
                trivial_mask |= d <= atol
        mode_str = "any pair (i<j)"
    else:
        raise ValueError(
            f"check_pairs must be one of {{'all', 'adjacent', 'last_only'}}; "
            f"got {check_pairs!r}"
        )

    trivial_indices = np.where(trivial_mask)[0].tolist()
    logger.info(
        "Trivial-copy series by encoding (%s, atol=%.1e): %d / %d",
        mode_str, atol, len(trivial_indices), n,
    )
    return trivial_indices


# ---------------------------------------------------------------
# Sequence-based filtering  (method="sequence")
# ---------------------------------------------------------------

def find_trivial_copy_indices_seq(
        dataset: SeriesEncodingsDataset,
        seq_lookup: Dict[str, str],
        *,
        check_pairs: str = "all",
        encoding_fallback: bool = True,
        enc_atol: float = 1e-5,
        check_gid_repeats: bool = True,
) -> List[int]:
    """
    Find series with at least one trivial-copy step in sequence space.

    A series is dropped when **any pair** of its time steps maps to the same
    protein sequence under ``seq_lookup`` (i.e. zero amino-acid mutations in
    the Spike segment). With ``check_pairs='all'`` (default), every
    ``(i, j)`` with ``i < j`` is scanned, which catches both adjacent
    zero-jump steps and non-adjacent loops such as ``[A, B, A, C, A, D]``
    that produce self-crossing trajectories on the UMAP plot.

    A second cheap pass (``check_gid_repeats``) flags series in which the
    same ``genome_id`` appears at two different positions. This typically
    indicates a 2-cycle in the predecessor map (e.g. ``pred[X]=Y`` and
    ``pred[Y]=X``) that ``find_trivial_copy_indices_*`` would otherwise miss
    when sequences happen to be near-identical.

    When a genome id is missing from the MSA-derived ``seq_lookup``, the
    sequence comparison cannot be done. With ``encoding_fallback=True`` (the
    default) we fall back to comparing the **embeddings** for that pair with
    tolerance ``enc_atol`` so missing-MSA gaps don't silently let duplicates
    through.

    Args:
        dataset: Dataset with sample_gids and (for fallback) sample_encodings.
        seq_lookup: genome_id -> protein sequence mapping.
        check_pairs: One of ``{"all", "adjacent", "last_only"}``. ``"all"``
            (default) scans every ``(i, j)`` with ``i < j``; ``"adjacent"``
            scans only consecutive pairs; ``"last_only"`` scans only the
            target/last-context pair (legacy behaviour).
        encoding_fallback: If True, when at least one gid in a pair is missing
            from ``seq_lookup``, compare embeddings with tolerance ``enc_atol``.
        enc_atol: L2 tolerance for the encoding fallback comparison.
        check_gid_repeats: If True, also drop any series where the same gid
            appears at two distinct positions (catches predecessor-map cycles
            independently of MSA availability).

    Returns:
        List of series indices to drop.
    """
    sample_gids = dataset.sample_gids
    n = len(sample_gids)
    if n == 0:
        return []

    encs = getattr(dataset, "sample_encodings", None)
    if encoding_fallback and encs is None:
        logger.warning(
            "encoding_fallback=True but dataset has no sample_encodings; disabling fallback."
        )
        encoding_fallback = False

    if check_pairs not in ("all", "adjacent", "last_only"):
        raise ValueError(
            f"check_pairs must be one of {{'all', 'adjacent', 'last_only'}}; "
            f"got {check_pairs!r}"
        )

    n_seq_hits = 0
    n_enc_hits = 0
    n_gid_hits = 0
    n_missing_no_fallback = 0

    trivial_indices: List[int] = []
    for idx, gids in enumerate(sample_gids):
        k = len(gids)
        if check_pairs == "all":
            pairs = [(i, j) for i in range(k - 1) for j in range(i + 1, k)]
        elif check_pairs == "adjacent":
            pairs = [(t, t + 1) for t in range(k - 1)]
        else:  # "last_only"
            pairs = [(k - 2, k - 1)]

        is_trivial = False
        # Cheap pass first: same gid at two positions -> series is degenerate
        # regardless of sequence/encoding equality (this catches predecessor
        # cycles even when MSA lookups would be inconclusive).
        if check_gid_repeats:
            for i, j in pairs:
                if str(gids[i]) == str(gids[j]):
                    is_trivial = True
                    n_gid_hits += 1
                    break
        if not is_trivial:
            for i, j in pairs:
                seq_i = seq_lookup.get(gids[i])
                seq_j = seq_lookup.get(gids[j])
                if seq_i is not None and seq_j is not None:
                    if seq_i == seq_j:
                        is_trivial = True
                        n_seq_hits += 1
                        break
                else:
                    if encoding_fallback:
                        d = float(np.linalg.norm(encs[idx, j] - encs[idx, i]))
                        if d <= enc_atol:
                            is_trivial = True
                            n_enc_hits += 1
                            break
                    else:
                        n_missing_no_fallback += 1
        if is_trivial:
            trivial_indices.append(idx)

    mode = {
        "all": "any pair (i<j)",
        "adjacent": "any adjacent pair",
        "last_only": "y == x[-1]",
    }[check_pairs]
    logger.info(
        "Trivial-copy series by sequence (%s): %d / %d "
        "[gid-hits=%d, seq-hits=%d, enc-fallback-hits=%d, "
        "missing-skipped-pairs=%d, enc_atol=%.1e]",
        mode, len(trivial_indices), n,
        n_gid_hits, n_seq_hits, n_enc_hits, n_missing_no_fallback, enc_atol,
    )
    return trivial_indices


def find_val_seq_matches_in_train(
        train_dataset: SeriesEncodingsDataset,
        val_dataset: SeriesEncodingsDataset,
        seq_lookup: Dict[str, str],
) -> List[int]:
    """
    Find validation samples whose target protein sequence is identical
    to any protein sequence present in the training dataset (any position).

    Uses O(1) set lookups instead of brute-force distance computation.

    Args:
        train_dataset: Training dataset (sample_gids used).
        val_dataset: Validation dataset (sample_gids used).
        seq_lookup: genome_id -> protein sequence mapping.

    Returns:
        List of validation indices whose target sequence is in training.
    """
    train_seqs: set = set()
    for gids in train_dataset.sample_gids:
        for gid in gids:
            seq = seq_lookup.get(gid)
            if seq is not None:
                train_seqs.add(seq)

    matched_indices: List[int] = []
    for idx, gids in enumerate(val_dataset.sample_gids):
        target_seq = seq_lookup.get(gids[-1])
        if target_seq is not None and target_seq in train_seqs:
            matched_indices.append(idx)

    logger.info(
        "Validation target sequences matched in train: %d / %d",
        len(matched_indices),
        len(val_dataset.sample_gids),
    )
    return matched_indices


def prepare_data_series(cfg: Union[DictConfig, ListConfig]) -> None:
    """
    Create k-series from raw data and perform train/validation split.

    Supports two split strategies via ``cfg.split_criteria``:

    * ``"by_date"`` (default) -- temporal cutoff on ``cfg.split_threshold``.
    * ``"by_clade"`` -- hold out an entire WHO lineage specified by
      ``cfg.val_clade``.  Uses ``cfg.clade_col`` (default ``lineage``)
      as the metadata column name.

    Parameters
    ----------
    cfg : Union[DictConfig, ListConfig]
        Loaded configuration object.
    """
    split_criteria = getattr(cfg, "split_criteria", "by_date")
    split_mode = getattr(cfg, "split_mode", "strict")

    # Determine which metadata columns to load
    metadata_cols = ["collection_date"]
    clade_col = getattr(cfg, "clade_col", "lineage")
    if split_criteria == "by_clade":
        metadata_cols.append(clade_col)
    if split_mode == "strict_leaf_prospective" or getattr(cfg, "test_leaves_only", False):
        metadata_cols.append("is_internal")

    # Load dataset
    collision_source: str = getattr(cfg, "metadata_collision_source", "csv")
    if collision_source not in ("fasta", "csv"):
        raise ValueError(
            f"metadata_collision_source must be 'fasta' or 'csv', got {collision_source!r}"
        )

    dmg = DataManager()
    dmg.load_msa_from_file(cfg.msa_path)

    # Detect columns present in both df_labels (from FASTA) and metadata_cols.
    # pandas would otherwise suffix both copies with _x/_y on merge.
    overlap = [c for c in metadata_cols if c in dmg.df_labels.columns]
    if overlap:
        if collision_source == "csv":
            # CSV wins: drop FASTA-derived values so the CSV ones come through.
            logger.info(
                "Column collision on %s — keeping CSV values "
                "(set metadata_collision_source: fasta to prefer FASTA headers).",
                overlap,
            )
            dmg.df_labels = dmg.df_labels.drop(columns=overlap)
        else:  # "fasta"
            # FASTA wins: skip those columns when loading from CSV.
            logger.info(
                "Column collision on %s — keeping FASTA-header values "
                "(set metadata_collision_source: csv to prefer the metadata CSV).",
                overlap,
            )
            metadata_cols = [c for c in metadata_cols if c not in overlap]

    if metadata_cols:
        dmg.add_metadata_from_csv(
            cfg.metadata_msa_path,
            cols_to_extract=metadata_cols,
        )
    else:
        logger.info(
            "All required metadata columns already present in FASTA labels; "
            "skipping CSV merge.",
        )

    # Build or load predecessor links
    # predecessor_method = getattr(cfg, "predecessor_method", "by_phylo")
    # if predecessor_method == "by_evolocity":
    #     logger.info("Computing predecessors from evolocity velocity graph ...")
    #     pred_finder = PredecessorFinder(dmg)
    #     pred_finder.by_evolocities(
    #         adata_path=cfg.adata_path,
    #         min_score=getattr(cfg, "evolocity_min_score", 0.0),
    #     )
    #     # Save to JSON so downstream can reload without re-computing
    #     pred_dir = Path(cfg.predecessors_path).parent
    #     pred_dir.mkdir(parents=True, exist_ok=True)
    #     pred_finder.write_predecessors_dict(cfg.predecessors_path)
    #     logger.info(f"Evolocity predecessors saved to '{cfg.predecessors_path}'")

    # Order data into series
    seriesObj = SeriesFactory(dmg)
    precomputed_path = resolve_precomputed_splits_path(cfg)

    if precomputed_path is not None:
        loaded = load_precomputed_splits(precomputed_path)
        loaded = apply_precomputed_split_filters(
            loaded,
            encodings_dir=str(cfg.encodings_dir),
            filter_encodings=bool(
                getattr(cfg, "precomputed_splits_filter_encodings", True)
            ),
            filter_last_element=bool(
                getattr(cfg, "precomputed_splits_filter_last_element", False)
            ),
        )
        kseries_lst = loaded["train"] + loaded["cal"] + loaded["val"]
        logger.info(
            "Precomputed splits mode: %d total series after filters (k=%d)",
            len(kseries_lst),
            cfg.series_len,
        )
        data_splitter = SeriesSplitter(df_metadata=seriesObj.dataset.df_labels)
        data_splitter.train_series_list = loaded["train"]
        data_splitter.cal_series_list = loaded["cal"]
        data_splitter.val_series_list = loaded["val"]
        data_splitter.split_stats = {
            **(loaded.get("split_stats") or {}),
            "precomputed_splits": True,
            "precomputed_splits_path": str(precomputed_path),
            **(loaded.get("filter_stats") or {}),
        }
        decon_stats = loaded.get("decontamination") or {}
    else:
        seriesObj.load_predecessors_dict(cfg.predecessors_path)
        stride_kwargs = stride_kwargs_from_cfg(cfg)
        kseries_lst = seriesObj.get_k_series(k=cfg.series_len, **stride_kwargs)

        logger.info(
            f"\nTotal count of considered leafs: {len(seriesObj.predecessors_dict)}"
            f"\nTotal count of k-series: {len(kseries_lst)}, k={cfg.series_len}, "
            f"series_stride={stride_kwargs['stride']}, series_hop_strides={stride_kwargs['hop_strides']}"
        )

        # Filter out series that don't have matched encoding
        seriesObj.filter_k_series_by_encodings(cfg.encodings_dir)
        # Exclude series if last element seen in another series
        kseries_lst = seriesObj.filter_k_series_by_last_element(inplace=True)

        # Data split
        logger.info(f"Splitting series by {split_criteria} with threshold {cfg.split_threshold}")
        data_splitter = SeriesSplitter(
            df_metadata=seriesObj.dataset.df_labels
        )

        if split_criteria == "by_date":
            cal_threshold = getattr(cfg, "cal_threshold", None)
            split_kwargs = dict(
                test_context_mode=str(getattr(cfg, "test_context_mode", "penultimate")),
                dedupe_test_by_target=bool(
                    getattr(
                        cfg,
                        "dedupe_test_by_target",
                        split_mode == "strict_leaf_prospective",
                    )
                ),
            )
            if getattr(cfg, "test_leaves_only", None) is not None:
                split_kwargs["test_leaves_only"] = bool(cfg.test_leaves_only)
            if cal_threshold:
                data_splitter.split_series_by_two_dates(
                    series_list=kseries_lst,
                    cal_threshold_date=str(cal_threshold),
                    test_threshold_date=str(cfg.split_threshold),
                    split_mode=split_mode,
                    **split_kwargs,
                )
            else:
                data_splitter.split_series_by_date(
                    series_list=kseries_lst,
                    threshold_date=cfg.split_threshold,
                    split_mode=split_mode,
                )
        elif split_criteria == "by_clade":
            val_clade = cfg.val_clade
            full_separation = getattr(cfg, "clade_full_separation", True)
            logger.info(
                f"Splitting by clade: val_clade='{val_clade}', "
                f"col='{clade_col}', full_separation={full_separation}"
            )
            data_splitter.split_series_by_clade(
                series_list=kseries_lst,
                val_clade=val_clade,
                clade_col=clade_col,
                full_separation=full_separation,
            )
        else:
            raise ValueError(
                f"Unknown split_criteria '{split_criteria}'. "
                "Use 'by_date' or 'by_clade'."
            )

        # Decontaminate: remove val series whose target gid is in train,
        # or whose context is identical to a training context
        decon_stats = data_splitter.decontaminate_val_series()

    logger.info(
        f"Split result: {len(data_splitter.train_series_list)} train, "
        f"{len(getattr(data_splitter, 'cal_series_list', []) or [])} cal, "
        f"{len(data_splitter.val_series_list)} val"
    )
    if data_splitter.split_stats:
        logger.info("Split stats: %s", data_splitter.split_stats)
    if decon_stats:
        logger.info("Decontamination stats: %s", decon_stats)

    split_payload = {
        "train": data_splitter.train_series_list,
        "val": data_splitter.val_series_list,
        "split_stats": data_splitter.split_stats,
        "decontamination": decon_stats,
    }
    cal_list = getattr(data_splitter, "cal_series_list", None)
    if cal_list is not None:
        split_payload["cal"] = cal_list

    # Save train and validation datasets
    with open(os.path.join(cfg.processed_dir, "splits_raw.pkl"), "wb") as fh:
        pickle.dump(split_payload, fh)
    logger.info(
        "Series train / validation split saved into '%s'",
        os.path.join(cfg.processed_dir, "splits_raw.pkl"),
    )


def finalize_data_series(cfg: Union[DictConfig, ListConfig], split_filename="splits_raw.pkl") -> None:
    """
        Match encodings to train/validation splits and save final datasets.

        The ``filter_method`` config key controls how duplicate / trivial-copy
        detection is performed:

        * ``"sequence"`` (default) -- exact protein-sequence string comparison.
          Requires ``cfg.msa_path``.  Fast (set-based, O(1) lookups).
        * ``"encoding"`` -- approximate comparison of embedding vectors with a
          tolerance of ``trivial_enc_atol``.  Slower but does not need the
          original MSA.

        Trivial-copy detection (the per-split step that removes degenerate
        series) is tunable via three keys:

        * ``trivial_check_pairs`` (default ``"all"``): one of
          ``{"all", "adjacent", "last_only"}``. ``"all"`` scans every
          ``(i, j)`` with ``i < j`` and catches non-adjacent loops such as
          ``[A, B, A, C, A, D]`` that produce self-crossing trajectories on
          the UMAP plot. ``"adjacent"`` only checks consecutive ``(i, i+1)``
          pairs (the previous default). ``"last_only"`` is the legacy
          ``y == x[-1]`` behaviour.
          For backwards compatibility the older boolean
          ``trivial_check_all_pairs`` (``True``→``"adjacent"``,
          ``False``→``"last_only"``) is still honoured if ``trivial_check_pairs``
          is not set.
        * ``trivial_check_gid_repeats`` (default ``True``, sequence mode):
          additionally drop any series in which the same ``genome_id``
          appears at two distinct positions. This catches predecessor-map
          2-cycles (e.g. ``pred[X]=Y`` and ``pred[Y]=X``) that produce
          alternating ``[A, B, A, B, …]`` series.
        * ``trivial_encoding_fallback`` (default ``True``, sequence mode):
          when at least one gid in a checked pair is missing from the MSA,
          fall back to comparing embeddings with tolerance ``trivial_enc_atol``.
        * ``trivial_enc_atol`` (default ``1e-5``): L2 tolerance for the
          encoding-based comparison (used both as fallback and as the primary
          test in ``filter_method='encoding'``).

        Parameters
        ----------
        cfg : Union[DictConfig, ListConfig]
            Loaded configuration object.
        split_filename : str, optional
            Name of the raw split file, by default "splits_raw.pkl".
        """
    filter_method = getattr(cfg, "filter_method", "sequence")
    if filter_method not in ("sequence", "encoding"):
        raise ValueError(f"Unknown filter_method '{filter_method}'. Use 'sequence' or 'encoding'.")
    # Resolve check_pairs with backwards compatibility for the old bool flag.
    check_pairs_cfg = getattr(cfg, "trivial_check_pairs", None)
    if check_pairs_cfg is None:
        legacy = getattr(cfg, "trivial_check_all_pairs", None)
        if legacy is None:
            trivial_check_pairs = "all"
        else:
            trivial_check_pairs = "adjacent" if bool(legacy) else "last_only"
            logger.warning(
                "Config key 'trivial_check_all_pairs' is deprecated; please use "
                "trivial_check_pairs ∈ {'all','adjacent','last_only'}. "
                "Mapped %s → %r for now.", bool(legacy), trivial_check_pairs,
            )
    else:
        trivial_check_pairs = str(check_pairs_cfg)
        if trivial_check_pairs not in ("all", "adjacent", "last_only"):
            raise ValueError(
                f"Unknown trivial_check_pairs={trivial_check_pairs!r}; "
                "use one of 'all', 'adjacent', 'last_only'."
            )
    trivial_enc_atol = float(getattr(cfg, "trivial_enc_atol", 1e-5))
    trivial_encoding_fallback = bool(getattr(cfg, "trivial_encoding_fallback", True))
    trivial_check_gid_repeats = bool(getattr(cfg, "trivial_check_gid_repeats", True))

    # Load dataset splits (w/o matched encodings)
    data_splitter = SeriesSplitter()
    data_splitter.load_split(os.path.join(cfg.processed_dir, split_filename))
    train, val = data_splitter.get_train_val_split()
    cal = data_splitter.get_cal_split() or []

    # Create dataset from matched encodings
    # Train dataset
    train_dataset = SeriesEncodingsDataset(train)
    train_dataset.match_encodings_from_dir(cfg.encodings_dir)
    extra_stats = getattr(cfg, "extra_encoding_stats", None)
    if extra_stats:
        train_dataset.match_extra_encodings_from_dir(cfg.encodings_dir, extra_stats)
    block_names = getattr(cfg, "block_names", None)
    block_pooling = str(getattr(cfg, "block_pooling", "average_token"))
    if block_names:
        train_dataset.match_block_encodings_from_dir(
            cfg.encodings_dir, block_names, block_pooling=block_pooling
        )
    logger.info(
        f"Train dataset length: {len(train_dataset)}, "
        f"Latent dimension: {train_dataset.get_latent_dim()}"
    )

    val_dataset = None
    if val:
        val_dataset = SeriesEncodingsDataset(val, split="val")
        val_dataset.match_encodings_from_dir(cfg.encodings_dir)
        if extra_stats:
            val_dataset.match_extra_encodings_from_dir(cfg.encodings_dir, extra_stats)
        if block_names:
            val_dataset.match_block_encodings_from_dir(
                cfg.encodings_dir, block_names, block_pooling=block_pooling
            )
    else:
        logger.info(
            "Validation split is empty; skipping val encoding match and filtering."
        )

    # Build sequence lookup once if using sequence-based filtering
    seq_lookup = None
    if filter_method == "sequence":
        logger.info("Filter method: sequence (exact string comparison)")
        dmg = DataManager()
        dmg.load_msa_from_file(cfg.msa_path)
        seq_lookup = build_seq_lookup(dmg)
        logger.info(f"Sequence lookup built: {len(seq_lookup)} genome_ids")
    else:
        logger.info("Filter method: encoding (approximate vector comparison)")

    # -- Step 1: remove trivial-copy series from both splits ---
    # With the default 'all' mode we scan every (i, j) pair (i < j), so we
    # filter out any series where ANY two time steps are identical in
    # gid / sequence / encoding. This catches:
    #   * adjacent zero-mutation jumps that any SKIP_CONFIGS pair would expose;
    #   * non-adjacent loops like [A, B, A, C, A, D] that produce
    #     self-crossing trajectories in the UMAP plot;
    #   * 2-cycles in predecessors_dict (same gid at two positions) — caught
    #     by the gid-repeat pre-pass even when MSA lookups are inconclusive.
    # For sequence-based filtering we additionally fall back to encoding
    # distance whenever one of the gids is missing from the MSA.
    if filter_method == "sequence":
        train_trivial = find_trivial_copy_indices_seq(
            train_dataset,
            seq_lookup,
            check_pairs=trivial_check_pairs,
            encoding_fallback=trivial_encoding_fallback,
            enc_atol=trivial_enc_atol,
            check_gid_repeats=trivial_check_gid_repeats,
        )
        val_trivial = (
            find_trivial_copy_indices_seq(
                val_dataset,
                seq_lookup,
                check_pairs=trivial_check_pairs,
                encoding_fallback=trivial_encoding_fallback,
                enc_atol=trivial_enc_atol,
                check_gid_repeats=trivial_check_gid_repeats,
            )
            if val_dataset is not None
            else []
        )
    else:
        train_trivial = find_trivial_copy_indices_enc(
            train_dataset,
            atol=trivial_enc_atol,
            check_pairs=trivial_check_pairs,
        )
        val_trivial = (
            find_trivial_copy_indices_enc(
                val_dataset,
                atol=trivial_enc_atol,
                check_pairs=trivial_check_pairs,
            )
            if val_dataset is not None
            else []
        )

    train_dataset.remove_by_index(train_trivial)
    logger.info(f"Train dataset length after trivial-copy removal: {len(train_dataset)}")
    if val_dataset is not None:
        val_dataset.remove_by_index(val_trivial)
        logger.info(
            f"Validation dataset length after trivial-copy removal: {len(val_dataset)}"
        )

        # -- Step 2: exclude val series whose target matches any train series --
        logger.info(
            "Exclude series from validation if target matches any element in training series."
        )
        if filter_method == "sequence":
            val_ids_to_exclude = find_val_seq_matches_in_train(
                train_dataset, val_dataset, seq_lookup
            )
        else:
            val_ids_to_exclude = find_approx_val_matches_in_train(
                train_dataset, val_dataset, atol=1e-5
            )

        val_dataset.remove_by_index(val_ids_to_exclude)
        logger.info(
            f"Validation dataset length after cross-split filtering: {len(val_dataset)}, "
            f"Latent dimension: {train_dataset.get_latent_dim()}"
        )

    # Fit default StdScaler
    if cfg.fit_scaler:
        train_dataset.fit_scaler()
        train_dataset.save_to_file(os.path.join(cfg.processed_dir, "train_dataset.npz"), save_scaler=True)
    else:
        train_dataset.save_to_file(os.path.join(cfg.processed_dir, "train_dataset.npz"), save_scaler=False)

    def _copy_train_scaler_to(ds):
        ds.scaler = train_dataset.scaler
        ds.block_scaler = train_dataset.block_scaler
        if extra_stats:
            ds.extra_encoding_stats = list(train_dataset.extra_encoding_stats)
            ds.extra_stat_encodings = {
                s: enc.copy() for s, enc in train_dataset.extra_stat_encodings.items()
            }
        if block_names:
            ds.block_encoding_names = list(train_dataset.block_encoding_names)
            ds.block_encodings = {
                n: enc.copy() for n, enc in train_dataset.block_encodings.items()
            }
            ds.block_pooling = train_dataset.block_pooling

    cal_dataset = None
    if cal:
        cal_dataset = SeriesEncodingsDataset(cal, split="cal")
        cal_dataset.match_encodings_from_dir(cfg.encodings_dir)
        if extra_stats:
            cal_dataset.match_extra_encodings_from_dir(cfg.encodings_dir, extra_stats)
        if block_names:
            cal_dataset.match_block_encodings_from_dir(
                cfg.encodings_dir, block_names, block_pooling=block_pooling
            )
        if filter_method == "sequence":
            cal_trivial = find_trivial_copy_indices_seq(
                cal_dataset,
                seq_lookup,
                check_pairs=trivial_check_pairs,
                encoding_fallback=trivial_encoding_fallback,
                enc_atol=trivial_enc_atol,
                check_gid_repeats=trivial_check_gid_repeats,
            )
        else:
            cal_trivial = find_trivial_copy_indices_enc(
                cal_dataset,
                atol=trivial_enc_atol,
                check_pairs=trivial_check_pairs,
            )
        cal_dataset.remove_by_index(cal_trivial)
        if filter_method == "sequence":
            cal_exclude = find_val_seq_matches_in_train(train_dataset, cal_dataset, seq_lookup)
        else:
            cal_exclude = find_approx_val_matches_in_train(
                train_dataset, cal_dataset, atol=1e-5
            )
        cal_dataset.remove_by_index(cal_exclude)
        _copy_train_scaler_to(cal_dataset)
        cal_dataset.save_to_file(
            os.path.join(cfg.processed_dir, "cal_dataset.npz"), save_scaler=True
        )
        logger.info(f"Calibration dataset length: {len(cal_dataset)}")
    
    val_path = os.path.join(cfg.processed_dir, "test_dataset.npz")
    if val_dataset is not None:
        _copy_train_scaler_to(val_dataset)
        logger.info(
            f"Validation dataset length:  {len(val_dataset)},"
            f"Latent dimension: {train_dataset.get_latent_dim()}"
        )
        val_dataset.save_to_file(val_path, save_scaler=True)
    else:
        k = int(train_dataset.sample_encodings.shape[1])
        latent_dim = int(train_dataset.get_latent_dim())
        empty_val = SeriesEncodingsDataset([], split="val")
        empty_val.sample_encodings = np.zeros((0, k, latent_dim), dtype=np.float32)
        _copy_train_scaler_to(empty_val)
        empty_val.save_to_file(val_path, save_scaler=True)
        logger.info(
            "Saved empty test_dataset.npz (0 samples, k=%d, latent_dim=%d)",
            k,
            latent_dim,
        )

    # Size of train and validation datasets
    logger.info(f"----- Size of cleaned train and validation datasets ------")
    logger.info(f"Train dataset length: {len(train_dataset)}")
    logger.info(
        "Validation dataset length: %d",
        len(val_dataset) if val_dataset is not None else 0,
    )

    del train_dataset
    if val_dataset is not None:
        del val_dataset

    # Verify that we can read the data
    logger.info(f"----- Verify reading of train dataset ------")

    train_dataset = SeriesEncodingsDataset.load_from_file(
        os.path.join(cfg.processed_dir, "train_dataset.npz"),
        load_scaler=True
    )
    logger.info(
        f"Train dataset length: {len(train_dataset)},"
        f"Latent dimension: {train_dataset.get_latent_dim()}"
    )


def write_used_config(
        cfg: Union[DictConfig, ListConfig],
        output_dir: Path,
        *,
        source_cfg: Optional[str] = None,
) -> Path:
    """
    Persist the resolved configuration used for this run into ``output_dir``.

    Writes ``config.json`` next to the generated dataset artifacts so the run
    can be reproduced from the same directory.
    """
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        payload = {"config": payload}
    if source_cfg is not None:
        payload = {"_source_cfg": str(source_cfg), **payload}

    config_path = Path(output_dir) / "config.json"
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote used config → %s", config_path)
    return config_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--step",
        choices=("all", "prepare", "finalize"),
        default="all",
        help=(
            "Pipeline stage: prepare (splits_raw.pkl only), finalize "
            "(match encodings + NPZ export), or all (default)."
        ),
    )
    parser.add_argument(
        "--only-finalize",
        action="store_true",
        help="Alias of --step finalize (kept for older command lines).",
    )
    cli_args = parser.parse_args()
    if cli_args.only_finalize:
        cli_args.step = "finalize"
    cfg = load_config(cli_args.cfg)
    
    # Output directory
    series_split_dir = Path(cfg.processed_dir)
    series_split_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot the exact configuration used for this run next to the outputs
    write_used_config(cfg, series_split_dir, source_cfg=cli_args.cfg)
    if cli_args.step in ("all", "prepare"):
        prepare_data_series(cfg)
    if cli_args.step in ("all", "finalize"):
        finalize_data_series(cfg)


if __name__ == "__main__":
    main()
