#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     series_splitter.py
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
""" Module to do train-validation split of the dataset.
"""
import collections
import os
import pickle
import random
import re

import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Any, List, Optional, Tuple, Union

import logging

from recomb2027.data.utils import joblib_load_numpy2_compat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def normalize_date(x):
    x = str(x)
    if re.fullmatch(r"\d{4}$", x):
        return x + "-01-01"
    elif re.fullmatch(r"\d{4}-\d{2}$", x):
        return x + "-01"
    return x


MAT_PROSPECTIVE_SPLIT_MODE = "strict_leaf_prospective"
TEST_CONTEXT_PENULTIMATE = "penultimate"
TEST_CONTEXT_ALL = "all"


def as_bool_series(series: pd.Series) -> pd.Series:
    """Coerce metadata booleans stored as str/bool."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def dedupe_series_by_target(
    series_list: List[List[str]],
) -> Tuple[List[List[str]], int]:
    """Keep the first series for each unique target (last genome_id)."""
    seen: set[str] = set()
    out: List[List[str]] = []
    for row in series_list:
        target = row[-1]
        if target in seen:
            continue
        seen.add(target)
        out.append(row)
    return out, len(series_list) - len(out)


def resolve_mat_prospective_split_options(
    split_mode: str,
    *,
    test_leaves_only: Optional[bool] = None,
    test_context_mode: Optional[str] = None,
    dedupe_test_by_target: Optional[bool] = None,
) -> Tuple[str, bool, str, bool]:
    """Map config into effective temporal split flags.

    ``strict_leaf_prospective`` enables leaf-only test targets, optional
    all-context dating, and one series per test leaf by default.
    """
    if split_mode == MAT_PROSPECTIVE_SPLIT_MODE:
        temporal_mode = "strict"
        leaves_only = True if test_leaves_only is None else bool(test_leaves_only)
        ctx_mode = test_context_mode or TEST_CONTEXT_PENULTIMATE
        dedupe = True if dedupe_test_by_target is None else bool(dedupe_test_by_target)
    else:
        temporal_mode = split_mode
        leaves_only = bool(test_leaves_only)
        ctx_mode = test_context_mode or TEST_CONTEXT_PENULTIMATE
        dedupe = bool(dedupe_test_by_target)
    if ctx_mode not in (TEST_CONTEXT_PENULTIMATE, TEST_CONTEXT_ALL):
        raise ValueError(
            f"Unknown test_context_mode={ctx_mode!r}; "
            f"use {TEST_CONTEXT_PENULTIMATE!r} or {TEST_CONTEXT_ALL!r}."
        )
    return temporal_mode, leaves_only, ctx_mode, dedupe


class SeriesSplitter:
    def __init__(self, df_metadata=None):
        if df_metadata is not None:
            self.df_metadata = df_metadata.copy()
            self._check_datetime_format()
        else:
            self.df_metadata = None
        self.train_series_list = None
        self.val_series_list = None
        self.split_stats: Optional[dict] = None

    def _date_lookup(self) -> dict[str, pd.Timestamp]:
        lookup: dict[str, pd.Timestamp] = {}
        for _, row in self.df_metadata.iterrows():
            gid = str(row["genome_id"])
            dt = row["collection_date"]
            if pd.notna(dt):
                lookup[gid] = pd.Timestamp(dt)
        return lookup

    def _observed_leaf_genome_ids(self) -> set[str]:
        if "is_internal" not in self.df_metadata.columns:
            logger.warning(
                "is_internal column missing from metadata; "
                "cannot restrict test to observed leaves."
            )
            return set(self.df_metadata["genome_id"].astype(str))
        internal = as_bool_series(self.df_metadata["is_internal"])
        return set(
            self.df_metadata.loc[~internal, "genome_id"].astype(str).tolist()
        )

    def _test_context_on_or_before_mask(
        self,
        kX: np.ndarray,
        threshold_date: str,
        *,
        test_context_mode: str,
    ) -> np.ndarray:
        if test_context_mode == TEST_CONTEXT_PENULTIMATE:
            return np.ones(len(kX), dtype=bool)
        threshold_ts = pd.Timestamp(threshold_date)
        dates = self._date_lookup()
        mask = np.ones(len(kX), dtype=bool)
        for i, row in enumerate(kX):
            for gid in row[:-1]:
                dt = dates.get(str(gid))
                if dt is None or pd.isna(dt) or dt > threshold_ts:
                    mask[i] = False
                    break
        return mask

    def _check_datetime_format(self, complete_not_full_values=True):
        if self.df_metadata['collection_date'].dtype == object:
            if complete_not_full_values:
                self.df_metadata['collection_date'] = self.df_metadata['collection_date'].astype(str).apply(normalize_date)

            self.df_metadata['collection_date'] = pd.to_datetime(
                self.df_metadata['collection_date'],
                errors='coerce',      # invalid dates become NaT
                format='mixed'        # infer date format for each element
        )

    def get_train_val_split(self):
        return self.train_series_list, self.val_series_list

    def load_metadata_from_csv(self, metadata_file, sep=",", cols_to_extract=None, dtype=str):
        """ Load metadata from a CSV file."""
        self.df_metadata = pd.read_csv(metadata_file, sep=sep, dtype=dtype)
        if cols_to_extract is not None and isinstance(cols_to_extract, collections.abc.Iterable):
            self.df_metadata = self.df_metadata[list(set(cols_to_extract))]

        self._check_datetime_format(complete_not_full_values=True)

    def split_series_by_date(
            self,
            series_list,
            threshold_date='2022-05-01',
            split_mode='strict',
            only_by_last_element=None,
    ) -> Tuple[Any, Optional[Any]]:
        """Split series into train / validation by a temporal cutoff.

        Parameters
        ----------
        series_list : list of list of str
            k-series of genome IDs.
        threshold_date : str
            Cutoff date.  Validation targets must be after this date.
        split_mode : str
            How context elements are constrained relative to *threshold_date*:

            * ``"strict"`` -- target (last element) after T **and**
              second-to-last element before T.  Guarantees the prediction
              crosses the temporal boundary.
            * ``"soft"`` -- only the target must be after T; context
              elements are unconstrained.  Produces a larger validation
              set that includes series with post-cutoff context.
            * ``"all_after"`` -- the *first* element of the series must be
              after T (entire series is post-cutoff).
        only_by_last_element : bool or None
            **Deprecated** -- kept for backward compatibility.
            ``True`` maps to ``split_mode="strict"``,
            ``False`` maps to ``split_mode="all_after"``.
            Ignored when *split_mode* is explicitly set to something other
            than ``"strict"`` by the caller.
        """
        # Backward compatibility
        if only_by_last_element is not None:
            split_mode = "strict" if only_by_last_element else "all_after"

        kX = np.stack(series_list)
        last = kX[:, -1].tolist()

        # --- Validation selection ---
        gids_after_T = set(
            self.df_metadata.loc[
                (self.df_metadata['genome_id'].isin(last))
                & (self.df_metadata['collection_date'] > threshold_date),
                'genome_id',
            ]
        )

        if split_mode == "strict":
            prev = kX[:, -2].tolist()
            gids_before_T = set(
                self.df_metadata.loc[
                    (self.df_metadata['genome_id'].isin(prev))
                    & (self.df_metadata['collection_date'] <= threshold_date),
                    'genome_id',
                ]
            )
            val_mask = (
                np.isin(kX[:, -1], list(gids_after_T))
                & np.isin(kX[:, -2], list(gids_before_T))
            )
        elif split_mode == "soft":
            val_mask = np.isin(kX[:, -1], list(gids_after_T))
        elif split_mode == "all_after":
            first = kX[:, 0].tolist()
            first_after_T = set(
                self.df_metadata.loc[
                    (self.df_metadata['genome_id'].isin(first))
                    & (self.df_metadata['collection_date'] > threshold_date),
                    'genome_id',
                ]
            )
            val_mask = np.isin(kX[:, 0], list(first_after_T))
        else:
            raise ValueError(
                f"Unknown split_mode '{split_mode}'. "
                "Use 'strict', 'soft', or 'all_after'."
            )

        self.val_series_list = kX[val_mask].tolist()

        # --- Training selection: target <= T ---
        gids_on_or_before_T = set(
            self.df_metadata.loc[
                (self.df_metadata['genome_id'].isin(last))
                & (self.df_metadata['collection_date'] <= threshold_date),
                'genome_id',
            ]
        )
        train_mask = np.isin(kX[:, -1], list(gids_on_or_before_T))
        self.train_series_list = kX[train_mask].tolist()

        logger.info(
            "split_series_by_date (mode=%s, threshold=%s): "
            "%d train, %d val",
            split_mode, threshold_date,
            len(self.train_series_list), len(self.val_series_list),
        )

        return self.train_series_list, self.val_series_list

    def split_series_by_two_dates(
        self,
        series_list,
        cal_threshold_date: str,
        test_threshold_date: str,
        split_mode: str = "strict",
        *,
        test_leaves_only: Optional[bool] = None,
        test_context_mode: str = TEST_CONTEXT_PENULTIMATE,
        dedupe_test_by_target: Optional[bool] = None,
    ) -> Tuple[Any, Any, Any]:
        """Three-way split: train / cal / test by two temporal cutoffs.

        * **train** — target (last gid) date ≤ ``cal_threshold_date``
        * **cal** — target date in (cal_threshold, test_threshold] (strict:
          penultimate ≤ cal_threshold)
        * **test** — target date > ``test_threshold`` (strict: penultimate ≤
          test_threshold)

        Optional MAT prospective evaluation flags (also enabled via
        ``split_mode='strict_leaf_prospective'`` through
        :func:`resolve_mat_prospective_split_options`):

        * ``test_leaves_only`` — keep test series whose target is an observed
          leaf (``is_internal == False`` in metadata).
        * ``test_context_mode`` — ``penultimate`` (default strict) or ``all``
          (every context genome dated ≤ test threshold).
        * ``dedupe_test_by_target`` — one series per unique test target.
        """
        temporal_mode, leaves_only, ctx_mode, dedupe = resolve_mat_prospective_split_options(
            split_mode,
            test_leaves_only=test_leaves_only,
            test_context_mode=test_context_mode,
            dedupe_test_by_target=dedupe_test_by_target,
        )

        if not series_list:
            logger.warning(
                "split_series_by_two_dates received an empty series_list; returning empty splits"
            )
            self.train_series_list = []
            self.cal_series_list = []
            self.val_series_list = []
            self.split_stats = {"train": 0, "cal": 0, "test": 0}
            return [], [], []

        kX = np.stack(series_list)
        last = kX[:, -1].tolist()
        prev = kX[:, -2].tolist()

        def _gids_after(threshold: str) -> set:
            return set(
                self.df_metadata.loc[
                    (self.df_metadata["genome_id"].isin(last))
                    & (self.df_metadata["collection_date"] > threshold),
                    "genome_id",
                ]
            )

        def _gids_on_or_before(threshold: str) -> set:
            return set(
                self.df_metadata.loc[
                    (self.df_metadata["genome_id"].isin(last))
                    & (self.df_metadata["collection_date"] <= threshold),
                    "genome_id",
                ]
            )

        def _prev_on_or_before(threshold: str) -> set:
            return set(
                self.df_metadata.loc[
                    (self.df_metadata["genome_id"].isin(prev))
                    & (self.df_metadata["collection_date"] <= threshold),
                    "genome_id",
                ]
            )

        gids_train = _gids_on_or_before(cal_threshold_date)
        train_mask = np.isin(kX[:, -1], list(gids_train))

        if temporal_mode == "strict":
            gids_cal_tgt = _gids_after(cal_threshold_date) & _gids_on_or_before(
                test_threshold_date
            )
            gids_cal_prev = _prev_on_or_before(cal_threshold_date)
            cal_mask = np.isin(kX[:, -1], list(gids_cal_tgt)) & np.isin(
                kX[:, -2], list(gids_cal_prev)
            )

            gids_test_tgt = _gids_after(test_threshold_date)
            gids_test_prev = _prev_on_or_before(test_threshold_date)
            test_mask = np.isin(kX[:, -1], list(gids_test_tgt)) & np.isin(
                kX[:, -2], list(gids_test_prev)
            )
        elif temporal_mode == "soft":
            gids_cal_tgt = _gids_after(cal_threshold_date) & _gids_on_or_before(
                test_threshold_date
            )
            cal_mask = np.isin(kX[:, -1], list(gids_cal_tgt))
            test_mask = np.isin(kX[:, -1], list(_gids_after(test_threshold_date)))
        else:
            raise ValueError(
                f"Unknown split_mode '{split_mode}' / temporal_mode '{temporal_mode}'. "
                "Use 'strict', 'soft', or 'strict_leaf_prospective'."
            )

        n_test_before_filters = int(test_mask.sum())
        if leaves_only:
            leaf_ids = self._observed_leaf_genome_ids()
            test_mask = test_mask & np.isin(kX[:, -1], list(leaf_ids))

        n_test_after_leaf = int(test_mask.sum())
        if ctx_mode == TEST_CONTEXT_ALL:
            test_mask = test_mask & self._test_context_on_or_before_mask(
                kX, test_threshold_date, test_context_mode=ctx_mode
            )

        n_test_after_context = int(test_mask.sum())

        self.train_series_list = kX[train_mask].tolist()
        self.cal_series_list = kX[cal_mask].tolist()
        self.val_series_list = kX[test_mask].tolist()

        n_test_deduped = 0
        if dedupe and self.val_series_list:
            self.val_series_list, n_test_deduped = dedupe_series_by_target(
                self.val_series_list
            )

        self.split_stats = {
            "split_mode": split_mode,
            "temporal_mode": temporal_mode,
            "test_leaves_only": leaves_only,
            "test_context_mode": ctx_mode,
            "dedupe_test_by_target": dedupe,
            "test_before_filters": n_test_before_filters,
            "test_after_leaf_filter": n_test_after_leaf,
            "test_after_context_filter": n_test_after_context,
            "test_deduped": n_test_deduped,
            "train": len(self.train_series_list),
            "cal": len(self.cal_series_list),
            "test": len(self.val_series_list),
        }

        logger.info(
            "split_series_by_two_dates (mode=%s, cal=%s, test=%s): "
            "%d train, %d cal, %d test",
            split_mode,
            cal_threshold_date,
            test_threshold_date,
            len(self.train_series_list),
            len(self.cal_series_list),
            len(self.val_series_list),
        )
        if leaves_only or ctx_mode == TEST_CONTEXT_ALL or dedupe:
            logger.info(
                "MAT prospective test filters: raw_test=%d, after_leaf=%d, "
                "after_context=%d, deduped=%d, final_test=%d",
                n_test_before_filters,
                n_test_after_leaf,
                n_test_after_context,
                n_test_deduped,
                len(self.val_series_list),
            )
        return self.train_series_list, self.cal_series_list, self.val_series_list

    def get_cal_split(self):
        return getattr(self, "cal_series_list", None)

    def split_series_by_clade(
            self,
            series_list,
            val_clade,
            clade_col='who_lineage',
            full_separation=True
    ) -> Tuple[List[List[str]], List[List[str]]]:

        self.train_series_list = []
        self.val_series_list = []

        for kseries in series_list:
            if not kseries or len(kseries) < 2:
                continue

            df_subset = self.df_metadata[self.df_metadata['genome_id'].isin(kseries)]

            in_val_clade = df_subset[clade_col].isin([val_clade])
            in_other_clade = ~in_val_clade

            if not in_val_clade.any():
                # no genomes from validation clade
                self.train_series_list.append(kseries)
                continue

            # at least one genome in validation clade
            if full_separation and in_other_clade.any():
                # ignore in case need all genomes in validation clade only, but some are in other clades
                continue
            # add to validation
            self.val_series_list.append(kseries)

        return self.train_series_list, self.val_series_list

    def decontaminate_val_series(self) -> dict:
        """Remove validation series that leak information from training.

        Two filters are applied sequentially:
          A) Target-in-train: the validation target genome_id (last element)
             appears anywhere (any position) in any training series.
          B) Duplicate context: the validation context (first k-1 genome_ids)
             is identical to the context of some training series.

        Returns a dict with removal statistics.
        """
        if self.train_series_list is None or self.val_series_list is None:
            raise ValueError("Run a split method first.")

        n_val_before = len(self.val_series_list)

        # --- Fix A: target genome_id seen in training -----------------------
        train_all_gids = set(
            gid for series in self.train_series_list for gid in series
        )
        val_after_a = [
            s for s in self.val_series_list if s[-1] not in train_all_gids
        ]
        n_removed_a = n_val_before - len(val_after_a)

        # --- Fix B: identical context in training ---------------------------
        train_contexts = set(
            tuple(s[:-1]) for s in self.train_series_list
        )
        val_after_b = [
            s for s in val_after_a if tuple(s[:-1]) not in train_contexts
        ]
        n_removed_b = len(val_after_a) - len(val_after_b)

        self.val_series_list = val_after_b

        stats = {
            "val_before": n_val_before,
            "removed_target_in_train": n_removed_a,
            "removed_duplicate_context": n_removed_b,
            "val_after": len(self.val_series_list),
        }
        logger.info(
            "Decontamination: %d val series -> %d "
            "(removed %d target-in-train, %d duplicate-context)",
            stats["val_before"],
            stats["val_after"],
            stats["removed_target_in_train"],
            stats["removed_duplicate_context"],
        )
        return stats

    def write_splits(self, filepath):
        """ Write splits into file.
        """
        with open(filepath, 'wb') as f:
            split_dict = {
                "train": self.train_series_list,
                "val": self.val_series_list,
            }
            cal = getattr(self, "cal_series_list", None)
            if cal is not None:
                split_dict["cal"] = cal
            pickle.dump(split_dict, f)
            logger.info(f"Series train / validation split saved into '{filepath}'")

    def load_split(self, filepath):
        """ Load saved series train/validation split
        """
        with open(filepath, 'rb') as f:
            split_dict = pickle.load(f)
            self.train_series_list = split_dict["train"]
            self.val_series_list = split_dict["val"]
            self.cal_series_list = split_dict.get("cal")
            logger.info(f"Series train / validation split loaded from '{filepath}'")


class SeriesDataset(Dataset):
    """ Class to construct a dataset for the models.
    """
    def __init__(self,
                 xb_gid, yb_gid,
                 df_encodings,
                 split='train',
                 scaler=None,
                 fix_context=False,
                 #predict_diff=False
                 ):
        """
        Args:
            X: np.array of inputs
            x: np.array of targets
            split: ['train', 'val'] data split
            df_encodings: dataframe with encodings for each genome_id in X, y
            scaler: which scaler used to re-scale thre dataset
            fix_context: use full context as input and only last element as a target
        """
        self.fix_context = fix_context
        self.xb_gid = xb_gid
        self.yb_gid = yb_gid
        self.split = split
        self.scaler = scaler    # main data scaler
        self.transform_scaler = None    # 2nd data scaler to re-scale after data transform
        self.X_ = None
        self.y_= None

        # Preload encoding lookup into a dictionary for fast access
        encoding_dict = df_encodings['encodings'].to_dict()

        # Use list comprehension to build arrays efficiently
        self.X_ = np.stack([
            np.stack([encoding_dict[gid] for gid in X_row]) for X_row in xb_gid
        ])
        B, T, D = self.X_.shape

        if scaler is not None:
            if split == 'train':
                # learn scaling on training split
                X_flat = self.X_.reshape(-1, D)
                self.X = scaler.fit_transform(X_flat).reshape(-1, T, D)
            else:
                # use learned scaling for validation split
                X_flat = self.X_.reshape(-1, D)
                self.X = scaler.transform(X_flat).reshape(-1, T, D)

        if self.yb_gid is not None:
            self.y_ = np.stack([
                np.stack([encoding_dict[gid] for gid in y_row]) for y_row in yb_gid
            ])

            y_flat = self.y_.reshape(-1, D)
            self.y = scaler.transform(y_flat).reshape(-1, T, D)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.yb_gid is None:
            return self.X[idx]

        if self.fix_context:
            return self.X[idx], self.y[idx, -1]
        else:
            return self.X[idx], self.y[idx]

    def get_latent_dim(self):
        return self.X.shape[2]

    def get_block_size(self):
        return self.X.shape[1]

    def get_original_xy_gid(self):
        if self.yb_gid is None:
            return self.xb_gid
        else:
            return self.xb_gid, self.yb_gid

    def get_original_xy(self):
        """ Get unscaled data. """
        if self.yb_gid is None:
            return self.X_
        else:
            return self.X_, self.y_

    def get_xy(self):
        """ Get rescaled data. """
        if self.yb_gid is None:
            return self.X
        else:
            return self.X, self.y

    def transform_x(self, transform_func, transform_scaler=None):
        """ Transform X"""
        B, T, D = self.X.shape
        X_flat = self.X.reshape(-1, D)
        # apply transformation to data
        X_flat = transform_func(X_flat)
        # rescalse
        if transform_scaler is not None:
            if self.split == "train":
                X_flat = transform_scaler.fit_transform(X_flat)
            else:
                X_flat = transform_scaler.transform(X_flat)

        self.X = X_flat.reshape(-1, T, D)

    def transform_y(self, transform_func, transform_scaler):
        """ Transform Y"""
        B, T, D = self.y.shape
        y_flat = self.y.reshape(-1, D)

        y_flat = transform_func(y_flat)
        if transform_scaler is not None:
            if self.split == "train":
                y_flat = transform_scaler.fit_transform(y_flat)
            else:
                y_flat = transform_scaler.transform(y_flat)

        self.y = y_flat.reshape(-1, T, D)

    def transform_to_predict_diff(self):
        # modify y to become a shift
        assert self.y.shape == self.X.shape

        self.y = self.y - self.X

    def save_to_file(self, filepath, save_scaler=True):
        """Save dataset arrays and metadata to an NPZ file."""
        logger.info(f"Serialize series object into: '{filepath}'")
        np.savez_compressed(
            filepath,
            # rescaled X,y
            X=self.X,
            y=self.y,
            # original X, y
            X_=self.X_,
            y_=self.y_,
            # genome ids for X,y
            xb_gid=np.array(self.xb_gid, dtype=object),
            yb_gid=np.array(self.yb_gid, dtype=object),
            #'train' or 'val' split label
            split=self.split,
            fix_context=self.fix_context
        )
        if save_scaler and self.scaler is not None:
            joblib.dump(self.scaler, str(filepath)+".scaler")

    def save_transform_scaler(self, filepath):
        if self.transform_scaler is not None:
            joblib.dump(self.transform_scaler, str(filepath)+".scaler")

    @classmethod
    def load_from_file(cls, filepath, load_scaler=True):
        """Load dataset arrays and metadata from file.

        Args:
            filepath: path to .npz file saved with `save_to_file()`
            scaler: optional scaler to attach
        """
        logger.info(f"Read series dataset from file: '{filepath}'")
        data = np.load(filepath, allow_pickle=True)
        obj = cls.__new__(cls)

        # Directly assign loaded attributes
        obj.X = data["X"]
        obj.y = data["y"]
        obj.X_ = data["X_"]
        obj.y_ = data["y_"]
        obj.xb_gid = data["xb_gid"].tolist()
        obj.yb_gid = data["yb_gid"].tolist()
        obj.split = str(data["split"])
        obj.fix_context = bool(data["fix_context"])

        # Optionally load scaler from a separate .scale
        scaler_path = str(filepath)+".scaler"
        logger.info(f"Load scaler from: '{scaler_path}'")
        if load_scaler and os.path.exists(scaler_path):
            obj.scaler = joblib_load_numpy2_compat(scaler_path)

        return obj

    @classmethod
    def load_transform_scaler(cls, filepath):
        # Optionally load scaler from a separate .scale
        scaler_path = str(filepath)+".scaler"
        logger.info(f"Load scaler from: '{scaler_path}'")
        if os.path.exists(scaler_path):
           scaler = joblib_load_numpy2_compat(scaler_path)
        else:
            scaler = None

        return scaler

class VectorAsSequenceDataset(Dataset):
    """ Class to construct a dataset for BiLSTM models.
        It presents each input data vector as a sequence for the model.
    """
    def __init__(self, x_data, y_data):
        # Convert to torch tensors if not already
        if not torch.is_tensor(x_data):
            x_data = torch.tensor(x_data, dtype=torch.float32)
        if not torch.is_tensor(y_data):
            y_data = torch.tensor(y_data, dtype=torch.float32)

        self.data = x_data         #[n_samples, seq_len, 1]
        self.targets = y_data      #[n_samples, seq_len, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Shape to [seq_len, 1] for LSTM
        x = self.data[idx].unsqueeze(-1)      # [seq_len, 1]
        y = self.targets[idx].unsqueeze(-1)   # [seq_len, 1]
        return x, y
