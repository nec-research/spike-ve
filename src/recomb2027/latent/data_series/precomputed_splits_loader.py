#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     precomputed_splits_loader.py
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
"""Load precomputed train/cal/test k-series (e.g. drift/jump builder pickle)."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from .series_factory import SeriesFactory

logger = logging.getLogger(__name__)


def resolve_precomputed_splits_path(cfg) -> Optional[Path]:
    """Resolve optional precomputed splits pickle from dataset config."""
    if not bool(getattr(cfg, "use_precomputed_splits", False)):
        return None
    for key in ("precomputed_splits_path", "drift_jump_splits_path"):
        raw = getattr(cfg, key, None)
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if path.is_file():
            return path
        logger.warning("use_precomputed_splits: %s not found: %s", key, path)
    return None


def load_precomputed_splits(path: str | Path) -> Dict[str, Any]:
    """Load train/cal/val k-series lists from a pickle file.

    Supports:

    * ``drift_jump_splits.pkl`` from ``run_usher_mat_drift_jump`` (uses ``test``)
    * Standard ``SeriesSplitter.write_splits`` output (uses ``val``)
    """
    path = Path(path)
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"precomputed splits must be a dict, got {type(payload)}")

    train = list(payload.get("train") or [])
    cal = list(payload.get("cal") or [])
    val = payload.get("val")
    if val is None:
        val = payload.get("test")
    val = list(val or [])

    out: Dict[str, Any] = {
        "train": train,
        "cal": cal,
        "val": val,
        "split_stats": dict(payload.get("split_stats") or {}),
        "decontamination": dict(payload.get("decontamination") or {}),
        "source_path": str(path),
    }
    if "drift_k_series" in payload:
        out["n_drift_k_series"] = len(payload["drift_k_series"])
    if "jump_k_series" in payload:
        out["n_jump_k_series"] = len(payload["jump_k_series"])
    logger.info(
        "Loaded precomputed splits from %s: train=%d cal=%d val=%d",
        path,
        len(train),
        len(cal),
        len(val),
    )
    return out


def apply_precomputed_split_filters(
    loaded: Dict[str, Any],
    *,
    encodings_dir: str,
    filter_encodings: bool = True,
    filter_last_element: bool = False,
) -> Dict[str, Any]:
    """Optionally filter precomputed split lists before NPZ export."""
    train = list(loaded["train"])
    cal = list(loaded["cal"])
    val = list(loaded["val"])
    stats: Dict[str, Any] = {
        "train_before": len(train),
        "cal_before": len(cal),
        "val_before": len(val),
    }

    if filter_encodings:
        available = SeriesFactory.scan_available_genome_ids(encodings_dir)
        train = SeriesFactory.filter_series_list_by_encodings(
            train, encodings_dir, available_ids=available
        )
        cal = SeriesFactory.filter_series_list_by_encodings(
            cal, encodings_dir, available_ids=available
        )
        val = SeriesFactory.filter_series_list_by_encodings(
            val, encodings_dir, available_ids=available
        )

    if filter_last_element:
        combined = train + cal + val
        factory = SeriesFactory.__new__(SeriesFactory)
        factory.lst_k_series = combined
        filtered = factory.filter_k_series_by_last_element(inplace=False)
        filtered_set = {tuple(row) for row in filtered}
        train = [row for row in train if tuple(row) in filtered_set]
        cal = [row for row in cal if tuple(row) in filtered_set]
        val = [row for row in val if tuple(row) in filtered_set]

    stats.update(
        {
            "train_after": len(train),
            "cal_after": len(cal),
            "val_after": len(val),
            "filter_encodings": filter_encodings,
            "filter_last_element": filter_last_element,
        }
    )
    logger.info(
        "Precomputed split filters: train %d→%d cal %d→%d val %d→%d",
        stats["train_before"],
        stats["train_after"],
        stats["cal_before"],
        stats["cal_after"],
        stats["val_before"],
        stats["val_after"],
    )
    return {
        **loaded,
        "train": train,
        "cal": cal,
        "val": val,
        "filter_stats": stats,
    }
