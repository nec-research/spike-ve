#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     unified_v3_lineage_substitution.py
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
"""Multi-draw B3 / unified-v3 inference via lineage-aware prefix edits.

Modes
-----
``last_step``
    Port of GPT lineage substitution: replace the last prefix genome with up to
    ``n_samples`` same-lineage BLOSUM-nearest candidates (date after penultimate).

``prefix_swap``
    Find donor **series** whose last-seen mean is L2-closest to the query's
    last-seen (same lineage). Build
    ``modified_prefix = donor[0:T-2] + query[T-2]``, then predict.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from recomb2027.data.seq_pairwise_distance import calculate_true_phylo_distance
from recomb2027.latent.data_series.horizon_collate import horizon_meta_to_tensor
from recomb2027.latent.data_series.msa_substitution_features import (
    PREFIX_FEATURE_DIM,
    PrefixSubstitutionState,
    build_sample_substitution_record,
    materialize_prefix_features,
    transition_substitutions,
)
from recomb2027.latent.data_series.pca2d_features import (
    concat_mean_pca2d_prefix_features,
)
from recomb2027.latent.inference.gpt_lineage_substitution import (
    LineageSubstitutionIndex,
    series_last_prefix_idx,
    to_gid_str,
)
from recomb2027.latent.predictors.train_residual import (
    _iter_leaf_datasets,
)
from recomb2027.latent.predictors.train_residual import (
    _select_step_tensor,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrefixSwapDonor:
    donor_split: str
    donor_series_idx: int
    distance: float
    lineage: str


@dataclass(frozen=True)
class SeriesRef:
    split: str
    series_idx: int
    leaf: Any


def unwrap_leaf(ds) -> Any:
    leaves = _iter_leaf_datasets(ds)
    if not leaves:
        raise ValueError("no leaf dataset found")
    return leaves[0]


def build_gid_latent_pool(
    *leaves: Any,
    pca2d_name: str = "pca2d_s1",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``gid → raw mean`` and ``gid → raw pca2d flat`` from series cubes."""
    mean_map: dict[str, np.ndarray] = {}
    pca_map: dict[str, np.ndarray] = {}
    for leaf in leaves:
        if leaf is None:
            continue
        enc = np.asarray(leaf.sample_encodings, dtype=np.float32)
        names = list(getattr(leaf, "get_pca2d_encoding_names", lambda: [])())
        key = None
        pca_all = None
        if names:
            key = pca2d_name if pca2d_name in names else names[0]
            pca_all = np.asarray(leaf.pca2d_encodings[key], dtype=np.float32)
        for i, row_gids in enumerate(leaf.sample_gids):
            for j, gid in enumerate(row_gids):
                g = to_gid_str(gid)
                mean_map[g] = enc[i, j].astype(np.float32)
                if pca_all is not None:
                    pca_map[g] = pca_all[i, j].astype(np.float32)
    return mean_map, pca_map


def scale_mean_vec(leaf, vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    if leaf.scaler is None:
        return v[0]
    cube = leaf._scale_mean_cube(v, leaf.scaler, fit=False)
    return np.asarray(cube, dtype=np.float32).reshape(-1)


def substitution_feats_for_prefix_gids(
    gids_prefix: Sequence[str],
    *,
    gid_to_seq: Mapping[str, str],
    seq_len: int,
) -> np.ndarray:
    """``[L, 5]`` features after observing all transitions inside the prefix."""
    gids = [to_gid_str(g) for g in gids_prefix]
    state = PrefixSubstitutionState.empty(seq_len)
    if len(gids) < 2:
        return materialize_prefix_features(state, seq_len=seq_len)
    for t in range(len(gids) - 1):
        rec = build_sample_substitution_record(
            [gids[t], gids[t + 1]],
            gid_to_seq=dict(gid_to_seq),
            seq_len=seq_len,
        )
        if rec["n_steps"] >= 1 and rec["target_subs"]:
            state.apply_transition(rec["target_subs"][0], step_idx=t)
        else:
            a = gid_to_seq.get(gids[t]) or ""
            b = gid_to_seq.get(gids[t + 1]) or ""
            if a and b:
                state.apply_transition(
                    transition_substitutions(a, b, seq_len=seq_len), step_idx=t
                )
    return materialize_prefix_features(state, seq_len=seq_len)


def edge_phylo_target_jump(
    gids_full: Sequence[str],
    *,
    last_idx: int,
    gid_to_seq: Mapping[str, str],
    use_log1p: bool,
) -> float:
    """Phylo distance on the predicted jump (last-seen → future), matching head edge."""
    gids = [to_gid_str(g) for g in gids_full]
    if last_idx + 1 >= len(gids):
        return 0.0
    a = gid_to_seq.get(gids[last_idx])
    b = gid_to_seq.get(gids[last_idx + 1])
    if not a or not b:
        return 0.0
    d = float(calculate_true_phylo_distance(a, b))
    return float(np.log1p(d)) if use_log1p else d


def horizon_for_target_jump(
    gids_full: Sequence[str],
    *,
    last_idx: int,
    gid_to_date: Mapping[str, pd.Timestamp],
    gid_to_seq: Mapping[str, str],
    cfg,
    use_log1p_edge: bool,
) -> torch.Tensor:
    """Last-step horizon vector for the predicted jump (matches eval collate)."""
    embed_dim = int(cfg.get("horizon_embed_dim", 4) or 4)
    graph_n = 1 + len(cfg.get("multi_horizon_paths") or [])
    gids = [to_gid_str(g) for g in gids_full]
    edge_tau = edge_phylo_target_jump(
        gids,
        last_idx=last_idx,
        gid_to_seq=gid_to_seq,
        use_log1p=use_log1p_edge,
    )
    dt_days = 0.0
    if last_idx + 1 < len(gids):
        d0 = gid_to_date.get(gids[last_idx], pd.NaT)
        d1 = gid_to_date.get(gids[last_idx + 1], pd.NaT)
        if not pd.isna(d0) and not pd.isna(d1):
            dt_days = max(0.0, float((d1 - d0).total_seconds()) / 86400.0)
    meta = {
        "edge_tau": float(edge_tau),
        "n_edges": 1,
        "dt_days": float(dt_days),
        "graph_id": 0,
    }
    return horizon_meta_to_tensor(meta, embed_dim=embed_dim, graph_n=graph_n)


def build_prefix_x_from_scaled(
    mean_scaled_full: np.ndarray,
    pca_scaled_full: np.ndarray,
    *,
    gatt_plus: bool = True,
    pca2d_gatt_plus: bool = True,
) -> np.ndarray:
    return concat_mean_pca2d_prefix_features(
        mean_scaled_full[:-1],
        pca_scaled_full[:-1],
        gatt_plus=gatt_plus,
        pca2d_gatt_plus=pca2d_gatt_plus,
    )


def build_prefix_swap_donor_refs(
    *,
    query_leaf,
    query_split_name: str,
    donor_leaves: Mapping[str, Any],
) -> list[SeriesRef]:
    """Collect donor series without double-counting the query split.

    When ``donor_leaves`` already includes the query leaf under
    ``query_split_name`` (e.g. ``--donor-pool train,cal,test``), every query
    series must appear once. Re-appending the query leaf duplicated slots and
    collapsed top-M clouds to ~M/2 unique donors.
    """
    refs: list[SeriesRef] = []
    for split_name, leaf in donor_leaves.items():
        if leaf is None:
            continue
        for j in range(len(leaf)):
            refs.append(SeriesRef(split=split_name, series_idx=j, leaf=leaf))
    query_already_in_pool = (
        query_split_name in donor_leaves
        and donor_leaves.get(query_split_name) is not None
    )
    if not query_already_in_pool:
        for j in range(len(query_leaf)):
            refs.append(
                SeriesRef(split=query_split_name, series_idx=j, leaf=query_leaf)
            )
    return refs


def plan_prefix_swap_donors(
    query_leaf,
    query_idx: int,
    donor_refs: Sequence[SeriesRef],
    *,
    gid_to_lineage: Mapping[str, str],
    n_samples: int,
) -> list[PrefixSwapDonor]:
    """Top-n same-lineage donors by L2 on scaled last-seen mean."""
    K = int(query_leaf.sample_encodings.shape[1])
    last_idx = series_last_prefix_idx(K)
    q_gids = [to_gid_str(g) for g in query_leaf.sample_gids[query_idx]]
    q_lin = str(gid_to_lineage.get(q_gids[last_idx], "")).strip()
    if not q_lin or q_lin.lower() in {"nan", "none"}:
        return []

    q_mean_raw = np.asarray(
        query_leaf.sample_encodings[query_idx, last_idx], dtype=np.float32
    )
    q_mean = scale_mean_vec(query_leaf, q_mean_raw)

    scored: list[PrefixSwapDonor] = []
    for ref in donor_refs:
        if ref.leaf is query_leaf and ref.series_idx == query_idx:
            continue
        d_gids = [to_gid_str(g) for g in ref.leaf.sample_gids[ref.series_idx]]
        if len(d_gids) != K:
            continue
        d_lin = str(gid_to_lineage.get(d_gids[last_idx], "")).strip()
        if d_lin != q_lin:
            continue
        d_mean_raw = np.asarray(
            ref.leaf.sample_encodings[ref.series_idx, last_idx], dtype=np.float32
        )
        # Scale with the query leaf's train scaler so distances are comparable.
        d_mean = scale_mean_vec(query_leaf, d_mean_raw)
        dist = float(
            np.linalg.norm(d_mean.astype(np.float64) - q_mean.astype(np.float64))
        )
        scored.append(
            PrefixSwapDonor(
                donor_split=ref.split,
                donor_series_idx=int(ref.series_idx),
                distance=dist,
                lineage=q_lin,
            )
        )
    scored.sort(key=lambda c: (c.distance, c.donor_split, c.donor_series_idx))
    return scored[: int(n_samples)]


def materialize_prefix_swap_series(
    query_leaf,
    query_idx: int,
    donor_leaf,
    donor_idx: int,
    *,
    pca2d_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Raw mean/pca2d/gids: donor[0:T-2] + query[T-2]; future = query future."""
    K = int(query_leaf.sample_encodings.shape[1])
    last_idx = series_last_prefix_idx(K)
    mean_q = np.asarray(query_leaf.sample_encodings[query_idx], dtype=np.float32).copy()
    mean_d = np.asarray(donor_leaf.sample_encodings[donor_idx], dtype=np.float32)
    names = list(query_leaf.get_pca2d_encoding_names())
    key = pca2d_name if pca2d_name in names else names[0]
    pca_q = np.asarray(
        query_leaf.pca2d_encodings[key][query_idx], dtype=np.float32
    ).copy()
    pca_d = np.asarray(donor_leaf.pca2d_encodings[key][donor_idx], dtype=np.float32)

    mean_out = mean_q.copy()
    pca_out = pca_q.copy()
    if last_idx > 0:
        mean_out[:last_idx] = mean_d[:last_idx]
        pca_out[:last_idx] = pca_d[:last_idx]
    mean_out[last_idx] = mean_q[last_idx]
    pca_out[last_idx] = pca_q[last_idx]

    gids_q = [to_gid_str(g) for g in query_leaf.sample_gids[query_idx]]
    gids_d = [to_gid_str(g) for g in donor_leaf.sample_gids[donor_idx]]
    gids_out = list(gids_q)
    if last_idx > 0:
        gids_out[:last_idx] = gids_d[:last_idx]
    gids_out[last_idx] = gids_q[last_idx]
    return mean_out, pca_out, gids_out


def materialize_last_step_series(
    query_leaf,
    query_idx: int,
    substitute_gid: str,
    *,
    mean_pool: Mapping[str, np.ndarray],
    pca_pool: Mapping[str, np.ndarray],
    pca2d_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    K = int(query_leaf.sample_encodings.shape[1])
    last_idx = series_last_prefix_idx(K)
    g = to_gid_str(substitute_gid)
    if g not in mean_pool or g not in pca_pool:
        raise KeyError(f"substitute {g} missing from latent pool")
    mean_out = np.asarray(query_leaf.sample_encodings[query_idx], dtype=np.float32).copy()
    names = list(query_leaf.get_pca2d_encoding_names())
    key = pca2d_name if pca2d_name in names else names[0]
    pca_out = np.asarray(
        query_leaf.pca2d_encodings[key][query_idx], dtype=np.float32
    ).copy()
    mean_out[last_idx] = mean_pool[g]
    pca_out[last_idx] = pca_pool[g]
    gids = [to_gid_str(x) for x in query_leaf.sample_gids[query_idx]]
    gids[last_idx] = g
    return mean_out, pca_out, gids


def _as_numpy_tensorish(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def unpack_wrapped_v3_item(
    item: Any,
    *,
    cfg,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Unpack one wrapped dataset item → ``(x, edge, horizon, subst)`` last-step."""
    if not isinstance(item, (list, tuple)):
        raise TypeError(f"unexpected dataset item type {type(item)}")
    parts = list(item)
    x = parts[0]
    xb = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    edge_phylo = horizon = substitution_feats = None
    embed_dim = int(cfg.get("horizon_embed_dim", 4) or 4)
    use_edge = bool(cfg.get("append_edge_phylo_to_head", False))
    use_subst = bool(cfg.get("append_msa_substitution", False))

    for part in parts[2:]:
        if isinstance(part, dict):
            horizon = horizon_meta_to_tensor(
                part,
                embed_dim=embed_dim,
                graph_n=1 + len(cfg.get("multi_horizon_paths") or []),
            ).unsqueeze(0).to(device)
            continue
        if isinstance(part, list):
            # all-step horizon metas, substitution step list, or activation masks
            if part and isinstance(part[0], dict):
                horizon = horizon_meta_to_tensor(
                    part[-1],
                    embed_dim=embed_dim,
                    graph_n=1 + len(cfg.get("multi_horizon_paths") or []),
                ).unsqueeze(0).to(device)
                continue
            if part and use_subst:
                feats = _as_numpy_tensorish(part[-1])
                # Substitution is [L, F]; target-activation masks are [L] — skip those.
                if feats.ndim == 2 and int(feats.shape[-1]) == int(PREFIX_FEATURE_DIM):
                    substitution_feats = (
                        torch.as_tensor(feats, dtype=torch.float32, device=device)
                        .unsqueeze(0)
                    )
                continue
            continue
        if isinstance(part, torch.Tensor):
            t = part
            if t.dim() == 1 and t.numel() == 1 and use_edge:
                edge_phylo = t.view(1, 1).to(device=device, dtype=torch.float32)
            elif t.dim() == 2 and t.shape[-1] == 1 and use_edge:
                # [T,1] all-step → last
                edge_phylo = t[-1:].to(device=device, dtype=torch.float32)
            elif t.dim() == 1 and t.numel() == embed_dim:
                horizon = t.unsqueeze(0).to(device=device, dtype=torch.float32)
            continue
        if isinstance(part, (float, int, np.floating, np.integer)) and use_edge:
            edge_phylo = torch.tensor(
                [[float(part)]], device=device, dtype=torch.float32
            )

    if edge_phylo is not None:
        edge_phylo = _select_step_tensor(edge_phylo, -1, last_only=True)
        if edge_phylo.dim() == 1:
            edge_phylo = edge_phylo.view(1, -1)
    if horizon is not None:
        horizon = _select_step_tensor(horizon, -1, last_only=True)
        if horizon.dim() == 1:
            horizon = horizon.unsqueeze(0)
    return xb, edge_phylo, horizon, substitution_feats


@torch.no_grad()
def forward_v3_tensors(
    model,
    xb: torch.Tensor,
    *,
    edge_phylo: Optional[torch.Tensor] = None,
    horizon: Optional[torch.Tensor] = None,
    substitution_feats: Optional[torch.Tensor] = None,
    absolute: bool = True,
    source_mean_scaled: Optional[np.ndarray] = None,
    source_pca2d_scaled: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    dist = model.predict_distribution(
        xb,
        horizon=horizon,
        edge_phylo=edge_phylo,
        substitution_feats=substitution_feats,
    )
    pred_mean = dist["pred_mean"][0].detach().cpu().numpy().astype(np.float32)
    pred_flat = dist["pred_flat"][0].detach().cpu().numpy().astype(np.float32)
    mu_flat = dist["mu_flat"][0].detach().cpu().numpy().astype(np.float32)
    out: dict[str, np.ndarray] = {
        "pred_mean_diff": pred_mean,
        "pred_pca2d_diff": pred_flat,
        "mu_pca2d_diff": mu_flat,
    }
    if "activation_probs" in dist:
        out["activation_probs"] = (
            dist["activation_probs"][0].detach().cpu().numpy().astype(np.float32)
        )
    if absolute:
        if source_mean_scaled is None or source_pca2d_scaled is None:
            raise ValueError("absolute preds require source_*_scaled arrays")
        out["pred_mean_abs"] = (source_mean_scaled + pred_mean).astype(np.float32)
        out["pred_pca2d_abs"] = (source_pca2d_scaled + pred_flat).astype(np.float32)
        out["mu_pca2d_abs"] = (source_pca2d_scaled + mu_flat).astype(np.float32)
        out["source_mean_scaled"] = np.asarray(source_mean_scaled, dtype=np.float32)
        out["source_pca2d_scaled"] = np.asarray(source_pca2d_scaled, dtype=np.float32)
    return out


@torch.no_grad()
def forward_v3_on_series(
    model,
    leaf,
    *,
    mean_raw_full: np.ndarray,
    pca_raw_full: np.ndarray,
    gids_full: Sequence[str],
    gid_to_seq: Mapping[str, str],
    gid_to_date: Mapping[str, pd.Timestamp],
    device: torch.device,
    cfg,
    pca2d_name: str = "pca2d_s1",
    absolute: bool = True,
) -> dict[str, np.ndarray]:
    """Run ``predict_distribution`` on one modified series."""
    if leaf.scaler is not None:
        mean_s = leaf._scale_mean_cube(
            np.asarray(mean_raw_full, dtype=np.float32), leaf.scaler, fit=False
        )
    else:
        mean_s = np.asarray(mean_raw_full, dtype=np.float32)
    pca_s = leaf._scale_pca2d_cube(np.asarray(pca_raw_full, dtype=np.float32), fit=False)
    x = build_prefix_x_from_scaled(
        mean_s,
        pca_s,
        gatt_plus=str(getattr(leaf, "input_feature_mode", "gatt_plus")) == "gatt_plus",
        pca2d_gatt_plus=bool(getattr(leaf, "pca2d_gatt_plus", True)),
    )
    xb = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)

    K = int(mean_raw_full.shape[0])
    last_idx = series_last_prefix_idx(K)
    gids_list = [to_gid_str(g) for g in gids_full]
    gids_prefix = gids_list[: last_idx + 1]
    use_log1p = bool(cfg.get("edge_phylo_use_log1p", True))

    edge_phylo = None
    if bool(cfg.get("append_edge_phylo_to_head", False)):
        ep = edge_phylo_target_jump(
            gids_list,
            last_idx=last_idx,
            gid_to_seq=gid_to_seq,
            use_log1p=use_log1p,
        )
        edge_phylo = torch.tensor([[ep]], device=device, dtype=torch.float32)

    horizon = None
    if bool(cfg.get("horizon_embedding", True)):
        horizon = horizon_for_target_jump(
            gids_list,
            last_idx=last_idx,
            gid_to_date=gid_to_date,
            gid_to_seq=gid_to_seq,
            cfg=cfg,
            use_log1p_edge=use_log1p,
        ).unsqueeze(0).to(device)

    substitution_feats = None
    if bool(cfg.get("append_msa_substitution", False)):
        seq_len = int(model.seq_len)
        feats = substitution_feats_for_prefix_gids(
            gids_prefix, gid_to_seq=gid_to_seq, seq_len=seq_len
        )
        substitution_feats = torch.from_numpy(feats).unsqueeze(0).to(device).float()

    return forward_v3_tensors(
        model,
        xb,
        edge_phylo=edge_phylo,
        horizon=horizon,
        substitution_feats=substitution_feats,
        absolute=absolute,
        source_mean_scaled=mean_s[last_idx],
        source_pca2d_scaled=pca_s[last_idx],
    )


def resolve_donor_leaf(
    donor: PrefixSwapDonor,
    *,
    donor_leaves: Mapping[str, Any],
    query_leaf,
    query_split_name: str,
) -> Any:
    """Resolve the dataset leaf for a planned donor.

    Prefer ``donor_leaves[split]`` when present so series indices always refer to
    the leaf that was scored. Fall back to ``query_leaf`` for same-split donors
    that were appended because the query split was omitted from the donor pool.
    """
    leaf = donor_leaves.get(donor.donor_split)
    if leaf is not None:
        return leaf
    if donor.donor_split == query_split_name:
        return query_leaf
    raise KeyError(f"donor split {donor.donor_split!r} not in donor_leaves")


@torch.no_grad()
def run_lineage_substitution_v3(
    model,
    *,
    query_leaf,
    donor_leaves: Mapping[str, Any],
    mode: str,
    n_samples: int,
    cfg,
    device: torch.device,
    gid_to_lineage: Mapping[str, str],
    gid_to_date: Mapping[str, pd.Timestamp],
    gid_to_seq: Mapping[str, str],
    pca2d_name: str = "pca2d_s1",
    absolute: bool = True,
    max_series: Optional[int] = None,
    query_split_name: str = "test",
    query_wrapped: Any = None,
) -> dict[str, Any]:
    """Run multi-draw substitution over all query series."""
    if mode not in ("last_step", "prefix_swap"):
        raise ValueError(f"unknown mode {mode!r}")
    n = len(query_leaf)
    if max_series is not None:
        n = min(n, int(max_series))
    K = int(query_leaf.sample_encodings.shape[1])
    last_idx = series_last_prefix_idx(K)
    mean_dim = int(query_leaf.sample_encodings.shape[-1])
    names = list(query_leaf.get_pca2d_encoding_names())
    key = pca2d_name if pca2d_name in names else names[0]
    pca_dim = int(query_leaf.pca2d_encodings[key].shape[-1])
    M = int(n_samples)

    pred_mean = np.full((n, M, mean_dim), np.nan, dtype=np.float32)
    pred_pca = np.full((n, M, pca_dim), np.nan, dtype=np.float32)
    baseline_mean = np.full((n, mean_dim), np.nan, dtype=np.float32)
    baseline_pca = np.full((n, pca_dim), np.nan, dtype=np.float32)
    target_mean = np.full((n, mean_dim), np.nan, dtype=np.float32)
    target_pca = np.full((n, pca_dim), np.nan, dtype=np.float32)
    distances = np.full((n, M), np.nan, dtype=np.float32)
    donor_series_idx = np.full((n, M), -1, dtype=np.int32)
    donor_split = np.full((n, M), "", dtype=object)
    substitute_gids = np.full((n, M), "", dtype=object)
    # Optional site-activation head (same layout as infer_unified_pca2d_site_v3).
    act_dim: Optional[int] = None
    baseline_act: Optional[np.ndarray] = None
    pred_act: Optional[np.ndarray] = None

    pool_leaves = [query_leaf] + [v for v in donor_leaves.values() if v is not None]
    mean_pool, pca_pool = build_gid_latent_pool(*pool_leaves, pca2d_name=key)

    donor_refs = build_prefix_swap_donor_refs(
        query_leaf=query_leaf,
        query_split_name=query_split_name,
        donor_leaves=donor_leaves,
    )

    sub_index = None
    if mode == "last_step":
        sub_index = LineageSubstitutionIndex(
            gid_to_lineage=gid_to_lineage,
            gid_to_date=gid_to_date,
            gid_to_seq=gid_to_seq,
            gid_to_encoding=mean_pool,
        )

    for i in tqdm(range(n), desc=f"v3 lineage-subst [{mode}]", unit="series"):
        mean_raw = np.asarray(query_leaf.sample_encodings[i], dtype=np.float32)
        pca_raw = np.asarray(query_leaf.pca2d_encodings[key][i], dtype=np.float32)
        gids = [to_gid_str(g) for g in query_leaf.sample_gids[i]]

        mean_s = (
            query_leaf._scale_mean_cube(mean_raw, query_leaf.scaler, fit=False)
            if query_leaf.scaler is not None
            else mean_raw
        )
        pca_s = query_leaf._scale_pca2d_cube(pca_raw, fit=False)
        target_mean[i] = mean_s[-1]
        target_pca[i] = pca_s[-1]
        src_mean = mean_s[last_idx]
        src_pca = pca_s[last_idx]

        if query_wrapped is not None:
            xb, edge_phylo, horizon, subst = unpack_wrapped_v3_item(
                query_wrapped[i], cfg=cfg, device=device
            )
            base = forward_v3_tensors(
                model,
                xb,
                edge_phylo=edge_phylo,
                horizon=horizon,
                substitution_feats=subst,
                absolute=absolute,
                source_mean_scaled=src_mean,
                source_pca2d_scaled=src_pca,
            )
        else:
            base = forward_v3_on_series(
                model,
                query_leaf,
                mean_raw_full=mean_raw,
                pca_raw_full=pca_raw,
                gids_full=gids,
                gid_to_seq=gid_to_seq,
                gid_to_date=gid_to_date,
                device=device,
                cfg=cfg,
                pca2d_name=key,
                absolute=absolute,
            )
        if absolute:
            baseline_mean[i] = base["pred_mean_abs"]
            baseline_pca[i] = base["pred_pca2d_abs"]
        else:
            baseline_mean[i] = base["pred_mean_diff"]
            baseline_pca[i] = base["pred_pca2d_diff"]
        if "activation_probs" in base:
            act = np.asarray(base["activation_probs"], dtype=np.float32).reshape(-1)
            if act_dim is None:
                act_dim = int(act.shape[0])
                baseline_act = np.full((n, act_dim), np.nan, dtype=np.float32)
                pred_act = np.full((n, M, act_dim), np.nan, dtype=np.float32)
            if baseline_act is not None and act.shape[0] == act_dim:
                baseline_act[i] = act

        if mode == "last_step":
            assert sub_index is not None
            cands = [
                c
                for c in sub_index.select_substitutes(gids, last_idx, M)
                if to_gid_str(c.gid) in pca_pool and to_gid_str(c.gid) in mean_pool
            ][:M]
            for slot, cand in enumerate(cands):
                try:
                    m_raw, p_raw, g_mod = materialize_last_step_series(
                        query_leaf,
                        i,
                        cand.gid,
                        mean_pool=mean_pool,
                        pca_pool=pca_pool,
                        pca2d_name=key,
                    )
                    pred = forward_v3_on_series(
                        model,
                        query_leaf,
                        mean_raw_full=m_raw,
                        pca_raw_full=p_raw,
                        gids_full=g_mod,
                        gid_to_seq=gid_to_seq,
                        gid_to_date=gid_to_date,
                        device=device,
                        cfg=cfg,
                        pca2d_name=key,
                        absolute=absolute,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("series %d slot %d failed: %s", i, slot, exc)
                    continue
                pred_mean[i, slot] = (
                    pred["pred_mean_abs"] if absolute else pred["pred_mean_diff"]
                )
                pred_pca[i, slot] = (
                    pred["pred_pca2d_abs"] if absolute else pred["pred_pca2d_diff"]
                )
                distances[i, slot] = float(cand.distance)
                substitute_gids[i, slot] = to_gid_str(cand.gid)
                if pred_act is not None and "activation_probs" in pred:
                    pred_act[i, slot] = np.asarray(
                        pred["activation_probs"], dtype=np.float32
                    ).reshape(-1)
        else:
            donors = plan_prefix_swap_donors(
                query_leaf,
                i,
                donor_refs,
                gid_to_lineage=gid_to_lineage,
                n_samples=M,
            )
            for slot, donor in enumerate(donors):
                try:
                    d_leaf = resolve_donor_leaf(
                        donor,
                        donor_leaves=donor_leaves,
                        query_leaf=query_leaf,
                        query_split_name=query_split_name,
                    )
                    m_raw, p_raw, g_mod = materialize_prefix_swap_series(
                        query_leaf,
                        i,
                        d_leaf,
                        donor.donor_series_idx,
                        pca2d_name=key,
                    )
                    pred = forward_v3_on_series(
                        model,
                        query_leaf,
                        mean_raw_full=m_raw,
                        pca_raw_full=p_raw,
                        gids_full=g_mod,
                        gid_to_seq=gid_to_seq,
                        gid_to_date=gid_to_date,
                        device=device,
                        cfg=cfg,
                        pca2d_name=key,
                        absolute=absolute,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("series %d donor slot %d failed: %s", i, slot, exc)
                    continue
                pred_mean[i, slot] = (
                    pred["pred_mean_abs"] if absolute else pred["pred_mean_diff"]
                )
                pred_pca[i, slot] = (
                    pred["pred_pca2d_abs"] if absolute else pred["pred_pca2d_diff"]
                )
                distances[i, slot] = float(donor.distance)
                donor_series_idx[i, slot] = int(donor.donor_series_idx)
                donor_split[i, slot] = donor.donor_split
                substitute_gids[i, slot] = to_gid_str(
                    d_leaf.sample_gids[donor.donor_series_idx][last_idx]
                )
                if pred_act is not None and "activation_probs" in pred:
                    pred_act[i, slot] = np.asarray(
                        pred["activation_probs"], dtype=np.float32
                    ).reshape(-1)

    out: dict[str, Any] = {
        "pred_samples_mean": pred_mean,
        "pred_samples_pca2d": pred_pca,
        "pred_baseline_mean": baseline_mean,
        "pred_baseline_pca2d": baseline_pca,
        "target_mean": target_mean,
        "target_pca2d": target_pca,
        "distances": distances,
        "donor_series_idx": donor_series_idx,
        "donor_split": donor_split,
        "substitute_gids": substitute_gids,
        "n_samples": M,
        "mode": mode,
        "last_idx": last_idx,
        "n_series": n,
    }
    if baseline_act is not None and pred_act is not None:
        out["pred_baseline_activation_probs"] = baseline_act
        out["pred_samples_activation_probs"] = pred_act
    return out
