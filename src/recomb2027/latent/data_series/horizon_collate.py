#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     horizon_collate.py
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
"""Batch horizon metadata helpers for pca2d v3 trainers."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from recomb2027.latent.data_series.jepa_collate import make_dict_target_collate_fn


def horizon_meta_to_tensor(
    meta: dict[str, float | int],
    *,
    embed_dim: int = 4,
    graph_n: int = 1,
) -> torch.Tensor:
    """Map raw horizon dict to a fixed-size feature vector."""
    edge_tau = float(meta.get("edge_tau", 0.0))
    n_edges = float(meta.get("n_edges", 1.0))
    dt_days = float(meta.get("dt_days", 0.0))
    graph_id = int(meta.get("graph_id", 0) or 0)
    if embed_dim <= 1:
        return torch.tensor([edge_tau], dtype=torch.float32)
    out = torch.zeros(embed_dim, dtype=torch.float32)
    out[0] = edge_tau
    if embed_dim >= 2:
        out[1] = n_edges
    if embed_dim >= 3:
        out[2] = float(np.log1p(max(dt_days, 0.0)))
    if embed_dim >= 4 and graph_n > 1:
        gid = min(max(graph_id, 0), graph_n - 1)
        out[3] = float(gid) / float(max(graph_n - 1, 1))
    elif embed_dim >= 4:
        out[3] = float(graph_id)
    return out


def stack_horizon_batch(
    metas: list[dict[str, float | int]],
    *,
    embed_dim: int = 4,
    graph_n: int = 1,
) -> torch.Tensor:
    rows = [
        horizon_meta_to_tensor(m, embed_dim=embed_dim, graph_n=graph_n)
        for m in metas
    ]
    return torch.stack(rows, dim=0)


def stack_horizon_steps_batch(
    metas: list[list[dict[str, float | int]]],
    *,
    embed_dim: int = 4,
    graph_n: int = 1,
) -> torch.Tensor:
    """Stack per-sample step lists to ``[B, T, embed_dim]``."""
    rows = [
        stack_horizon_batch(step_list, embed_dim=embed_dim, graph_n=graph_n)
        for step_list in metas
    ]
    return torch.stack(rows, dim=0)


def _stack_horizon_metas(
    metas: list,
    *,
    embed_dim: int,
    graph_n: int,
) -> torch.Tensor:
    if not metas:
        raise ValueError("empty horizon meta batch")
    if isinstance(metas[0], list):
        return stack_horizon_steps_batch(
            metas, embed_dim=embed_dim, graph_n=graph_n
        )
    return stack_horizon_batch(metas, embed_dim=embed_dim, graph_n=graph_n)


def make_horizon_collate_fn(
    *,
    use_weights: bool = False,
    use_edge_phylo: bool = False,
    use_substitution: bool = False,
    use_target_activation: bool = False,
    use_drift_jump: bool = False,
    use_msa_tokens: bool = False,
    msa_seq_len: int = 712,
    horizon_embed_dim: int = 4,
    graph_n: int = 1,
):
    """Collate (x, y, meta) or (x, y, w, meta) / edge / sidecar variants.

    Optional dataset tail fields (outermost last when multiple are enabled)::

        (..., subst_steps, target_act, drift_jump, a0, a1)

    ``subst_steps``: list of ``[L, F]`` prefix features (or list-of-lists for all-step).
    ``target_act``: list of ``[L]`` binary masks.
    ``drift_jump``: list of dict rows with ``is_jump``, ``activation_mask``, etc.
    ``a0`` / ``a1``: ``int64 [L]`` parent/child MSA tokens (last hop), or lists thereof
    when ``all_step_loss`` is enabled. Requires ``use_msa_tokens=True``.
    """
    base_collate = make_dict_target_collate_fn()

    def _looks_like_token_array(x) -> bool:
        return (
            isinstance(x, np.ndarray)
            and x.ndim == 1
            and x.dtype.kind in "iu"
            and int(x.shape[0]) == int(msa_seq_len)
        )

    def _looks_like_token_steps(x) -> bool:
        return isinstance(x, list) and bool(x) and _looks_like_token_array(x[0])

    def _extract_core(batch_item):
        subst = None
        target_act = None
        drift_jump = None
        a0 = None
        a1 = None
        if isinstance(batch_item, (list, tuple)):
            items = list(batch_item)
            # extract msa tokens first (they appear as two trailing token arrays or lists of token arrays)
            if use_msa_tokens and len(items) >= 2:
                maybe_a1 = items[-1]
                maybe_a0 = items[-2]
                if _looks_like_token_array(maybe_a1) and _looks_like_token_array(maybe_a0):
                    a1 = items.pop()
                    a0 = items.pop()
                elif _looks_like_token_steps(maybe_a1) and _looks_like_token_steps(maybe_a0):
                    a1 = items.pop()
                    a0 = items.pop()
            # then extract drift/jump labels by scanning from the end to handle
            # cases where other trailing items may be present
            if use_drift_jump:
                for i in range(len(items) - 1, -1, -1):
                    maybe_dj = items[i]
                    if (
                        isinstance(maybe_dj, list)
                        and maybe_dj
                        and isinstance(maybe_dj[0], dict)
                        and "is_jump" in maybe_dj[0]
                    ):
                        drift_jump = items.pop(i)
                        break
            if use_target_activation and len(items) >= 2:
                maybe_tgt = items[-1]
                maybe_sub = items[-2]
                sub_ok = isinstance(maybe_sub, list) and (
                    not maybe_sub
                    or (
                        isinstance(maybe_sub[0], np.ndarray)
                        and maybe_sub[0].ndim == 2
                    )
                )
                tgt_ok = (
                    isinstance(maybe_tgt, list)
                    and (
                        not maybe_tgt
                        or (
                            isinstance(maybe_tgt[0], np.ndarray)
                            and maybe_tgt[0].ndim == 1
                        )
                    )
                ) or (
                    isinstance(maybe_tgt, np.ndarray) and maybe_tgt.ndim == 1
                )
                if sub_ok and tgt_ok:
                    target_act = maybe_tgt
                    subst = maybe_sub
                    items = items[:-2]
                elif use_substitution and sub_ok:
                    subst = maybe_sub
                    items = items[:-1]
            elif use_substitution and items and isinstance(items[-1], list):
                tail = items[-1]
                if not tail or (
                    isinstance(tail[0], np.ndarray) and tail[0].ndim == 2
                ):
                    subst = items.pop()
            batch_item = tuple(items)
        if len(batch_item) == 2:
            return batch_item[0], batch_item[1], None, None, None, subst, target_act, drift_jump, a0, a1
        if len(batch_item) == 3:
            a, b, c = batch_item
            if isinstance(c, dict):
                return a, b, None, None, c, subst, target_act, drift_jump, a0, a1
            if isinstance(c, torch.Tensor) and c.dim() == 0:
                return a, b, c, None, None, subst, target_act, drift_jump, a0, a1
            if isinstance(c, torch.Tensor) and c.numel() == 1:
                return a, b, None, c, None, subst, target_act, drift_jump, a0, a1
            return a, b, None, None, c, subst, target_act, drift_jump, a0, a1
        if len(batch_item) == 4:
            a, b, c, d = batch_item
            if isinstance(c, torch.Tensor) and c.dim() == 0:
                return a, b, c, d, None, subst, target_act, drift_jump, a0, a1
            if isinstance(d, dict):
                return a, b, c, None, d, subst, target_act, drift_jump, a0, a1
            return a, b, c, d, None, subst, target_act, drift_jump, a0, a1
        if len(batch_item) == 5:
            x, y, w, edge, meta = batch_item
            return x, y, w, edge, meta, subst, target_act, drift_jump, a0, a1
        raise ValueError(f"Unexpected batch item length {len(batch_item)}")

    def _stack_substitution(substs: list) -> torch.Tensor:
        """Stack per-sample step lists → ``[B, T, L, F]`` or ``[B, L, F]``."""
        if not substs:
            raise ValueError("empty substitution batch")
        if isinstance(substs[0], list):
            max_t = max(len(s) for s in substs)
            rows = []
            for steps in substs:
                if not steps:
                    raise ValueError("empty substitution step list")
                L, F = steps[0].shape
                buf = np.zeros((max_t, L, F), dtype=np.float32)
                for t, arr in enumerate(steps):
                    buf[t] = np.asarray(arr, dtype=np.float32)
                if len(steps) < max_t:
                    buf[len(steps) :] = buf[len(steps) - 1]
                rows.append(buf)
            return torch.from_numpy(np.stack(rows, axis=0))
        return torch.stack(
            [torch.as_tensor(s, dtype=torch.float32) for s in substs], dim=0
        )

    def _stack_target_activation(masks: list) -> torch.Tensor:
        if isinstance(masks[0], list):
            max_t = max(len(s) for s in masks)
            rows = []
            for steps in masks:
                L = steps[0].shape[0] if steps else 0
                buf = np.zeros((max_t, L), dtype=np.float32)
                for t, arr in enumerate(steps):
                    buf[t] = np.asarray(arr, dtype=np.float32)
                if steps and len(steps) < max_t:
                    buf[len(steps) :] = buf[len(steps) - 1]
                rows.append(buf)
            return torch.from_numpy(np.stack(rows, axis=0))
        return torch.stack(
            [torch.as_tensor(s, dtype=torch.float32) for s in masks], dim=0
        )

    def _neutral_drift_jump_step(L: int, *, n_templates: int = 0) -> dict:
        row = {
            "is_jump": 0.0,
            "is_saltation": 0.0,
            "sample_weight": 1.0,
            "activation_mask": np.zeros(L, dtype=np.float32),
        }
        if n_templates > 0:
            row["template_soft"] = np.zeros(n_templates, dtype=np.float32)
            row["template_mask"] = np.zeros(L, dtype=np.float32)
            row["residual_mask"] = np.zeros(L, dtype=np.float32)
        return row

    def _infer_n_templates(steps_batch: list) -> int:
        for steps in steps_batch:
            if steps and "template_soft" in steps[0]:
                return int(len(steps[0]["template_soft"]))
        return 0

    def _stack_drift_jump(steps_batch: list) -> tuple:
        max_t = max(len(s) for s in steps_batch)
        n_templates = _infer_n_templates(steps_batch)
        meta_rows = []
        mask_rows = []
        soft_rows = []
        tmpl_mask_rows = []
        residual_mask_rows = []
        for steps in steps_batch:
            L = steps[0]["activation_mask"].shape[0] if steps else 0
            meta = np.zeros((max_t, 3), dtype=np.float32)
            mk = np.zeros((max_t, L), dtype=np.float32)
            soft = (
                np.zeros((max_t, n_templates), dtype=np.float32)
                if n_templates > 0
                else None
            )
            tmpl_mk = (
                np.zeros((max_t, L), dtype=np.float32) if n_templates > 0 else None
            )
            res_mk = (
                np.zeros((max_t, L), dtype=np.float32) if n_templates > 0 else None
            )
            for t, row in enumerate(steps):
                meta[t, 0] = float(row["is_jump"])
                meta[t, 1] = float(row["is_saltation"])
                meta[t, 2] = float(row["sample_weight"])
                mk[t] = np.asarray(row["activation_mask"], dtype=np.float32)
                if n_templates > 0:
                    soft[t] = np.asarray(row.get("template_soft"), dtype=np.float32)
                    tmpl_mk[t] = np.asarray(row.get("template_mask"), dtype=np.float32)
                    res_mk[t] = np.asarray(row.get("residual_mask"), dtype=np.float32)
            if steps and len(steps) < max_t:
                meta[len(steps) :] = meta[len(steps) - 1]
                mk[len(steps) :] = mk[len(steps) - 1]
                if n_templates > 0:
                    soft[len(steps) :] = soft[len(steps) - 1]
                    tmpl_mk[len(steps) :] = tmpl_mk[len(steps) - 1]
                    res_mk[len(steps) :] = res_mk[len(steps) - 1]
            meta_rows.append(meta)
            mask_rows.append(mk)
            if n_templates > 0:
                soft_rows.append(soft)
                tmpl_mask_rows.append(tmpl_mk)
                residual_mask_rows.append(res_mk)
        out = (
            torch.from_numpy(np.stack(meta_rows, axis=0)),
            torch.from_numpy(np.stack(mask_rows, axis=0)),
        )
        if n_templates > 0:
            out = out + (
                torch.from_numpy(np.stack(soft_rows, axis=0)),
                torch.from_numpy(np.stack(tmpl_mask_rows, axis=0)),
                torch.from_numpy(np.stack(residual_mask_rows, axis=0)),
            )
        return out

    def _fill_drift_jump_batch(steps_batch: list) -> list:
        """Ensure one drift/jump step-list per sample (batch-aligned)."""
        valid = next((steps for steps in steps_batch if steps), None)
        if valid is None:
            raise ValueError("drift/jump collate: no labels found in batch")
        L = int(valid[0]["activation_mask"].shape[0])
        n_templates = _infer_n_templates(steps_batch)
        max_t = max(len(steps or ()) or 1 for steps in steps_batch)
        filled: list = []
        for steps in steps_batch:
            if not steps:
                steps = [
                    _neutral_drift_jump_step(L, n_templates=n_templates)
                    for _ in range(max_t)
                ]
            elif len(steps) < max_t:
                pad = steps[-1]
                steps = list(steps) + [pad] * (max_t - len(steps))
            filled.append(steps)
        return filled

    def _append_drift_jump_tensors(out: list[Any], drift_jumps: list) -> None:
        stacked = _stack_drift_jump(_fill_drift_jump_batch(drift_jumps))
        out.append(stacked[0])
        out.append(stacked[1])
        if len(stacked) > 2:
            out.extend(stacked[2:])

    def _stack_msa_tokens(a0_batch: list, a1_batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(a0_batch[0], list):
            max_t = max(len(s) for s in a0_batch)
            a0_rows = []
            a1_rows = []
            for a0_steps, a1_steps in zip(a0_batch, a1_batch):
                buf0 = np.zeros((max_t, msa_seq_len), dtype=np.int64)
                buf1 = np.zeros((max_t, msa_seq_len), dtype=np.int64)
                for t, (a0, a1) in enumerate(zip(a0_steps, a1_steps)):
                    buf0[t] = np.asarray(a0, dtype=np.int64)
                    buf1[t] = np.asarray(a1, dtype=np.int64)
                if a0_steps and len(a0_steps) < max_t:
                    buf0[len(a0_steps) :] = buf0[len(a0_steps) - 1]
                    buf1[len(a1_steps) :] = buf1[len(a1_steps) - 1]
                a0_rows.append(buf0)
                a1_rows.append(buf1)
            return (
                torch.from_numpy(np.stack(a0_rows, axis=0)),
                torch.from_numpy(np.stack(a1_rows, axis=0)),
            )
        return (
            torch.stack(
                [torch.as_tensor(a, dtype=torch.long) for a in a0_batch], dim=0
            ),
            torch.stack(
                [torch.as_tensor(a, dtype=torch.long) for a in a1_batch], dim=0
            ),
        )

    def collate(batch):
        xs, ys, weights, edges, metas, substs, targets, drift_jumps, a0s, a1s = [], [], [], [], [], [], [], [], [], []
        for item in batch:
            x, y, w, edge, meta, subst, target_act, drift_jump, a0, a1 = _extract_core(item)
            xs.append(x)
            ys.append(y)
            if w is not None:
                weights.append(w)
            if edge is not None:
                edges.append(edge)
            if meta is not None:
                metas.append(meta)
            if subst is not None:
                substs.append(subst)
            if target_act is not None:
                targets.append(target_act)
            if use_drift_jump:
                drift_jumps.append(drift_jump)
            if a0 is not None:
                a0s.append(a0)
            if a1 is not None:
                a1s.append(a1)
        xb, yb = base_collate(list(zip(xs, ys)))
        out: list[Any] = [xb, yb]
        if weights:
            out.append(torch.stack(weights))
        if edges:
            out.append(torch.stack(edges))
        if metas:
            out.append(
                _stack_horizon_metas(
                    metas, embed_dim=horizon_embed_dim, graph_n=graph_n
                )
            )
        if substs:
            out.append(_stack_substitution(substs))
        if use_drift_jump:
            if len(drift_jumps) != len(xs):
                raise ValueError(
                    f"drift/jump collate: expected {len(xs)} labels, got {len(drift_jumps)}"
                )
            _append_drift_jump_tensors(out, drift_jumps)
        if targets:
            out.append(_stack_target_activation(targets))
        if a0s:
            a0b, a1b = _stack_msa_tokens(a0s, a1s)
            out.append(a0b)
            out.append(a1b)
        return tuple(out)

    if (
        not use_weights
        and not use_edge_phylo
        and not use_substitution
        and not use_target_activation
        and not use_drift_jump
        and not use_msa_tokens
    ):
        def collate_simple(batch):
            xs, ys, metas, substs, targets, drift_jumps, a0s, a1s = [], [], [], [], [], [], [], []
            for item in batch:
                x, y, w, edge, meta, subst, target_act, drift_jump, a0, a1 = _extract_core(item)
                xs.append(x)
                ys.append(y)
                if meta is not None:
                    metas.append(meta)
                if subst is not None:
                    substs.append(subst)
                if target_act is not None:
                    targets.append(target_act)
                if use_drift_jump:
                    drift_jumps.append(drift_jump)
                if a0 is not None:
                    a0s.append(a0)
                if a1 is not None:
                    a1s.append(a1)
            xb, yb = base_collate(list(zip(xs, ys)))
            out: list[Any] = [xb, yb]
            if metas:
                out.append(
                    _stack_horizon_metas(
                        metas, embed_dim=horizon_embed_dim, graph_n=graph_n
                    )
                )
            if substs:
                out.append(_stack_substitution(substs))
            if use_drift_jump:
                if len(drift_jumps) != len(xs):
                    raise ValueError(
                        f"drift/jump collate: expected {len(xs)} labels, got {len(drift_jumps)}"
                    )
                _append_drift_jump_tensors(out, drift_jumps)
            if targets:
                out.append(_stack_target_activation(targets))
            if a0s:
                a0b, a1b = _stack_msa_tokens(a0s, a1s)
                out.append(a0b)
                out.append(a1b)
            return tuple(out)

        return collate_simple

    return collate


def horizon_from_batch(
    batch,
    *,
    cfg,
    device: torch.device,
    step_index: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Extract horizon tensor from collated batch; fallback to config constant."""
    if isinstance(batch, (list, tuple)):
        embed_dim = int(cfg.get("horizon_embed_dim", 4) or 4)
        for item in reversed(batch):
            if not isinstance(item, torch.Tensor) or item.dim() not in (2, 3):
                continue
            if item.dim() == 2 and item.shape[1] == 1:
                continue
            if item.dim() == 2 and item.shape[1] == embed_dim:
                return item.to(device)
            if item.dim() == 3 and item.shape[2] == embed_dim:
                if step_index is not None:
                    return item[:, int(step_index), :].to(device)
                return item[:, -1, :].to(device)

    if not bool(cfg.get("horizon_embedding", True)):
        return None
    embed_dim = int(cfg.get("horizon_embed_dim", 4) or 4)
    if embed_dim <= 0:
        return None
    batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else 1
    tau = float(cfg.get("horizon_tau", 1.0))
    if embed_dim == 1:
        return torch.full((batch_size, 1), tau, device=device, dtype=torch.float32)
    out = torch.zeros(batch_size, embed_dim, device=device, dtype=torch.float32)
    idx = min(int(tau) - 1, embed_dim - 1)
    out[:, idx] = 1.0
    return out
