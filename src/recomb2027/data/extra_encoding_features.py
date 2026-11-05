#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     extra_encoding_features.py
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
"""Optional ESM stat blocks concatenated to pooled-mean embeddings for model inputs.

Primary latent (``mean`` / ``average_token``) drives targets, velocities, and the
fitted ``StandardScaler``.  Extra stats (``std``, ``median``, ``argmax_pos``, …)
are appended per timestep without temporal differencing.

Scaling policy
--------------
* ``std``, ``median`` (and other value stats if enabled): same ``StandardScaler``
  as the mean embedding.
* ``argmax_pos``, ``argmin_pos``: **not** scaled (already in ``[0, 1]`` when
  encoded with position normalisation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.preprocessing import StandardScaler

from recomb2027.data.esm_pooling import (
    EXTRA_STAT_NAMES,
    POSITION_STAT_NAMES,
    VALUE_STAT_NAMES,
)
from recomb2027.data.spike_blocks import (
    block_target_name,
    is_block_target,
    parse_block_target_name,
)

# Stats selectable as *extra* channels (mean is the primary NPZ embedding).
ALLOWED_EXTRA_STATS: tuple[str, ...] = EXTRA_STAT_NAMES

# Extra stats that reuse the mean embedding scaler (user requirement: std, median).
# Block means use the same scaler as the global mean embedding.
STATS_SCALE_WITH_MEAN: frozenset[str] = frozenset({"std", "median"})

PRIMARY_POOLING_DIRNAME = "average_token"


def parse_extra_encoding_stats(
    raw: Union[None, str, Sequence[str], Sequence[Sequence[str]]],
) -> list[str]:
    """Normalise config / CLI into an ordered, deduplicated stat list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [s.strip() for s in raw.replace(",", " ").split() if s.strip()]
    elif raw and isinstance(raw[0], (list, tuple)):
        items = [str(s).strip() for s in raw[0]]
    else:
        items = [str(s).strip() for s in raw]
    out: list[str] = []
    seen: set[str] = set()
    for s in items:
        if not s:
            continue
        if s not in ALLOWED_EXTRA_STATS:
            raise ValueError(
                f"Unknown extra_encoding_stat {s!r}; choose from {list(ALLOWED_EXTRA_STATS)}"
            )
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def stat_encodings_dir(primary_encodings_dir: Union[str, Path], stat: str) -> Path:
    """Resolve directory for one extra stat given the mean-pooling path.

    ``.../average_token/33`` → ``.../std/33`` (replaces pooling folder name).
    """
    if stat not in ALLOWED_EXTRA_STATS:
        raise ValueError(f"stat must be one of {ALLOWED_EXTRA_STATS}; got {stat!r}")
    p = Path(primary_encodings_dir)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part == PRIMARY_POOLING_DIRNAME:
            parts[i] = stat
            return Path(*parts)
    return p.parent.parent / stat / p.name


def extra_encoding_width(primary_dim: int, extra_stats: Sequence[str]) -> int:
    """Total width of extra stat channels (each stat contributes ``primary_dim``)."""
    return int(primary_dim) * len(extra_stats)


def parse_block_names(
    raw: Union[None, str, Sequence[str], Sequence[Sequence[str]]],
) -> list[str]:
    """Normalise block name list from config."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [s.strip() for s in raw.replace(",", " ").split() if s.strip()]
    elif raw and isinstance(raw[0], (list, tuple)):
        items = [str(s).strip() for s in raw[0]]
    else:
        items = [str(s).strip() for s in raw]
    out: list[str] = []
    seen: set[str] = set()
    for name in items:
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def block_encoding_width(primary_dim: int, block_names: Sequence[str]) -> int:
    return int(primary_dim) * len(block_names)


def expected_model_input_dim(
    *,
    primary_dim: int,
    extra_stats: Sequence[str] = (),
    block_names: Sequence[str] = (),
    gatt_plus: bool = False,
    append_edge_phylo: bool = False,
    feature_mode: str = "absolute",
) -> int:
    """Input feature width for LSTM / GPT / gated models.

    ``feature_mode`` is one of ``absolute``, ``gatt_plus`` / ``diff``, or gated
    decomposition modes that only extend the primary block with velocity.
    """
    d = int(primary_dim)
    extra_w = extra_encoding_width(d, extra_stats)
    block_w = block_encoding_width(d, block_names)
    fm = "gatt_plus" if gatt_plus else feature_mode
    if fm in ("gatt_plus", "diff"):
        width = 2 * d + extra_w + block_w
    elif fm in ("mag_scalar", "direction_magnitude"):
        width = 2 * d + 1 + extra_w + block_w
    else:
        width = d + extra_w + block_w
    if append_edge_phylo:
        width += 1
    return width


def scale_block_array(
    enc: np.ndarray,
    scaler: Optional[StandardScaler],
    *,
    fit: bool = False,
) -> np.ndarray:
    """Scale block mean cube with a dedicated block ``StandardScaler`` when provided."""
    arr = np.asarray(enc, dtype=np.float32)
    if scaler is None:
        return arr.astype(np.float32)
    dims = arr.shape
    flat = arr.reshape(-1, dims[-1])
    if fit:
        flat = scaler.fit_transform(flat)
    else:
        flat = scaler.transform(flat)
    return flat.reshape(dims).astype(np.float32)


def scale_extra_stat_array(
    enc: np.ndarray,
    stat: str,
    scaler: Optional[StandardScaler],
    *,
    fit: bool = False,
) -> np.ndarray:
    """Scale one stat cube ``[..., D]``; position stats pass through unchanged."""
    arr = np.asarray(enc, dtype=np.float32)
    if stat in STATS_SCALE_WITH_MEAN and scaler is not None:
        dims = arr.shape
        flat = arr.reshape(-1, dims[-1])
        if fit:
            flat = scaler.fit_transform(flat)
        else:
            flat = scaler.transform(flat)
        return flat.reshape(dims).astype(np.float32)
    return arr.astype(np.float32)


def scale_mean_and_extra_cubes(
    mean_enc: np.ndarray,
    extra_by_stat: dict[str, np.ndarray],
    scaler: StandardScaler,
    *,
    fit_mean: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Scale mean cube with ``scaler``; apply same scaler to value extra stats."""
    dims = mean_enc.shape
    flat = mean_enc.reshape(-1, dims[-1])
    if fit_mean:
        mean_out = scaler.fit_transform(flat).reshape(dims).astype(np.float32)
    else:
        mean_out = scaler.transform(flat).reshape(dims).astype(np.float32)
    extra_out = {
        stat: scale_extra_stat_array(enc, stat, scaler, fit=False)
        for stat, enc in extra_by_stat.items()
    }
    return mean_out, extra_out


def concat_prefix_step_features(
    x_abs_primary: np.ndarray,
    extra_steps: Optional[dict[str, np.ndarray]],
    extra_stats: Sequence[str],
    *,
    input_feature_mode: str = "absolute",
    edge_feat: Optional[np.ndarray] = None,
    block_steps: Optional[dict[str, np.ndarray]] = None,
    block_names: Sequence[str] = (),
) -> np.ndarray:
    """Build per-step model input from primary absolutes + optional extra stats.

    Velocities are computed **only** on the primary (mean) block.
    """
    x_abs = np.asarray(x_abs_primary, dtype=np.float32)
    fm = input_feature_mode
    if fm in ("gatt_plus", "diff"):
        velocity = np.zeros_like(x_abs)
        if x_abs.shape[0] > 1:
            velocity[1:] = x_abs[1:] - x_abs[:-1]
        x = np.concatenate([x_abs, velocity], axis=-1)
    elif fm == "absolute":
        x = x_abs
    else:
        raise ValueError(
            f"concat_prefix_step_features does not handle feature_mode={fm!r}; "
            "use AugmentedDataset for mag_scalar / direction_magnitude."
        )

    if extra_stats and extra_steps:
        parts = []
        for stat in extra_stats:
            if stat not in extra_steps:
                raise KeyError(f"extra step missing stat {stat!r}")
            parts.append(np.asarray(extra_steps[stat], dtype=np.float32))
        if parts:
            x = np.concatenate([x, *parts], axis=-1)

    if block_names and block_steps:
        parts = []
        for name in block_names:
            if name not in block_steps:
                raise KeyError(f"block step missing {name!r}")
            parts.append(np.asarray(block_steps[name], dtype=np.float32))
        if parts:
            x = np.concatenate([x, *parts], axis=-1)

    if edge_feat is not None:
        ef = np.asarray(edge_feat, dtype=np.float32).reshape(-1, 1)
        x = np.concatenate([x, ef], axis=-1)
    return x.astype(np.float32)


def extra_stats_npz_keys(stats: Sequence[str]) -> list[str]:
    return [f"extra_enc_{s}" for s in stats]


def block_enc_npz_keys(block_names: Sequence[str]) -> list[str]:
    return [f"block_enc_{name}" for name in block_names]


def triple_branch_input_layout(
    *,
    primary_dim: int,
    extra_stats: Sequence[str],
    block_names: Sequence[str] = (),
    gatt_plus: bool = False,
    append_edge_phylo: bool = False,
) -> dict[str, Optional[Tuple[int, int]]]:
    """Channel slices for triple-branch GPT input (primary / position / block).

    Block means are appended after extra stats and before optional edge phylo.
    """
    dual = dual_branch_input_layout(
        primary_dim=primary_dim,
        extra_stats=extra_stats,
        gatt_plus=gatt_plus,
        append_edge_phylo=False,
    )
    d = int(primary_dim)
    offset = dual["position_slice"][1]

    block_start = offset
    block_end = block_start + block_encoding_width(d, block_names)
    offset = block_end

    edge_phylo_slice = None
    if append_edge_phylo:
        edge_phylo_slice = (offset, offset + 1)

    return {
        "primary_slice": dual["primary_slice"],
        "position_slice": dual["position_slice"],
        "block_slice": (block_start, block_end),
        "edge_phylo_slice": edge_phylo_slice,
    }


def normalise_triple_branch_layout(
    raw: Union[None, dict[str, Sequence[int]]],
) -> dict[str, Optional[Tuple[int, int]]]:
    """Normalise ``triple_branch_layout`` from config / checkpoint hparams."""
    if not raw:
        return {
            "primary_slice": (0, 0),
            "position_slice": (0, 0),
            "block_slice": (0, 0),
            "edge_phylo_slice": None,
        }
    primary = tuple(int(x) for x in raw["primary_slice"])
    position = tuple(int(x) for x in raw["position_slice"])
    block = tuple(int(x) for x in raw["block_slice"])
    edge = raw.get("edge_phylo_slice")
    edge_slice = None if edge is None else (int(edge[0]), int(edge[1]))
    return {
        "primary_slice": (primary[0], primary[1]),
        "position_slice": (position[0], position[1]),
        "block_slice": (block[0], block[1]),
        "edge_phylo_slice": edge_slice,
    }


def serialise_triple_branch_layout(
    layout: dict[str, Optional[Tuple[int, int]]],
) -> dict[str, Optional[list[int]]]:
    out: dict[str, Optional[list[int]]] = {
        "primary_slice": list(layout["primary_slice"]),
        "position_slice": list(layout["position_slice"]),
        "block_slice": list(layout["block_slice"]),
    }
    edge = layout.get("edge_phylo_slice")
    out["edge_phylo_slice"] = None if edge is None else list(edge)
    return out


def extra_stat_block_slices(
    *,
    primary_dim: int,
    extra_stats: Sequence[str],
    gatt_plus: bool = False,
) -> dict[str, tuple[int, int]]:
    """Half-open channel slice ``[start, end)`` for each extra stat in model input."""
    d = int(primary_dim)
    offset = 2 * d if gatt_plus else d
    return {
        str(stat): (offset + i * d, offset + (i + 1) * d)
        for i, stat in enumerate(extra_stats)
    }


def position_stat_projection_slices(
    *,
    primary_dim: int,
    extra_stats: Sequence[str],
    gatt_plus: bool = False,
) -> dict[str, tuple[int, int]]:
    """Slices for ``argmax_pos`` / ``argmin_pos`` blocks only (for learned projection)."""
    blocks = extra_stat_block_slices(
        primary_dim=primary_dim,
        extra_stats=extra_stats,
        gatt_plus=gatt_plus,
    )
    return {s: blocks[s] for s in extra_stats if s in POSITION_STAT_NAMES and s in blocks}


def normalise_position_stat_slices(
    raw: Union[None, dict, Sequence[Sequence[Union[str, int]]]],
) -> dict[str, tuple[int, int]]:
    """Accept ``{stat: [start, end]}`` or ``[[stat, start, end], ...]`` from config / hparams."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): (int(v[0]), int(v[1])) for k, v in raw.items()}
    out: dict[str, tuple[int, int]] = {}
    for row in raw:
        stat, start, end = row[0], int(row[1]), int(row[2])
        out[str(stat)] = (start, end)
    return out


def dual_branch_input_layout(
    *,
    primary_dim: int,
    extra_stats: Sequence[str],
    gatt_plus: bool = False,
    append_edge_phylo: bool = False,
) -> dict[str, Optional[Tuple[int, int]]]:
    """Channel slices for dual-branch GPT input (primary vs position-stat blocks).

    Primary branch: mean (+ velocity if ``gatt_plus``) plus any value extra stats
    (``std``, ``median``, …) that appear **before** the first position stat in
    ``extra_stats`` order.  Position branch: contiguous ``argmax_pos`` / ``argmin_pos``
    blocks.  Optional edge phylo is appended on the primary branch at forward time.
    """
    d = int(primary_dim)
    base = 2 * d if gatt_plus else d
    offset = base
    primary_end = base
    position_start: Optional[int] = None
    position_end = base

    for stat in extra_stats:
        if stat in POSITION_STAT_NAMES:
            if position_start is None:
                position_start = offset
            position_end = offset + d
        else:
            primary_end = offset + d
        offset += d

    if position_start is None:
        position_start = primary_end
        position_end = primary_end

    edge_phylo_slice = None
    if append_edge_phylo:
        edge_phylo_slice = (offset, offset + 1)

    return {
        "primary_slice": (0, primary_end),
        "position_slice": (position_start, position_end),
        "edge_phylo_slice": edge_phylo_slice,
    }


def normalise_dual_branch_layout(
    raw: Union[None, dict[str, Sequence[int]]],
) -> dict[str, Optional[Tuple[int, int]]]:
    """Normalise ``dual_branch_layout`` from config / checkpoint hparams."""
    if not raw:
        return {
            "primary_slice": (0, 0),
            "position_slice": (0, 0),
            "edge_phylo_slice": None,
        }
    primary = tuple(int(x) for x in raw["primary_slice"])
    position = tuple(int(x) for x in raw["position_slice"])
    edge = raw.get("edge_phylo_slice")
    edge_slice = None if edge is None else (int(edge[0]), int(edge[1]))
    return {
        "primary_slice": (primary[0], primary[1]),
        "position_slice": (position[0], position[1]),
        "edge_phylo_slice": edge_slice,
    }


def serialise_dual_branch_layout(
    layout: dict[str, Optional[Tuple[int, int]]],
) -> dict[str, Optional[list[int]]]:
    """JSON/checkpoint-friendly layout dict."""
    out: dict[str, Optional[list[int]]] = {
        "primary_slice": list(layout["primary_slice"]),
        "position_slice": list(layout["position_slice"]),
    }
    edge = layout.get("edge_phylo_slice")
    out["edge_phylo_slice"] = None if edge is None else list(edge)
    return out
