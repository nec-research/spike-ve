#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     spike_blocks.py
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
"""Spike protein domain block boundaries for MSA-aligned ESM pooling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_spike_block_boundaries(path: str | Path) -> dict[str, Any]:
    """Load ``spike_block_boundaries.yaml`` (or compatible config)."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Block boundaries config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    blocks = cfg.get("blocks") or []
    if not blocks:
        raise ValueError(f"No blocks defined in {cfg_path}")
    for block in blocks:
        for key in ("name", "msa_start", "msa_end"):
            if key not in block:
                raise ValueError(f"Block entry missing {key!r}: {block}")
    return cfg


def validate_block_boundaries(cfg: dict[str, Any], *, msa_length: int | None = None) -> None:
    """Check blocks are contiguous and within ``[0, msa_length)``."""
    blocks = cfg.get("blocks") or []
    if not blocks:
        raise ValueError("No blocks to validate")
    msa_len = int(msa_length if msa_length is not None else cfg.get("msa_length", 0))
    prev_end = None
    prev_name = None
    for block in blocks:
        start = int(block["msa_start"])
        end = int(block["msa_end"])
        name = str(block["name"])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid block range [{start}, {end}) for {name!r}")
        if msa_len and end > msa_len:
            raise ValueError(
                f"Block {name!r} end {end} exceeds msa_length={msa_len}"
            )
        if prev_end is not None and start != prev_end:
            raise ValueError(
                f"Non-contiguous blocks: gap/overlap between "
                f"{prev_name!r} and {name!r}"
            )
        prev_end = end
        prev_name = name


def block_output_dirname(block_name: str) -> str:
    """Directory name for one block under the encodings root."""
    return f"block_{block_name}"


BLOCK_TARGET_PREFIX = "block_"


def block_target_name(block_name: str) -> str:
    """Prediction / JEPA target key for one domain block mean."""
    return f"{BLOCK_TARGET_PREFIX}{block_name}"


def is_block_target(name: str) -> bool:
    return str(name).startswith(BLOCK_TARGET_PREFIX)


def parse_block_target_name(target: str) -> str:
    """``block_RBD_a`` → ``RBD_a``."""
    if not is_block_target(target):
        raise ValueError(f"Not a block target: {target!r}")
    return str(target)[len(BLOCK_TARGET_PREFIX) :]


def block_encodings_dir(
    primary_encodings_dir: str | Path,
    block_name: str,
    *,
    block_pooling: str = "average_token",
) -> Path:
    """Resolve block mean dir from global mean encodings dir.

    ``.../filtered_msa_stats/average_token/33``
    → ``.../filtered_msa_stats/block_RBD_a/average_token/33``
    """
    p = Path(primary_encodings_dir)
    layer = p.name
    root = p.parent.parent
    return root / block_output_dirname(str(block_name)) / str(block_pooling) / layer


DEFAULT_SPIKE_BLOCK_NAMES: tuple[str, ...] = (
    "NTD",
    "RBD_a",
    "RBD_b",
    "RBD_c",
    "S1_C",
    "S2",
)


def max_block_msa_end(cfg: dict[str, Any]) -> int:
    """Largest ``msa_end`` across all blocks in a boundaries config."""
    blocks = cfg.get("blocks") or []
    if not blocks:
        return 0
    return max(int(b["msa_end"]) for b in blocks)


def clamp_block_slice(
    msa_start: int,
    msa_end: int,
    truncate_len: int,
    *,
    block_name: str | None = None,
    warn_on_clamp: bool = True,
    _warned: set[str] | None = None,
) -> tuple[int, int] | None:
    """Clamp block ``[msa_start, msa_end)`` to the encoded AA length.

    Returns ``None`` when the block starts at or after ``truncate_len`` (no rows to pool).
    """
    start = int(msa_start)
    end = min(int(msa_end), int(truncate_len))
    if start >= end:
        return None
    if warn_on_clamp and end < int(msa_end):
        key = block_name or f"{start}:{msa_end}"
        warned = _warned if _warned is not None else set()
        if key not in warned:
            logger.warning(
                "Block %s slice clamped from [%s, %s) to [%s, %s) because "
                "truncate_len=%s (increase --seq-length to avoid this).",
                block_name or key,
                msa_start,
                msa_end,
                start,
                end,
                truncate_len,
            )
            warned.add(key)
    return start, end


def block_names_from_config(cfg: dict | None) -> list[str]:
    if not cfg:
        return []
    blocks = cfg.get("blocks") or []
    return [str(b["name"]) for b in blocks]


def block_boundaries_from_config(cfg: dict) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for b in cfg.get("blocks") or []:
        name = str(b["name"])
        out[name] = (int(b["msa_start"]), int(b["msa_end"]))
    return out


def _build_gid_index(enc_dir: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for pt in enc_dir.glob("*.pt"):
        try:
            rec = torch.load(pt, map_location="cpu", weights_only=False)
            label = rec.get("label", pt.stem)
            idx[str(label)] = pt
        except Exception:
            idx[pt.stem] = pt
    return idx


def _load_pooled_vector(pt_path: Path) -> np.ndarray:
    rec = torch.load(pt_path, map_location="cpu", weights_only=False)
    enc = rec["encodings"]
    if isinstance(enc, dict):
        enc = next(iter(enc.values()))
    arr = enc.detach().cpu().numpy() if hasattr(enc, "detach") else np.asarray(enc)
    return arr.astype(np.float32).reshape(-1)


def _resolve_encodings_dir(
    encodings_dir: Path,
    block_name: str,
    *,
    block_pooling: str,
    layer: int,
) -> Path:
    candidates = [
        encodings_dir / block_output_dirname(block_name) / block_pooling / str(layer),
        encodings_dir / "blocks" / block_name / block_pooling / str(layer),
        encodings_dir / block_name / block_pooling / str(layer),
    ]
    for p in candidates:
        if p.is_dir():
            return p
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No block encoding directory for block={block_name!r} under {encodings_dir}. "
        f"Tried: {tried}"
    )


def load_block_encodings_for_gids(
    gids: list[str],
    *,
    encodings_dir: Path,
    block_name: str,
    block_pooling: str = "average_token",
    layer: int = 33,
    gid_index: dict[str, Path] | None = None,
) -> np.ndarray:
    """Load ``[N, D]`` block-pooled encodings for genome ids."""
    enc_dir = _resolve_encodings_dir(
        encodings_dir, block_name, block_pooling=block_pooling, layer=layer
    )
    if gid_index is None:
        gid_index = _build_gid_index(enc_dir)
    rows = []
    missing = []
    for gid in gids:
        p = gid_index.get(str(gid))
        if p is None:
            missing.append(str(gid))
            continue
        rows.append(_load_pooled_vector(p))
    if missing:
        raise KeyError(
            f"Missing block={block_name!r} encodings for {len(missing)} gids "
            f"(first: {missing[:3]}) in {enc_dir}"
        )
    return np.stack(rows, axis=0)
