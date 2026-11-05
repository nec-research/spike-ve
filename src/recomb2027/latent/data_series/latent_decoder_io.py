#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     latent_decoder_io.py
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
"""Load raw ESM1B pooled encodings and build decoder training pair tables."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_NAMES: tuple[str, ...] = (
    "NTD",
    "RBD_a",
    "RBD_b",
    "RBD_c",
    "S1_C",
    "S2",
)


def parse_msa_gids(path: str | Path) -> dict[str, str]:
    """Parse filtered MSA FASTA → ``{genome_id: aligned_sequence}``.

    Robust extraction: accepts key:value fields (genome_id, id, seqid, accession),
    EPI_ISL_* tokens, numeric dot ids, or falls back to the first header token.
    """
    path = Path(path)
    gid_to_seq: dict[str, str] = {}
    cur_gids: list[str] = []
    cur_parts: list[str] = []
    n_headers = 0
    log_every = 50_000
    logger.info("Parsing MSA FASTA: %s", path)

    id_key_re = re.compile(
        r"(?:genome_id|sequence_id|seqid|id|accession)[:=]\s*([^|]+)",
        flags=re.IGNORECASE,
    )
    epi_re = re.compile(r"(EPI[_-]ISL[_-]?\d+)", flags=re.IGNORECASE)
    numeric_dot_re = re.compile(r"(\d+\.\d+)")
    token_split_re = re.compile(r"[\|\s,;]+")

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                # flush previous record
                if cur_gids and cur_parts:
                    seq = "".join(cur_parts)
                    for gid in cur_gids:
                        gid_to_seq[gid] = seq
                header = line[1:]

                # try explicit key:value like genome_id:...
                m = id_key_re.search(header)
                cur_gids = []
                if m:
                    raw_val = m.group(1).strip()
                    parts = [p.strip() for p in raw_val.split(",") if p.strip()]
                    for p in parts:
                        # Prefer EPI_ISL tokens
                        mepi = epi_re.search(p)
                        if mepi:
                            cur_gids.append(mepi.group(1))
                            continue
                        # Prefer tokens that contain letters (preserve full accession like AAX12345.1)
                        if any(ch.isalpha() for ch in p):
                            cur_gids.append(p)
                            continue
                        # Fallback to numeric dot ids
                        mnum = numeric_dot_re.search(p)
                        if mnum:
                            cur_gids.append(mnum.group(1))
                            continue
                        # Last resort: keep raw token
                        cur_gids.append(p)
                else:
                    # no explicit key; try to find EPI_ISL or numeric ids anywhere
                    # Try to extract a sensible token: prefer EPI_ISL, then first token if it has letters,
                    # otherwise numeric suffix, otherwise first token.
                    mepi_any = epi_re.search(header)
                    if mepi_any:
                        cur_gids = [mepi_any.group(1)]
                    else:
                        # first token before separators
                        tok = token_split_re.split(header.strip())[0]
                        if ":" in tok:
                            tok = tok.split(":", 1)[1].strip()
                        if tok:
                            if any(ch.isalpha() for ch in tok):
                                cur_gids = [tok]
                            else:
                                mnum_any = numeric_dot_re.search(tok)
                                if mnum_any:
                                    cur_gids = [mnum_any.group(1)]
                                else:
                                    # fallback to token even if odd
                                    cur_gids = [tok]

                cur_parts = []
                n_headers += 1
                if n_headers % log_every == 0:
                    logger.info(
                        "  … parsed %d FASTA records (%d genome_ids so far)",
                        n_headers,
                        len(gid_to_seq),
                    )
            elif line:
                cur_parts.append(line)

    # flush last record
    if cur_gids and cur_parts:
        seq = "".join(cur_parts)
        for gid in cur_gids:
            gid_to_seq[gid] = seq

    logger.info(
        "MSA parse complete: %d FASTA records → %d genome_ids",
        n_headers,
        len(gid_to_seq),
    )
    return gid_to_seq


def parse_msa_collection_dates(path: str | Path) -> dict[str, str]:
    """Parse MSA FASTA headers → ``{genome_id: collection_date ISO string}``.

    Expects headers like ``...|genome_id:2697049.4653096|...|collection_date:2021-07-08|...``.
    Genome ids without a parseable date are omitted.
    """
    gid_to_date: dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            m_gid = re.search(r"genome_id:([^|]+)", header)
            m_date = re.search(r"collection_date:([^|]+)", header)
            if not m_gid or not m_date:
                continue
            date = m_date.group(1).strip()
            if not date or date.lower() in {"nan", "none", "unknown", ""}:
                continue
            for gid in re.findall(r"(\d+\.\d+)", m_gid.group(1)):
                gid_to_date[gid] = date
    return gid_to_date


def _gid_from_pt_name(name: str) -> str | None:
    m = re.search(r"genome_id_([\d]+\.[\d]+)", name)
    return m.group(1) if m else None


def resolve_genome_id(genome_id: str, keys: set[str] | dict[str, object]) -> str | None:
    """Resolve a (possibly truncated) genome id against encoding/MSA keys."""
    gid = str(genome_id).strip()
    if not gid or gid.lower() == "nan":
        return None
    key_set = keys if isinstance(keys, set) else set(keys)
    if gid in key_set:
        return gid

    prefix_matches = [key for key in key_set if key.startswith(gid)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    reverse_matches = [key for key in key_set if gid.startswith(key)]
    if len(reverse_matches) == 1:
        return reverse_matches[0]
    # Fallback: if gid contains a numeric suffix like 'AAX12345.1' try matching '12345.1'
    m = re.search(r'(\d+\.\d+)$', gid)
    if m:
        suf = m.group(1)
        if suf in key_set:
            return suf
    return None


def build_gid_pt_index(enc_dir: Path) -> dict[str, Path]:
    """Map genome_id → ``.pt`` path under one encoding directory."""
    index: dict[str, Path] = {}
    if not enc_dir.is_dir():
        raise FileNotFoundError(f"Encoding directory not found: {enc_dir}")
    for fpath in enc_dir.glob("*.pt"):
        gid = _gid_from_pt_name(fpath.name)
        if gid is not None:
            index.setdefault(gid, fpath)
    return index


def load_raw_pooled_pt(fpath: Path, *, layer: int) -> np.ndarray:
    import torch

    rec = torch.load(fpath, map_location="cpu", weights_only=False)
    enc = rec["encodings"]
    if isinstance(enc, dict):
        enc = enc[int(layer)]
    arr = enc.detach().cpu().numpy() if hasattr(enc, "detach") else np.asarray(enc)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 1280:
        raise ValueError(f"Expected 1280-d vector in {fpath}; got shape {arr.shape}")
    return arr


@dataclass
class RawEmbeddingStore:
    """In-memory raw mean + block vectors indexed by genome_id."""

    gids: np.ndarray
    gid_to_row: dict[str, int]
    mean: np.ndarray
    blocks: dict[str, np.ndarray] = field(default_factory=dict)
    embed_dim: int = 1280

    @classmethod
    def from_encodings_root(
        cls,
        encodings_root: str | Path,
        *,
        layer: int = 33,
        block_names: Sequence[str] = DEFAULT_BLOCK_NAMES,
        mean_pooling: str = "average_token",
    ) -> RawEmbeddingStore:
        root = Path(encodings_root)
        mean_dir = root / mean_pooling / str(layer)
        mean_index = build_gid_pt_index(mean_dir)
        if not mean_index:
            raise FileNotFoundError(f"No .pt files under {mean_dir}")

        block_indices: dict[str, dict[str, Path]] = {}
        for name in block_names:
            bdir = root / f"block_{name}" / mean_pooling / str(layer)
            block_indices[name] = build_gid_pt_index(bdir)

        common = set(mean_index)
        for name, idx in block_indices.items():
            common &= set(idx)
        gids = sorted(common)
        if not gids:
            raise RuntimeError("No genome_ids with mean + all block encodings")

        n = len(gids)
        d = 1280
        mean_mat = np.zeros((n, d), dtype=np.float32)
        block_mats = {name: np.zeros((n, d), dtype=np.float32) for name in block_names}
        gid_to_row = {g: i for i, g in enumerate(gids)}

        for i, gid in enumerate(gids):
            mean_mat[i] = load_raw_pooled_pt(mean_index[gid], layer=layer)
            for name in block_names:
                block_mats[name][i] = load_raw_pooled_pt(block_indices[name][gid], layer=layer)
            if (i + 1) % 20000 == 0:
                logger.info("Loaded %d / %d encodings", i + 1, n)

        logger.info(
            "RawEmbeddingStore: %d genomes, mean dir=%s, blocks=%s",
            n,
            mean_dir,
            list(block_names),
        )
        return cls(
            gids=np.asarray(gids, dtype=object),
            gid_to_row=gid_to_row,
            mean=mean_mat,
            blocks=block_mats,
            embed_dim=d,
        )

    @classmethod
    def from_gid_subset(
        cls,
        encodings_root: str | Path,
        gids: Sequence[str],
        *,
        layer: int = 33,
        block_names: Sequence[str] = DEFAULT_BLOCK_NAMES,
        mean_pooling: str = "average_token",
    ) -> RawEmbeddingStore:
        """Load raw pooled vectors only for ``gids`` (with prefix resolution)."""
        root = Path(encodings_root)
        mean_dir = root / mean_pooling / str(layer)
        mean_index = build_gid_pt_index(mean_dir)
        if not mean_index:
            raise FileNotFoundError(f"No .pt files under {mean_dir}")

        block_indices: dict[str, dict[str, Path]] = {}
        for name in block_names:
            bdir = root / f"block_{name}" / mean_pooling / str(layer)
            block_indices[name] = build_gid_pt_index(bdir)

        resolved: list[tuple[str, str]] = []
        all_keys = set(mean_index)
        for name, idx in block_indices.items():
            all_keys &= set(idx)
        for gid in gids:
            resolved_gid = resolve_genome_id(str(gid), all_keys)
            if resolved_gid is not None:
                resolved.append((str(gid), resolved_gid))
        if not resolved:
            raise RuntimeError("No genome_ids from subset resolved in encodings_root")

        uniq_resolved = sorted({r for _, r in resolved})
        n = len(uniq_resolved)
        d = 1280
        mean_mat = np.zeros((n, d), dtype=np.float32)
        block_mats = {name: np.zeros((n, d), dtype=np.float32) for name in block_names}
        gid_to_row = {g: i for i, g in enumerate(uniq_resolved)}

        for i, gid in enumerate(uniq_resolved):
            mean_mat[i] = load_raw_pooled_pt(mean_index[gid], layer=layer)
            for name in block_names:
                block_mats[name][i] = load_raw_pooled_pt(block_indices[name][gid], layer=layer)

        logger.info(
            "RawEmbeddingStore subset: %d genomes (from %d requested)",
            n,
            len(set(str(g) for g in gids)),
        )
        return cls(
            gids=np.asarray(uniq_resolved, dtype=object),
            gid_to_row=gid_to_row,
            mean=mean_mat,
            blocks=block_mats,
            embed_dim=d,
        )

    def row(self, gid: str) -> int:
        return self.gid_to_row[str(gid)]

    def mean_vec(self, gid: str) -> np.ndarray:
        return self.mean[self.row(gid)]

    def block_vec(self, name: str, gid: str) -> np.ndarray:
        return self.blocks[name][self.row(gid)]


def split_gids(
    gids: Sequence[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold-out split by hashed genome_id (stable, reproducible)."""
    rng = np.random.default_rng(seed)
    gids_arr = np.asarray(sorted(set(str(g) for g in gids)), dtype=object)
    perm = rng.permutation(len(gids_arr))
    n_val = max(1, int(round(len(gids_arr) * val_fraction)))
    val_set = set(gids_arr[perm[:n_val]].tolist())
    train = np.asarray([g for g in gids_arr if g not in val_set], dtype=object)
    val = np.asarray([g for g in gids_arr if g in val_set], dtype=object)
    return train, val


def sample_raw_pairs(
    store: RawEmbeddingStore,
    parent_gids: Sequence[str],
    child_gids: Sequence[str],
    n_pairs: int,
    *,
    seed: int = 42,
    max_attempts_factor: int = 50,
) -> dict[str, np.ndarray]:
    """Sample random (parent, child) pairs with parent ≠ child; raw Δ vectors."""
    rng = np.random.default_rng(seed)
    parent_pool = np.asarray(list(parent_gids), dtype=object)
    child_pool = np.asarray(list(child_gids), dtype=object)
    if len(parent_pool) == 0 or len(child_pool) < 2:
        raise ValueError("Need at least one parent gid and two child gids")

    parent_rows = np.array([store.gid_to_row[str(g)] for g in parent_pool], dtype=np.int64)
    child_rows_all = np.array([store.gid_to_row[str(g)] for g in child_pool], dtype=np.int64)

    n = int(n_pairs)
    out_parent = np.empty(n, dtype=object)
    out_child = np.empty(n, dtype=object)
    delta_mean = np.zeros((n, store.embed_dim), dtype=np.float32)
    child_mean = np.zeros((n, store.embed_dim), dtype=np.float32)
    delta_blocks = {name: np.zeros((n, store.embed_dim), dtype=np.float32) for name in store.blocks}
    child_blocks = {name: np.zeros((n, store.embed_dim), dtype=np.float32) for name in store.blocks}

    filled = 0
    attempts = 0
    max_attempts = n * max_attempts_factor
    while filled < n and attempts < max_attempts:
        attempts += 1
        pi = int(rng.integers(0, len(parent_pool)))
        ci = int(rng.integers(0, len(child_pool)))
        if parent_pool[pi] == child_pool[ci]:
            continue
        pr = parent_rows[pi]
        cr = child_rows_all[ci]
        out_parent[filled] = parent_pool[pi]
        out_child[filled] = child_pool[ci]
        dmean = store.mean[cr] - store.mean[pr]
        delta_mean[filled] = dmean
        child_mean[filled] = store.mean[cr]
        for name, mat in store.blocks.items():
            delta_blocks[name][filled] = mat[cr] - mat[pr]
            child_blocks[name][filled] = mat[cr]
        filled += 1

    if filled < n:
        raise RuntimeError(f"Could only sample {filled}/{n} pairs after {attempts} attempts")

    result: dict[str, np.ndarray] = {
        "parent_gid": out_parent,
        "child_gid": out_child,
        "delta_mean": delta_mean,
        "child_mean": child_mean,
        "delta_norm": np.linalg.norm(delta_mean, axis=1).astype(np.float32),
    }
    for name in store.blocks:
        result[f"delta_block_{name}"] = delta_blocks[name]
        result[f"child_block_{name}"] = child_blocks[name]
    return result


def save_pair_table(path: str | Path, table: dict[str, np.ndarray], *, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **table)
    if meta is not None:
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    logger.info("Wrote %d pairs → %s", len(table["parent_gid"]), path)


def load_pair_table(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _to_gid_str(gid) -> str:
    if isinstance(gid, bytes):
        return gid.decode()
    return str(gid)


def build_decoder_infer_pairs_from_series(
    series_gids: np.ndarray,
    store: RawEmbeddingStore,
    *,
    block_names: Sequence[str] | None = None,
    pair_mode: str = "last_edge",
    max_series: int | None = None,
) -> dict[str, np.ndarray]:
    """Build raw Δ tables from phylogenetic series rows + ``RawEmbeddingStore``.

    ``pair_mode``:
      - ``last_edge``: one (parent, child) per series row — parent = ``row[-2]``,
        child = ``row[-1]`` (matches ``infer_latent_decoder`` batch eval).
      - ``all_pairs``: every consecutive timestep along each series.
    """
    if pair_mode not in ("last_edge", "all_pairs"):
        raise ValueError(f"pair_mode must be 'last_edge' or 'all_pairs'; got {pair_mode!r}")
    if block_names is not None:
        names = list(block_names)
    elif store.blocks:
        names = list(store.blocks.keys())
    else:
        names = []
    d = int(store.embed_dim)

    parent_gids: list[str] = []
    child_gids: list[str] = []
    delta_mean_rows: list[np.ndarray] = []
    child_mean_rows: list[np.ndarray] = []
    delta_block_rows: dict[str, list[np.ndarray]] = {n: [] for n in names}
    child_block_rows: dict[str, list[np.ndarray]] = {n: [] for n in names}
    skipped = 0

    n_rows = len(series_gids)
    if max_series is not None:
        n_rows = min(n_rows, int(max_series))

    for si in range(n_rows):
        row = [_to_gid_str(g) for g in np.asarray(series_gids[si], dtype=object).ravel()]
        if len(row) < 2:
            skipped += 1
            continue
        if pair_mode == "last_edge":
            pairs = [(row[-2], row[-1])]
        else:
            pairs = [(row[t], row[t + 1]) for t in range(len(row) - 1)]

        for parent_gid, child_gid in pairs:
            if parent_gid == child_gid:
                skipped += 1
                continue
            if parent_gid not in store.gid_to_row or child_gid not in store.gid_to_row:
                skipped += 1
                continue
            pr = store.row(parent_gid)
            cr = store.row(child_gid)
            parent_gids.append(parent_gid)
            child_gids.append(child_gid)
            dm = store.mean[cr] - store.mean[pr]
            delta_mean_rows.append(dm.astype(np.float32))
            child_mean_rows.append(store.mean[cr].astype(np.float32))
            for name in names:
                if name not in store.blocks:
                    raise KeyError(f"block {name!r} missing from RawEmbeddingStore")
                blk = store.blocks[name]
                delta_block_rows[name].append((blk[cr] - blk[pr]).astype(np.float32))
                child_block_rows[name].append(blk[cr].astype(np.float32))

    if not parent_gids:
        raise RuntimeError(
            "No decoder infer pairs built (check encodings_root overlap with series gids)"
        )

    n = len(parent_gids)
    out: dict[str, np.ndarray] = {
        "parent_gid": np.asarray(parent_gids, dtype=object),
        "child_gid": np.asarray(child_gids, dtype=object),
        "delta_mean": np.stack(delta_mean_rows, axis=0).astype(np.float32),
        "child_mean": np.stack(child_mean_rows, axis=0).astype(np.float32),
        "delta_norm": np.linalg.norm(np.stack(delta_mean_rows, axis=0), axis=1).astype(np.float32),
        "n_skipped": np.asarray(skipped, dtype=np.int64),
    }
    for name in names:
        out[f"delta_block_{name}"] = np.stack(delta_block_rows[name], axis=0).astype(np.float32)
        out[f"child_block_{name}"] = np.stack(child_block_rows[name], axis=0).astype(np.float32)
    logger.info(
        "Built %d decoder infer pairs (mode=%s, skipped=%d)",
        n,
        pair_mode,
        skipped,
    )
    return out
