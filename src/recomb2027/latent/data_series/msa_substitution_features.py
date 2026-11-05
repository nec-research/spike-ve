#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     msa_substitution_features.py
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
"""Per-sample sparse MSA substitution sidecar for pca2d residual heads (B1).

Builds prefix-only substitution state (no leakage from target genomes) and
records target-transition Hamming distances for validation.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from tqdm import tqdm

from recomb2027.latent.data_series.latent_decoder_io import (
    parse_msa_gids,
    resolve_genome_id,
)
from recomb2027.latent.data_series.stem_predecessor import resolve_n_jobs

logger = logging.getLogger(__name__)

# Worker globals for multiprocessing (set via ``_init_b1_worker``).
_WORKER_GID_TO_SEQ: Optional[dict[str, str]] = None
_WORKER_SEQ_LEN: int = 0

AA_ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {c: i for i, c in enumerate(AA_ALPHABET)}
PREFIX_FEATURE_DIM = 5


def aa_to_index(aa: str) -> int:
    return AA_TO_IDX.get(str(aa).upper(), 0)


def hamming_distance(seq_a: str, seq_b: str, *, seq_len: Optional[int] = None) -> int:
    """Count differing non-gap positions between two aligned sequences."""
    L = seq_len if seq_len is not None else min(len(seq_a), len(seq_b))
    dist = 0
    for i in range(L):
        a = seq_a[i] if i < len(seq_a) else "-"
        b = seq_b[i] if i < len(seq_b) else "-"
        if a in "-." or b in "-.":
            continue
        if a.upper() != b.upper():
            dist += 1
    return dist


def transition_substitutions(
    seq_from: str,
    seq_to: str,
    *,
    seq_len: Optional[int] = None,
) -> list[tuple[int, int, int]]:
    """Return ``[(site, from_aa_idx, to_aa_idx), ...]`` for one transition."""
    L = seq_len if seq_len is not None else min(len(seq_from), len(seq_to))
    out: list[tuple[int, int, int]] = []
    for i in range(L):
        a = seq_from[i] if i < len(seq_from) else "-"
        b = seq_to[i] if i < len(seq_to) else "-"
        # include indels (gap '.' or '-') as legitimate transitions;
        # map gaps/unknowns to index 0 via aa_to_index and record any change.
        if a.upper() != b.upper():
            out.append((i, aa_to_index(a), aa_to_index(b)))
    return out


@dataclass
class PrefixSubstitutionState:
    """Cumulative prefix substitution state at end of prefix ``[0..t]``."""

    cumulative_count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    steps_since_last: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    last_from: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    last_to: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    substituted_last: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))

    @classmethod
    def empty(cls, seq_len: int) -> "PrefixSubstitutionState":
        L = int(seq_len)
        return cls(
            cumulative_count=np.zeros(L, dtype=np.int32),
            steps_since_last=np.full(L, 255, dtype=np.int32),
            last_from=np.zeros(L, dtype=np.uint8),
            last_to=np.zeros(L, dtype=np.uint8),
            substituted_last=np.zeros(L, dtype=np.uint8),
        )

    def apply_transition(
        self,
        subs: Sequence[tuple[int, int, int]],
        *,
        step_idx: int,
    ) -> None:
        """Update state after observing prefix transition ``step_idx``."""
        self.steps_since_last = np.minimum(self.steps_since_last + 1, 255)
        self.substituted_last.fill(0)
        for site, from_idx, to_idx in subs:
            self.cumulative_count[site] += 1
            self.steps_since_last[site] = 0
            self.last_from[site] = int(from_idx)
            self.last_to[site] = int(to_idx)
            self.substituted_last[site] = 1


def materialize_prefix_features(
    state: PrefixSubstitutionState,
    *,
    seq_len: int,
) -> np.ndarray:
    """Dense ``[L, 5]`` float32: sub_last, cum_count, steps_since, from, to."""
    L = int(seq_len)
    st = state if state.cumulative_count.size == L else PrefixSubstitutionState.empty(L)
    out = np.zeros((L, 5), dtype=np.float32)
    out[:, 0] = st.substituted_last.astype(np.float32)
    out[:, 1] = st.cumulative_count.astype(np.float32)
    out[:, 2] = st.steps_since_last.astype(np.float32)
    out[:, 3] = st.last_from.astype(np.float32)
    out[:, 4] = st.last_to.astype(np.float32)
    return out


def _resolve_gid(gid: str, gid_to_seq: dict[str, str]) -> Optional[str]:
    return resolve_genome_id(gid, gid_to_seq)


def build_sample_substitution_record(
    sample_gids: Sequence[str],
    *,
    gid_to_seq: dict[str, str],
    seq_len: int,
    active_indices: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Build per-step prefix + target substitution record for one series.

    ``active_indices`` defaults to full timeline ``0..T-1``.
    """
    gids = [str(g) for g in sample_gids]
    T = len(gids) - 1
    if T <= 0:
        return {
            "n_steps": 0,
            "transition_hamming": np.zeros(0, dtype=np.float32),
            "prefix_states": [],
            "target_subs": [],
        }

    active = list(active_indices) if active_indices is not None else list(range(len(gids)))
    if len(active) < 2:
        return {
            "n_steps": 0,
            "transition_hamming": np.zeros(0, dtype=np.float32),
            "prefix_states": [],
            "target_subs": [],
        }

    resolved = [_resolve_gid(gids[i], gid_to_seq) for i in active]
    seqs = [gid_to_seq[r] if r else "" for r in resolved]

    n_trans = len(active) - 1
    hamming = np.zeros(n_trans, dtype=np.float32)
    prefix_states: list[dict[str, np.ndarray]] = []
    target_subs: list[list[tuple[int, int, int]]] = []

    state = PrefixSubstitutionState.empty(seq_len)
    prefix_states.append(
        {
            "sub_last": state.substituted_last.copy(),
            "cum_count": state.cumulative_count.copy(),
            "steps_since": state.steps_since_last.copy(),
            "last_from": state.last_from.copy(),
            "last_to": state.last_to.copy(),
        }
    )

    for t in range(n_trans):
        seq_from = seqs[t]
        seq_to = seqs[t + 1]
        if not seq_from or not seq_to:
            hamming[t] = float("nan")
            subs: list[tuple[int, int, int]] = []
        else:
            hamming[t] = float(hamming_distance(seq_from, seq_to, seq_len=seq_len))
            subs = transition_substitutions(seq_from, seq_to, seq_len=seq_len)

        target_subs.append(subs)

        if seq_from and seq_to:
            state.apply_transition(subs, step_idx=t)
        prefix_states.append(
            {
                "sub_last": state.substituted_last.copy(),
                "cum_count": state.cumulative_count.copy(),
                "steps_since": state.steps_since_last.copy(),
                "last_from": state.last_from.copy(),
                "last_to": state.last_to.copy(),
            }
        )

    return {
        "n_steps": n_trans,
        "transition_hamming": hamming,
        "prefix_states": prefix_states,
        "target_subs": target_subs,
    }


def _pack_sparse_records(records: list[dict[str, Any]], *, seq_len: int) -> dict[str, Any]:
    """Pack list of per-sample records into compressed NPZ-friendly arrays."""
    n = len(records)
    max_steps = max((int(r["n_steps"]) for r in records), default=0)
    hamming = np.full((n, max(max_steps, 1)), np.nan, dtype=np.float32)

    entry_ptr = [0]
    entry_sample: list[int] = []
    entry_step: list[int] = []
    sites_p: list[int] = []
    from_p: list[int] = []
    to_p: list[int] = []
    cum_p: list[int] = []
    since_p: list[int] = []
    sub_last_p: list[int] = []

    target_entry_ptr = [0]
    target_entry_sample: list[int] = []
    target_entry_step: list[int] = []
    sites_t: list[int] = []
    from_t: list[int] = []
    to_t: list[int] = []

    for si, rec in enumerate(records):
        h = rec["transition_hamming"]
        if h.size:
            hamming[si, : h.size] = h

        for step_idx, state in enumerate(rec["prefix_states"]):
            entry_sample.append(int(si))
            entry_step.append(int(step_idx))
            active = np.union1d(
                np.flatnonzero(state["sub_last"]),
                np.flatnonzero(state["cum_count"]),
            )
            for site in active:
                sites_p.append(int(site))
                from_p.append(int(state["last_from"][site]))
                to_p.append(int(state["last_to"][site]))
                cum_p.append(int(state["cum_count"][site]))
                since_p.append(min(int(state["steps_since"][site]), 255))
                sub_last_p.append(int(state["sub_last"][site]))
            entry_ptr.append(len(sites_p))

        for step_idx, subs in enumerate(rec["target_subs"]):
            if not subs:
                continue
            target_entry_sample.append(int(si))
            target_entry_step.append(int(step_idx))
            for site, f_idx, t_idx in subs:
                sites_t.append(int(site))
                from_t.append(int(f_idx))
                to_t.append(int(t_idx))
            target_entry_ptr.append(len(sites_t))

    return {
        "n_samples": np.int32(n),
        "max_steps": np.int32(max_steps),
        "seq_len": np.int32(seq_len),
        "transition_hamming": hamming,
        "prefix_entry_ptr": np.asarray(entry_ptr, dtype=np.int64),
        "prefix_entry_sample": np.asarray(entry_sample, dtype=np.int32),
        "prefix_entry_step": np.asarray(entry_step, dtype=np.uint8),
        "prefix_site": np.asarray(sites_p, dtype=np.int16),
        "prefix_from": np.asarray(from_p, dtype=np.uint8),
        "prefix_to": np.asarray(to_p, dtype=np.uint8),
        "prefix_cum_count": np.asarray(cum_p, dtype=np.uint16),
        "prefix_steps_since": np.asarray(since_p, dtype=np.uint8),
        "prefix_sub_last": np.asarray(sub_last_p, dtype=np.uint8),
        "target_entry_ptr": np.asarray(target_entry_ptr, dtype=np.int64),
        "target_entry_sample": np.asarray(target_entry_sample, dtype=np.int32),
        "target_entry_step": np.asarray(target_entry_step, dtype=np.uint8),
        "target_site": np.asarray(sites_t, dtype=np.int16),
        "target_from": np.asarray(from_t, dtype=np.uint8),
        "target_to": np.asarray(to_t, dtype=np.uint8),
    }


def _init_b1_worker(gid_to_seq: dict[str, str], seq_len: int) -> None:
    global _WORKER_GID_TO_SEQ, _WORKER_SEQ_LEN
    _WORKER_GID_TO_SEQ = gid_to_seq
    _WORKER_SEQ_LEN = int(seq_len)


def _build_records_chunk(rows: list[list[str]]) -> tuple[list[dict[str, Any]], list[float]]:
    if _WORKER_GID_TO_SEQ is None:
        raise RuntimeError("B1 worker not initialized")
    records: list[dict[str, Any]] = []
    hamming: list[float] = []
    for row in rows:
        rec = build_sample_substitution_record(
            row,
            gid_to_seq=_WORKER_GID_TO_SEQ,
            seq_len=_WORKER_SEQ_LEN,
        )
        records.append(rec)
        for h in rec["transition_hamming"]:
            if h == h:
                hamming.append(float(h))
    return records, hamming


def _chunk_sample_rows(
    sample_gids: np.ndarray,
    *,
    chunk_size: int,
) -> list[list[list[str]]]:
    rows = [list(row) for row in sample_gids]
    if not rows:
        return []
    size = max(1, int(chunk_size))
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def _build_records_parallel(
    sample_gids: np.ndarray,
    *,
    gid_to_seq: dict[str, str],
    seq_len: int,
    n_jobs: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    n_samples = int(sample_gids.shape[0])
    n_workers = resolve_n_jobs(n_jobs)
    chunk_size = max(64, min(4000, n_samples // max(n_workers * 8, 1)))
    chunks = _chunk_sample_rows(sample_gids, chunk_size=chunk_size)
    logger.info(
        "B1 sidecar: parallel build with %d workers, %d chunks (~%d series/chunk)",
        n_workers,
        len(chunks),
        chunk_size,
    )
    records: list[dict[str, Any]] = []
    all_hamming: list[float] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_b1_worker,
        initargs=(gid_to_seq, seq_len),
    ) as pool:
        for recs, hams in tqdm(
            pool.map(_build_records_chunk, chunks, chunksize=1),
            total=len(chunks),
            desc="B1 prefix chunks",
            unit="chunk",
            mininterval=2.0,
        ):
            records.extend(recs)
            all_hamming.extend(hams)
    return records, all_hamming


def build_substitution_sidecar_from_gids(
    sample_gids: np.ndarray,
    *,
    msa_path: Optional[Union[str, Path]] = None,
    gid_to_seq: Optional[dict[str, str]] = None,
    seq_len: int,
    n_jobs: int = 1,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build sidecar arrays + summary stats from ``sample_gids`` ``[N, series_len]``."""
    n_samples = int(sample_gids.shape[0])
    if gid_to_seq is None:
        if msa_path is None:
            raise ValueError("Either msa_path or gid_to_seq must be provided")
        msa_path = Path(msa_path)
        logger.info(
            "B1 sidecar: parsing MSA FASTA (%s); this may take several minutes…",
            msa_path,
        )
        t0 = time.perf_counter()
        gid_to_seq = parse_msa_gids(msa_path)
        logger.info(
            "B1 sidecar: MSA loaded — %d genome_ids in %.1fs",
            len(gid_to_seq),
            time.perf_counter() - t0,
        )
    if not gid_to_seq:
        src = msa_path if msa_path is not None else "<gid_to_seq>"
        raise ValueError(f"No sequences available for sidecar build: {src}")

    n_workers = resolve_n_jobs(n_jobs)
    logger.info(
        "B1 sidecar: building prefix substitution records for %d series "
        "(seq_len=%d, workers=%d)…",
        n_samples,
        seq_len,
        n_workers,
    )
    t1 = time.perf_counter()
    if n_workers <= 1 or n_samples < 256:
        records: list[dict[str, Any]] = []
        all_hamming: list[float] = []
        for row in tqdm(
            sample_gids,
            desc="B1 prefix records",
            unit="series",
            mininterval=2.0,
        ):
            rec = build_sample_substitution_record(
                list(row), gid_to_seq=gid_to_seq, seq_len=seq_len
            )
            records.append(rec)
            for h in rec["transition_hamming"]:
                if h == h:
                    all_hamming.append(float(h))
    else:
        records, all_hamming = _build_records_parallel(
            sample_gids,
            gid_to_seq=gid_to_seq,
            seq_len=seq_len,
            n_jobs=n_workers,
        )
    logger.info(
        "B1 sidecar: built %d records, %d transitions in %.1fs",
        len(records),
        len(all_hamming),
        time.perf_counter() - t1,
    )

    logger.info("B1 sidecar: packing sparse arrays…")
    t2 = time.perf_counter()
    packed = _pack_sparse_records(records, seq_len=seq_len)
    logger.info("B1 sidecar: packed sparse arrays in %.1fs", time.perf_counter() - t2)
    stats = {
        "n_samples": float(len(records)),
        "seq_len": float(seq_len),
        "n_transitions": float(len(all_hamming)),
        "hamming_mean": float(np.mean(all_hamming)) if all_hamming else float("nan"),
        "hamming_median": float(np.median(all_hamming)) if all_hamming else float("nan"),
        "hamming_p95": float(np.percentile(all_hamming, 95)) if all_hamming else float("nan"),
    }
    return packed, stats


def save_substitution_sidecar(
    payload: dict[str, Any],
    path: Union[str, Path],
    *,
    stats: Optional[dict[str, float]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = dict(payload)
    if stats:
        for k, v in stats.items():
            save_dict[f"stat_{k}"] = np.float32(v)
    logger.info("B1 sidecar: writing compressed NPZ → %s", path)
    t0 = time.perf_counter()
    np.savez_compressed(path, **save_dict)
    logger.info(
        "B1 sidecar: saved %s (%.1fs)",
        path,
        time.perf_counter() - t0,
    )
    return path


def load_substitution_sidecar(path: Union[str, Path]) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def summarize_sidecar_hamming(payload: dict[str, Any]) -> dict[str, float]:
    h = np.asarray(payload.get("transition_hamming"), dtype=np.float64)
    flat = h[np.isfinite(h)]
    if flat.size == 0:
        return {
            "hamming_mean": float("nan"),
            "hamming_median": float("nan"),
            "hamming_p95": float("nan"),
            "n_transitions": 0.0,
        }
    return {
        "hamming_mean": float(flat.mean()),
        "hamming_median": float(np.median(flat)),
        "hamming_p95": float(np.percentile(flat, 95)),
        "n_transitions": float(flat.size),
    }
