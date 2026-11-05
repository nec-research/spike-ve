#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     splits.py
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
"""FASTA parsing and train/test split loading for decoder training."""

from __future__ import annotations

import os
import pickle
import random


def parse_header_metadata(header: str) -> dict:
    """Parse one FASTA header into a dict of ``key -> value``."""
    h = header.lstrip(">").strip()
    fields: dict = {}
    for part in h.split("|"):
        part = part.strip()
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        fields[key.strip().lower()] = val.strip()
    return fields


def parse_fasta_with_metadata(in_fasta: str) -> tuple[list[str], list[dict]]:
    """Parse a FASTA file, returning aligned ``(sequences, metadata)`` lists."""
    seqs: list[str] = []
    metas: list[dict] = []
    cur_seq: list[str] = []
    cur_meta: dict | None = None
    with open(in_fasta, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_meta is not None:
                    seqs.append("".join(cur_seq).upper())
                    metas.append(cur_meta)
                    cur_seq = []
                cur_meta = parse_header_metadata(line)
            else:
                cur_seq.append(line)
        if cur_meta is not None:
            seqs.append("".join(cur_seq).upper())
            metas.append(cur_meta)
    if not seqs:
        raise ValueError(f"No sequences parsed from {in_fasta}")
    return seqs, metas


def select_holdout_split(
    n_total: int,
    limit: int,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], int]:
    """Train on ``limit`` sequences; test on new sequences from the rest."""
    if n_total < 2:
        raise ValueError(f"Need at least 2 sequences, got {n_total}.")
    if not 0 < limit < n_total:
        raise ValueError(
            f"holdout split requires limit in (0, {n_total}), got {limit}."
        )
    if not 0.0 < test_fraction <= 1.0:
        raise ValueError(
            f"test_fraction must be in (0, 1], got {test_fraction}."
        )

    rng = random.Random(seed)
    train_global = sorted(rng.sample(range(n_total), limit))
    train_set = set(train_global)
    remaining = [i for i in range(n_total) if i not in train_set]
    n_test_requested = max(1, int(round(limit * test_fraction)))
    if not remaining:
        raise ValueError(
            "No sequences remain outside the train pool for holdout test."
        )
    if n_test_requested > len(remaining):
        print(
            f"Warning: requested {n_test_requested} test sequences "
            f"(limit * test_fraction) but only {len(remaining)} remain "
            f"outside train; using all {len(remaining)}."
        )
        n_test = len(remaining)
    else:
        n_test = n_test_requested
    test_global = sorted(rng.sample(remaining, n_test))
    return train_global, test_global, n_test_requested


def parse_collection_date(meta: dict):
    """``collection_date`` of one FASTA record as ``datetime.date`` (``YYYY-MM`` / ``YYYY`` -> first day), or None."""
    from datetime import date

    dstr = str(meta.get("collection_date") or meta.get("date") or "").strip()
    if not dstr or dstr.lower() == "nan":
        return None
    try:
        return date.fromisoformat(dstr)
    except Exception:
        parts = dstr.split("-")
        try:
            if len(parts) == 1:
                return date(int(parts[0]), 1, 1)
            if len(parts) == 2:
                return date(int(parts[0]), int(parts[1]), 1)
        except Exception:
            pass
    return None


def temporal_split_indices(
    metas: list[dict],
    cutoff,
    *,
    n_test: int,
    seed: int = 42,
    max_train: int | None = None,
) -> tuple[list[int], list[int]]:
    """Date-based split of a whole FASTA (no splits file).

    train = every record with ``collection_date <= cutoff``; test = ``n_test`` records drawn at random (seeded)
    from those with ``collection_date > cutoff``. Records without a parsable date are used for neither.
    ``max_train`` randomly subsamples the train pool (seeded). Returns sorted FASTA indices ``(train, test)``.
    """
    if n_test < 1:
        raise ValueError(f"n_test must be >= 1, got {n_test}")
    before: list[int] = []
    after: list[int] = []
    undated = 0
    for i, m in enumerate(metas):
        d = parse_collection_date(m)
        if d is None:
            undated += 1
        elif d <= cutoff:
            before.append(i)
        else:
            after.append(i)
    if not before:
        raise ValueError(f"No sequences with collection_date <= {cutoff}")
    if not after:
        raise ValueError(f"No sequences with collection_date > {cutoff} to build the test set from")
    rng = random.Random(seed)
    if max_train is not None and 0 < max_train < len(before):
        before = sorted(rng.sample(before, max_train))
    if n_test > len(after):
        print(f"Warning: requested n_test={n_test} but only {len(after)} sequences are after {cutoff}; using all.", flush=True)
        n_test = len(after)
    test = sorted(rng.sample(after, n_test))
    print(
        f"Temporal split at {cutoff}: train={len(before)} (<= cutoff), test={len(test)} sampled from "
        f"{len(after)} after cutoff; {undated} undated records unused.",
        flush=True,
    )
    return before, test


def _dedupe_preserve_order(items: list) -> list:
    seen: set = set()
    out: list = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _flatten_split_values(raw, key: str) -> list:
    if raw is None:
        raise ValueError(f"Split key {key!r} is missing or None.")
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"Split {key!r} must be a list, got {type(raw).__name__}."
        )
    flat: list = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    if not flat:
        raise ValueError(f"Split {key!r} is empty after flattening.")
    return flat


def _is_index_split(flat: list) -> bool:
    for value in flat:
        if isinstance(value, (list, tuple, dict)):
            return False
        try:
            int(value)
        except (TypeError, ValueError):
            return False
    return True


def _coerce_split_index_list(flat: list, key: str) -> list[int]:
    try:
        idxs = [int(i) for i in flat]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Split {key!r} must be a sequence of integers, "
            f"got {type(flat[0]).__name__}."
        ) from exc
    if len(idxs) < 2:
        raise ValueError(
            f"Split {key!r} must have at least 2 indices, got {len(idxs)}."
        )
    return sorted(set(idxs))


def build_genome_id_to_index(
    metas: list[dict],
    id_field: str = "genome_id",
) -> dict[str, int]:
    """Map ``genome_id`` metadata values to global FASTA row indices."""
    field = id_field.lower()
    out: dict[str, int] = {}
    dup_in_fasta = 0
    missing_field = 0
    for i, meta in enumerate(metas):
        gid = meta.get(id_field) or meta.get(field)
        if gid is None:
            missing_field += 1
            continue
        gid_s = str(gid).strip()
        if gid_s in out:
            dup_in_fasta += 1
            continue
        out[gid_s] = i
    if not out:
        raise ValueError(
            f"No FASTA records with metadata field {id_field!r}."
        )
    if dup_in_fasta:
        print(
            f"Warning: {dup_in_fasta} duplicate {id_field} values in FASTA; "
            "using the first occurrence for each."
        )
    if missing_field:
        print(
            f"Note: {missing_field} FASTA records lack {id_field!r} metadata."
        )
    return out


def _genome_ids_from_flat(flat: list, key: str, id_field: str) -> list[str]:
    ids: list[str] = []
    field = id_field.lower()
    for value in flat:
        if isinstance(value, dict):
            gid = value.get(id_field) or value.get(field)
            if gid is None:
                raise ValueError(
                    f"Split {key!r} entry dict missing {id_field!r}: {value!r}."
                )
            ids.append(str(gid).strip())
        else:
            ids.append(str(value).strip())
    ids = _dedupe_preserve_order(ids)
    if len(ids) < 2:
        raise ValueError(
            f"Split {key!r} must have at least 2 unique IDs after "
            f"flatten/dedupe, got {len(ids)}."
        )
    return ids


def _genome_ids_to_indices(
    genome_ids: list[str],
    key: str,
    id_to_index: dict[str, int],
) -> list[int]:
    missing = [gid for gid in genome_ids if gid not in id_to_index]
    if missing:
        raise ValueError(
            f"Split {key!r} has {len(missing)} {key} genome_ids not found "
            f"in FASTA (e.g. {missing[:5]})."
        )
    return sorted(id_to_index[gid] for gid in genome_ids)


def _split_values_to_indices(
    raw,
    key: str,
    *,
    id_to_index: dict[str, int] | None,
    id_field: str,
) -> tuple[list[int], int, int, str]:
    flat = _flatten_split_values(raw, key)
    n_flat = len(flat)
    if _is_index_split(flat):
        idxs = _coerce_split_index_list(flat, key)
        return idxs, n_flat, len(idxs), "index"

    if id_to_index is None:
        raise ValueError(
            f"Split {key!r} contains genome IDs; pass FASTA metadata via "
            f"--fasta (parsed for {id_field!r})."
        )
    genome_ids = _genome_ids_from_flat(flat, key, id_field)
    idxs = _genome_ids_to_indices(genome_ids, key, id_to_index)
    return idxs, n_flat, len(genome_ids), "genome_id"


def load_splits_from_pkl(
    path: str,
    *,
    train_key: str = "train",
    test_key: str = "test",
    test_key_fallback: str = "val",
    n_fasta: int | None = None,
    metas: list[dict] | None = None,
    id_field: str = "genome_id",
) -> tuple[list[int], list[int], dict]:
    """Load train/test global FASTA indices from a pickle dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Splits pickle not found: {path}")

    with open(path, "rb") as fh:
        data = pickle.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected dict in {path}, got {type(data).__name__}."
        )

    if train_key not in data:
        raise ValueError(
            f"Train key {train_key!r} not in pickle keys: {sorted(data.keys())}."
        )

    resolved_test_key = test_key
    if test_key not in data:
        if test_key_fallback in data:
            resolved_test_key = test_key_fallback
            print(
                f"Note: test key {test_key!r} not found; "
                f"using {test_key_fallback!r}."
            )
        else:
            raise ValueError(
                f"Test key {test_key!r} (and fallback {test_key_fallback!r}) "
                f"not in pickle keys: {sorted(data.keys())}."
            )

    id_to_index = (
        build_genome_id_to_index(metas, id_field=id_field) if metas is not None else None
    )

    train_idxs, train_flat, train_unique, train_format = _split_values_to_indices(
        data[train_key],
        train_key,
        id_to_index=id_to_index,
        id_field=id_field,
    )
    test_idxs, test_flat, test_unique, test_format = _split_values_to_indices(
        data[resolved_test_key],
        resolved_test_key,
        id_to_index=id_to_index,
        id_field=id_field,
    )
    if train_format != test_format:
        raise ValueError(
            f"Train split format ({train_format}) != test split format ({test_format})."
        )

    train_set = set(train_idxs)
    overlap = train_set & set(test_idxs)
    n_overlap_removed = 0
    if overlap:
        n_overlap_removed = len(overlap)
        test_idxs = sorted(i for i in test_idxs if i not in train_set)
        print(
            f"Warning: train and {resolved_test_key!r} overlap on "
            f"{n_overlap_removed} FASTA indices; removed them from "
            f"{resolved_test_key!r} (train kept)."
        )
        if len(test_idxs) < 2:
            raise ValueError(
                f"After removing {n_overlap_removed} overlapping indices from "
                f"{resolved_test_key!r}, only {len(test_idxs)} sequences remain "
                f"(need at least 2)."
            )

    if n_fasta is not None:
        for name, idxs in (("train", train_idxs), ("test", test_idxs)):
            bad = [i for i in idxs if i < 0 or i >= n_fasta]
            if bad:
                raise ValueError(
                    f"{name} split has {len(bad)} out-of-range indices for "
                    f"FASTA size {n_fasta} (e.g. {bad[:5]})."
                )

    meta = {
        "splits_pkl": os.path.abspath(path),
        "splits_train_key": train_key,
        "splits_test_key": resolved_test_key,
        "splits_id_field": id_field,
        "splits_format": train_format,
        "n_train": len(train_idxs),
        "n_test": len(test_idxs),
        "n_train_flat": train_flat,
        "n_test_flat": test_flat,
        "n_train_unique_ids": train_unique,
        "n_test_unique_ids": test_unique,
        "n_test_overlap_removed": n_overlap_removed,
    }
    print(
        f"Loaded splits from {path}: train={len(train_idxs)}, "
        f"test={len(test_idxs)} "
        f"(keys {train_key!r}, {resolved_test_key!r}; "
        f"flat {train_flat}+{test_flat} -> unique {train_unique}+{test_unique})"
    )
    return train_idxs, test_idxs, meta
