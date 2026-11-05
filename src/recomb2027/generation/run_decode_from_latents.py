#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     run_decode_from_latents.py
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
"""Decode latents from an NPZ to aligned FASTA sequences (inference CLI).

Example (fused PCA-2D + mean decoder, as in the RECOMB 2027 paper):

    python -m recomb2027.generation.run_decode_from_latents \
        --latents-npz preds/<variant>/a4b4_wo_edges_cal_test.npz \
        --latent-key test_pred_pca2d_flat_raw --mean-latent-key test_pred_mean_raw \
        --decoder-checkpoint nar_fused/fused_v1_AB/best.pt \
        --decode-batch-size 256 --device cuda:0 --esm-repr-layer 33 --cycle-metric both \
        --source-fasta phylotree_matched/filtered_msa_S1_dedup.fasta \
        --fasta-out decoded.fasta --output decoded.csv

Supported decoder input modes (from the decoder checkpoint): ``pca2d_mean`` (fused), ``pca2d``
and ``raw``. Projected-latent decoders are not part of this package.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recomb2027.core.gpu_utils import (
    estimate_decoder_batch_size,
    get_gpu_free_memory,
    parse_gpu_ids,
    validate_gpu_ids,
)
from recomb2027.generation.latent_decoder.esm_loader import load_esm
from recomb2027.generation.latent_decoder.decoder_loaders import (
    load_sequence_decoder,
    resolve_device,
    resolve_esm_repr_layer,
)
from recomb2027.generation.latent_decoder.sequence_decoder import detokenize

RAW_ESM_DIM = 1280

LINEAGE_SUBST_BASELINE_KEYS = (
    "pred_baseline_mean_raw",
    "pred_baseline_mean",
    "test_pred_baseline_mean",
    "cal_pred_baseline_mean",
)
LINEAGE_SUBST_SAMPLES_KEYS = (
    "pred_samples_mean_raw",
    "pred_samples_mean",
    "test_pred_samples_mean",
    "cal_pred_samples_mean",
)


@dataclass(frozen=True)
class DecodeRecord:
    row: int
    query_id: str
    slot: int
    source_gid: str = ""


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


class StageLogger:
    """Numbered stage output with per-stage and total elapsed time."""

    def __init__(self, *, total_stages: int) -> None:
        self.total_stages = int(total_stages)
        self._stage_idx = 0
        self._run_t0 = time.monotonic()
        self._stage_t0 = self._run_t0

    def begin(self, name: str, *, detail: str = "") -> None:
        if self._stage_idx > 0:
            elapsed = time.monotonic() - self._stage_t0
            print(f"  finished in {_fmt_duration(elapsed)}", flush=True)
        self._stage_idx += 1
        self._stage_t0 = time.monotonic()
        header = f"[{self._stage_idx}/{self.total_stages}] {name}"
        print(f"\n=== {header} ===", flush=True)
        if detail:
            print(f"  {detail}", flush=True)

    def info(self, message: str) -> None:
        print(f"  {message}", flush=True)

    def finish(self) -> None:
        if self._stage_idx > 0:
            elapsed = time.monotonic() - self._stage_t0
            print(f"  finished in {_fmt_duration(elapsed)}", flush=True)
        total = time.monotonic() - self._run_t0
        print(
            f"\n=== Complete ({self._stage_idx}/{self.total_stages} stages, "
            f"total {_fmt_duration(total)}) ===",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode latents from NPZ to aligned FASTA sequences.",
    )
    p.add_argument("--latents-npz", type=str, required=True)
    p.add_argument(
        "--latent-key",
        type=str,
        default=None,
        help="Array key in npz (default: only key, or first key).",
    )
    p.add_argument("--decoder-checkpoint", type=str, required=True)
    p.add_argument(
        "--latents-are-pca2d",
        action="store_true",
        help="NPZ holds PCA2D flattened latents (used when decoder.input_mode == 'pca2d').",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--row-start", type=int, default=0)
    p.add_argument("--decode-batch-size", type=int, default=8)
    p.add_argument(
        "--auto-decode-batch-size",
        action="store_true",
        help=(
            "Estimate a safe decode batch size from free GPU memory "
            "(capped by --decode-batch-size)."
        ),
    )
    p.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids for parallel decode "
            "(one model replica per GPU, e.g. '0,1,2,3')."
        ),
    )
    p.add_argument("--esm-batch-size", type=int, default=8)
    p.add_argument("--esm-repr-layer", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--fasta-out",
        type=str,
        required=True,
        help="Output FASTA of decoded sequences.",
    )
    p.add_argument("--output", type=str, default=None, help="Optional summary CSV.")
    p.add_argument(
        "--skip-cycle",
        action="store_true",
        help="Skip ESM re-embed cycle check (torch + numpy only).",
    )
    p.add_argument(
        "--pca2d-basis",
        type=str,
        default=None,
        help="Path to fitted PCA2D basis .pt for cycle consistency check in PCA2D mode.",
    )
    p.add_argument(
        "--cycle-metric",
        choices=["auto", "pca2d", "mean", "both"],
        default="auto",
        help=(
            "Cycle-consistency metric. 'pca2d': cosine on flattened per-site PCA2D projections "
            "(PCA2D decoders only; strict). 'mean': cosine between mean-pooled 1280-d ESM "
            "re-embedding and a mean-pooled reference (same metric as mean decoders, so results "
            "are comparable across decoders). 'both': report both. 'auto' (default): pca2d, plus "
            "mean when a mean reference is available (--mean-reference-key or --clean-baseline). "
            "Mean/raw decoders always use the mean metric."
        ),
    )
    p.add_argument(
        "--mean-reference-key",
        type=str,
        default=None,
        help=(
            "NPZ key with mean-pooled raw ESM vectors (N,1280) used as reference for the 'mean' "
            "cycle metric of PCA2D decoders. Default: <split>_pred_mean_raw if present."
        ),
    )
    p.add_argument(
        "--mean-latent-key",
        type=str,
        default=None,
        help=(
            "Fused (pca2d+mean) decoders only: NPZ key with the mean-pooled raw ESM vectors fed to the "
            "decoder alongside the PCA2D latents. Default: <split>_pred_mean_raw."
        ),
    )
    p.add_argument(
        "--fused-mode",
        choices=["both", "pca2d_only", "mean_only"],
        default="both",
        help="Fused decoders only: ablate an input (replaced by its learned null embedding). Single-GPU only.",
    )
    p.add_argument(
        "--skip-scale-check",
        action="store_true",
        help="Disable the warning for latents in the wrong units (checked against the decoder's own normalisation).",
    )
    p.add_argument(
        "--clean-baseline",
        action="store_true",
        help=(
            "Ignore the predicted latents and decode CLEAN ground-truth latents of the source "
            "genomes (--source-gids-key), built from --enc-2d-dir. PCA2D decoders get true "
            "PCA2D projections; mean decoders get true mean-pooled ESM vectors. Cycle metrics "
            "then measure decoder error only (no predictor error)."
        ),
    )
    p.add_argument(
        "--source-gids-key",
        type=str,
        default=None,
        help="NPZ key with source genome ids (default: <split>_source_gids).",
    )
    p.add_argument(
        "--enc-2d-dir",
        type=str,
        default=None,
        help="Per-site ESM 2D encodings dir (needed for --clean-baseline).",
    )
    p.add_argument(
        "--source-fasta",
        type=str,
        default=None,
        help=(
            "Aligned FASTA with genome_id headers. If given with source gids, report identity / "
            "non-gap accuracy of decoded sequences vs the SOURCE sequence."
        ),
    )
    p.add_argument(
        "--reuse-decoded-fasta",
        action="store_true",
        help="If --fasta-out already exists and contains the expected number of records, skip decoding and reuse it.",
    )
    p.add_argument(
        "--lineage-subst",
        action="store_true",
        help=(
            "Decode lineage-subst cloud NPZ: unmodified baseline plus each finite "
            "prefix-swap / last_step sample slot per series row."
        ),
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split prefix for lineage-subst NPZ keys (default: split field or test).",
    )
    p.add_argument(
        "--prefer-raw",
        action="store_true",
        default=True,
        help="Prefer *_raw mean arrays in lineage-subst NPZ (default).",
    )
    p.add_argument(
        "--no-prefer-raw",
        action="store_true",
        help="Use scaled mean arrays instead of *_raw in lineage-subst NPZ.",
    )
    p.add_argument(
        "--lineage-input",
        type=str,
        choices=["auto", "mean", "pca2d", "fused"],
        default="auto",
        help=(
            "Which cloud arrays to decode in lineage-subst mode: 'mean' (*_mean_raw), 'pca2d' (*_pca2d_flat_raw) or "
            "'fused' (packed [pca2d | mean]). 'auto' (default) follows the decoder checkpoint's input mode."
        ),
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip unmodified baseline when decoding lineage-subst cloud NPZ.",
    )
    return p.parse_args()


def _finite_vec(v: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(v)).all())


def _first_key(store: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in store:
            return key
    raise KeyError(f"none of {keys}; have {sorted(store)}")


def _is_lineage_subst_npz(keys: list[str]) -> bool:
    has_baseline = any(k in keys for k in LINEAGE_SUBST_BASELINE_KEYS)
    has_samples = any(k in keys for k in LINEAGE_SUBST_SAMPLES_KEYS)
    return has_baseline and has_samples


def _resolve_lineage_subst_split(store: dict[str, Any], split: Optional[str]) -> str:
    if split:
        return str(split)
    if "split" in store:
        return str(np.asarray(store["split"]).item())
    return "test"


def _lineage_subst_source_gids(
    store: dict[str, Any],
    *,
    split: str,
) -> Optional[np.ndarray]:
    for key in ("source_gids", f"{split}_source_gids", "test_source_gids"):
        if key in store:
            return np.asarray(store[key])
    return None


def _lineage_modality_arrays(
    store: dict[str, Any],
    modality: str,
    *,
    split: str,
    prefer_raw: bool,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Resolve ``(baseline (N, D), samples (N, M, D), baseline_key, samples_key)`` for 'mean' or 'pca2d'."""
    if modality == "mean":
        raw_b, raw_s = "pred_baseline_mean_raw", "pred_samples_mean_raw"
        stem_b, stem_s = "pred_baseline_mean", "pred_samples_mean"
    elif modality == "pca2d":
        raw_b, raw_s = "pred_baseline_pca2d_flat_raw", "pred_samples_pca2d_flat_raw"
        stem_b, stem_s = "pred_baseline_pca2d", "pred_samples_pca2d"
    else:
        raise ValueError(f"unknown lineage modality {modality!r}")
    if prefer_raw and raw_b in store:
        baseline_key = raw_b
    else:
        baseline_key = _first_key(store, stem_b, f"{split}_{stem_b}", f"test_{stem_b}")
    if prefer_raw and raw_s in store:
        samples_key = raw_s
    else:
        samples_key = _first_key(store, stem_s, f"{split}_{stem_s}", f"test_{stem_s}")
    baseline = np.asarray(store[baseline_key], dtype=np.float32)
    samples = np.asarray(store[samples_key], dtype=np.float32)
    if baseline.ndim != 2:
        raise ValueError(f"Expected baseline (N, D), got {baseline_key} shape {baseline.shape}")
    if samples.ndim != 3:
        raise ValueError(f"Expected samples (N, M, D), got {samples_key} shape {samples.shape}")
    if baseline.shape[0] != samples.shape[0]:
        raise ValueError(
            f"Row count mismatch: {baseline_key} n={baseline.shape[0]} vs {samples_key} n={samples.shape[0]}"
        )
    if baseline.shape[1] != samples.shape[2]:
        raise ValueError(
            f"Latent dim mismatch: {baseline_key} D={baseline.shape[1]} vs {samples_key} D={samples.shape[2]}"
        )
    return baseline, samples, baseline_key, samples_key


def _load_lineage_subst_latents(
    path: str,
    *,
    split: Optional[str] = None,
    prefer_raw: bool = True,
    row_start: int = 0,
    row_limit: Optional[int] = None,
    include_baseline: bool = True,
    kind: str = "mean",
) -> tuple[np.ndarray, list[DecodeRecord], dict[str, Any]]:
    """Flatten a lineage-subst cloud into decodable vectors plus one :class:`DecodeRecord` per vector.

    ``kind``: 'mean' -> (n, 1280); 'pca2d' -> (n, L*k) (mean vectors, when present, returned as
    ``meta['mean_reference']`` for the pooled cycle metric); 'fused' -> packed (n, L*k + 1280) = [pca2d | mean].
    A slot is decoded only if every array it needs is finite (padding slots are NaN).
    """
    if kind not in ("mean", "pca2d", "fused"):
        raise ValueError(f"kind must be mean/pca2d/fused, got {kind!r}")
    data = np.load(path, allow_pickle=True)
    try:
        store = {k: data[k] for k in data.files}
        sp = _resolve_lineage_subst_split(store, split)

        parts: list[tuple[np.ndarray, np.ndarray]] = []  # (baseline, samples) in packing order
        keys_b: list[str] = []
        keys_s: list[str] = []
        modalities = {"mean": ["mean"], "pca2d": ["pca2d"], "fused": ["pca2d", "mean"]}[kind]
        for mod in modalities:
            bl, sm, kb, ks = _lineage_modality_arrays(store, mod, split=sp, prefer_raw=prefer_raw)
            parts.append((bl, sm))
            keys_b.append(kb)
            keys_s.append(ks)
        aux_mean: tuple[np.ndarray, np.ndarray] | None = None
        if kind == "pca2d":  # optional pooled-mean reference for the cycle check
            try:
                bl, sm, _, _ = _lineage_modality_arrays(store, "mean", split=sp, prefer_raw=prefer_raw)
                aux_mean = (bl, sm)
            except KeyError:
                aux_mean = None
        n_total = int(parts[0][0].shape[0])
        for bl, sm in parts + ([aux_mean] if aux_mean else []):
            if bl.shape[0] != n_total or sm.shape[1] != parts[0][1].shape[1]:
                raise ValueError("Inconsistent N / n_samples between lineage-subst arrays")
        n_slots = int(parts[0][1].shape[1])
        mean_dim = int(parts[-1][0].shape[1]) if kind in ("mean", "fused") else 0

        start = max(0, int(row_start))
        end = n_total if row_limit is None else min(n_total, start + int(row_limit))
        source_gids = _lineage_subst_source_gids(store, split=sp)

        def _vec(which: int, row: int, slot: int | None) -> np.ndarray:
            chunks = [
                (p_[0][row] if slot is None else p_[1][row, slot]).astype(np.float32).reshape(-1)
                for p_ in parts
            ]
            return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

        latents: list[np.ndarray] = []
        aux: list[np.ndarray] = []
        records: list[DecodeRecord] = []
        skipped_baseline = 0
        skipped_slots = 0
        for row in range(start, end):
            source_gid = ""
            if source_gids is not None and 0 <= row < len(source_gids):
                source_gid = str(np.asarray(source_gids[row]))

            def _emit(slot: int | None) -> bool:
                v = _vec(0, row, slot)
                if not _finite_vec(v):
                    return False
                a_ = None
                if aux_mean is not None:
                    a_ = (aux_mean[0][row] if slot is None else aux_mean[1][row, slot]).astype(np.float32).reshape(-1)
                    if not _finite_vec(a_):
                        a_ = None
                if aux_mean is not None and a_ is None:
                    return False
                latents.append(v)
                if a_ is not None:
                    aux.append(a_)
                records.append(
                    DecodeRecord(
                        row=row,
                        query_id="baseline" if slot is None else f"slot_{slot}",
                        slot=-1 if slot is None else int(slot),
                        source_gid=source_gid,
                    )
                )
                return True

            if include_baseline and not _emit(None):
                skipped_baseline += 1
            for slot in range(n_slots):
                if not _emit(slot):
                    skipped_slots += 1

        if not latents:
            raise ValueError(f"No finite lineage-subst latents in rows {start}:{end} of {path}")

        meta = {
            "mode": str(np.asarray(store["mode"]).item()) if "mode" in store else "",
            "split": sp,
            "kind": kind,
            "baseline_key": "+".join(keys_b),
            "samples_key": "+".join(keys_s),
            "n_series_rows": n_total,
            "row_start": start,
            "row_end": end,
            "n_slots_per_row": n_slots,
            "skipped_baseline": skipped_baseline,
            "skipped_slots": skipped_slots,
            "prefer_raw": bool(prefer_raw),
            "mean_dim": mean_dim,
            "mean_reference": np.stack(aux, axis=0) if aux else None,
        }
        return np.stack(latents, axis=0), records, meta
    finally:
        data.close()


def _load_latents(path: str, key: str | None) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    if not keys:
        raise ValueError(f"No arrays in {path}")
    if key is None:
        if len(keys) == 1:
            key = keys[0]
        else:
            raise ValueError(f"--latent-key required; npz keys: {keys}")
    if key not in data:
        raise KeyError(f"Key {key!r} not in {path}; available: {keys}")
    z = np.asarray(data[key], dtype=np.float32)
    if z.ndim == 3:
        z = z.reshape(z.shape[0], -1)
    elif z.ndim != 2:
        raise ValueError(f"Expected (N, D) or (N, L, k) latents, got shape {z.shape}")
    return z


def _prepare_decoder_latents(
    z_np: np.ndarray,
    *,
    input_mode: str,
    latent_dim: int,
    latents_are_pca2d: bool = False,
    site_cond_dim: int | None = None,
    seq_len: int | None = None,
) -> torch.Tensor:
    d_in = z_np.shape[1]

    if input_mode == "pca2d_mean":
        # fused decoder: packed [L*k | mean_dim] float rows
        if d_in != latent_dim:
            raise ValueError(f"Fused decoder expects packed width {latent_dim} = L*k + mean_dim, got {d_in}.")
        return torch.from_numpy(np.ascontiguousarray(z_np, dtype=np.float32))

    if input_mode == "pca2d":
        # support two PCA2D layouts:
        # - flattened NPZ arrays (N, L*k) -> reshape to (N, L, k) when site_cond info provided
        if site_cond_dim is not None and seq_len is not None:
            expected = int(seq_len) * int(site_cond_dim)
            if d_in == expected:
                return torch.from_numpy(z_np.reshape(-1, int(seq_len), int(site_cond_dim)))
        # fallback: expect latent_dim to match columns
        if d_in != latent_dim:
            raise ValueError(f"PCA2D decoder expects latent_dim={latent_dim} or flattened (L*k) when site_cond provided, got {d_in}.")
        return torch.from_numpy(z_np)

    if input_mode == "raw":
        if d_in != latent_dim:
            raise ValueError(
                f"Raw decoder expects latent_dim={latent_dim}, got {d_in}."
            )
        return torch.from_numpy(z_np)

    raise ValueError(
        f"Unsupported decoder input_mode {input_mode!r}. Supported: pca2d_mean, pca2d, raw."
    )


@torch.inference_mode()
def decode_latents(
    model,
    latents: torch.Tensor,
    device: torch.device,
    batch_size: int,
    *,
    log_progress: bool = True,
    worker_label: str = "",
    on_batch: Optional[Callable[[int, list[str]], None]] = None,
    keep_in_memory: bool = False,
) -> list[str]:
    model.eval()
    out: list[str] = []
    n = int(latents.shape[0])
    if n == 0:
        return out
    log_every = max(1, n // 20)
    prefix = f"[{worker_label}] " if worker_label else ""
    t0 = time.monotonic()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if log_progress and (start == 0 or end == n or start % log_every == 0):
            elapsed = time.monotonic() - t0
            rate = end / elapsed if elapsed > 0 else 0.0
            eta = (n - end) / rate if rate > 0 else 0.0
            pct = 100.0 * end / n
            print(
                f"  {prefix}decode {end:,}/{n:,} ({pct:.1f}%) "
                f"elapsed {_fmt_duration(elapsed)} "
                f"eta {_fmt_duration(eta)}",
                flush=True,
            )
        z = latents[start:end].to(device, non_blocking=device.type == "cuda")
        tokens = model.generate(z, greedy=True)
        batch_seqs = detokenize(tokens)
        if on_batch is not None:
            on_batch(start, batch_seqs)
        if keep_in_memory:
            out.extend(batch_seqs)
    if log_progress:
        elapsed = time.monotonic() - t0
        print(
            f"  {prefix}decode done: {n:,} sequences in {_fmt_duration(elapsed)}",
            flush=True,
        )
    return out


def _resolve_decode_batch_size(
    *,
    requested: int,
    auto: bool,
    seq_len: int,
    gpu_ids: list[int],
    d_model: int,
    n_layers: int,
) -> int:
    if requested <= 0:
        raise ValueError(f"--decode-batch-size must be positive, got {requested}")
    if not auto:
        return int(requested)

    if not gpu_ids:
        estimated = estimate_decoder_batch_size(
            seq_len,
            free_memory_mib=8000,
            d_model=d_model,
            n_layers=n_layers,
            max_batch_size=int(requested),
        )
        print(
            f"Auto decode batch size (CPU fallback): {estimated} "
            f"(cap={requested})"
        )
        return estimated

    per_gpu: list[int] = []
    for gpu_id in gpu_ids:
        device = torch.device(f"cuda:{gpu_id}")
        free_mib = get_gpu_free_memory(device)
        per_gpu.append(
            estimate_decoder_batch_size(
                seq_len,
                free_mib,
                d_model=d_model,
                n_layers=n_layers,
                max_batch_size=int(requested),
            )
        )
    estimated = min(per_gpu)
    return estimated


def _split_chunks(n: int, n_workers: int) -> list[tuple[int, int]]:
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    base = n // n_workers
    rem = n % n_workers
    chunks: list[tuple[int, int]] = []
    start = 0
    for worker in range(n_workers):
        size = base + (1 if worker < rem else 0)
        if size > 0:
            chunks.append((start, start + size))
            start += size
    return chunks


def _decode_latents_worker(payload: dict[str, Any]) -> dict[str, Any]:
    gpu_id = int(payload["gpu_id"])
    worker_label = f"GPU {gpu_id}"
    chunk_start = int(payload["chunk_start"])
    chunk_end = int(payload["chunk_end"])
    part_path = Path(payload["fasta_part_path"])
    print(
        f"  [{worker_label}] starting shard {chunk_start:,}:{chunk_end:,} "
        f"({chunk_end - chunk_start:,} sequences)",
        flush=True,
    )
    print(f"  [{worker_label}] streaming FASTA -> {part_path}", flush=True)
    device = torch.device(f"cuda:{gpu_id}")
    z_np = np.asarray(payload["z_np"], dtype=np.float32)
    print(f"  [{worker_label}] loading decoder checkpoint...", flush=True)
    model, meta = load_sequence_decoder(payload["decoder_checkpoint"], device=device)
    input_mode = meta.get("input_mode", "raw")
    latent_dim = int(meta.get("latent_dim", z_np.shape[1]))
    latents = _prepare_decoder_latents(
        z_np,
        input_mode=input_mode,
        latent_dim=latent_dim,
        latents_are_pca2d=bool(payload.get("latents_are_pca2d", False)),
        site_cond_dim=meta.get("site_cond_dim"),
        seq_len=meta.get("seq_len"),
    )
    records = _records_from_payload(payload.get("records"))
    writer = FastaStreamWriter(
        FastaWriteContext(
            path=part_path,
            records=records,
            row_offset=int(payload["row_offset"]),
            record_index_offset=chunk_start,
        )
    )

    def on_batch(local_start: int, sequences: list[str]) -> None:
        writer.write_batch(chunk_start + local_start, sequences)

    decode_latents(
        model,
        latents,
        device,
        int(payload["decode_batch_size"]),
        log_progress=True,
        worker_label=worker_label,
        on_batch=on_batch,
    )
    writer.close()
    return {
        "gpu_id": gpu_id,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "n_written": writer.n_written,
        "part_path": str(part_path),
    }


def decode_latents_multi_gpu(
    z_np: np.ndarray,
    *,
    gpu_ids: list[int],
    decoder_checkpoint: str,
    decode_batch_size: int,
    latents_are_pca2d: bool = False,
    fasta_out: Path,
    records: Optional[list[DecodeRecord]],
    row_offset: int,
    log: StageLogger,
) -> int:
    chunks = _split_chunks(int(z_np.shape[0]), len(gpu_ids))
    active = list(zip(gpu_ids[: len(chunks)], chunks))
    payloads: list[dict[str, Any]] = []
    for gpu_id, (start, end) in active:
        part_path = _fasta_part_path(fasta_out, int(gpu_id))
        payloads.append(
            {
                "gpu_id": int(gpu_id),
                "chunk_start": int(start),
                "chunk_end": int(end),
                "z_np": z_np[start:end],
                "decoder_checkpoint": decoder_checkpoint,
                "decode_batch_size": int(decode_batch_size),
                "latents_are_pca2d": bool(latents_are_pca2d),
                "fasta_part_path": str(part_path),
                "records": _records_chunk(records, start, end),
                "row_offset": int(row_offset),
            }
        )

    log.info(
        "parallel decode on GPUs "
        f"{[gpu for gpu, _ in active]} "
        f"({', '.join(f'{s:,}:{e:,}' for _, (s, e) in active)})"
    )
    log.info(f"streaming shard FASTAs -> {fasta_out.parent}/{fasta_out.name}.part.gpu*")
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    results_by_gpu: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as pool:
        futures = {
            pool.submit(_decode_latents_worker, payload): payload for payload in payloads
        }
        for fut in as_completed(futures):
            result = fut.result()
            results_by_gpu[int(result["gpu_id"])] = result
            log.info(
                f"GPU {result['gpu_id']} shard "
                f"{result['chunk_start']:,}:{result['chunk_end']:,} complete "
                f"({int(result['n_written']):,} records -> {result['part_path']})"
            )

    part_paths = [Path(results_by_gpu[int(gpu_id)]["part_path"]) for gpu_id, _ in active]
    log.info(f"merging {len(part_paths)} shard FASTAs -> {fasta_out}")
    return merge_fasta_parts(part_paths, fasta_out, log=log)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=1)
    b = F.normalize(b.float(), dim=1)
    return (1.0 - (a * b).sum(dim=1)).clamp(min=0.0, max=2.0)


def _resolve_pca2d_basis(args, meta: dict, log) -> Optional[dict[str, Any]]:
    """PCA2D basis: --pca2d-basis > checkpoint training_config > meta > NPZ basis_path > default."""
    basis_path = None
    if args.pca2d_basis:
        basis_path = Path(args.pca2d_basis)
    elif meta.get("training_config", {}).get("pca2d_basis"):
        basis_path = Path(meta["training_config"]["pca2d_basis"])
    elif meta.get("pca2d_basis"):
        basis_path = Path(meta["pca2d_basis"])
    else:
        try:
            with np.load(args.latents_npz, allow_pickle=True) as npz_probe:
                if "basis_path" in npz_probe:
                    basis_path = Path(str(npz_probe["basis_path"].item()))
        except Exception:
            pass
    if basis_path is not None and basis_path.exists():
        from recomb2027.latent.data_series.pca2d_features import load_pca2d_basis

        log.info(f"loaded PCA2D basis -> {basis_path}")
        return load_pca2d_basis(basis_path)
    log.info("warning: PCA2D basis could not be resolved; PCA2D cycle check will be skipped")
    return None


def _dist_stats(d: torch.Tensor) -> dict:
    return {
        "cycle_cos_dist_mean": float(d.mean().item()),
        "cycle_cos_dist_median": float(d.median().item()),
        "cycle_cos_dist_p90": float(torch.quantile(d, 0.9).item()),
        "cycle_distances": d.numpy(),
    }


@torch.no_grad()
def embed_sequences_at_layer(
    seqs: list[str],
    *,
    device: torch.device,
    repr_layer: int,
    batch_size: int,
    log: Optional[StageLogger] = None,
) -> torch.Tensor:
    model, alphabet, batch_converter, _ = load_esm(device)
    chunks: list[torch.Tensor] = []
    n = len(seqs)
    log_every = max(1, n // 10)
    t0 = time.monotonic()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if log is not None and (start == 0 or end == n or start % log_every == 0):
            elapsed = time.monotonic() - t0
            rate = end / elapsed if elapsed > 0 else 0.0
            eta = (n - end) / rate if rate > 0 else 0.0
            pct = 100.0 * end / n
            log.info(
                f"re-embed {end:,}/{n:,} ({pct:.1f}%) "
                f"elapsed {_fmt_duration(elapsed)} eta {_fmt_duration(eta)}"
            )
        batch = seqs[start:end]
        pairs = [(f"s{i}", s) for i, s in enumerate(batch)]
        _, _, tokens = batch_converter(pairs)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[repr_layer])
        rep = out["representations"][repr_layer]
        mask = (
            (tokens != alphabet.padding_idx)
            & (tokens != alphabet.cls_idx)
            & (tokens != alphabet.eos_idx)
        )
        mask_f = mask.unsqueeze(-1).to(rep.dtype)
        summed = (rep * mask_f).sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(rep.dtype)
        chunks.append((summed / counts).float().cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def cycle_metrics(
    latents: torch.Tensor,
    decoded_seqs: list[str],
    *,
    input_mode: str,
    device: torch.device,
    esm_batch_size: int,
    esm_repr_layer: int,
    pca2d_basis: Optional[dict[str, Any]] = None,
    cycle_metric: str = "auto",
    mean_reference: Optional[torch.Tensor] = None,
    log: Optional[StageLogger] = None,
) -> dict:
    """Cycle consistency: decoded sequence -> ESM re-embedding -> cosine distance to the input.

    For PCA2D decoders two metrics are available (``cycle_metric``):

    * ``pca2d``: cosine between the flattened per-site PCA2D projections of the re-embedding and
      the decoder input latents (strict, centred, per-site).
    * ``mean``: cosine between the mean-pooled raw 1280-d re-embedding and ``mean_reference``
      (N, 1280). This is exactly the metric used by mean decoders, so numbers are comparable.

    Result keys ``cycle_cos_dist_*`` / ``cycle_distances`` describe the primary metric (pca2d if
    computed, otherwise mean); ``metrics[name]`` holds every computed metric.
    """
    if input_mode == "pca2d":
        if pca2d_basis is None:
            raise ValueError(
                "PCA2D cycle check requires a fitted PCA2D basis (--pca2d-basis, "
                "or saved in checkpoint/NPZ)."
            )
        if cycle_metric == "auto":
            cycle_metric = "both" if mean_reference is not None else "pca2d"
        want_pca2d = cycle_metric in ("pca2d", "both")
        want_mean = cycle_metric in ("mean", "both")
        if want_mean and mean_reference is None:
            raise ValueError(
                "--cycle-metric mean/both needs a mean reference "
                "(--mean-reference-key or --clean-baseline)."
            )
        seq_len = int(pca2d_basis.get("seq_len", 712))
        k = int(pca2d_basis.get("k", 4))
        W = torch.as_tensor(pca2d_basis["W"], dtype=torch.float32, device=device)
        mu = torch.as_tensor(pca2d_basis["mu"], dtype=torch.float32, device=device)

        model, alphabet, batch_converter, _ = load_esm(device)
        chunks: list[torch.Tensor] = []
        mean_chunks: list[torch.Tensor] = []
        n = len(decoded_seqs)
        log_every = max(1, n // 10)
        t0 = time.monotonic()
        for start in range(0, n, esm_batch_size):
            end = min(start + esm_batch_size, n)
            if log is not None and (start == 0 or end == n or start % log_every == 0):
                elapsed = time.monotonic() - t0
                rate = end / elapsed if elapsed > 0 else 0.0
                eta = (n - end) / rate if rate > 0 else 0.0
                pct = 100.0 * end / n
                log.info(
                    f"re-embed {end:,}/{n:,} ({pct:.1f}%) "
                    f"elapsed {_fmt_duration(elapsed)} eta {_fmt_duration(eta)}"
                )
            batch = decoded_seqs[start:end]
            pairs = [(f"s{i}", s) for i, s in enumerate(batch)]
            _, _, tokens = batch_converter(pairs)
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=[esm_repr_layer])
            rep = out["representations"][esm_repr_layer]
            if want_mean:
                # identical pooling to embed_sequences_at_layer (pad/cls/eos masked, gaps kept)
                tmask = (
                    (tokens != alphabet.padding_idx)
                    & (tokens != alphabet.cls_idx)
                    & (tokens != alphabet.eos_idx)
                )
                tmask_f = tmask.unsqueeze(-1).to(rep.dtype)
                cnt = tmask.sum(dim=1, keepdim=True).clamp(min=1).to(rep.dtype)
                mean_chunks.append(((rep * tmask_f).sum(dim=1) / cnt).float().cpu())
            # Token 0 is <cls>, tokens 1 : 1 + seq_len are residue tokens
            tok_rep = rep[:, 1 : 1 + seq_len, :]
            if tok_rep.shape[1] < seq_len:
                pad = torch.zeros(
                    tok_rep.shape[0],
                    seq_len - tok_rep.shape[1],
                    tok_rep.shape[2],
                    device=device,
                    dtype=tok_rep.dtype,
                )
                tok_rep = torch.cat([tok_rep, pad], dim=1)
            elif tok_rep.shape[1] > seq_len:
                tok_rep = tok_rep[:, :seq_len, :]
            # Project (B, seq_len, 1280) -> (B, seq_len, k) on the fly
            if want_pca2d:
                proj_batch = (tok_rep - mu.view(1, 1, -1)) @ W.t()
                chunks.append(proj_batch.float().cpu())

        metrics: dict[str, dict] = {}
        if want_pca2d:
            re_z = torch.cat(chunks, dim=0)
            lat_tensor = latents.cpu()
            if lat_tensor.ndim == 3:
                lat_flat = lat_tensor.reshape(lat_tensor.shape[0], -1)
            elif lat_tensor.ndim == 2:
                lat_flat = lat_tensor
            else:
                raise ValueError(f"Unexpected latents shape: {lat_tensor.shape}")
            re_flat = re_z.reshape(re_z.shape[0], -1)
            metrics["pca2d"] = _dist_stats(_cosine_distance(lat_flat, re_flat))
        if want_mean:
            re_mean = torch.cat(mean_chunks, dim=0)
            ref = torch.as_tensor(mean_reference).float().cpu()
            if ref.shape != re_mean.shape:
                raise ValueError(
                    f"mean reference shape {tuple(ref.shape)} != re-embedding {tuple(re_mean.shape)}"
                )
            metrics["mean"] = _dist_stats(_cosine_distance(ref, re_mean))
        primary = "pca2d" if "pca2d" in metrics else "mean"
        res = dict(metrics[primary])
        res["primary_metric"] = primary
        res["metrics"] = metrics
        return res

    re_emb = embed_sequences_at_layer(
        decoded_seqs,
        device=device,
        repr_layer=esm_repr_layer,
        batch_size=esm_batch_size,
        log=log,
    )
    re_z = re_emb
    d = _cosine_distance(latents.cpu(), re_z.cpu())
    res = _dist_stats(d)
    res["primary_metric"] = "mean"
    res["metrics"] = {"mean": _dist_stats(d)}
    return res


def _format_fasta_header(
    record: Optional[DecodeRecord],
    *,
    row: int,
    cycle_dist: Optional[float] = None,
) -> str:
    parts: list[str] = [f"row={row}"]
    if record is not None:
        parts.extend(
            [
                f"query_id={record.query_id}",
                f"slot={record.slot}",
            ]
        )
        if record.source_gid:
            parts.append(f"source_gid={record.source_gid}")
    if cycle_dist is not None:
        parts.append(f"cycle_cos_dist={cycle_dist:.6f}")
    return ">" + " ".join(parts)


@dataclass
class FastaWriteContext:
    path: Path
    records: Optional[list[DecodeRecord]]
    row_offset: int = 0
    record_index_offset: int = 0


class FastaStreamWriter:
    """Append decoded sequences to FASTA, flushing after every batch."""

    def __init__(self, ctx: FastaWriteContext) -> None:
        self.ctx = ctx
        self.path = Path(ctx.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.n_written = 0

    def write_batch(self, global_start: int, sequences: list[str]) -> None:
        for offset, seq in enumerate(sequences):
            global_idx = int(global_start) + offset
            record: Optional[DecodeRecord] = None
            if self.ctx.records is not None:
                local_idx = global_idx - int(self.ctx.record_index_offset)
                record = self.ctx.records[local_idx]
            row = record.row if record is not None else self.ctx.row_offset + global_idx
            self._fh.write(f"{_format_fasta_header(record, row=row)}\n{seq}\n")
            self.n_written += 1
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _fasta_part_path(fasta_out: Path, gpu_id: int) -> Path:
    return fasta_out.parent / f"{fasta_out.name}.part.gpu{gpu_id}"


def _count_fasta_records(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


def read_fasta_sequences(path: Path) -> list[str]:
    seqs: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            seq = line.rstrip("\n")
            if seq:
                seqs.append(seq)
    return seqs


def merge_fasta_parts(
    part_paths: list[Path],
    out_path: Path,
    *,
    log: Optional[StageLogger] = None,
    delete_parts: bool = True,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for part in part_paths:
            n_part = _count_fasta_records(part)
            if log is not None:
                log.info(f"appended shard {part.name} ({n_part:,} records)")
            with part.open(encoding="utf-8") as in_fh:
                shutil.copyfileobj(in_fh, out_fh)
            n_total += n_part
            if delete_parts:
                part.unlink(missing_ok=True)
    return n_total


def rewrite_fasta_cycle_dists(
    path: Path,
    *,
    records: Optional[list[DecodeRecord]],
    row_offset: int,
    dists: np.ndarray,
) -> None:
    seqs = read_fasta_sequences(path)
    if len(seqs) != len(dists):
        raise ValueError(
            f"FASTA record count {len(seqs)} != cycle distance count {len(dists)}"
        )
    tmp = path.with_suffix(path.suffix + ".cycle.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for i, seq in enumerate(seqs):
            record = records[i] if records is not None else None
            row = record.row if record is not None else row_offset + i
            fh.write(
                f"{_format_fasta_header(record, row=row, cycle_dist=float(dists[i]))}\n"
                f"{seq}\n"
            )
    tmp.replace(path)


def _records_chunk(
    records: Optional[list[DecodeRecord]],
    chunk_start: int,
    chunk_end: int,
) -> Optional[list[dict[str, Any]]]:
    if records is None:
        return None
    return [asdict(records[i]) for i in range(chunk_start, chunk_end)]


def _records_from_payload(payload: Optional[list[dict[str, Any]]]) -> Optional[list[DecodeRecord]]:
    if payload is None:
        return None
    return [DecodeRecord(**item) for item in payload]


def main() -> None:
    args = parse_args()
    total_stages = 5 if args.skip_cycle else 6
    log = StageLogger(total_stages=total_stages)

    log.begin(
        "Setup",
        detail=f"latents={args.latents_npz}",
    )
    gpu_ids = parse_gpu_ids(gpus=args.gpus, device=args.device)
    validate_gpu_ids(gpu_ids)
    if gpu_ids:
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = resolve_device(args.device)
    log.info(f"device={device}")
    if len(gpu_ids) > 1:
        log.info(f"multi-GPU decode enabled on {gpu_ids}")
    log.info(f"decoder checkpoint={args.decoder_checkpoint}")
    log.info(f"fasta out={args.fasta_out}")
    if args.output:
        log.info(f"summary csv={args.output}")

    prefer_raw = bool(args.prefer_raw) and not bool(args.no_prefer_raw)
    lineage_subst = bool(args.lineage_subst)
    records: list[DecodeRecord] | None = None
    lineage_meta: dict[str, Any] | None = None

    log.begin("Load latents")
    with np.load(args.latents_npz, allow_pickle=True) as probe:
        npz_keys = list(probe.keys())
    if args.latent_key is None and _is_lineage_subst_npz(npz_keys):
        lineage_subst = True
        log.info("detected lineage-subst cloud NPZ")

    if lineage_subst:
        if args.latent_key is not None:
            raise ValueError(
                "Use either --latent-key for a flat (N, D) array or lineage-subst "
                "cloud decoding (omit --latent-key / pass --lineage-subst), not both."
            )
        lineage_kind = args.lineage_input
        if lineage_kind == "auto":
            _probe_model, _probe_meta = load_sequence_decoder(args.decoder_checkpoint, device=torch.device("cpu"))
            _im = _probe_meta.get("input_mode", "raw")
            lineage_kind = {"pca2d_mean": "fused", "pca2d": "pca2d"}.get(_im, "mean")
            del _probe_model, _probe_meta
            log.info(f"--lineage-input auto -> {lineage_kind!r} (decoder input_mode={_im!r})")
        z_np, records, lineage_meta = _load_lineage_subst_latents(
            args.latents_npz,
            split=args.split,
            prefer_raw=prefer_raw,
            row_start=args.row_start,
            row_limit=args.limit,
            include_baseline=not bool(args.no_baseline),
            kind=lineage_kind,
        )
        n_total = int(lineage_meta["n_series_rows"])
        start = int(lineage_meta["row_start"])
        end = int(lineage_meta["row_end"])
        log.info(
            f"cloud shape={z_np.shape} "
            f"(series rows {start}:{end} of {n_total})"
        )
        log.info(
            f"keys: baseline={lineage_meta['baseline_key']}, "
            f"samples={lineage_meta['samples_key']}"
        )
        if lineage_meta.get("mode"):
            log.info(f"lineage-subst mode={lineage_meta['mode']}")
        skipped_baseline = int(lineage_meta.get("skipped_baseline", 0))
        skipped_slots = int(lineage_meta.get("skipped_slots", 0))
        if skipped_baseline or skipped_slots:
            log.info(
                f"skipped non-finite entries: baseline={skipped_baseline}, "
                f"slots={skipped_slots}"
            )
        n_decode = int(z_np.shape[0])
        log.info(f"ready to decode {n_decode:,} latent vectors")
    else:
        latent_key = args.latent_key
        z_np = _load_latents(args.latents_npz, latent_key)
        n_total = z_np.shape[0]
        start = max(0, int(args.row_start))
        end = n_total if args.limit is None else min(n_total, start + args.limit)
        z_np = z_np[start:end]
        log.info(f"flat array key={latent_key!r}")
        log.info(f"shape={z_np.shape} (rows {start}:{end} of {n_total})")
        if (
            args.latents_are_pca2d
            and not args.clean_baseline
            and not str(latent_key).endswith("_raw")
        ):
            log.info(
                f"WARNING: --latents-are-pca2d with key {latent_key!r}: decoders are trained on "
                f"raw (aa-mu)@W.T projections (std ~9). Keys without '_raw' (e.g. test_pred_pca2d) "
                f"are usually in scaled/normalised units (std ~0.6) and will decode to garbage. "
                f"Use e.g. test_pred_pca2d_flat_raw. Got std={float(np.std(z_np)):.3f}."
            )
        log.info(f"ready to decode {z_np.shape[0]:,} latent vectors")

    log.begin("Load decoder")
    model, meta = load_sequence_decoder(args.decoder_checkpoint, device=device)
    input_mode = meta.get("input_mode", "raw")
    latent_dim = int(meta.get("latent_dim", z_np.shape[1]))
    layer = resolve_esm_repr_layer(args.esm_repr_layer, meta)
    log.info(
        f"seq_len={model.seq_len} layer={layer} "
        f"input_mode={input_mode} latent_dim={latent_dim}"
    )

    pca2d_basis = None
    is_fused = input_mode == "pca2d_mean"
    has_pca = input_mode in ("pca2d", "pca2d_mean")  # decoder consumes per-site PCA2D latents
    if is_fused and len(gpu_ids) > 1 and args.fused_mode != "both":
        raise ValueError("--fused-mode other than 'both' is only supported on a single GPU")
    if is_fused:
        model.set_inference_mode(args.fused_mode)
        log.info(f"fused decoder inference mode = {args.fused_mode}")
    if has_pca and (not args.skip_cycle or args.clean_baseline):
        pca2d_basis = _resolve_pca2d_basis(args, meta, log)

    # ---- optional ground-truth side data (source gids, clean latents, mean reference) ----
    source_gids: list[str] | None = None
    source_seqs: list[str | None] | None = None
    mean_reference: torch.Tensor | None = None
    if not lineage_subst:
        from recomb2027.generation.latent_decoder import cycle_eval as _ce

        gids_key = args.source_gids_key or _ce.default_source_gids_key(args.latent_key)
        if args.clean_baseline or args.source_fasta:
            with np.load(args.latents_npz, allow_pickle=True) as npz_g:
                if gids_key not in npz_g:
                    raise ValueError(
                        f"source gids key {gids_key!r} not in NPZ (use --source-gids-key). "
                        f"Keys: {list(npz_g.keys())}"
                    )
                all_gids = [str(g) for g in np.asarray(npz_g[gids_key]).reshape(-1)]
            source_gids = all_gids[start:end]
            if len(source_gids) != z_np.shape[0]:
                raise ValueError(
                    f"{len(source_gids)} source gids vs {z_np.shape[0]} latent rows "
                    f"(key {gids_key!r})"
                )

        if args.clean_baseline:
            if not args.enc_2d_dir:
                raise ValueError("--clean-baseline requires --enc-2d-dir")
            if input_mode not in ("pca2d", "pca2d_mean", "raw"):
                raise ValueError(f"--clean-baseline unsupported for input_mode={input_mode!r}")
            if has_pca and pca2d_basis is None:
                raise ValueError("--clean-baseline with a PCA2D decoder needs a PCA2D basis")
            log.info(f"CLEAN BASELINE: decoding true latents of {len(source_gids):,} source rows")
            clean = _ce.load_source_latents(
                source_gids,
                Path(args.enc_2d_dir),
                layer=int(layer),
                seq_len=int(model.seq_len),
                basis=pca2d_basis if has_pca else None,
                log=log,
            )
            keep = np.asarray(clean["found"], dtype=bool)
            if has_pca:
                z_np = clean["pca2d"][keep]
                args.latents_are_pca2d = True
            else:
                z_np = clean["mean"][keep]
            mean_reference = torch.as_tensor(clean["mean"][keep])
            source_gids = [g for g, k_ in zip(source_gids, keep) if k_]
            log.info(f"clean latents shape={z_np.shape} std={float(np.std(z_np)):.3f}")
        elif input_mode == "pca2d" and args.cycle_metric in ("auto", "mean", "both"):  # pure PCA2D decoder
            ref_key = args.mean_reference_key or _ce.default_mean_reference_key(args.latent_key)
            with np.load(args.latents_npz, allow_pickle=True) as npz_r:
                if ref_key in npz_r:
                    mean_reference = torch.as_tensor(np.asarray(npz_r[ref_key])[start:end])
                    log.info(f"mean-pooled cycle reference key={ref_key!r} shape={tuple(mean_reference.shape)}")
                elif args.cycle_metric in ("mean", "both") or args.mean_reference_key:
                    raise ValueError(f"mean reference key {ref_key!r} not in NPZ")

        if args.source_fasta and source_gids is not None:
            source_seqs = _ce.load_source_sequences(Path(args.source_fasta), source_gids)
            n_ok = sum(s is not None for s in source_seqs)
            log.info(f"source sequences found for {n_ok:,}/{len(source_seqs):,} rows")
    elif args.clean_baseline:
        raise ValueError("--clean-baseline is not supported with lineage-subst clouds")
    elif input_mode == "pca2d" and lineage_meta.get("mean_reference") is not None:
        mean_reference = torch.as_tensor(lineage_meta["mean_reference"])
        log.info(f"mean-pooled cycle reference from cloud, shape={tuple(mean_reference.shape)}")

    # ---- fused decoder: pack [PCA2D sites | mean vector]; keep the parts for the cycle check ----
    cycle_latents_np: np.ndarray | None = None
    if is_fused and lineage_subst:
        if lineage_meta.get("kind") != "fused":
            raise ValueError(
                f"fused decoder needs --lineage-input fused (or auto), got {lineage_meta.get('kind')!r}"
            )
        n_mean = int(lineage_meta["mean_dim"])
        cycle_latents_np = np.ascontiguousarray(z_np[:, :-n_mean], dtype=np.float32)
        mean_reference = torch.as_tensor(np.ascontiguousarray(z_np[:, -n_mean:]))
        log.info(
            f"fused cloud packed latents shape={z_np.shape} "
            f"(L*k={cycle_latents_np.shape[1]} + mean={n_mean})"
        )
    elif is_fused:
        if mean_reference is None:  # predicted-latent run: mean vectors come from the NPZ
            from recomb2027.generation.latent_decoder import cycle_eval as _ce2

            mkey = args.mean_latent_key or _ce2.default_mean_reference_key(args.latent_key)
            with np.load(args.latents_npz, allow_pickle=True) as npz_m:
                if mkey not in npz_m:
                    raise ValueError(f"mean latent key {mkey!r} not in NPZ (use --mean-latent-key)")
                mean_np_full = np.asarray(npz_m[mkey])
            mean_reference = torch.as_tensor(mean_np_full[start:end])
            log.info(f"fused decoder: mean input key={mkey!r} shape={tuple(mean_reference.shape)}")
        mean_np = mean_reference.float().numpy()
        if mean_np.shape[0] != z_np.shape[0]:
            raise ValueError(f"mean rows {mean_np.shape[0]} != PCA2D rows {z_np.shape[0]}")
        cycle_latents_np = np.ascontiguousarray(z_np.reshape(z_np.shape[0], -1), dtype=np.float32)
        z_np = np.concatenate([cycle_latents_np, mean_np], axis=1)
        log.info(f"fused packed latents shape={z_np.shape} (L*k={cycle_latents_np.shape[1]} + mean={mean_np.shape[1]})")

    if not args.skip_scale_check:
        from recomb2027.generation.latent_decoder import cycle_eval as _ce3

        for msg in _ce3.input_scale_warnings(model, z_np):
            log.info(f"WARNING: {msg}")
    if not is_fused and args.fused_mode != "both":
        log.info(f"WARNING: --fused-mode={args.fused_mode} ignored (decoder is not a fused decoder)")

    decode_batch_size = _resolve_decode_batch_size(
        requested=int(args.decode_batch_size),
        auto=bool(args.auto_decode_batch_size),
        seq_len=int(model.seq_len),
        gpu_ids=gpu_ids,
        d_model=int(meta.get("d_model", 256)),
        n_layers=int(meta.get("n_layers", 4)),
    )
    if args.auto_decode_batch_size:
        if gpu_ids:
            per_gpu = []
            for gpu_id in gpu_ids:
                free_mib = get_gpu_free_memory(torch.device(f"cuda:{gpu_id}"))
                per_gpu.append(
                    estimate_decoder_batch_size(
                        int(model.seq_len),
                        free_mib,
                        d_model=int(meta.get("d_model", 256)),
                        n_layers=int(meta.get("n_layers", 4)),
                        max_batch_size=int(args.decode_batch_size),
                    )
                )
            log.info(
                f"auto decode batch size={decode_batch_size} "
                f"(per-GPU={per_gpu}, cap={args.decode_batch_size})"
            )
        else:
            log.info(
                f"auto decode batch size={decode_batch_size} "
                f"(cap={args.decode_batch_size})"
            )
    else:
        log.info(f"decode batch size={decode_batch_size}")

    log.begin(
        "Decode sequences",
        detail=f"N={z_np.shape[0]:,} batch_size={decode_batch_size}",
    )
    fasta_path = Path(args.fasta_out)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)

    latents = _prepare_decoder_latents(
        z_np,
        input_mode=input_mode,
        latent_dim=latent_dim,
        latents_are_pca2d=args.latents_are_pca2d,
        site_cond_dim=meta.get("site_cond_dim") if 'meta' in locals() else None,
        seq_len=meta.get("seq_len") if 'meta' in locals() else None,
    )

    can_reuse = False
    if args.reuse_decoded_fasta and fasta_path.is_file():
        existing_count = _count_fasta_records(fasta_path)
        if existing_count == z_np.shape[0]:
            can_reuse = True
            n_decoded = existing_count
            log.info(f"reusing existing decoded FASTA ({n_decoded:,} records) -> {fasta_path}")
        else:
            log.info(
                f"--reuse-decoded-fasta set but {fasta_path} has {existing_count:,} records "
                f"(expected {z_np.shape[0]:,}); re-decoding"
            )

    if not can_reuse:
        log.info(f"streaming FASTA -> {fasta_path}")
        if len(gpu_ids) > 1:
            n_decoded = decode_latents_multi_gpu(
                z_np,
                gpu_ids=gpu_ids,
                decoder_checkpoint=args.decoder_checkpoint,
                decode_batch_size=decode_batch_size,
                latents_are_pca2d=args.latents_are_pca2d,
                fasta_out=fasta_path,
                records=records,
                row_offset=int(start),
                log=log,
            )
        else:
            writer = FastaStreamWriter(
                FastaWriteContext(
                    path=fasta_path,
                    records=records,
                    row_offset=int(start),
                )
            )

            def on_batch(global_start: int, sequences: list[str]) -> None:
                writer.write_batch(global_start, sequences)

            decode_latents(
                model,
                latents,
                device,
                batch_size=decode_batch_size,
                on_batch=on_batch,
            )
            writer.close()
            n_decoded = writer.n_written

        log.info(f"streamed {n_decoded:,} records to {fasta_path}")

    cycle: dict | None = None
    dists = None
    if not args.skip_cycle:
        if has_pca and pca2d_basis is None:
            log.info("PCA2D cycle check skipped (no PCA2D basis found)")
        else:
            log.begin(
                "Cycle consistency",
                detail=(
                    f"re-embed with ESM layer={layer}, batch_size={args.esm_batch_size}, "
                    f"metric={args.cycle_metric}"
                ),
            )
            decoded_seqs = read_fasta_sequences(fasta_path)
            if len(decoded_seqs) != n_decoded:
                raise ValueError(
                    f"FASTA record count {len(decoded_seqs)} != decoded count {n_decoded}"
                )
            cycle = cycle_metrics(
                torch.from_numpy(cycle_latents_np) if is_fused else latents,
                decoded_seqs,
                input_mode="pca2d" if is_fused else input_mode,
                device=device,
                esm_batch_size=args.esm_batch_size,
                esm_repr_layer=layer,
                pca2d_basis=pca2d_basis,
                cycle_metric=args.cycle_metric,
                mean_reference=mean_reference,
                log=log,
            )
            dists = cycle["cycle_distances"]
            for name, m in cycle["metrics"].items():
                tag = " (primary)" if name == cycle["primary_metric"] else ""
                log.info(
                    f"cycle cos dist [{name}]{tag}: mean={m['cycle_cos_dist_mean']:.6f} "
                    f"median={m['cycle_cos_dist_median']:.6f} "
                    f"p90={m['cycle_cos_dist_p90']:.6f}"
                )
            log.info("updating FASTA headers with cycle_cos_dist")
            rewrite_fasta_cycle_dists(
                fasta_path,
                records=records,
                row_offset=int(start),
                dists=dists,
            )
    else:
        log.info("cycle check skipped (--skip-cycle)")

    truth_stats: dict | None = None
    if source_seqs is not None:
        from recomb2027.generation.latent_decoder.cycle_eval import sequence_accuracy

        decoded_for_truth = read_fasta_sequences(fasta_path)
        truth_stats = sequence_accuracy(decoded_for_truth, source_seqs)
        log.info(
            f"vs SOURCE sequence (n={truth_stats['n_with_truth']:,}): "
            f"identity={truth_stats['identity_mean']:.4f} "
            f"non-gap acc={truth_stats['nongap_acc_mean']:.4f} "
            f"exact={truth_stats['exact_match_frac']:.4f}"
        )

    n_unique = len(set(read_fasta_sequences(fasta_path)))
    log.info(f"unique decoded sequences: {n_unique:,} / {n_decoded:,}")

    log.begin("Write outputs")
    log.info(f"FASTA already streamed ({n_decoded:,} records) -> {fasta_path}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["which", "key", "value"])
            summary = {
                "n_rows": n_decoded,
                "row_start": start,
                "row_end": end,
                "n_total": n_total,
                "latent_dim": latent_dim,
                "input_mode": input_mode,
                "esm_repr_layer": layer,
                "decoder_checkpoint": args.decoder_checkpoint,
                "n_unique_decoded": n_unique,
            }
            if lineage_meta is not None:
                summary.update(
                    {
                        "lineage_subst": True,
                        "lineage_subst_mode": lineage_meta.get("mode", ""),
                        "lineage_subst_split": lineage_meta.get("split", ""),
                        "baseline_key": lineage_meta.get("baseline_key", ""),
                        "samples_key": lineage_meta.get("samples_key", ""),
                        "n_slots_per_row": lineage_meta.get("n_slots_per_row", 0),
                        "skipped_baseline": lineage_meta.get("skipped_baseline", 0),
                        "skipped_slots": lineage_meta.get("skipped_slots", 0),
                    }
                )
            if cycle is not None:
                summary.update(
                    {
                        "cycle_cos_dist_mean": cycle["cycle_cos_dist_mean"],
                        "cycle_cos_dist_median": cycle["cycle_cos_dist_median"],
                        "cycle_cos_dist_p90": cycle["cycle_cos_dist_p90"],
                        "cycle_primary_metric": cycle["primary_metric"],
                    }
                )
                for name, m in cycle["metrics"].items():
                    for stat in ("mean", "median", "p90"):
                        summary[f"cycle_{name}_cos_dist_{stat}"] = m[f"cycle_cos_dist_{stat}"]
            summary["clean_baseline"] = bool(args.clean_baseline)
            if truth_stats is not None:
                summary.update(
                    {
                        "src_identity_mean": truth_stats["identity_mean"],
                        "src_nongap_acc_mean": truth_stats["nongap_acc_mean"],
                        "src_exact_match_frac": truth_stats["exact_match_frac"],
                        "src_n_with_truth": truth_stats["n_with_truth"],
                    }
                )
            for k, v in summary.items():
                w.writerow(["summary", k, v])
            if dists is not None:
                for i in range(n_decoded):
                    if records is not None:
                        record = records[i]
                        row = record.row
                        row_key = (
                            f"{row}_{record.query_id}_cycle_cos_dist"
                        )
                        w.writerow(
                            [
                                "row",
                                row_key,
                                float(dists[i]),
                            ]
                        )
                    else:
                        row = start + i
                        w.writerow(["row", f"{row}_cycle_cos_dist", float(dists[i])])
        log.info(f"wrote summary CSV -> {out_path}")

    log.finish()


if __name__ == "__main__":
    main()
