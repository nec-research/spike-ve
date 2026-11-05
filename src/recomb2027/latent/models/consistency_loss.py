#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     consistency_loss.py
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
"""Mean <-> PCA-2D consistency loss, ridge readout and the helpers it uses.

``resolve_ridge_readout`` fits (or loads) the frozen ridge map from the PCA-2D delta to the mean delta; it is
used by both the anchor and the residual trainer (and by the ``prefit`` stage of ``run_experiment.sh``)."""
from __future__ import annotations

import numpy as np
import logging
import torch
import hashlib
import json
import time
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from sklearn.linear_model import Ridge
from recomb2027.latent.datasets.analyse_branching_sequences import hamming_distance
from recomb2027.latent.predictors.predictor_utils import (
    parse_pca2d_splits,
    build_datasets,
)
from omegaconf import OmegaConf
from recomb2027.config_io import load_config
from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.latent.data_series.pca2d_features import PCA2D_S1_TARGET


# ============================================================================
# (merged from pca2d_sparsity.py)
# ============================================================================

def site_energy(delta_flat: np.ndarray, *, seq_len: int, k: int) -> np.ndarray:
    """Squared L2 energy per site, shape ``[L]``."""
    sites = np.asarray(delta_flat, dtype=np.float64).reshape(seq_len, k)
    return np.sum(sites * sites, axis=-1)


def topk_energy_fraction(energies: np.ndarray, top_k: int) -> float:
    """Fraction of total per-sample energy in the top ``top_k`` sites."""
    total = float(np.sum(energies))
    if total <= 1e-12:
        return 0.0
    kk = min(int(top_k), energies.size)
    sorted_e = np.sort(energies)[::-1]
    return float(sorted_e[:kk].sum() / total)


# ============================================================================
# (merged from mean_pca2d_consistency.py)
# ============================================================================

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgeRecord:
    g_prev: str
    g_next: str
    delta_mean: np.ndarray
    delta_pca2d: np.ndarray
    aa_hamming: int
    is_saltation: bool
    top1_pca2d_energy_frac: float


def _load_mean_vector(path: Path, *, layer: int) -> np.ndarray:
    rec = torch.load(path, map_location="cpu", weights_only=False)
    enc = rec["encodings"]
    if isinstance(enc, dict):
        key = layer if layer in enc else str(layer)
        if key not in enc:
            raise KeyError(f"Layer {layer!r} missing in {path}")
        enc = enc[key]
    vec = enc.detach().cpu().numpy() if hasattr(enc, "detach") else np.asarray(enc)
    return np.asarray(vec, dtype=np.float64).reshape(-1)


def fit_delta_ridge(
    delta_pca2d: np.ndarray,
    delta_mean: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Return ``coef`` with shape ``[D, L*k]`` so ``delta_mean ≈ delta_pca2d @ coef.T``."""
    x = np.asarray(delta_pca2d, dtype=np.float64)
    y = np.asarray(delta_mean, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"expected [N, *]; got x={x.shape} y={y.shape}")
    model = Ridge(alpha=float(alpha), fit_intercept=False)
    model.fit(x, y)
    return np.asarray(model.coef_, dtype=np.float64)


def collect_series_edges(
    ds,
    *,
    pca2d_name: str,
    basis: dict[str, Any],
    gid_index_2d: dict[str, Path],
    gid_index_mean: dict[str, Path],
    gid_to_seq: dict[str, str],
    layer: int,
    saltation_min_aa: int = 8,
    max_edges: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[EdgeRecord]:
    """Collect consecutive-step Δmean and Δpca2d on a series dataset."""
    rng = rng or np.random.default_rng(0)
    seq_len = int(basis["seq_len"])
    k = int(basis["k"])
    layer_i = int(layer)
    pca_cube = ds.pca2d_encodings.get(str(pca2d_name))
    if pca_cube is None:
        raise KeyError(f"dataset missing pca2d encodings for {pca2d_name!r}")

    mean_enc = getattr(ds, "sample_encodings", None)
    if mean_enc is not None:
        mean_enc = np.asarray(mean_enc, dtype=np.float64)

    mean_cache: dict[str, np.ndarray] = {}
    pca_cache: dict[str, np.ndarray] = {}

    def _mean_from_disk(gid: str) -> Optional[np.ndarray]:
        if gid not in mean_cache:
            if gid not in gid_index_mean:
                return None
            try:
                mean_cache[gid] = _load_mean_vector(gid_index_mean[gid], layer=layer_i)
            except (OSError, KeyError, ValueError):
                return None
        return mean_cache[gid]

    def _pca_flat(row: int, col: int) -> np.ndarray:
        key = (row, col)
        if key not in pca_cache:
            pca_cache[key] = np.asarray(pca_cube[row, col], dtype=np.float64)
        return pca_cache[key]

    edges: list[EdgeRecord] = []
    gids_arr = np.asarray(ds.sample_gids, dtype=object)
    n_series = int(gids_arr.shape[0]) if gids_arr.ndim >= 1 else 0
    ragged = gids_arr.ndim == 1
    if ragged:
        k_series = len(gids_arr[0]) if n_series else 0
    else:
        k_series = int(gids_arr.shape[1])

    for i in range(n_series):
        row = gids_arr[i]
        row_k = len(row) if ragged else k_series
        for t in range(1, row_k):
            gp = str(row[t - 1] if ragged else gids_arr[i, t - 1])
            gn = str(row[t] if ragged else gids_arr[i, t])
            if gp == gn:
                continue
            if mean_enc is not None:
                m_prev = mean_enc[i, t - 1].reshape(-1)
                m_next = mean_enc[i, t].reshape(-1)
            else:
                m_prev = _mean_from_disk(gp)
                m_next = _mean_from_disk(gn)
                if m_prev is None or m_next is None:
                    continue
                if gp not in gid_index_2d or gn not in gid_index_2d:
                    continue
            delta_mean = m_next - m_prev
            delta_pca = _pca_flat(i, t) - _pca_flat(i, t - 1)

            s1 = gid_to_seq.get(gp)
            s2 = gid_to_seq.get(gn)
            if s1 is None or s2 is None:
                aa_h = 0
            else:
                aa_h, _ = hamming_distance(s1, s2)
            is_salt = int(aa_h) >= int(saltation_min_aa)
            energy = site_energy(delta_pca, seq_len=seq_len, k=k)
            top1_frac = topk_energy_fraction(energy, 1)

            edges.append(
                EdgeRecord(
                    g_prev=gp,
                    g_next=gn,
                    delta_mean=delta_mean,
                    delta_pca2d=delta_pca,
                    aa_hamming=int(aa_h),
                    is_saltation=is_salt,
                    top1_pca2d_energy_frac=float(top1_frac),
                )
            )

    if max_edges is not None and len(edges) > int(max_edges):
        idx = rng.choice(len(edges), size=int(max_edges), replace=False)
        edges = [edges[int(j)] for j in idx]
    return edges


# ============================================================================
# (merged from mean_pca2d_consistency_loss.py)
# ============================================================================

def _cfg_section(cfg, key: str, fallback_key: str) -> dict[str, Any]:
    raw = cfg.get(key) or cfg.get(fallback_key) or {}
    if OmegaConf.is_config(raw):
        return OmegaConf.to_container(raw, resolve=True) or {}
    return dict(raw)


def mean_pca2d_v4_cfg(cfg) -> dict[str, Any]:
    """Resolve v4 consistency / readout settings from predictor YAML."""
    return _cfg_section(cfg, "mean_pca2d_v4", "mean_pca2d_consistency")


class FrozenPca2dDeltaReadout(nn.Module):
    """Linear readout ``delta_mean = delta_pca2d_flat @ coef.T`` (coef frozen)."""

    def __init__(self, coef: Union[np.ndarray, torch.Tensor], *, trainable: bool = False):
        super().__init__()
        t = torch.as_tensor(coef, dtype=torch.float32)
        if t.ndim != 2:
            raise ValueError(f"ridge coef must be [mean_dim, flat_dim]; got {tuple(t.shape)}")
        self.mean_dim = int(t.shape[0])
        self.flat_dim = int(t.shape[1])
        self.linear = nn.Linear(self.flat_dim, self.mean_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(t)
        self.linear.weight.requires_grad_(bool(trainable))
        # Provenance (not part of the state dict).
        self.sha256 = ridge_sha256(t.detach().cpu().numpy())
        self.source: Optional[str] = None
        self.meta: dict[str, Any] = {}

    def forward(self, delta_pca2d_flat: torch.Tensor) -> torch.Tensor:
        return self.linear(delta_pca2d_flat)

    @classmethod
    def from_path(cls, path: Union[str, Path], *, trainable: bool = False) -> "FrozenPca2dDeltaReadout":
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"ridge coef not found: {p}")
        meta: dict[str, Any] = {}
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                arr = z["coef"]
                if "meta_json" in z.files:
                    meta = json.loads(str(z["meta_json"]))
        elif p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            arr = data.get("ridge_coef") or data.get("coef")
            if arr is None:
                raise KeyError(f"JSON {p} missing ridge_coef/coef")
            arr = np.asarray(arr, dtype=np.float64)
        else:
            arr = np.load(p)
        readout = cls(arr, trainable=trainable)
        readout.source = str(p)
        readout.meta = meta
        return readout


def ridge_sha256(coef: Union[np.ndarray, torch.Tensor]) -> str:
    """Stable content hash of a ridge matrix (float32 bytes + shape), 16 hex chars."""
    arr = np.ascontiguousarray(
        torch.as_tensor(coef, dtype=torch.float32).detach().cpu().numpy()
    )
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def save_ridge_coef(
    coef: np.ndarray,
    path: Union[str, Path],
    *,
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"coef": np.asarray(coef, dtype=np.float64)}
    if meta is not None:
        meta = dict(meta)
        meta["coef_sha256"] = ridge_sha256(np.asarray(coef))
        payload["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(out, **payload)
    return out


def _file_identity(path: Optional[Union[str, Path]]) -> Optional[dict[str, Any]]:
    """Cheap identity of an input file: size + mtime (content hash for small files)."""
    if not path:
        return None
    p = Path(str(path)).expanduser()
    if not p.is_file():
        return {"path": str(p), "missing": True}
    st = p.stat()
    ident: dict[str, Any] = {"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    if st.st_size <= 5_000_000:  # e.g. the PCA basis
        ident["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        ident.pop("mtime_ns")  # content hash is authoritative; mtime changes on copy
    return ident


def ridge_fit_fingerprint(cfg, data_cfg, v4: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Everything that determines the fitted ridge matrix (None if inputs unknowable)."""
    if data_cfg is None:
        return None
    fit_split = str(v4.get("fit_split", "train"))
    processed = data_cfg.get("processed_dir")
    series_npz = Path(str(processed)) / f"{fit_split}_dataset.npz" if processed else None
    fields = {
        "ridge_alpha": float(v4.get("ridge_alpha", 1.0)),
        "fit_split": fit_split,
        "max_fit_edges": v4.get("max_fit_edges"),
        "saltation_min_aa": int(v4.get("saltation_min_aa", 8)),
        "seed": int(cfg.get("seed", 42)),
        "pca2d_target_name": str(cfg.get("pca2d_target_name", PCA2D_S1_TARGET)),
        "scale_pca2d": bool(cfg.get("scale_pca2d", True)),
        "pca2d_scaler_mode": str(cfg.get("pca2d_scaler_mode", "flat") or "flat"),
        "series_npz": _file_identity(series_npz),
        "pca2d_basis": _file_identity(cfg.get("pca2d_basis_path") or data_cfg.get("pca2d_basis_path")),
        "msa": _file_identity(cfg.get("msa_path") or data_cfg.get("msa_path")),
    }
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return {"fingerprint": digest, "fit_settings": fields}


def fit_ridge_coef_from_series(
    cfg,
    data_cfg,
    *,
    fit_split: str = "train",
    ridge_alpha: float = 1.0,
    max_edges: Optional[int] = 8000,
    saltation_min_aa: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Fit ridge ``A`` on consecutive series edges (train split)."""
    pca2d_name = str(cfg.get("pca2d_target_name", PCA2D_S1_TARGET))
    train_ds, cal_ds, test_ds, basis_path = parse_pca2d_splits(
        build_datasets(cfg, data_cfg)
    )
    split_map = {"train": train_ds, "test": test_ds}
    if cal_ds is not None:
        split_map["cal"] = cal_ds
    if fit_split not in split_map:
        raise KeyError(f"fit_split {fit_split!r} not in {list(split_map)}")

    from recomb2027.latent.data_series.esm_2d_io import build_gid_file_index
    from recomb2027.latent.data_series.pca2d_features import load_pca2d_basis
    from recomb2027.latent.datasets.analyse_branching_sequences import parse_msa_fasta

    basis = load_pca2d_basis(basis_path)
    layer = int(data_cfg.get("layer", basis.get("layer", 33)))
    enc_2d_dir = Path(str(cfg.get("enc_2d_dir") or data_cfg.get("enc_2d_dir")))
    enc_mean_dir = Path(str(data_cfg.get("encodings_dir")))
    msa_path = Path(str(data_cfg.get("msa_path")))
    gid_index_2d = build_gid_file_index(enc_2d_dir)
    gid_index_mean = build_gid_file_index(enc_mean_dir)
    gid_to_seq = parse_msa_fasta(str(msa_path))
    rng = np.random.default_rng(int(seed))

    edges = collect_series_edges(
        split_map[fit_split],
        pca2d_name=pca2d_name,
        basis=basis,
        gid_index_2d=gid_index_2d,
        gid_index_mean=gid_index_mean,
        gid_to_seq=gid_to_seq,
        layer=layer,
        saltation_min_aa=saltation_min_aa,
        max_edges=max_edges,
        rng=rng,
    )
    if not edges:
        raise RuntimeError(f"No edges for ridge fit on split {fit_split!r}")
    x_fit = np.stack([e.delta_pca2d for e in edges], axis=0)
    y_fit = np.stack([e.delta_mean for e in edges], axis=0)
    return fit_delta_ridge(x_fit, y_fit, alpha=float(ridge_alpha))


def _log_readout(readout: FrozenPca2dDeltaReadout, how: str) -> FrozenPca2dDeltaReadout:
    logger.info(
        "Ridge readout %s: shape=%dx%d sha256=%s source=%s",
        how,
        readout.mean_dim,
        readout.flat_dim,
        readout.sha256,
        readout.source or "<in-memory fit>",
    )
    return readout


def resolve_ridge_readout(
    cfg,
    *,
    device: torch.device,
    data_cfg: Optional[Any] = None,
) -> FrozenPca2dDeltaReadout:
    """Load a cached ridge matrix or fit one, logging provenance and hash.

    ``mean_pca2d_v4.ridge_cache_policy`` controls a stale / unverifiable cache
    (fit-settings fingerprint missing or different from the current inputs):
    ``warn`` (default) keeps it and logs a warning, ``refit`` refits and overwrites
    the cache, ``error`` raises. Caches written by this version embed the fingerprint
    (ridge_alpha, fit split, series NPZ, PCA basis, MSA, ...); older caches have none
    and are treated as unverifiable.
    """
    v4 = mean_pca2d_v4_cfg(cfg)
    trainable = bool(v4.get("trainable_readout", False))
    path = v4.get("ridge_coef_path")
    if path:
        readout = FrozenPca2dDeltaReadout.from_path(str(path), trainable=trainable)
        return _log_readout(readout, "loaded (explicit ridge_coef_path)").to(device)

    if data_cfg is None:
        ds_path = cfg.get("dataset_config") or cfg.get("dataset_config_path")
        if ds_path:
            data_cfg = load_config(str(ds_path))

    cache = v4.get("ridge_coef_cache")
    cache_path = Path(str(cache)).expanduser() if cache else None
    policy = str(v4.get("ridge_cache_policy", "warn") or "warn").lower()
    if policy not in ("warn", "refit", "error"):
        raise ValueError(f"ridge_cache_policy must be warn|refit|error; got {policy!r}")
    expected = ridge_fit_fingerprint(cfg, data_cfg, v4)

    if cache_path is not None and cache_path.is_file():
        readout = FrozenPca2dDeltaReadout.from_path(cache_path, trainable=trainable)
        cached_fp = (readout.meta or {}).get("fingerprint")
        if expected is None:
            problem = None  # cannot verify without dataset settings
        elif cached_fp is None:
            problem = "cache has no fit fingerprint (legacy file); freshness cannot be verified"
        elif cached_fp != expected["fingerprint"]:
            problem = (
                f"cache fingerprint {cached_fp} != current {expected['fingerprint']} "
                "(dataset / PCA basis / MSA / ridge settings changed since it was fit)"
            )
        else:
            problem = None
        if problem is None:
            return _log_readout(readout, "loaded (cache, fingerprint ok)").to(device)
        msg = f"ridge_coef_cache {cache_path}: {problem}"
        if policy == "error":
            raise RuntimeError(msg)
        if policy == "warn":
            logger.warning("%s. Keeping it (ridge_cache_policy=warn).", msg)
            return _log_readout(readout, "loaded (cache, UNVERIFIED)").to(device)
        logger.warning("%s. Refitting (ridge_cache_policy=refit).", msg)

    if data_cfg is None:
        raise ValueError(
            "mean_pca2d_v4: no usable ridge_coef_path/ridge_coef_cache and dataset_config "
            "not set for on-the-fly ridge fit"
        )

    coef = fit_ridge_coef_from_series(
        cfg,
        data_cfg,
        fit_split=str(v4.get("fit_split", "train")),
        ridge_alpha=float(v4.get("ridge_alpha", 1.0)),
        max_edges=v4.get("max_fit_edges"),
        saltation_min_aa=int(v4.get("saltation_min_aa", 8)),
        seed=int(cfg.get("seed", 42)),
    )
    readout = FrozenPca2dDeltaReadout(coef, trainable=trainable)
    if cache_path is not None:
        meta = dict(expected or {})
        meta["fit_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_ridge_coef(coef, cache_path, meta=meta)
        readout.source = str(cache_path)
        readout.meta = meta
        logger.info("Wrote ridge coef cache %s (sha256=%s)", cache_path, readout.sha256)
    return _log_readout(readout, "fitted").to(device)


def delta_consistency_mse(
    pred_delta_mean: torch.Tensor,
    delta_pca2d_flat: torch.Tensor,
    readout: FrozenPca2dDeltaReadout,
    *,
    stop_grad_pca2d: bool = True,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``MSE(pred_delta_mean, sg(readout(delta_pca2d)))``."""
    src = delta_pca2d_flat.detach() if stop_grad_pca2d else delta_pca2d_flat
    target = readout(src)
    per = (pred_delta_mean - target).pow(2).mean(dim=-1)
    if sample_weight is not None:
        w = sample_weight.reshape(-1).to(dtype=per.dtype)
        return (per * w).sum() / w.sum().clamp(min=1e-12)
    return per.mean()


def flat_pca2d_predictions(
    model,
    pred_pca: torch.Tensor,
    *,
    readout: FrozenPca2dDeltaReadout,
) -> torch.Tensor:
    """Ensure pca2d preds are flat ``L*k`` (decode bottleneck if needed)."""
    pred_pca = pred_pca.float()
    if int(pred_pca.shape[-1]) == int(readout.flat_dim):
        return pred_pca
    bn_dim = int(getattr(model, "hparams", {}).get("pca2d_bottleneck_dim", 0) or 0)
    if bn_dim > 0 and int(pred_pca.shape[-1]) == bn_dim:
        decode = getattr(model, "decode_pca2d_bottleneck", None)
        if decode is not None:
            return decode(pred_pca)
    raise ValueError(
        "pca2d prediction dim "
        f"{int(pred_pca.shape[-1])} incompatible with ridge readout flat_dim="
        f"{readout.flat_dim} (bottleneck_dim={bn_dim})"
    )


def gpt_mean_pca2d_consistency_loss(
    model,
    xb: torch.Tensor,
    pred_dict: dict[str, torch.Tensor],
    *,
    readout: FrozenPca2dDeltaReadout,
    pca2d_target_name: str = PCA2D_S1_TARGET,
    only_last: bool = False,
    stop_grad_pca2d: bool = True,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """A4 consistency: ``MSE(Δmean_hat, sg(A vec(Δpca_hat)))`` in diff space."""
    if PRIMARY_TARGET not in pred_dict or pca2d_target_name not in pred_dict:
        return pred_dict[PRIMARY_TARGET].new_zeros(())
    pred_mean = pred_dict[PRIMARY_TARGET].float()
    pred_pca = flat_pca2d_predictions(
        model, pred_dict[pca2d_target_name], readout=readout
    )
    if only_last:
        pred_mean = pred_mean[:, -1, :]
        pred_pca = pred_pca[:, -1, :]
    return delta_consistency_mse(
        pred_mean,
        pred_pca,
        readout,
        stop_grad_pca2d=stop_grad_pca2d,
        sample_weight=sample_weight,
    )


def attach_consistency_to_loss(
    loss: torch.Tensor,
    consistency: Optional[torch.Tensor],
    weight: float,
    info: Optional[dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    if consistency is None or weight <= 0.0:
        return loss
    if info is not None:
        info["consistency_mse"] = consistency.detach()
    return loss + float(weight) * consistency
