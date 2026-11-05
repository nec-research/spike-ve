#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     visualize_infer_umap.py
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
"""UMAP plot: inference predictions vs ground truth on a Spike embedding background.

Uses :func:`recomb2027.latent.plotting.plot_umap_inference_predictions`
(same idea as ``plot_series_w_predictions``: fit UMAP on background, transform pred/GT).

Background options (pick one):

1. **Directory of .pt files** — same layout as ``UMAPEvolutionPlotter`` / GA runs
   (dict with ``encodings`` + ``label`` with genome_id). Typical resolved path:
   ``.../ESM1B/msa_who_lineages/average_token/33``.

2. **Train dataset npz** — randomly samples rows from ``sample_encodings`` (unscaled
   raw vectors inside the npz; if your pipeline stored scaled, use ``--bg-pt-dir``).

**Lineages:** pass ``--metadata-csv`` — either a table (``genome_id``, ``lineage``)
or a GA-style **FASTA** (``.fa`` / ``.fasta``) with ``key:value|`` headers.

Example::

    python -m recomb2027.latent.predictors.visualize_infer_umap \\
        --npz val_preds_skip0.npz \\
        --metadata-csv /path/to/df_msa_metadata.csv \\
        --output-html umap_val_skip0.html

    python -m recomb2027.latent.predictors.visualize_infer_umap \\
        --npz val_preds_skip0.npz \\
        --bg-pt-dir /mnt/.../encodings/ESM1B/msa_who_lineages/average_token/33 \\
        --metadata-csv /path/to/df_msa_metadata.csv \\
        --output-html umap_val_skip0.html

    # Only points with low embedding error (L2 norm of pred − target):
    python -m recomb2027.latent.predictors.visualize_infer_umap \\
        --npz val_preds_skip0.npz --max-pred-error 15 --output-html umap_low_err.html

    # Only predictions whose **source** is lineage BA.2 (prefix match):
    python -m recomb2027.latent.predictors.visualize_infer_umap \\
        --npz val_preds_skip0.npz --metadata-csv /path/to/df_msa_metadata.csv \\
        --filter-source-lineage BA.2 --filter-source-lineage-match prefix \\
        --output-html umap_ba2_src.html

    # Only rows where the GT jump has more than 5 MSA mutations (needs --msa-path if npz has no n_mutations):
    python -m recomb2027.latent.predictors.visualize_infer_umap \\
        --npz val_preds_skip0.npz --min-gt-mutations 5 \\
        --metadata-csv /path/to/df_msa_metadata.csv --output-html umap_high_mut.html
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

from recomb2027.core.paths import repo_root
import numpy as np
import pandas as pd
import torch

REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger(__name__)

TRAIN_DS_DEFAULT = (
    ""    
)

METADATA_CSV_DEFAULT = (
    ""
)

try:
    import recomb2027.generation.constants as _sgc
except ImportError:
    _sgc = None


def _file_first_nonempty_starts_with_gt(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s:
                    return s.startswith(">")
    except OSError:
        return False
    return False


def _is_fasta_metadata_file(path: Path) -> bool:
    """True when ``path`` should be read as FASTA headers (not pandas CSV)."""
    suf = path.suffix.lower()
    if suf in (".fa", ".faa", ".fasta", ".fas"):
        return True
    if suf in (".csv", ".tsv"):
        return False
    return _file_first_nonempty_starts_with_gt(path)


def load_gid_lineage_map(
    csv_path: Path,
    gid_col: str = "genome_id",
    lineage_col: str = "lineage",
) -> dict[str, str]:
    """Map genome_id -> lineage string from a CSV table or GA-style FASTA headers."""
    path = Path(csv_path)
    if _is_fasta_metadata_file(path):
        return _load_gid_lineage_map_fasta(path, gid_col, lineage_col)

    df = pd.read_csv(path, low_memory=False)
    if gid_col not in df.columns:
        raise ValueError(f"Column {gid_col!r} not in {csv_path}; columns={list(df.columns)[:30]}")
    if lineage_col not in df.columns:
        raise ValueError(
            f"Column {lineage_col!r} not in {csv_path}; try --metadata-lineage-col"
        )
    out: dict[str, str] = {}
    for _, row in df[[gid_col, lineage_col]].iterrows():
        gid = str(row[gid_col]).strip()
        if not gid or gid == "nan":
            continue
        out[gid] = str(row[lineage_col]) if pd.notna(row[lineage_col]) else "n/a"
    return out


def _parse_pt_label(label: str) -> dict:
    """Parse GA-style .pt label string into metadata dict (genome_id, lineage)."""
    sep = "|" if _sgc is None else _sgc.FASTA_FILE_FIELD_SEP
    vsep = ":" if _sgc is None else _sgc.FASTA_FILE_VALUE_SEP
    msep = "," if _sgc is None else _sgc.FASTA_FILE_MULTIVALUED_SEP
    gid_key = "genome_id" if _sgc is None else _sgc.FASTA_FILE_FIELD_GENOME_ID
    data_dict: dict = {}
    fields = label.split(sep)
    for field_str in fields:
        if vsep in field_str:
            key, value = field_str.split(vsep, 1)
            data_dict[key.strip()] = value.strip()
    if gid_key in data_dict:
        gids = str(data_dict[gid_key]).split(msep)
        data_dict[gid_key] = gids[0].strip()
    elif fields:
        data_dict["genome_id"] = fields[0]

    # Lineage: prefer Pango ('lineage') over WHO labels by default, but keep
    # WHO as a fallback so legacy GA exports without a Pango field still work.
    # When a metadata CSV is loaded downstream it always overrides this value
    # (see load_background_from_pt_dir), so the .pt-embedded label here is just
    # the no-CSV fallback.
    for k in ("lineage", "pango_lineage", "PANGO_lineage", "who_lineage"):
        if k in data_dict:
            data_dict["_lineage"] = data_dict[k]
            break
    return data_dict


def _load_gid_lineage_map_fasta(
    fasta_path: Path,
    gid_col: str,
    lineage_col: str,
) -> dict[str, str]:
    """Parse ``genome_id`` / lineage from GA-style ``key:value|`` FASTA headers."""
    gid_key = "genome_id" if _sgc is None else _sgc.FASTA_FILE_FIELD_GENOME_ID
    out: dict[str, str] = {}
    n_headers = 0
    with fasta_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            n_headers += 1
            parsed = _parse_pt_label(line[1:].strip())
            gid_raw = parsed.get(gid_col) or parsed.get(gid_key) or parsed.get("genome_id")
            if not gid_raw:
                continue
            gid = str(gid_raw).strip()
            if not gid or gid == "nan":
                continue
            lin = parsed.get(lineage_col)
            if lin is None:
                lin = parsed.get("_lineage") or parsed.get("lineage")
            if lin is None:
                lin_s = "n/a"
            else:
                lin_s = str(lin).strip() if str(lin).strip() != "nan" else "n/a"
            out[gid] = lin_s
    if not out and n_headers:
        logger.warning(
            "Parsed %d FASTA headers from %s but no genome ids matched "
            "(try --metadata-gid-col / inspect header tag names)",
            n_headers,
            fasta_path,
        )
    logger.info(
        "Loaded genome→lineage from FASTA headers %s (%d genomes)",
        fasta_path,
        len(out),
    )
    return out


def load_background_from_pt_dir(
    bg_dir: Path,
    n_samples: int,
    lineage_map: dict[str, str] | None,
    random_state: int = 42,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load mean-pooled embeddings + genome_id / lineage from GA-style .pt files."""
    rng = random.Random(random_state)
    files = [f for f in os.listdir(bg_dir) if f.endswith(".pt")]
    if not files:
        raise ValueError(f"No .pt files in {bg_dir}")
    pick = rng.sample(files, k=min(n_samples, len(files)))
    rows_em: list[np.ndarray] = []
    rows_gid: list[str] = []
    rows_lin: list[str] = []
    for fname in pick:
        path = bg_dir / fname
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception as e:
            logger.warning("Skip %s: %s", fname, e)
            continue
        if not isinstance(obj, dict) or "encodings" not in obj or "label" not in obj:
            continue
        emb = obj["encodings"]
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        meta = _parse_pt_label(str(obj["label"]).replace("'", ""))
        gid = str(meta.get("genome_id", fname))
        # Metadata CSV (if provided) is the source of truth — the user picks
        # the column via --metadata-lineage-col (default 'lineage'). Only fall
        # back to whatever the .pt label embedded when no CSV is supplied or
        # the genome id is missing from the lookup.
        if lineage_map is not None:
            lin = lineage_map.get(gid)
            if lin is None:
                lin = meta.get("_lineage", "n/a")
        else:
            lin = meta.get("_lineage", "n/a")
        rows_em.append(emb)
        rows_gid.append(gid)
        rows_lin.append(str(lin))
    if len(rows_em) < 10:
        raise ValueError(f"Too few valid .pt encodings from {bg_dir}")
    return np.stack(rows_em, axis=0), rows_gid, rows_lin


