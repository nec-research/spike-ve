#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     esm_2d_io.py
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
"""Load per-residue ESM token embeddings from ``encode_proteins_from_fasta`` 2D outputs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch


def build_gid_file_index(enc_dir: Path, layer: int | None = None) -> dict[str, Path]:
    """Map genome_id strings to ``.pt`` paths under ``enc_dir`` (or ``enc_dir / str(layer)``)."""
    index: dict[str, Path] = {}
    if not enc_dir.is_dir():
        return index
    import logging

    logger = logging.getLogger(__name__)

    target_dir = enc_dir
    if layer is not None and (enc_dir / str(layer)).is_dir():
        target_dir = enc_dir / str(layer)
    logger.info("Building GID -> .pt file index from: %s", target_dir)

    pt_files = list(target_dir.glob("*.pt"))
    if not pt_files and target_dir == enc_dir:
        # Check subdirectories if no .pt files at root
        pt_files = list(target_dir.glob("*/*.pt"))

    total = 0
    for fpath in pt_files:
        total += 1
        stem = fpath.name[:-3]
        keys: list[str] = []

        # Files named exactly by the label (no '|' delimiter).
        if "|" not in stem and stem:
            keys.append(stem)

        # ``label_to_filename`` usually yields a ``genome_id_<value>`` / ``genome_id:<value>`` field.
        for field in stem.split("|"):
            if field.startswith("genome_id_"):
                keys.append(field[len("genome_id_") :])
                break
            if field.startswith("genome_id:"):
                keys.append(field.split(":", 1)[1])
                break

        # Accession-style ids (e.g. EPI_ISL_12345).
        m = re.search(r"(EPI[_-]ISL[_-]?\d+)", stem, flags=re.IGNORECASE)
        if m:
            keys.append(m.group(1))

        # Legacy BV-BRC style ids (e.g. 11320.1234) anywhere in the filename.
        keys.extend(re.findall(r"(\d+\.\d+)", fpath.name))

        # Slow fallback: open the .pt and read the embedded label.
        if not keys:
            try:
                rec = torch.load(fpath, map_location="cpu")
                label = rec.get("label") if isinstance(rec, dict) else None
                if label and isinstance(label, str):
                    lab = label.replace("'", "")
                    for part in lab.split("|"):
                        if part.lower().startswith("genome_id"):
                            if ":" in part:
                                val = part.split(":", 1)[1].strip()
                            elif "_" in part:
                                val = part.split("_", 1)[1].strip()
                            else:
                                val = part
                            if val:
                                keys.append(val)
                                break
                    if not keys:
                        m2 = re.search(r"(EPI[_-]ISL[_-]?\d+)", lab, flags=re.IGNORECASE)
                        if m2:
                            keys.append(m2.group(1))
            except Exception as exc:
                logger.debug("Could not read .pt to extract gid: %s (%s)", fpath, exc)

        for gid in keys:
            index.setdefault(gid, fpath)

        if total % 1000 == 0:
            logger.info("Scanned %d files, indexed %d unique gids so far.", total, len(index))
    logger.info("Finished indexing %d files -> %d unique gids.", total, len(index))
    return index


def load_2d_encoding(fpath: Path, layer: int, *, drop_cls: bool = True) -> np.ndarray:
    """Return AA token matrix ``[L, D]`` from a 2D encoding record."""
    rec = torch.load(fpath, map_location="cpu", weights_only=False)
    enc = rec["encodings"]
    if isinstance(enc, dict):
        enc = enc[layer]
    arr = enc.detach().cpu().numpy() if hasattr(enc, "detach") else np.asarray(enc)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D encoding in {fpath}; got {arr.shape}")
    if drop_cls:
        arr = arr[1:, :]
    return arr.astype(np.float32)


def pad_or_truncate_aa_matrix(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Crop or zero-pad ``[L, D]`` to ``[seq_len, D]``."""
    l, d = arr.shape
    if l >= seq_len:
        return arr[:seq_len].astype(np.float32)
    out = np.zeros((seq_len, d), dtype=np.float32)
    out[:l] = arr
    return out
