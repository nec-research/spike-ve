#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     decoder_loaders.py
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
"""Load sequence decoder checkpoints (projected-decoder mode is not part of recomb2027)."""

from __future__ import annotations

from pathlib import Path

import torch

from recomb2027.generation.latent_decoder.sequence_decoder import SequenceDecoder

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {dev} but CUDA is not available.")
    return dev


def resolve_checkpoint_path(path: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve a checkpoint path relative to repo root when not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = repo_root or REPO_ROOT
    candidate = root / p
    if candidate.exists():
        return candidate
    return p


@torch.no_grad()
def load_sequence_decoder(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
):
    """Load a trained ``SequenceDecoder`` from ``best.pt``.

    Returns ``(model, meta)`` where ``meta`` is the full checkpoint dict.
    """
    device = resolve_device(device)
    path = resolve_checkpoint_path(checkpoint_path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state.get("decoder_state_dict", state)

    # "dima_nar_fused" / "dima_nar" / "dima" are the model_type strings of checkpoints written before the rename.
    if state.get("model_type") in ("nar_fused", "dima_nar_fused"):
        from recomb2027.generation.latent_decoder.nar_fused_decoder import PCA2DMeanFusedDecoder

        model = PCA2DMeanFusedDecoder(
            seq_len=int(state["seq_len"]),
            k=int(state["k"]),
            mean_dim=int(state.get("mean_dim", 1280)),
            vocab_size=int(state.get("vocab_size", 22)),
            d_model=int(state.get("d_model", 256)),
            n_layers=int(state.get("n_layers", 3)),
            n_heads=int(state.get("n_heads", 4)),
            ff_dim=int(state.get("ff_dim", 1024)),
            dropout=float(state.get("dropout", 0.1)),
            bos_id=int(state.get("bos_id", 21)),
            n_global=int(state.get("n_global", 4)),
        )
        model.load_state_dict(state["model_state_dict"])  # strict; normalisation buffers are inside
        model = model.to(device).eval()
        state["site_cond_dim"] = model.k
        state["latent_dim"] = model.latent_dim  # packed width [L*k | D]
        state["input_mode"] = "pca2d_mean"
        return model, state

    if state.get("model_type") in ("nar", "dima_nar"):
        from recomb2027.generation.latent_decoder.nar_decoder import PCA2DNARDecoder

        k = int(state.get("k", state.get("site_cond_dim", 4)))
        seq_len = int(state.get("seq_len", 712))
        nar_sd = dict(state["model_state_dict"])
        # Legacy checkpoints: scalar global ``latent_scale`` and no ``site_mean`` buffer.
        ls = nar_sd.get("latent_scale")
        if ls is not None and ls.dim() == 0:
            nar_sd["latent_scale"] = ls.reshape(1).expand(k).clone()
        nar_sd.setdefault("site_mean", torch.zeros(seq_len, k))
        model = PCA2DNARDecoder(
            seq_len=seq_len,
            k=k,
            vocab_size=int(state.get("vocab_size", 22)),
            d_model=int(state.get("d_model", 256)),
            n_layers=int(state.get("n_layers", 3)),
            n_heads=int(state.get("n_heads", 4)),
            ff_dim=int(state.get("ff_dim", 1024)),
            dropout=float(state.get("dropout", 0.1)),
            bos_id=int(state.get("bos_id", 21)),
        )
        model.load_state_dict(nar_sd)  # strict
        model = model.to(device).eval()
        state.setdefault("site_cond_dim", model.k)
        state.setdefault("input_mode", "pca2d")
        return model, state

    # Check if checkpoint is an encoder-decoder model
    is_encoder_decoder = (
        state.get("model_type") in ("encoder_decoder", "dima")
        or "encoder.input_proj.weight" in sd
        or (isinstance(state.get("model_state_dict"), dict) and "encoder.input_proj.weight" in state["model_state_dict"])
    )
    if is_encoder_decoder:
        from recomb2027.generation.latent_decoder.encoder_decoder import PCA2DEncoderDecoder
        encdec_sd = state.get("model_state_dict", sd)
        k = int(state.get("k", state.get("site_cond_dim", 4)))
        seq_len = int(state.get("seq_len", 712))
        d_model = int(state.get("d_model", 256))
        enc_layers = int(state.get("enc_layers", 3))
        dec_layers = int(state.get("dec_layers", 4))
        n_heads = int(state.get("n_heads", 4))
        ff_dim = int(state.get("ff_dim", 1024))
        dropout = float(state.get("dropout", 0.1))
        vocab_size = int(state.get("vocab_size", 22))
        bos_id = int(state.get("bos_id", 21))
        model = PCA2DEncoderDecoder(
            seq_len=seq_len,
            k=k,
            vocab_size=vocab_size,
            d_model=d_model,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            bos_id=bos_id,
        )
        model.load_state_dict(encdec_sd, strict=False)
        model = model.to(device).eval()
        if "site_cond_dim" not in state:
            state["site_cond_dim"] = k
        if "input_mode" not in state:
            state["input_mode"] = "pca2d"
        return model, state

    latent_dim = state.get("latent_dim")
    seq_len = state.get("seq_len")
    if latent_dim is None or seq_len is None:
        raise KeyError(
            "Decoder checkpoint missing latent_dim/seq_len. "
            f"Keys present: {sorted(state.keys())}"
        )

    model = SequenceDecoder(
        latent_dim=int(latent_dim),
        seq_len=int(seq_len),
        d_model=int(state.get("d_model", 256)),
        n_layers=int(state.get("n_layers", 4)),
        n_heads=int(state.get("n_heads", 4)),
        ff_dim=int(state.get("ff_dim", 1024)),
        dropout=float(state.get("dropout", 0.1)),
        vocab_size=int(state.get("vocab_size", 22)),
        bos_id=int(state.get("bos_id", 21)),
    )
    # If checkpoint indicates site conditioning, enable it before loading weights,
    # then load state dict with strict=False to allow older checkpoints.
    site_dim = state.get("site_cond_dim") or (state.get("training_config") or {}).get("site_cond_dim")
    if site_dim:
        try:
            model.enable_site_conditioning(int(site_dim))
        except Exception:
            pass
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model, state


def resolve_esm_repr_layer(
    cli_layer: int | None,
    decoder_meta: dict,
) -> int:
    if cli_layer is not None:
        return int(cli_layer)
    layer = decoder_meta.get("esm_repr_layer")
    if layer is not None:
        return int(layer)
    training = decoder_meta.get("training_config") or {}
    if training.get("esm_repr_layer") is not None:
        return int(training["esm_repr_layer"])
    return 34
