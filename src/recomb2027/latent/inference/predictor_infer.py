#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     predictor_infer.py
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

"""
This is the entry point to run the full procedure for predicting
the location of future protein variant in the latent space.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

import logging

from torch.utils.data import DataLoader

from recomb2027.latent.inference.gpt_lineage_substitution import (
    LineageSubstitutionIndex,
    build_substituted_model_input,
    prefix_edge_distances_with_substitute,
    scale_substitute_block,
    scale_substitute_extra,
    scale_substitute_mean,
    scaled_prefix_components_from_dataset,
    series_last_prefix_idx,
    to_gid_str,
)
from recomb2027.data.seq_pairwise_distance import (
    fast_pairwise_blosum_distance,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@torch.no_grad()
def infer_bilstm_encodings(
        bilstm_model: torch.nn.Module,
        dataset,
        device: torch.device,
        batch_size: int = 128,
        use_amp: bool = True
):
    """
    Adjust encodings with trained BiLSTM model.

    Args:
        bilstm_model: Trained BiLSTM model.
        dataset: VectorAsSequenceDataset
        device: torch.device("cuda") or torch.device("cpu").
        batch_size: Batch size for inference.
        use_amp: Use Automatic Mixed Precision on CUDA for faster inference.

    Returns:
        torch.Tensor: [N, T] or [N, *] encoded outputs (mean over last dim).
    """
    bilstm_model.eval()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=2
    )

    outputs = []

    autocast_enabled = (use_amp and device.type == "cuda")

    with torch.cuda.amp.autocast(enabled=autocast_enabled):
        for batch_x, _ in tqdm(data_loader, desc="Inferring BiLSTM encodings", unit="batch"):
            batch_x = batch_x.to(device, non_blocking=True)
            _, lstm_out = bilstm_model(batch_x)

            # Move to CPU in one go after detaching
            outputs.append(lstm_out.detach().cpu())

    # Concatenate and mean over last dimension (per your original code)
    X_enc = torch.cat(outputs, dim=0).mean(dim=-1)
    print(f"Inference complete: Encodings shape = {X_enc.shape}")

    return X_enc


def make_infer_bilstm_encodings_fn(
    bilstm_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 128,
    use_amp: bool = True,

) -> Callable[[np.ndarray], torch.Tensor]:
    """
    Creates a callable BiLSTM inference function that accepts a 2D numpy array X.

    This wrapper allows easy integration into external pipelines where the
    returned function can be passed as a parameter.

    Example:
        infer_fn = make_infer_bilstm_encodings_fn(bilstm_model, device)
        encoded = infer_fn(X)  # where X is a numpy array

    Args:
        bilstm_model: Trained BiLSTM model.
        device: torch.device("cuda") or torch.device("cpu").
        batch_size: Batch size for inference.
        use_amp: Use Automatic Mixed Precision on CUDA for faster inference.

    Returns:
        Callable[[np.ndarray], torch.Tensor]:
            Function that takes X (np.ndarray [N, D]) and returns encoded torch.Tensor [N, *].
    """

    bilstm_model = bilstm_model.to(device)
    bilstm_model.eval()
    autocast_enabled = (use_amp and device.type == "cuda")

    @torch.no_grad()
    def infer_fn(X: np.ndarray) -> torch.Tensor:
        """Infer BiLSTM encodings for a given 2D NumPy array X."""
        assert isinstance(X, np.ndarray), "Input X must be a NumPy array"
        assert X.ndim == 2, f"Expected X to be 2D, got shape {X.shape}"

        N = X.shape[0]
        outputs = []
        X_tensor = torch.from_numpy(X).float()

        for i in tqdm(range(0, N, batch_size), desc="Inferring BiLSTM encodings", unit="batch"):
            batch_x = X_tensor[i:i + batch_size].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                _, lstm_out = bilstm_model(batch_x)
            outputs.append(lstm_out.detach().cpu())

        X_enc = torch.cat(outputs, dim=0).mean(dim=-1)
        print(f"Inference complete: Encodings shape = {X_enc.shape}")
        return X_enc

    return infer_fn


@torch.no_grad()
def predict_with_model(
        model,
        dataset,
        batch_size,
        device,
        only_last_element=True
):
    """
    Run inference on a dataset using a trained model.

    Args:
        model: trained PyTorch model
        dataset: torch.utils.data.Dataset
        batch_size: int
        device: torch.device
        only_last_element: predict full sequence, otherwise only last element

    Returns:
        predictions: torch.Tensor (all predicted vectors)
    """
    device  = torch.device(device)
    # Move model to device
    model.to(device)
    model.eval()
    # Run prediciton loop
    all_preds = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for xb, _ in tqdm(loader, desc="Predicting", unit="batch"):
        xb = xb.to(device).float()
        #with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            y_pred = model(xb)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            if isinstance(y_pred, dict):
                from recomb2027.core.constants import PRIMARY_TARGET
                y_pred = y_pred[PRIMARY_TARGET]
            if only_last_element:
                y_pred = y_pred[:, -1, :]

        y_pred = y_pred.cpu()
        all_preds.append(y_pred)

    all_preds = torch.cat(all_preds, dim=0)

    logger.info(f"\nPredictions completed. Output tensor shape: {all_preds.shape}.")

    # --- Move model back to CPU and clear GPU memory ---
    model.to("cpu")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    logger.info("Model moved to CPU and GPU cache cleared.")

    return all_preds


@torch.no_grad()
def predict_with_lineage_substitution(
    model,
    dataset,
    substitution_index: LineageSubstitutionIndex,
    *,
    m_substitutes: int,
    batch_size: int,
    device,
    target_mode: str = "diff",
    only_last_element: bool = True,
) -> dict:
    """Draw up to ``m_substitutes`` GPT predictions per series via lineage substitution.

    For each validation series, the last prefix genome is replaced with eligible
    same-lineage alternatives (ordered by BLOSUM distance from the penultimate
    step). Returns a dict with:

    - ``pred_samples``: ``[N, M, D]`` predictions (NaN rows when fewer than M candidates)
    - ``substitute_gids``: ``[N, M]`` object array of substitute genome ids
    - ``substitute_distances``: ``[N, M]`` float array (NaN when unused slot)
    - ``plans``: list of ``LineageSubstitutePlan`` per series
    """
    if m_substitutes <= 0:
        raise ValueError("m_substitutes must be >= 1")

    device = torch.device(device)
    model.to(device)
    model.eval()

    n_series = len(dataset)
    if n_series == 0:
        raise ValueError("dataset is empty")

    sample0 = dataset[0]
    if isinstance(sample0[0], torch.Tensor):
        feat_dim = int(sample0[0].shape[-1])
    else:
        feat_dim = int(np.asarray(sample0[0]).shape[-1])
    out_dim = int(getattr(model, "hparams", {}).get("output_dim", feat_dim // 2))

    pred_samples = np.full((n_series, m_substitutes, out_dim), np.nan, dtype=np.float32)
    substitute_gids = np.full((n_series, m_substitutes), "", dtype=object)
    substitute_distances = np.full((n_series, m_substitutes), np.nan, dtype=np.float32)

    input_feature_mode = getattr(dataset, "input_feature_mode", "absolute")
    append_edge = bool(getattr(dataset, "append_edge_phylo_distance", False))
    edge_log1p = bool(getattr(dataset, "edge_phylo_use_log1p", True))
    extra_stats = list(dataset.get_extra_encoding_stats())
    block_names = list(dataset.get_block_encoding_names())
    series_len = int(dataset.get_block_size())
    last_idx = series_last_prefix_idx(series_len)

    plans = []
    pending_x: list[np.ndarray] = []
    pending_meta: list[tuple[int, int, np.ndarray]] = []  # (series_idx, slot, last_x)

    def _flush_pending():
        nonlocal pending_x, pending_meta
        if not pending_x:
            return
        xb = torch.from_numpy(np.stack(pending_x, axis=0)).to(device).float()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            y_pred = model(xb)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            if isinstance(y_pred, dict):
                from recomb2027.core.constants import PRIMARY_TARGET

                y_pred = y_pred[PRIMARY_TARGET]
            if only_last_element:
                y_pred = y_pred[:, -1, :]
        y_pred = y_pred.detach().cpu().numpy()
        for row, (si, slot, last_x) in zip(y_pred, pending_meta):
            out_row = row.astype(np.float32)
            if target_mode == "diff":
                out_row = out_row + last_x
            pred_samples[si, slot] = out_row
        pending_x = []
        pending_meta = []

    for si in tqdm(range(n_series), desc="Lineage-substitution GPT", unit="series"):
        row_gids = [to_gid_str(g) for g in dataset.sample_gids[si]]
        plan = substitution_index.plan_for_series(
            si,
            row_gids,
            last_idx,
            m_substitutes,
            extra_stats=extra_stats,
            block_names=block_names,
        )
        plans.append(plan)

        if not plan.candidates:
            continue

        x_abs, extra_prefix, block_prefix = scaled_prefix_components_from_dataset(
            dataset, si,
        )
        edge_row = None
        if append_edge:
            if dataset.edge_distances is None:
                raise RuntimeError(
                    "append_edge_phylo_distance=True but dataset.edge_distances is None"
                )
            edge_row = dataset.edge_distances[si]

        anchor_gid = plan.anchor_gid
        anchor_seq = substitution_index.gid_to_seq.get(anchor_gid)

        for slot, cand in enumerate(plan.candidates):
            if anchor_seq is None or cand.gid not in substitution_index.gid_to_seq:
                sub_dist = cand.distance
            else:
                sub_dist = float(
                    fast_pairwise_blosum_distance(
                        anchor_seq, substitution_index.gid_to_seq[cand.gid],
                    )
                )
            edge_d = None
            if append_edge:
                edge_d = prefix_edge_distances_with_substitute(
                    edge_row,
                    last_idx=last_idx,
                    sub_distance=sub_dist,
                )
            enc_sub_raw = substitution_index.gid_to_encoding[cand.gid]
            enc_sub = scale_substitute_mean(dataset, enc_sub_raw)

            substitute_extra: dict[str, np.ndarray] = {}
            for stat in extra_stats:
                raw = substitution_index.gid_to_extra[stat][cand.gid]
                substitute_extra[stat] = scale_substitute_extra(dataset, stat, raw)

            substitute_blocks: dict[str, np.ndarray] = {}
            for name in block_names:
                raw = substitution_index.gid_to_blocks[name][cand.gid]
                substitute_blocks[name] = scale_substitute_block(dataset, name, raw)

            x_in = build_substituted_model_input(
                x_abs,
                substitute_encoding=enc_sub,
                last_idx=last_idx,
                input_feature_mode=input_feature_mode,
                edge_distances=edge_d,
                edge_phylo_use_log1p=edge_log1p,
                extra_prefix=extra_prefix if extra_stats else None,
                substitute_extra=substitute_extra if extra_stats else None,
                extra_stats=extra_stats,
                block_prefix=block_prefix if block_names else None,
                substitute_blocks=substitute_blocks if block_names else None,
                block_names=block_names,
            )
            pending_x.append(x_in)
            pending_meta.append((si, slot, enc_sub[:out_dim]))
            substitute_gids[si, slot] = cand.gid
            substitute_distances[si, slot] = cand.distance

            if len(pending_x) >= batch_size:
                _flush_pending()

    _flush_pending()

    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    logger.info(
        "Lineage-substitution predictions completed. "
        f"pred_samples shape={pred_samples.shape}, "
        f"filled={(np.isfinite(pred_samples).any(axis=-1)).sum()} / {n_series * m_substitutes} slots."
    )

    return {
        "pred_samples": pred_samples,
        "substitute_gids": substitute_gids,
        "substitute_distances": substitute_distances,
        "plans": plans,
        "last_idx": last_idx,
        "target_mode": target_mode,
    }
