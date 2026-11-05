#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     train_fused_decoder.py
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
#!/usr/bin/env python3
"""Train the fused non-autoregressive decoder: per-site PCA2D latents + mean-pooled ESM vector -> sequence.

See ``nar_fused_decoder.py`` for the fusion design. Optimiser defaults follow DiMA's transformer.yaml
(AdamW lr 1e-4, wd 1e-3, betas 0.9/0.98). Each modality gets DiMA-style noise augmentation in units of its own
between-sequence std, plus modality dropout (learned null embeddings).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from recomb2027.generation.latent_decoder import cycle_eval as ce
from recomb2027.generation.latent_decoder.nar_decoder import add_latent_noise
from recomb2027.generation.latent_decoder.nar_fused_decoder import PCA2DMeanFusedDecoder
from recomb2027.generation.latent_decoder.sequence_decoder import (
    DECODER_RESIDUES,
    _CHAR_TO_ID,
    tokenize_sequences,
)
from recomb2027.generation.latent_decoder.training.decoder_dataset import DecoderHoldoutData
from recomb2027.generation.latent_decoder.training.splits import (
    load_splits_from_pkl,
    parse_collection_date,
    parse_fasta_with_metadata,
    temporal_split_indices,
)
from recomb2027.latent.data_series.pca2d_features import load_pca2d_basis


def parse_args():
    p = argparse.ArgumentParser(description="Train fused PCA2D + mean non-autoregressive decoder.")
    p.add_argument("--fasta", required=True)
    p.add_argument("--pca2d-basis", required=True)
    p.add_argument("--enc-2d-dir", required=True, help="Per-genome 2D ESM encodings (layer dir or its parent).")
    p.add_argument("--splits-pkl", default=None)
    p.add_argument("--splits-train-key", default="train")
    p.add_argument("--splits-test-key", default="val")
    p.add_argument("--splits-id-field", default="genome_id")
    p.add_argument("--before-date", default=None,
                   help="Temporal cutoff YYYY-MM-DD (collection_date <= cutoff is 'before'; undated records are dropped). "
                        "WITHOUT --splits-pkl: train = all FASTA records on/before the date, test = --n-test records sampled "
                        "from those after it. WITH --splits-pkl: the split's train is filtered to <= date "
                        "(val unchanged unless --before-date-val).")
    p.add_argument("--n-test", type=int, default=2000,
                   help="Temporal mode (no --splits-pkl): number of test sequences sampled from after --before-date.")
    p.add_argument("--test-seed", type=int, default=42, help="Seed for the test (and --limit train) subsampling in temporal mode.")
    p.add_argument("--before-date-val", action="store_true",
                   help="With --splits-pkl: also apply --before-date to the validation split.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--embedding-cache-dir", default=None, help="Cache dir for PCA2D projections and mean vectors (default: FASTA dir).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.98])
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=3, help="Transformer layers (DiMA yaml: 3).")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n-global", type=int, default=4, help="Number of global tokens produced from the mean vector.")
    # noise augmentation (sigma in units of each modality's between-sequence std)
    p.add_argument("--noise-min", type=float, default=0.001)
    p.add_argument("--noise-max", type=float, default=0.5, help="PCA2D sites. Measured predictor displacement ~0.46 std.")
    p.add_argument("--mean-noise-min", type=float, default=0.001)
    p.add_argument("--mean-noise-max", type=float, default=0.75, help="Mean vector. Measured predictor displacement ~0.6 std.")
    p.add_argument("--modality-dropout", type=float, default=0.1, help="P(drop mean) = P(drop sites) = this value (never both). 0 disables.")
    p.add_argument("--val-noise", type=float, default=0.45, help="Sigma for noisy-val on sites.")
    p.add_argument("--val-mean-noise", type=float, default=0.6, help="Sigma for noisy-val on the mean.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def mean_vectors(gids, enc_dir, layer, seq_len, cache_file):
    """Mean-pooled raw ESM vector per gid from 2D encodings (cached)."""
    if cache_file is not None and cache_file.exists():
        c = torch.load(cache_file, map_location="cpu")
        if c["gids"] == list(gids):
            print(f"Loaded cached mean vectors from {cache_file}", flush=True)
            return c["means"], c["found"]
        print("Mean-vector cache gids mismatch; recomputing", flush=True)
    print(f"Computing mean-pooled vectors for {len(set(gids)):,} unique genomes (threaded file reads)...", flush=True)
    t0 = time.time()
    out = ce.load_source_latents(gids, Path(enc_dir), layer=layer, seq_len=seq_len, basis=None, num_workers=24)
    means = torch.from_numpy(out["mean"])
    found = torch.from_numpy(np.asarray(out["found"], dtype=bool))
    print(f"  done in {time.time() - t0:.0f}s; missing encodings: {int((~found).sum())}", flush=True)
    if cache_file is not None:
        torch.save({"gids": list(gids), "means": means, "found": found}, cache_file)
        print(f"Saved mean-vector cache -> {cache_file}", flush=True)
    return means, found


def filter_by_date(idxs, metas, cutoff):
    kept = []
    for i in idxs:
        d = parse_collection_date(metas[i])
        if d is not None and d <= cutoff:
            kept.append(i)
    return kept


def main():
    args = parse_args()
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    all_seqs, all_metas = parse_fasta_with_metadata(args.fasta)
    train_idx = test_idx = None
    if args.splits_pkl:
        train_idx, test_idx, _ = load_splits_from_pkl(
            args.splits_pkl, train_key=args.splits_train_key, test_key=args.splits_test_key,
            test_key_fallback="val", n_fasta=len(all_seqs), metas=all_metas, id_field=args.splits_id_field,
        )
        if args.before_date:
            from datetime import date

            cutoff = date.fromisoformat(args.before_date)
            n_tr0, n_te0 = len(train_idx), len(test_idx)
            train_idx = filter_by_date(train_idx, all_metas, cutoff)
            if args.before_date_val:
                test_idx = filter_by_date(test_idx, all_metas, cutoff)
            print(f"--before-date {args.before_date}: train {n_tr0} -> {len(train_idx)}; "
                  f"val {n_te0} -> {len(test_idx)}{'' if args.before_date_val else ' (unfiltered)'}", flush=True)
            if not train_idx or not test_idx:
                raise SystemExit("--before-date left an empty train or val set")
        if args.limit:
            train_idx = train_idx[: args.limit]
            test_idx = test_idx[: max(1, int(round(args.limit * args.test_fraction)))]
    if args.before_date and not args.splits_pkl:
        from datetime import date

        train_idx, test_idx = temporal_split_indices(
            all_metas, date.fromisoformat(args.before_date), n_test=args.n_test, seed=args.test_seed,
            max_train=args.limit,
        )
    data = DecoderHoldoutData(
        args.fasta, pca2d_basis=args.pca2d_basis, enc_2d_dir=args.enc_2d_dir, limit=args.limit,
        test_fraction=args.test_fraction, train_global_idxs=train_idx, test_global_idxs=test_idx,
        embedding_cache_dir=args.embedding_cache_dir,
    )
    layer = int(data.esm_repr_layer)
    enc_dir = Path(args.enc_2d_dir)
    if (enc_dir / str(layer)).is_dir() and next(enc_dir.glob("*.pt"), None) is None:
        enc_dir = enc_dir / str(layer)

    tr_gids = [str(all_metas[i].get("genome_id") or all_metas[i].get("id")) for i in data.train_global_idxs]
    te_gids = [str(all_metas[i].get("genome_id") or all_metas[i].get("id")) for i in data.test_global_idxs]
    cache_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else Path(args.fasta).resolve().parent
    cache_file = cache_dir / f"{Path(args.fasta).stem}.mean_layer{layer}.N{len(tr_gids) + len(te_gids)}.pt"
    seq_len = int(data.train_embeddings.shape[1])
    k = int(data.train_embeddings.shape[2])
    means, found = mean_vectors(tr_gids + te_gids, enc_dir, layer, seq_len, cache_file)
    tr_mean, te_mean = means[: len(tr_gids)], means[len(tr_gids):]
    tr_found, te_found = found[: len(tr_gids)], found[len(tr_gids):]
    D = int(tr_mean.shape[1])

    tr_tok = tokenize_sequences(data.train_sequences)
    te_tok = tokenize_sequences(data.test_sequences)
    tr_sites_all, te_sites_all = data.train_embeddings.float(), data.test_embeddings.float()

    # Integrity check: PCA2D rows and mean rows must belong to the same genomes. Exact identity:
    #   mean_i z_i == (mean - mu) @ W^T
    basis = load_pca2d_basis(args.pca2d_basis)
    W, mu_b = torch.from_numpy(basis["W"]), torch.from_numpy(basis["mu"])
    probe = [i for i in range(len(tr_gids)) if bool(tr_found[i])][:32]
    err = (tr_sites_all[probe].mean(1) - (tr_mean[probe] - mu_b) @ W.T).abs().max().item()
    print(f"alignment check (mean of PCA2D sites == (mean-mu)W^T) max abs err = {err:.2e}", flush=True)
    if err > 1e-2:
        raise RuntimeError(
            "PCA2D rows and mean-vector rows are misaligned (stale PCA2D cache for a different split?). "
            "Delete the cached *.pca2d_*.pt file or point --embedding-cache-dir to a fresh dir."
        )

    # Drop genomes without a 2D encoding (their zero vectors would corrupt statistics and training).
    if int((~tr_found).sum()) or int((~te_found).sum()):
        print(f"dropping {int((~tr_found).sum())} train / {int((~te_found).sum())} val rows without encodings", flush=True)
        tr_sites_all, tr_mean, tr_tok = tr_sites_all[tr_found], tr_mean[tr_found], tr_tok[tr_found]
        te_sites_all, te_mean, te_tok = te_sites_all[te_found], te_mean[te_found], te_tok[te_found]
    vocab = len(DECODER_RESIDUES) + 1
    gap_id = _CHAR_TO_ID["-"]
    print(f"train={len(tr_tok):,} val={len(te_tok):,} seq_len={seq_len} k={k} mean_dim={D}", flush=True)

    # ---- normalisation statistics (train only) ----
    site_mean = tr_sites_all.mean(0)
    latent_scale = (tr_sites_all - site_mean).reshape(-1, k).std(0)
    mean_mu = tr_mean.mean(0)
    mean_scale = tr_mean.std(0).clamp_min(1e-4)
    print("site latent_scale:", [round(x, 3) for x in latent_scale.tolist()],
          "| mean_scale per-dim min/median/max:",
          [round(float(f), 4) for f in (mean_scale.min(), mean_scale.median(), mean_scale.max())], flush=True)

    model = PCA2DMeanFusedDecoder(
        seq_len=seq_len, k=k, mean_dim=D, vocab_size=vocab, d_model=args.d_model, n_layers=args.layers,
        n_heads=args.n_heads, ff_dim=args.ff_dim, dropout=args.dropout, bos_id=vocab - 1, n_global=args.n_global,
        latent_scale=latent_scale, site_mean=site_mean, mean_mu=mean_mu, mean_scale=mean_scale,
    ).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    ls_dev, ms_dev = latent_scale.to(device), mean_scale.to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    train_loader = DataLoader(TensorDataset(tr_sites_all, tr_mean.float(), tr_tok.long()),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(te_sites_all, te_mean.float(), te_tok.long()),
                            batch_size=args.batch_size, shuffle=False)

    def run_eval(noisy: bool, mode: str):
        """mode: both | pca2d_only | mean_only. Returns (loss, nongap_acc)."""
        g = torch.Generator().manual_seed(0)
        model.eval()
        ls = nl = nc = nt = 0
        with torch.no_grad():
            for z, m, toks in val_loader:
                z, m, toks = z.to(device), m.to(device), toks.to(device)
                if noisy:
                    z = z + args.val_noise * ls_dev * torch.randn(z.shape, generator=g).to(device)
                    m = m + args.val_mean_noise * ms_dev * torch.randn(m.shape, generator=g).to(device)
                b = z.shape[0]
                dm = torch.full((b,), mode == "pca2d_only", dtype=torch.bool, device=device)
                ds = torch.full((b,), mode == "mean_only", dtype=torch.bool, device=device)
                logits = model(model.pack(z, m), drop_mean=dm, drop_sites=ds)
                ls += F.cross_entropy(logits.reshape(-1, vocab), toks.reshape(-1), reduction="sum").item()
                nl += toks.numel()
                msk = toks != gap_id
                nc += int(((logits.argmax(-1) == toks) & msk).sum().item())
                nt += int(msk.sum().item())
        return ls / nl, nc / max(1, nt)

    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tl = tn = 0
        for step, (z, m, toks) in enumerate(train_loader, 1):
            z, m, toks = z.to(device), m.to(device), toks.to(device)
            b = z.shape[0]
            z = add_latent_noise(z, ls_dev, args.noise_min, args.noise_max)
            m = add_latent_noise(m, ms_dev, args.mean_noise_min, args.mean_noise_max)
            u = torch.rand(b, device=device)
            p = args.modality_dropout
            dm, ds = u < p, (u >= p) & (u < 2 * p)
            logits = model(model.pack(z, m), drop_mean=dm, drop_sites=ds)
            loss = F.cross_entropy(logits.reshape(-1, vocab), toks.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * b
            tn += b
            if step % max(1, len(train_loader) // 5) == 0:
                print(f"  epoch {epoch} [{step}/{len(train_loader)}] loss {loss.item():.2e}", flush=True)
        sched.step()
        v_loss, v_acc = run_eval(False, "both")
        n_loss, n_acc = run_eval(True, "both")
        _, p_acc = run_eval(False, "pca2d_only")
        _, mo_acc = run_eval(False, "mean_only")
        rec = {"epoch": epoch, "train_loss": tl / tn, "val_loss": v_loss, "val_nongap": v_acc,
               "val_noisy_loss": n_loss, "val_noisy_nongap": n_acc,
               "val_pca2d_only_nongap": p_acc, "val_mean_only_nongap": mo_acc, "lr": opt.param_groups[0]["lr"]}
        history.append(rec)
        print(f"Epoch {epoch:02d}/{args.epochs} ({time.time() - t0:.0f}s) | train {rec['train_loss']:.2e} | "
              f"val {v_loss:.2e} acc {v_acc:.5f} | noisy {n_loss:.2e} acc {n_acc:.5f} | "
              f"pca2d-only {p_acc:.5f} | mean-only {mo_acc:.5f}", flush=True)

        ckpt = {
            "model_type": "nar_fused", "model_state_dict": model.state_dict(),
            "k": k, "seq_len": seq_len, "mean_dim": D, "vocab_size": vocab, "d_model": args.d_model,
            "n_layers": args.layers, "n_heads": args.n_heads, "ff_dim": args.ff_dim, "dropout": args.dropout,
            "bos_id": vocab - 1, "n_global": args.n_global, "esm_repr_layer": layer,
            "train_global_idxs": data.train_global_idxs, "test_global_idxs": data.test_global_idxs,
            "training_config": vars(args), "epoch": epoch, "val_loss": v_loss, "val_noisy_loss": n_loss,
            "val_nongap_acc": v_acc,
        }
        torch.save(ckpt, run_dir / "latest.pt")
        # select on NOISY val loss: inference inputs are predictions, not clean latents
        if n_loss < best:
            best = n_loss
            torch.save(ckpt, run_dir / "best.pt")
            print(f"  * new best (noisy val loss {n_loss:.3e}) -> {run_dir / 'best.pt'}", flush=True)
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Training completed. Checkpoints in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
