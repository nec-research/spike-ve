#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     summarize_experiment.py
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
"""Compare A4/B4 experiment runs (paper variants): latent skill, ridge provenance and decoded-sequence metrics.

For every results directory (``--run NAME=DIR``) it reads
  * ``a4b4_wo_edges_cal_test.json``        -> calibration / test latent skill,
  * ``pred_test_fused_decoded.fasta`` (+ ``validation_pairs_test.csv``)
                                           -> Hamming gain over copy-the-source and
                                              position-level precision / recall / F1,
and prints a markdown table (also written to ``--out-csv``). Gain CIs are paired
bootstrap (10,000 resamples) of ``H_src - H_dec`` per test series.

Example:
  python scripts/summarize_experiment.py \
    --pairs $SERIES/validation_pairs_test.csv \
    --run baseline=$P/preds/unified_pca2d_site_v4_a2_b2_b4_activation_mean_res_w_gaps_dedup_wo_edges_gatefix_A4B4 \
    --run no_cons=$P/preds/unified_pca2d_site_v4_a2_b2_b4_abl_no_cons_dedup_wo_edges_gatefix_A4B4 \
    --run conscurr=$P/preds/unified_pca2d_site_v4_a2_b2_b4_abl_conscurr_dedup_wo_edges_gatefix_A4B4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_fasta_by_row(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    row, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if row is not None:
                    out[row] = "".join(buf)
                row = int(line.split("row=")[1].split()[0])
                buf = []
            else:
                buf.append(line)
    if row is not None:
        out[row] = "".join(buf)
    return out


def boot_ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def decoded_metrics(pairs: pd.DataFrame, fasta: Path) -> dict[str, float]:
    dec = read_fasta_by_row(fasta)
    src = np.array([np.frombuffer(s.encode(), np.uint8) for s in pairs.seq_source])
    tgt = np.array([np.frombuffer(s.encode(), np.uint8) for s in pairs.seq_target])
    rows = pairs.series_row.to_numpy() if "series_row" in pairs else np.arange(len(pairs))
    dcd = np.array([np.frombuffer(dec[int(i)].encode(), np.uint8) for i in range(len(pairs))])
    h_src = (src != tgt).sum(1)
    h_dec = (dcd != tgt).sum(1)
    gain = (h_src - h_dec).astype(float)
    true_edit = src != tgt
    pred_edit = src != dcd
    # site-level hit (paper definition): predicted edit at a site that truly changes
    tp = int((true_edit & pred_edit).sum())
    fp = int((~true_edit & pred_edit).sum())
    fn = int((true_edit & ~pred_edit).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    lo, hi = boot_ci(gain)
    return {
        "H_src": float(h_src.mean()),
        "H_dec": float(h_dec.mean()),
        "gain": float(gain.mean()),
        "gain_lo": lo,
        "gain_hi": hi,
        "precision": prec,
        "recall": rec,
        "F1": 2 * prec * rec / max(prec + rec, 1e-12),
        "unique_decoded": len(set(dec.values())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pairs", required=True, help="validation_pairs_test.csv")
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--latent-json", default="a4b4_wo_edges_cal_test.json")
    ap.add_argument("--fasta", default="pred_test_fused_decoded.fasta")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs, usecols=["seq_source", "seq_target"])
    rows = []
    for spec in args.run:
        name, d = spec.split("=", 1)
        d = Path(d)
        rec: dict[str, object] = {"run": name}
        jp = d / args.latent_json
        if jp.is_file():
            sp = json.loads(jp.read_text())["splits"]
            rec.update(
                cal_skill_mu=sp["cal"]["skill_mu"], cal_skill_full=sp["cal"]["skill_full"],
                test_skill_mu=sp["test"]["skill_mu"], test_skill_full=sp["test"]["skill_full"],
            )
        fp = d / args.fasta
        if fp.is_file():
            rec.update(decoded_metrics(pairs, fp))
        else:
            print(f"[warn] {name}: no decoded FASTA at {fp}")
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 250, "display.max_columns", 50,
                  "display.float_format", lambda v: f"{v:.3f}")
    print(df.to_string())
    if args.out_csv:
        df.to_csv(args.out_csv)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
