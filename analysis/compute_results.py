#!/usr/bin/env python3
"""Recompute every number, table and figure used in the paper from the results directory.

Reads (read-only):
  <RES>/b3_mean_res_cal_test.json                          latent-space skill
  <RES>/evaluation_report_fused.csv                        per-series decoder / kNN report (1,422 test series)
  <RES>/b3_test_fused_decoded.csv                          decoder summary (unique decodes, cycle distances)
  <RES>/evaluation_plots_fused*/{cohort_summary.json,paired_decoder_vs_knn_by_bin.csv}
  <RES>/evaluation_prefix_swap_fused/{prefix_swap_evaluation_report.csv,prefix_swap_slots_long.csv}
  <D>/phylotree_matched/filtered_msa_S1_dedup.fasta        aligned S1 sequences (+ dates)
  <D>/latent_variant_prediction/models/checkpoints/...     parameter counts and checkpoint metadata

Writes (next to this script's parent directory):
  numbers.json, tables/*.tex, figures/*.pdf

Usage:  python3 analysis/compute_results.py
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = Path("")
RES = D / ("latent_variant_prediction/preds/"
           "unified_pca2d_site_v4_a2_b2_b4_abl_conscurr_dedup_wo_edges_gatefix_A4B4")
LATENT_JSON = "a4b4_wo_edges_cal_test.json"
DECODED_CSV = "a4b4_wo_edges_test_fused_decoded.csv"
B_CKPT = "unified_pca2d_site_v4_a2_b2_b4_abl_conscurr_dedup_wo_edges_gatefix_A4B4/Pca2dSiteUnifiedV4_best_model.pt"
A_CKPT = ("gpt_jepa_mean_pca2dk4_s1_site_pool_attn_abc_flat_decode_rank256_site_gate_w_gaps_dedup_A4_abl_conscurr/"
          "GPTModelPca2dSite_best_model.pt")
DEC_CKPT = "latent_variant_prediction/dima_nar_fused/fused_v1_AB"
MSA = D / "phylotree_matched/filtered_msa_S1_dedup.fasta"
CKPT = D / "latent_variant_prediction/models/checkpoints/layer33"
PAPER = Path(__file__).resolve().parents[1]
TAB = PAPER / "tables"
FIG = PAPER / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

CUTOFF = "2022-12-01"
RNG = np.random.default_rng(0)
N_BOOT = 10_000
BIN_EDGES = [0, 1, 2, 3, 5, 10_000]
BIN_LABELS = ["1", "2", "3", "4--5", "$\\geq$6"]
BIN_LABELS_PLAIN = ["1", "2", "3", "4-5", "6+"]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150})

N: dict = {}  # everything that ends up in numbers.json
PREV = PAPER / "analysis" / "preview"
PREV.mkdir(exist_ok=True)


def save(fig, name, **kw):
    fig.savefig(FIG / f"{name}.pdf", **kw)
    fig.savefig(PREV / f"{name}.png", dpi=130, **kw)



def f(x, nd=3):
    return f"{x:.{nd}f}"


def ham(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))


def boot_ci(x: np.ndarray, stat=np.mean):
    x = np.asarray(x, dtype=float)
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    vals = stat(x[idx], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def prf(tp, fn, fp):
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return rec, prec, f1


# ----------------------------------------------------------------------------------------------
# 1. Latent-space skill
# ----------------------------------------------------------------------------------------------
lat = json.loads((RES / LATENT_JSON).read_text())
N["latent"] = {
    sp: {k: lat["splits"][sp][k] for k in ("n", "persist_mse", "mu_mse", "full_mse", "skill_mu", "skill_full",
                                          "r2_pc_mean", "r2_site_mean")}
    for sp in ("cal", "test")
}
rows = []
for sp, name in (("cal", "Calibration"), ("test", "Test")):
    s = lat["splits"][sp]
    rows.append(f"{name} & {int(s['n']):,} & {f(s['persist_mse'])} & {f(s['mu_mse'])} & {f(s['full_mse'])} & "
                f"{f(s['skill_mu'])} & {f(s['skill_full'])} \\\\")
(TAB / "tab_latent.tex").write_text(
    "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    " & & \\multicolumn{3}{c}{MSE} & \\multicolumn{2}{c}{Skill} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
    "Split & $N$ & Persist. & A4 & A4+B4 & A4 & A4+B4 \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------------------------
# 2. Checkpoint metadata / parameter counts
# ----------------------------------------------------------------------------------------------
try:
    import torch

    b3 = torch.load(CKPT / B_CKPT, map_location="cpu", weights_only=False)
    a3 = torch.load(CKPT / A_CKPT, map_location="cpu", weights_only=False)
    dec = torch.load(D / DEC_CKPT / "best.pt", map_location="cpu", weights_only=False)
    n_dec_buf = sum(v.numel() for k, v in dec["model_state_dict"].items()
                    if k.split(".")[0] in ("site_mean", "latent_scale", "mean_mu", "mean_scale"))
    N["models"] = {
        "b3_best_epoch": int(b3["epoch"]),
        "b3_best_skill_full_cal": float(b3["best_val"]),
        "b3_trainable_params": int(sum(v.numel() for v in b3["trainable_state_dict"].values())),
        "a3_params": int(sum(v.numel() for v in a3["model_state_dict"].values())),
        "a3_best_epoch": int(a3["best_epoch"]),
        "b3_total_params": int(sum(v.numel() for v in b3["model_state_dict"].values())),
        "a3_val_select": float(a3["loss"]),
        "decoder_params": int(sum(v.numel() for v in dec["model_state_dict"].values()) - n_dec_buf),
        "decoder_epoch": int(dec["epoch"]),
        "decoder_train_genomes": len(dec["train_global_idxs"]),
        "decoder_val_genomes": len(dec["test_global_idxs"]),
        "decoder_val_nongap_acc": float(dec["val_nongap_acc"]),
    }
    hist = json.loads((D / DEC_CKPT / "training_history.json").read_text())
    last = hist[-1]
    N["decoder_training"] = {k: last[k] for k in ("val_nongap", "val_noisy_nongap", "val_pca2d_only_nongap",
                                                  "val_mean_only_nongap")}
except Exception as exc:  # pragma: no cover
    N["models_error"] = repr(exc)

# ----------------------------------------------------------------------------------------------
# 3. Cohort description
# ----------------------------------------------------------------------------------------------
d = pd.read_csv(RES / "evaluation_report_fused.csv")
n = len(d)
gt = d.gt_hamming_source_target.to_numpy()
d["h_src"] = gt
d["h_dec"] = d.decoded_hamming_to_target.to_numpy()
d["bin"] = pd.cut(d.h_src, bins=BIN_EDGES, labels=BIN_LABELS_PLAIN, right=True)
N["cohort"] = {
    "n_series": n,
    "n_unique_sources": int(d.seq_source.nunique()),
    "n_unique_targets": int(d.seq_target.nunique()),
    "source_date_min": d.date_source.min(), "source_date_max": d.date_source.max(),
    "target_date_min": d.date_target.min(), "target_date_max": d.date_target.max(),
    "gt_hamming_mean": float(gt.mean()), "gt_hamming_median": float(np.median(gt)),
    "gt_hamming_ge10": int((gt >= 10).sum()), "gt_hamming_max": int(gt.max()),
    "gt_hamming_le3_frac": float((gt <= 3).mean()),
    "total_required_mutations": int(gt.sum()),
    "same_lineage_frac": float((d.lineage_source == d.lineage_target).mean()),
    "top_target_lineages": d.lineage_target.value_counts().head(6).to_dict(),
    "n_distinct_target_lineages": int(d.lineage_target.nunique()),
    "unique_continuation_n": int((d.n_branch_targets_for_prefix == 1).sum()),
    "branch_ge2_n": int((d.n_branch_targets_for_prefix >= 2).sum()),
    "branch_ge4_n": int((d.n_branch_targets_for_prefix >= 4).sum()),
    "branch_max": int(d.n_branch_targets_for_prefix.max()),
    "bin_counts": d.bin.value_counts().reindex(BIN_LABELS_PLAIN).astype(int).to_dict(),
}

# ----------------------------------------------------------------------------------------------
# 4. Main sequence-level comparison on the full test cohort
# ----------------------------------------------------------------------------------------------
dec_zero_edit = float((d.decoded_hamming_to_source == 0).mean())
diff = d.h_src.to_numpy() - d.h_dec.to_numpy()  # >0: decoded sequence closer to the target than the source
N["hamming"] = {
    "source_to_target_mean": float(d.h_src.mean()),
    "decoder_to_target_mean": float(d.h_dec.mean()),
    "decoder_to_target_median": float(d.h_dec.median()),
    "mean_gain": float(diff.mean()),
    "mean_gain_ci": boot_ci(diff),
    "decoder_closer_frac": float((diff > 0).mean()),
    "decoder_equal_frac": float((diff == 0).mean()),
    "decoder_farther_frac": float((diff < 0).mean()),
    "decoder_equals_source_frac": dec_zero_edit,
    "decoder_equals_target_n": int((d.h_dec == 0).sum()),
    "decoder_mean_edits_from_source": float(d.decoded_hamming_to_source.mean()),
    "blast_id_dec": float(d.blast_identity_decoded_to_target.mean()),
    "blast_id_src": float(d.blast_identity_source_to_target.mean()),
    "decoder_matches_any_branch_n": int(d.decoder_matches_any_branch.astype(bool).sum()),
}

# position-level (change-from-source) confusion, micro-averaged over all 1,422 series
tp, fn, fp = d.decoder_position_tp.sum(), d.decoder_position_fn.sum(), d.decoder_position_fp.sum()
rec, prec, f1 = prf(tp, fn, fp)
N["decoder_position_micro"] = dict(tp=int(tp), fn=int(fn), fp=int(fp), recall=rec, precision=prec, f1=f1)
atp, afn, afp = d.decoder_tp.sum(), d.decoder_fn.sum(), d.decoder_fp_alleles_count.sum()
rec_a, prec_a, f1_a = prf(atp, afn, afp)
N["decoder_allele_micro"] = dict(tp=int(atp), fn=int(afn), fp=int(afp), recall=rec_a, precision=prec_a, f1=f1_a)
# fraction of position-level TP that carry the wrong allele
N["decoder_wrong_allele_at_required_site"] = int(tp - atp)

# aggregate over the series with at least two required edits (the regime the forecaster is used in)
m2 = d.h_src >= 2
g2 = (d.h_src - d.h_dec).to_numpy()[m2.to_numpy()]
t2, f2_, p2 = (d[m2].decoder_position_tp.sum(), d[m2].decoder_position_fn.sum(), d[m2].decoder_position_fp.sum())
r2, pr2, f12 = prf(t2, f2_, p2)
N["ge2_edits"] = dict(n=int(m2.sum()), frac=float(m2.mean()), recall=r2, precision=pr2, f1=f12,
                      h_src=float(d[m2].h_src.mean()), h_dec=float(d[m2].h_dec.mean()),
                      gain=float(g2.mean()), gain_ci=boot_ci(g2),
                      closer=float((g2 > 0).mean()), farther=float((g2 < 0).mean()))
# of the required columns the decoder locates, the share that also carries the correct residue
N["allele_accuracy_at_located_sites"] = float(atp / tp)

# kNN: closest neighbour (1,166 series with a neighbour record) and consensus pool (all series)
has_knn = d.closest_knn_tp.notna()
N["knn"] = {"n_with_closest": int(has_knn.sum()), "n_without_closest": int((~has_knn).sum())}
k = d[has_knn].copy()
k["h_knn"] = [ham(a, b) for a, b in zip(k.knn_closest_sequence, k.seq_target)]
k["h_knn_src"] = [ham(a, b) for a, b in zip(k.knn_closest_sequence, k.seq_source)]
N["knn"]["closest_to_target_mean"] = float(k.h_knn.mean())
N["knn"]["source_to_target_mean_same_subset"] = float(k.h_src.mean())
N["knn"]["decoder_to_target_mean_same_subset"] = float(k.h_dec.mean())
N["knn"]["closest_equals_source_frac"] = float((k.h_knn_src == 0).mean())
kd = k.h_src.to_numpy() - k.h_knn.to_numpy()
N["knn"]["closest_gain_mean"] = float(kd.mean())
N["knn"]["closest_gain_ci"] = boot_ci(kd)
dd = k.h_src.to_numpy() - k.h_dec.to_numpy()
N["knn"]["decoder_gain_mean_same_subset"] = float(dd.mean())
N["knn"]["decoder_gain_ci_same_subset"] = boot_ci(dd)
paired = k.h_knn.to_numpy() - k.h_dec.to_numpy()  # >0: decoder closer than closest-kNN
N["knn"]["decoder_minus_knn_hamming_mean"] = float(-paired.mean())
N["knn"]["decoder_minus_knn_hamming_ci"] = tuple(-v for v in boot_ci(paired))[::-1]
N["knn"]["decoder_closer_than_knn_frac"] = float((paired > 0).mean())
N["knn"]["decoder_farther_than_knn_frac"] = float((paired < 0).mean())

ck_tp, ck_fn, ck_fp = k.closest_knn_tp.sum(), k.closest_knn_fn.sum(), k.closest_knn_fp.sum()
N["knn"]["closest_position_micro"] = dict(zip(("recall", "precision", "f1"), prf(ck_tp, ck_fn, ck_fp)),
                                          tp=int(ck_tp), fn=int(ck_fn), fp=int(ck_fp))
dtp, dfn, dfp = k.decoder_position_tp.sum(), k.decoder_position_fn.sum(), k.decoder_position_fp.sum()
N["knn"]["decoder_position_micro_same_subset"] = dict(zip(("recall", "precision", "f1"), prf(dtp, dfn, dfp)),
                                                      tp=int(dtp), fn=int(dfn), fp=int(dfp))
pt, pf, pp = d.knn_tp.sum(), d.knn_fn.sum(), d.knn_fp.sum()
N["knn"]["pool_position_micro"] = dict(zip(("recall", "precision", "f1"), prf(pt, pf, pp)),
                                       tp=int(pt), fn=int(pf), fp=int(pp))

# per-series macro-averaged F1 etc. as exported by the pipeline (paired comparison on the 1,166-series subset)
pb = pd.read_csv(RES / "evaluation_plots_fused/paired_decoder_vs_knn_by_bin.csv")
pbu = pd.read_csv(RES / "evaluation_plots_fused_unique/paired_decoder_vs_knn_by_bin.csv")
N["paired_all_bins"] = pb[pb["bin"] == "All"].to_dict(orient="records")
N["paired_unique_all_bins"] = pbu[pbu["bin"] == "All"].to_dict(orient="records")

# main table: every method scored on the same 1,422 series
def bold(v, best_, nd=3):
    t = f(v, nd)
    return f"\\textbf{{{t}}}" if abs(v - best_) < 1e-12 else t


dpm = N["decoder_position_micro"]
kp = N["knn"]
kcl, kpool = kp["closest_position_micro"], kp["pool_position_micro"]
best_p = max(dpm["precision"], kcl["precision"], kpool["precision"])
best_f1 = max(dpm["f1"], kcl["f1"], kpool["f1"])
best_h = min(float(d.h_dec.mean()), float(k.h_knn.mean()), float(d.h_src.mean()))
best_g = max(kp["closest_gain_mean"], N["hamming"]["mean_gain"], 0.0)


def gain_cell(g_, ci_):
    return f"{bold(g_, best_g, 2)} [${ci_[0]:+.2f}$, ${ci_[1]:+.2f}$]"


rows = [
    f"Copy last-seen sequence & 0.000 & --- & --- & {bold(float(d.h_src.mean()), best_h, 2)} & --- \\\\",
    f"kNN consensus pool ($k{{=}}20$) & \\textbf{{{f(kpool['recall'])}}} & {bold(kpool['precision'], best_p)} & "
    f"{bold(kpool['f1'], best_f1)} & --- & --- \\\\",
    f"Closest kNN to the forecast & {f(kcl['recall'])} & {bold(kcl['precision'], best_p)} & "
    f"{bold(kcl['f1'], best_f1)} & {bold(float(k.h_knn.mean()), best_h, 2)} & "
    f"{gain_cell(kp['closest_gain_mean'], kp['closest_gain_ci'])} \\\\",
    f"Latent forecast + decoder (ours) & {f(dpm['recall'])} & {bold(dpm['precision'], best_p)} & "
    f"{bold(dpm['f1'], best_f1)} & {bold(float(d.h_dec.mean()), best_h, 2)} & "
    f"{gain_cell(N['hamming']['mean_gain'], N['hamming']['mean_gain_ci'])} \\\\",
]
(TAB / "tab_main.tex").write_text(
    "\\begin{tabular}{lrrrrr}\n\\toprule\n"
    "Method & Recall & Precision & F1 & $H$ to target & Gain over copying [95\\% CI] \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------------------------
# 5. Stratification by number of true edits
# ----------------------------------------------------------------------------------------------
strat = []
for lab, lab_tex in zip(BIN_LABELS_PLAIN, BIN_LABELS):
    s = d[d.bin == lab]
    tp_, fn_, fp_ = s.decoder_position_tp.sum(), s.decoder_position_fn.sum(), s.decoder_position_fp.sum()
    r_, p_, f_ = prf(tp_, fn_, fp_)
    sk = s[s.closest_knn_tp.notna()]
    kr, kpr, kf = prf(sk.closest_knn_tp.sum(), sk.closest_knn_fn.sum(), sk.closest_knn_fp.sum())
    dr, dpr, df_ = prf(sk.decoder_position_tp.sum(), sk.decoder_position_fn.sum(), sk.decoder_position_fp.sum())
    g = (s.h_src - s.h_dec).to_numpy()
    strat.append(dict(bin=lab, bin_tex=lab_tex, n=len(s), n_knn=len(sk), dec_recall=r_, dec_precision=p_, dec_f1=f_,
                      knn_recall=kr, knn_precision=kpr, knn_f1=kf, dec_recall_sub=dr, dec_precision_sub=dpr,
                      dec_f1_sub=df_, h_src=float(s.h_src.mean()), h_dec=float(s.h_dec.mean()),
                      gain=float(g.mean()), gain_ci=boot_ci(g), closer=float((g > 0).mean()),
                      farther=float((g < 0).mean())))
N["stratified"] = strat
rows = []
for s in strat:
    rows.append(f"{s['bin_tex']} & {s['n']} & {f(s['dec_recall'])} & {f(s['dec_precision'])} & {f(s['dec_f1'])} & "
                f"{f(s['h_src'], 2)} & {f(s['h_dec'], 2)} & ${s['gain']:+.2f}$ [${s['gain_ci'][0]:+.2f}$, ${s['gain_ci'][1]:+.2f}$] \\\\")
(TAB / "tab_strat.tex").write_text(
    "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
    "True edits & $N$ & Recall & Precision & F1 & $H_{\\mathrm{src}}$ & $H_{\\mathrm{dec}}$ & Gain [95\\% CI] \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# paired decoder vs closest-kNN per bin (macro, per-series F1; as exported by the pipeline)
rows = []
for b in ["1", "2", "3", "4-5", "6+", "All"]:
    r = pb[(pb["bin"] == b) & (pb.definition == "pos")].set_index("metric")
    bt = {"4-5": "4--5", "6+": "$\\geq$6"}.get(b, b)
    rr, ff = r.loc["recall"], r.loc["f1"]
    rows.append(f"{bt} & {int(rr.n)} & {f(rr.mean_decoder)} / {f(rr.mean_knn)} & "
                f"${rr.mean_diff:+.3f}$ [${rr.ci_lo:+.3f}$, ${rr.ci_hi:+.3f}$] & {f(ff.mean_decoder)} / {f(ff.mean_knn)} & "
                f"${ff.mean_diff:+.3f}$ [${ff.ci_lo:+.3f}$, ${ff.ci_hi:+.3f}$] \\\\")
    if b == "6+":
        rows.append("\\midrule")
(TAB / "tab_paired.tex").write_text(
    "\\begin{tabular}{lrcccc}\n\\toprule\n"
    " & & \\multicolumn{2}{c}{Recall (dec./kNN)} & \\multicolumn{2}{c}{F1 (dec./kNN)} \\\\\n"
    "True edits & $N$ & means & diff. [95\\% CI] & means & diff. [95\\% CI] \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------------------------
# 6. Branching (several observed futures per prefix)
# ----------------------------------------------------------------------------------------------
cs_u = json.loads((RES / "evaluation_plots_fused_unique/cohort_summary.json").read_text())
cs_a = json.loads((RES / "evaluation_plots_fused/cohort_summary.json").read_text())
N["cohort_summary_all"] = {m["method"]: m for m in cs_a["methods_position"]}
N["cohort_summary_unique"] = {m["method"]: m for m in cs_u["methods_position"]}
N["cohort_summary_all_allele"] = {m["method"]: m for m in cs_a["methods_allele"]}
N["cohort_summary_unique_allele"] = {m["method"]: m for m in cs_u["methods_allele"]}
d["nb_cls"] = pd.cut(d.n_branch_targets_for_prefix, bins=[0, 1, 3, 10_000], labels=["1", "2-3", "4+"])
br = []
for lab in ["1", "2-3", "4+"]:
    s = d[d.nb_cls == lab]
    tp_, fn_, fp_ = s.decoder_position_tp.sum(), s.decoder_position_fn.sum(), s.decoder_position_fp.sum()
    r_, p_, f_ = prf(tp_, fn_, fp_)
    g = (s.h_src - s.h_dec).to_numpy()
    mh = s.decoder_min_hamming_to_branch_targets.to_numpy()
    br.append(dict(cls=lab, n=len(s), recall=r_, precision=p_, f1=f_, h_src=float(s.h_src.mean()),
                   h_dec=float(s.h_dec.mean()), gain=float(g.mean()), gain_ci=boot_ci(g),
                   h_dec_min_branch=float(mh.mean()), n_unique_prefix_targets=float(s.n_branch_targets_for_prefix.mean())))
N["branching"] = br
rows = []
for b in br:
    rows.append(f"{b['cls']} & {b['n']} & {f(b['recall'])} & {f(b['precision'])} & {f(b['f1'])} & "
                f"{f(b['h_src'], 2)} & {f(b['h_dec'], 2)} & ${b['gain']:+.2f}$ [${b['gain_ci'][0]:+.2f}$, ${b['gain_ci'][1]:+.2f}$] \\\\")
(TAB / "tab_branch.tex").write_text(
    "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
    "Futures per prefix & $N$ & Recall & Precision & F1 & $H_{\\mathrm{src}}$ & $H_{\\mathrm{dec}}$ & Gain [95\\% CI] \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# all-branch (shared-by-every-future) recall, position level, as exported
N["all_branch_position"] = {
    m["method"]: m for m in cs_a["methods_position"] if m["method"] in ("decoder_all_branches", "closest_knn_all_branches")
}

# ----------------------------------------------------------------------------------------------
# 7. Decoder behaviour: collapse and novelty
# ----------------------------------------------------------------------------------------------
seqs, dates = [], []
cur = None
with open(MSA) as fh:
    buf = []
    for line in fh:
        line = line.strip()
        if line.startswith(">"):
            if cur is not None:
                seqs.append("".join(buf).upper())
            meta = dict(x.split(":", 1) for x in line[1:].split("|"))
            dates.append(meta.get("collection_date", ""))
            cur = line
            buf = []
        elif line:
            buf.append(line)
    seqs.append("".join(buf).upper())
before = {s for s, dt in zip(seqs, dates) if dt and dt != "nan" and dt <= CUTOFF}
anyset = set(seqs)
dec_seqs = d.decoded_prediction_sequence.tolist()
uniq = set(dec_seqs)
cnt = Counter(dec_seqs)
N["decoder_behaviour"] = {
    "n_unique_decoded": len(uniq),
    "unique_decoded_frac": len(uniq) / n,
    "decoded_in_msa_any_date_frac": float(np.mean([s in anyset for s in dec_seqs])),
    "decoded_in_msa_before_cutoff_frac": float(np.mean([s in before for s in dec_seqs])),
    "unique_decoded_in_msa_before_cutoff_frac": float(np.mean([s in before for s in uniq])),
    "decoded_novel_vs_all_msa_frac": float(np.mean([s not in anyset for s in dec_seqs])),
    "most_common_decoded_count": cnt.most_common(1)[0][1],
    "top10_decoded_share": sum(c for _, c in cnt.most_common(10)) / n,
    "n_msa": len(seqs), "n_msa_before_cutoff": len(before),
}
summ = pd.read_csv(RES / DECODED_CSV)
sm = summ[summ.which == "summary"].set_index("key").value
N["decoder_behaviour"].update({
    "cycle_pca2d_cos_mean": float(sm["cycle_pca2d_cos_dist_mean"]),
    "cycle_mean_cos_mean": float(sm["cycle_mean_cos_dist_mean"]),
    "src_identity_mean": float(sm["src_identity_mean"]),
    "src_exact_match_frac": float(sm["src_exact_match_frac"]),
    "n_unique_decoded_csv": int(float(sm["n_unique_decoded"])),
})
# novelty among the decoded sequences that differ from the source
nz = d[d.decoded_hamming_to_source > 0]
N["decoder_behaviour"]["edited_n"] = len(nz)
N["decoder_behaviour"]["edited_in_msa_any_frac"] = float(np.mean([s in anyset for s in nz.decoded_prediction_sequence]))
N["decoder_behaviour"]["edited_in_msa_before_frac"] = float(np.mean([s in before for s in nz.decoded_prediction_sequence]))

# ----------------------------------------------------------------------------------------------
# 8. Site-level analysis (alignment columns; spike numbering via the reference record)
# ----------------------------------------------------------------------------------------------
ref = seqs[0]
col2res, c = {}, 0
for i, ch in enumerate(ref, start=1):
    if ch != "-":
        c += 1
        col2res[i] = c
N["reference_first_record_prefix"] = ref[:30]


def parse_list(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str) or not x:
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


req_mut = Counter()
tp_mut = Counter()
req_site = Counter()
tp_site = Counter()
for r in d.itertuples():
    req = parse_list(r.required_mutations)
    tpa = set(parse_list(r.decoder_tp_alleles))
    sites = parse_list(r.required_mutation_sites)
    changed_sites = {i + 1 for i, (a, b) in enumerate(zip(r.seq_source, r.decoded_prediction_sequence)) if a != b}
    for m in req:
        req_mut[m] += 1
        tp_mut[m] += m in tpa
    for s_ in sites:
        req_site[s_] += 1
        tp_site[s_] += s_ in changed_sites


def spike_name(m: str) -> str:
    a, pos, b = m[0], int(m[1:-1]), m[-1]
    rp = col2res.get(pos)
    return f"{a}{rp}{b}" if rp else f"{a}{pos}{b}*"


top = []
for m, cnum in req_mut.most_common(15):
    top.append(dict(mut=m, spike=spike_name(m), count=cnum, recovered=int(tp_mut[m]), frac=tp_mut[m] / cnum))
N["top_required_mutations"] = top
N["n_distinct_required_mutations"] = len(req_mut)
N["n_distinct_required_sites"] = len(req_site)
# recurrence concentration
tot_events = sum(req_mut.values())
N["top15_share_of_required_events"] = sum(c for _, c in req_mut.most_common(15)) / tot_events
rec_mut = [(m, c) for m, c in req_mut.items() if c >= 5]
N["recurrent_mut_ge5"] = dict(
    n=len(rec_mut), events=sum(c for _, c in rec_mut), events_frac=sum(c for _, c in rec_mut) / tot_events,
    recovered=int(sum(tp_mut[m] for m, _ in rec_mut)),
)
sing = [(m, c) for m, c in req_mut.items() if c == 1]
N["singleton_mut"] = dict(n=len(sing), events_frac=len(sing) / tot_events,
                          recovered=int(sum(tp_mut[m] for m, _ in sing)))
N["singleton_mut"]["recall"] = N["singleton_mut"]["recovered"] / max(1, len(sing))
N["recurrent_mut_ge5"]["recall"] = N["recurrent_mut_ge5"]["recovered"] / max(1, N["recurrent_mut_ge5"]["events"])
mid = [(m, c) for m, c in req_mut.items() if 2 <= c <= 4]
N["mid_mut_2_4"] = dict(n=len(mid), events=sum(c for _, c in mid), events_frac=sum(c for _, c in mid) / tot_events,
                        recovered=int(sum(tp_mut[m] for m, _ in mid)))
N["mid_mut_2_4"]["recall"] = N["mid_mut_2_4"]["recovered"] / max(1, N["mid_mut_2_4"]["events"])

# Six receptor-binding-domain sites (spike numbering) whose Omicron-lineage state is R408S, K417N, N440K,
# K444T, L452R, N460K. A required edit is "forward" if the target carries the Omicron state, else "reverse".
res2col = {v: k_ for k_, v in col2res.items()}
OMI = {408: "S", 417: "N", 440: "K", 444: "T", 452: "R", 460: "K"}
rbd_cols = {res2col[r_]: (r_, a_) for r_, a_ in OMI.items()}
fw = Counter()
for m, cnum in req_mut.items():
    pos = int(m[1:-1])
    if pos in rbd_cols:
        key = "forward" if m[-1] == rbd_cols[pos][1] else "reverse"
        fw[key + "_events"] += cnum
        fw[key + "_recovered"] += tp_mut[m]
N["rbd_six_sites"] = dict(fw)
N["rbd_six_sites"]["share_of_all_required_events"] = (fw["forward_events"] + fw["reverse_events"]) / tot_events
N["rbd_six_sites"]["forward_recall"] = fw["forward_recovered"] / max(1, fw["forward_events"])
N["rbd_six_sites"]["reverse_recall"] = fw["reverse_recovered"] / max(1, fw["reverse_events"])
# series with >=1 required edit at these sites, and their mean number of true edits
mask_rbd = d.required_mutation_sites.apply(lambda x: any(s_ in rbd_cols for s_ in parse_list(x)))
N["rbd_six_sites"]["n_series_with_edit"] = int(mask_rbd.sum())
N["rbd_six_sites"]["mean_gt_hamming_series_with_edit"] = float(d[mask_rbd].h_src.mean())
N["rbd_six_sites"]["mean_gt_hamming_series_without_edit"] = float(d[~mask_rbd].h_src.mean())
N["rbd_six_sites"]["same_lineage_frac_with_edit"] = float((d[mask_rbd].lineage_source == d[mask_rbd].lineage_target).mean())

rows = []
for t in top:
    rows.append(f"{t['spike']} & {t['mut']} & {t['count']} & {t['recovered']} & {f(t['frac'], 2)} \\\\")
(TAB / "tab_sites.tex").write_text(
    "\\begin{tabular}{llrrr}\n\\toprule\n"
    "Mutation (spike) & Mutation (column) & Required in & Recovered & Fraction \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n")

# recurrence-stratified table
rec_rows = []
for name, grp in (("1 (singleton)", sing), ("2--4", mid), ("$\\geq$5", rec_mut)):
    ev = sum(c for _, c in grp)
    rc = sum(tp_mut[m] for m, _ in grp)
    rec_rows.append(f"{name} & {len(grp):,} & {ev:,} & {rc:,} & {f(rc / ev)} \\\\")
(TAB / "tab_recurrence.tex").write_text(
    "\\begin{tabular}{lrrrr}\n\\toprule\n"
    "Test series carrying the mutation & Distinct mutations & Occurrences & Recovered & Recall \\\\\n\\midrule\n"
    + "\n".join(rec_rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------------------------
# 9. Prefix-swap ensembles
# ----------------------------------------------------------------------------------------------
ps = pd.read_csv(RES / "evaluation_prefix_swap_fused/prefix_swap_evaluation_report.csv")
slots = pd.read_csv(RES / "evaluation_prefix_swap_fused/prefix_swap_slots_long.csv")
sub = ps[ps.n_pred_queries > 1].copy()  # series with at least one donor slot
N["prefix_swap"] = {
    "n_series": len(ps), "n_with_slots": len(sub), "n_without_slots": int(len(ps) - len(sub)),
    "n_queries_median": float(ps.n_pred_queries.median()), "n_full_21": int((ps.n_pred_queries == 21).sum()),
    "unique_seq_mean": float(sub.n_pred_unique_sequences.mean()),
    "unique_seq_median": float(sub.n_pred_unique_sequences.median()),
    "frac_all_identical": float((sub.n_pred_unique_sequences == 1).mean()),
    "pairwise_hamming_mean": float(sub.pred_pairwise_hamming_mean.mean()),
    "variable_sites_mean": float(sub.n_pred_variable_sites.mean()),
    "wins_best_vs_base": int((sub.pred_best_slot_recall > sub.pred_baseline_recall).sum()),
    "ties_best_vs_base": int((sub.pred_best_slot_recall == sub.pred_baseline_recall).sum()),
    "loses_best_vs_base": int((sub.pred_best_slot_recall < sub.pred_baseline_recall).sum()),
}
slot_only = slots[(~slots.is_baseline) & (slots.row.isin(sub.row))]
base_only = slots[(slots.is_baseline) & (slots.row.isin(sub.row))]
N["prefix_swap"]["mean_slot_hamming_to_target"] = float(slot_only.hamming_to_target.mean())
N["prefix_swap"]["mean_baseline_hamming_to_target"] = float(base_only.hamming_to_target.mean())
N["prefix_swap"]["mean_slot_recall"] = float(slot_only.recall.mean())
N["prefix_swap"]["mean_slot_f1"] = float(slot_only.f1.mean())
N["prefix_swap"]["mean_baseline_recall"] = float(base_only.recall.mean())
N["prefix_swap"]["mean_baseline_f1"] = float(base_only.f1.mean())
# min hamming over slots (oracle) vs baseline
mh = slots[~slots.is_baseline].groupby("row").hamming_to_target.min()
bh = base_only.set_index("row").hamming_to_target
N["prefix_swap"]["oracle_slot_hamming_mean"] = float(mh.reindex(sub.row).mean())
N["prefix_swap"]["baseline_hamming_same_rows"] = float(bh.reindex(sub.row).mean())

ps_rows = []
for key, name in (("pred_baseline", "Baseline prefix (decoded forecast)"),
                  ("pred_pooled_slots", "Union of edits over donor slots"),
                  ("pred_best_slot", "Best single slot (oracle, uses target)")):
    tp_, fn_, fp_ = sub[key + "_tp"].sum(), sub[key + "_fn"].sum(), sub[key + "_fp"].sum()
    r_, p_, f_ = prf(tp_, fn_, fp_)
    N["prefix_swap"][key] = dict(tp=int(tp_), fn=int(fn_), fp=int(fp_), recall=r_, precision=p_, f1=f_)
    ps_rows.append(f"{name} & {int(tp_):,} & {int(fn_):,} & {int(fp_):,} & {f(r_)} & {f(p_)} & {f(f_)} \\\\")
(TAB / "tab_swap.tex").write_text(
    "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    "Prediction & TP & FN & FP & Recall & Precision & F1 \\\\\n\\midrule\n" + "\n".join(ps_rows) +
    "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------------------------
# 10. Figures
# ----------------------------------------------------------------------------------------------
# Fig 0 (headline): distance to target per method, gain histogram, accuracy by bin
fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.3))
meth = [("Copy\nlast-seen", float(d.h_src.mean()), boot_ci(d.h_src.to_numpy()), "#BBBBBB"),
        ("kNN\nretrieval", float(k.h_knn.mean()), boot_ci(k.h_knn.to_numpy()), "#F58518"),
        ("This work", float(d.h_dec.mean()), boot_ci(d.h_dec.to_numpy()), "#4C78A8")]
xs = np.arange(3)
ax[0].bar(xs, [m[1] for m in meth], 0.62, color=[m[3] for m in meth])
ax[0].errorbar(xs, [m[1] for m in meth], yerr=np.abs(np.array([[m[1] - m[2][0] for m in meth],
                                                               [m[2][1] - m[1] for m in meth]])),
               fmt="none", ecolor="#333333", capsize=2.5, lw=0.9)
for xi, m in zip(xs, meth):
    ax[0].text(xi, m[1] + 0.12, f"{m[1]:.2f}", ha="center", fontsize=7)
ax[0].set_xticks(xs)
ax[0].set_xticklabels([m[0] for m in meth])
ax[0].set_ylabel("Hamming distance to target")
ax[0].set_ylim(0, max(m[1] for m in meth) * 1.25)
ax[0].set_title("(a) Distance to the observed target")
vals = np.clip(diff, -6, 6)
ax[1].hist(vals, bins=np.arange(-6.5, 7.5, 1), color="#4C78A8", edgecolor="white")
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_xlabel("$H_{\\mathrm{src}}-H_{\\mathrm{dec}}$ (clipped at $\\pm$6)")
ax[1].set_ylabel("Test series")
ax[1].set_title("(b) Per-series gain over copying")
xb = np.arange(len(strat))
ax[2].bar(xb - 0.19, [s_["dec_recall"] for s_ in strat], 0.38, label="Recall", color="#4C78A8")
ax[2].bar(xb + 0.19, [s_["dec_precision"] for s_ in strat], 0.38, label="Precision", color="#F58518")
ax[2].set_xticks(xb)
ax[2].set_xticklabels([s_["bin"] for s_ in strat])
ax[2].set_xlabel("True edits (last-seen $\\to$ target)")
ax[2].set_ylim(0, 1)
ax[2].set_title("(c) Accuracy by step size")
ax[2].legend(frameon=False, loc="upper center", ncol=2, fontsize=6)
fig.tight_layout()
save(fig, "fig_headline")
plt.close(fig)

# Fig A: paired Hamming improvement histogram + by-bin recall/precision
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
vals = np.clip(diff, -6, 6)
bins = np.arange(-6.5, 7.5, 1)
ax[0].hist(vals, bins=bins, color="#4C78A8", edgecolor="white")
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set_xlabel("$H_{\\mathrm{src}}-H_{\\mathrm{dec}}$ (clipped at $\\pm$6)")
ax[0].set_ylabel("Test series")
ax[0].set_title("(a) Gain over copying the last-seen sequence")
x = np.arange(len(strat))
w = 0.38
ax[1].bar(x - w / 2, [s_["dec_recall"] for s_ in strat], w, label="Recall", color="#4C78A8")
ax[1].bar(x + w / 2, [s_["dec_precision"] for s_ in strat], w, label="Precision", color="#F58518")
ax[1].set_xticks(x)
ax[1].set_xticklabels([f"{s_['bin']}\n(n={s_['n']})" for s_ in strat])
ax[1].set_xlabel("True edits between last-seen and target")
ax[1].set_ylim(0, 1)
ax[1].set_title("(b) Position-level accuracy by number of edits")
ax[1].legend(frameon=False, loc="upper right")
fig.tight_layout()
save(fig, "fig_hamming_and_bins")
plt.close(fig)

# Fig C: prefix-swap summary
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
labs = ["Baseline", "Union of\nslots", "Best slot\n(oracle)"]
keys = ["pred_baseline", "pred_pooled_slots", "pred_best_slot"]
x = np.arange(3)
ax[0].bar(x - w / 2, [N["prefix_swap"][k_]["recall"] for k_ in keys], w, label="Recall", color="#4C78A8")
ax[0].bar(x + w / 2, [N["prefix_swap"][k_]["precision"] for k_ in keys], w, label="Precision", color="#F58518")
ax[0].set_xticks(x)
ax[0].set_xticklabels(labs)
ax[0].set_ylim(0, 1)
ax[0].set_title(f"(a) Position-level accuracy (n={len(sub)})")
ax[0].legend(frameon=False, loc="upper left", ncol=2)
uc = sub.n_pred_unique_sequences.value_counts().sort_index()
ax[1].bar(uc.index[:8], uc.values[:8], color="#72B7B2")
ax[1].set_xlabel("Distinct decoded sequences per series")
ax[1].set_ylabel("Test series")
ax[1].set_title("(b) Diversity of the swap ensemble")
fig.tight_layout()
save(fig, "fig_prefix_swap")
plt.close(fig)

# Fig D: site-level — required mutation columns and decoder recovery
cols = np.arange(1, 713)
req_arr = np.array([req_site.get(c_, 0) for c_ in cols])
tp_arr = np.array([tp_site.get(c_, 0) for c_ in cols])
fig, ax = plt.subplots(figsize=(6.6, 2.0))
ax.bar(cols, req_arr, width=1.0, color="#BBBBBB", label="Required (observed change)")
ax.bar(cols, tp_arr, width=1.0, color="#4C78A8", label="Decoder edits the column")
ax.set_xlabel("S1 alignment column")
ax.set_ylabel("Test series")
ax.set_xlim(0, 713)
ax.set_ylim(0, req_arr.max() * 1.18)
for c_ in np.argsort(-req_arr)[:5]:
    ax.annotate(f"{ref[c_]}{col2res.get(int(c_) + 1, '?')}", (c_ + 1, req_arr[c_]), ha="center", va="bottom", fontsize=6)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
save(fig, "fig_sites")
plt.close(fig)
N["sites_top_columns"] = [(int(c_), int(req_site[c_]), int(tp_site[c_])) for c_, _ in req_site.most_common(10)]

# Fig E: pipeline schematic
fig, ax = plt.subplots(figsize=(6.6, 2.1))
ax.axis("off")


def box(x0, y0, w_, h_, text, fc="#E8EEF7"):
    ax.add_patch(plt.Rectangle((x0, y0), w_, h_, fc=fc, ec="#33415C", lw=0.8))
    ax.text(x0 + w_ / 2, y0 + h_ / 2, text, ha="center", va="center", fontsize=6.5)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9, color="#33415C"))


box(0.00, 0.30, 0.16, 0.42, "Ancestor series\n$g_0,\\dots,g_3$\n(aligned S1)")
box(0.21, 0.30, 0.17, 0.42, "ESM-1b layer 33\nmean $m_t$ (1280)\n+ site PCA $P_t$\n($712\\times4$)")
box(0.43, 0.30, 0.17, 0.42, "A4 site-GPT\n(frozen) + B4\nsparse residual\n+ activation gate", fc="#FDE9D3")
box(0.65, 0.30, 0.15, 0.42, "Forecast\n$\\hat m_4,\\hat P_4$", fc="#FDE9D3")
box(0.85, 0.30, 0.15, 0.42, "Fused decoder\n(non-autoreg.)\n$\\to$ aligned\nsequence", fc="#DDEFD9")
for a_, b_ in ((0.16, 0.21), (0.38, 0.43), (0.60, 0.65), (0.80, 0.85)):
    arrow(a_, 0.51, b_, 0.51)
box(0.43, 0.02, 0.37, 0.16, "Baselines: copy last-seen; kNN over observed genomes", fc="#F2F2F2")
arrow(0.725, 0.30, 0.725, 0.18)
box(0.00, 0.84, 0.38, 0.14, "Prefix swap: replace $g_0,g_1,g_2$ by a donor series", fc="#F2F2F2")
arrow(0.19, 0.84, 0.19, 0.72)
ax.set_xlim(-0.02, 1.03)
ax.set_ylim(0, 1.0)
save(fig, "fig_pipeline", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 11. Previous-generation (A3+B3) run on the same cohort, for the architecture-comparison table
# ----------------------------------------------------------------------------------------------
PREV_RES = D / "latent_variant_prediction/preds/layer33_unified_pca2d_site_v3_a2_b2_b3_activation_mean_res_w_gaps_dedup"
try:
    pl = json.loads((PREV_RES / "b3_mean_res_cal_test.json").read_text())
    pd_ = pd.read_csv(PREV_RES / "evaluation_report_fused.csv")
    ptp, pfn, pfp = (pd_.decoder_position_tp.sum(), pd_.decoder_position_fn.sum(), pd_.decoder_position_fp.sum())
    pr, pp, pf1 = prf(ptp, pfn, pfp)
    pgain = (pd_.gt_hamming_source_target - pd_.decoded_hamming_to_target).to_numpy()
    N["prev_a3b3"] = {
        "skill_full_test": pl["splits"]["test"]["skill_full"],
        "skill_mu_test": pl["splits"]["test"]["skill_mu"],
        "skill_full_cal": pl["splits"]["cal"]["skill_full"],
        "skill_mu_cal": pl["splits"]["cal"]["skill_mu"],
        "recall": pr, "precision": pp, "f1": pf1,
        "hamming_dec": float(pd_.decoded_hamming_to_target.mean()),
        "gain": float(pgain.mean()), "gain_ci": boot_ci(pgain),
        "closer_frac": float((pgain > 0).mean()),
        "n_unique_decoded": int(pd_.decoded_prediction_sequence.nunique()),
    }
    rows = [
        ("A3 + B3 (earlier generation)", N["prev_a3b3"]["skill_full_test"], N["prev_a3b3"]["recall"],
         N["prev_a3b3"]["precision"], N["prev_a3b3"]["f1"], N["prev_a3b3"]["hamming_dec"],
         N["prev_a3b3"]["gain"], N["prev_a3b3"]["gain_ci"]),
        ("A4 + B4 (reported system)", lat["splits"]["test"]["skill_full"], dpm["recall"], dpm["precision"],
         dpm["f1"], float(d.h_dec.mean()), float(diff.mean()), N["hamming"]["mean_gain_ci"]),
    ]
    best = {i: max(r[i] for r in rows) for i in (1, 2, 3, 4, 6)}
    best[5] = min(r[5] for r in rows)

    def bf(v, i, nd=3):
        t = f(v, nd)
        return f"\\textbf{{{t}}}" if abs(v - best[i]) < 1e-12 else t

    tr = [f"{nm} & {bf(sk,1)} & {bf(r_,2)} & {bf(p_,3)} & {bf(f1_,4)} & {bf(h_,5,2)} & "
          f"{bf(g_,6,2)} [${ci[0]:+.2f}$, ${ci[1]:+.2f}$] \\\\"
          for nm, sk, r_, p_, f1_, h_, g_, ci in rows]
    (TAB / "tab_generations.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Forecaster & Latent skill & Recall & Precision & F1 & $H_{\\mathrm{dec}}$ & Gain [95\\% CI] \\\\\n"
        "\\midrule\n" + "\n".join(tr) + "\n\\bottomrule\n\\end{tabular}\n")
except Exception as exc:  # pragma: no cover
    N["prev_a3b3_error"] = repr(exc)

# ----------------------------------------------------------------------------------------------
(PAPER / "numbers.json").write_text(json.dumps(N, indent=2, default=str))
print(json.dumps(N, indent=2, default=str))
