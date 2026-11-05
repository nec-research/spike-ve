#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     plotting.py
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
""" Plotting utils for future prediction module.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
import pickle
import json

import random
from typing import List, Optional, Sequence

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from recomb2027.data.encodings import EncodingMatcher
from recomb2027.latent.io_utils import read_preds_and_gt_from_ndjson
from recomb2027.core.constants import PRIMARY_TARGET

_logger = logging.getLogger(__name__)
logger = _logger


def _umap_transform_rows(projector, X: np.ndarray) -> np.ndarray:
    """Project rows with a fitted UMAP model.

    ``UMAP.transform`` is not row-wise independent: identical latent vectors
    passed in one batch can land at different 2D coordinates (nearest-neighbor
    context is built jointly over the query batch).  Transform each unique row
    once and reuse the result so the same embedding always maps to one point.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {X.shape}")
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    out = np.empty((X.shape[0], 2), dtype=np.float64)
    cache: dict[bytes, np.ndarray] = {}
    for i in range(X.shape[0]):
        key = X[i].tobytes()
        if key not in cache:
            cache[key] = np.asarray(
                projector.transform(X[i : i + 1]), dtype=np.float64
            ).reshape(-1)
        out[i] = cache[key]
    return out


def _umap_series_figure_layout(
    *,
    title: str,
    fig_width: int,
    fig_height: int,
    n_legend_entries: int,
) -> dict:
    """Layout kwargs for UMAP trajectory figures.

    Places the legend **outside** the plot domain (right margin) with a compact
    semi-opaque panel so underlying samples and trajectories stay visible in the
    data area. Plotly cannot draw per-entry legend text boxes only; external
    placement avoids occluding the embedding.
    """
    n_leg = max(1, int(n_legend_entries))
    right_margin = int(min(420, max(240, 210 + n_leg * 6)))
    return dict(
        title=title,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        width=fig_width,
        height=fig_height,
        template="plotly_white",
        margin=dict(l=60, r=right_margin, t=80, b=60),
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11),
            itemsizing="constant",
            tracegroupgap=2,
        ),
        hovermode="closest",
    )


def write_plotly_umap_html(
    fig: go.Figure,
    path: str | Path,
    *,
    include_plotlyjs: str = "cdn",
) -> None:
    """Write UMAP figure HTML with legend/idle-cloud UX hooks."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True)
    sync_js = """
<script>
(function () {
  var gd = document.querySelector('.plotly-graph-div');
  if (!gd || !gd.data) return;

  function idleTraceIndices() {
    var out = [];
    gd.data.forEach(function (tr, i) {
      if (tr.name && String(tr.name).indexOf('(diffusion samples, idle)') >= 0) {
        out.push({ idx: i, group: tr.legendgroup });
      }
    });
    return out;
  }
  function mainTraceIndex(group) {
    for (var i = 0; i < gd.data.length; i++) {
      var tr = gd.data[i];
      if (tr.legendgroup !== group || tr.showlegend === false) continue;
      if (tr.mode && String(tr.mode).indexOf('lines') >= 0) return i;
    }
    return -1;
  }
  function syncIdleClouds() {
    var idle = idleTraceIndices();
    if (!idle.length) return;
    idle.forEach(function (item) {
      var mainIdx = mainTraceIndex(item.group);
      var mainVis = mainIdx >= 0 ? gd.data[mainIdx].visible : false;
      var hideIdle = mainVis === true;
      Plotly.restyle(gd, { visible: hideIdle ? 'legendonly' : true }, [item.idx]);
    });
  }

  function isPredictionLegendDummy(tr) {
    if (!tr || !tr.name) return false;
    var n = String(tr.name);
    return n.indexOf('Prediction') === 0 && n.indexOf('★') >= 0;
  }
  function predictionOverlayIndices() {
    var out = [];
    gd.data.forEach(function (tr, i) {
      if (!tr.name || tr.showlegend !== false) return;
      var n = String(tr.name);
      // Per-series dashed line + ★ (+ optional open source ring uses "(src)")
      if (n.indexOf('(pred)') >= 0) out.push(i);
    });
    return out;
  }
  function syncPredictionOverlaysFromDummy() {
    var dummyIdx = -1;
    for (var i = 0; i < gd.data.length; i++) {
      if (isPredictionLegendDummy(gd.data[i])) { dummyIdx = i; break; }
    }
    if (dummyIdx < 0) return;
    var vis = gd.data[dummyIdx].visible;
    var predLegendOn = !(vis === false || vis === 'legendonly');
    var idxs = predictionOverlayIndices();
    if (!idxs.length) return;
    idxs.forEach(function (idx) {
      var tr = gd.data[idx];
      var mainIdx = mainTraceIndex(tr.legendgroup);
      var seriesOn = mainIdx >= 0 && gd.data[mainIdx].visible === true;
      var want = predLegendOn && seriesOn;
      var cur = gd.data[idx].visible;
      var curOn = !(cur === false || cur === 'legendonly');
      if (want !== curOn) {
        Plotly.restyle(gd, { visible: want ? true : 'legendonly' }, [idx]);
      }
    });
  }

  gd.on('plotly_legendclick', function (ev) {
    window.setTimeout(function () {
      syncIdleClouds();
      // If the clicked legend item was the Prediction ★ dummy, sync overlays.
      try {
        var ci = ev && (ev.curveNumber != null ? ev.curveNumber : ev.expandedIndex);
        if (ci != null && isPredictionLegendDummy(gd.data[ci])) {
          syncPredictionOverlaysFromDummy();
        }
      } catch (e) {}
    }, 0);
    return true;
  });
  gd.on('plotly_legenddoubleclick', function () {
    window.setTimeout(syncIdleClouds, 0);
    return true;
  });
})();
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", sync_js + "\n</body>", 1)
    else:
        html += sync_js
    out.write_text(html, encoding="utf-8")


def _prepare_diffusion_cloud_proj_bundle(
    *,
    cloud_arr: np.ndarray,
    projector,
    pred_for_steps: dict | None,
    tr_raw: np.ndarray,
    T: int,
    exc_ix: int | None,
) -> dict | None:
    """Project diffusion cloud to UMAP and build hover customdata."""
    if cloud_arr is None or cloud_arr.shape[0] == 0:
        return None
    cloud_arr = np.asarray(cloud_arr, dtype=np.float64)
    finite = np.isfinite(cloud_arr).all(axis=1)
    if not np.any(finite):
        return None
    if not np.all(finite):
        cloud_arr = cloud_arr[finite]
    tgt_step_cloud = (
        int(pred_for_steps["tgt_step"]) if pred_for_steps is not None else T - 1
    )
    src_step_cloud = (
        int(pred_for_steps["src_step"]) if pred_for_steps is not None else max(0, T - 2)
    )
    proj_cloud = _umap_transform_rows(
        projector, np.asarray(cloud_arr, dtype=np.float64)
    )
    gt_vec = tr_raw[tgt_step_cloud] if 0 <= tgt_step_cloud < T else tr_raw[-1]
    norms = np.linalg.norm(cloud_arr - gt_vec.reshape(1, -1), axis=1)
    n_pts = int(cloud_arr.shape[0])
    cd_cloud = np.empty((n_pts, 5), dtype=object)
    for ii in range(n_pts):
        cd_cloud[ii, 0] = int(ii)
        cd_cloud[ii, 1] = int(src_step_cloud + 1)
        cd_cloud[ii, 2] = int(tgt_step_cloud + 1)
        cd_cloud[ii, 3] = int(T)
        cd_cloud[ii, 4] = float(norms[ii])

    if exc_ix is not None and 0 <= int(exc_ix) < n_pts:
        mask = np.ones(n_pts, dtype=bool)
        mask[int(exc_ix)] = False
        proj_plot = proj_cloud[mask]
        cd_plot = cd_cloud[mask]
    else:
        proj_plot = proj_cloud
        cd_plot = cd_cloud

    if proj_plot.shape[0] == 0:
        return None
    return {
        "proj_plot": proj_plot,
        "cd_plot": cd_plot,
        "src_step_cloud": src_step_cloud,
        "tgt_step_cloud": tgt_step_cloud,
    }


def prepare_json_data_for_plotting(json_file, encodings_dir):
    # Read json file and get list of dictionaries with series information
    series_gt_pred_list = read_preds_and_gt_from_ndjson(json_file)

    # Extract sample gids for each series
    x_gid_list = []
    y_gid_list = []
    y_pred_encodings_list = []
    y_gt_encodings_list = []
    for obj_dict in series_gt_pred_list:
        x_gid_list += obj_dict['series_gids']
        y_gid_list.append(obj_dict['gt_gid'])
        y_pred_encodings_list.append(obj_dict['predicted_encoding'])
        y_gt_encodings_list.append(obj_dict['gt_encoding'])

    # Match encodings to genome ids
    df_encodings = EncodingMatcher.read_from_dir(encodings_dir, encoding_key="encodings").set_index(keys="genome_id")
    #encoding_dict = df_encodings["encodings"].to_dict()
    # x_encodings = np.stack([
    #     #np.stack([encoding_dict[gid] for gid in sample_gids])
    #     np.stack([encoding_dict.get(gid, None) for gid in x_gid_list])
    # ])
    # Make dataframe gid, encoding instead. Allow missing values if encoding not found

    assert len(y_gid_list) == len(y_pred_encodings_list)

    df_pred = pd.DataFrame(
        data={
            'genome_id': y_gid_list,
            'encodings': y_pred_encodings_list,
            'label': 'prediction'
        }
    )

    assert len(y_gid_list) == len(y_gt_encodings_list)

    df_gt = pd.DataFrame(
        data={
            'genome_id': y_gid_list,
            'encodings': y_gt_encodings_list,
            'label': 'ground_truth'
        }
    )

    df_other = df_encodings[df_encodings.index.isin(x_gid_list)].reset_index()[['genome_id', 'encodings']]#.drop_duplicates()
    df_other['label'] = 'bg'

    #df_other.drop_duplicates(inplace=True)

    # df_other = df_encodings[~df_encodings.index.isin(y_gid_truth)].reset_index()[['genome_id', 'encodings']]
    # df_other['tobe_predicted'] = 'N'

    df_data = pd.concat([df_pred, df_gt, df_other], ignore_index=True)

    # TBD: merge with a dataframe of metadata to get collection_date. Or fill collection date from data saved in json
    df_data = df_data.merge(
        df_encodings.reset_index()[['genome_id', 'lineage', 'who_lineage']].drop_duplicates(),
        how='left',
        on='genome_id'
    )

    print(f"Total length of the combined dataframe: {len(df_data)}")

    return df_data


def prepare_objects_for_plotting(val_dataset, pred_list_scaled, yb_list_scaled, df_encodings, std_scaler):

    assert pred_list_scaled.shape == yb_list_scaled.shape
    
    x_gid, y_gid = val_dataset.get_original_xy_gid()
    y_gid_truth = np.array(y_gid)[:,-1].tolist()
    
    print(f"Number of predictions: {len(y_gid_truth)}")

    pred_list_unscaled = std_scaler.inverse_transform(pred_list_scaled)

    assert len(y_gid_truth) == len(pred_list_unscaled)
    
    df_pred = pd.DataFrame(
        data = {
            'genome_id': y_gid_truth,
            'encodings': pred_list_unscaled.tolist(),
            'tobe_predicted': 'prediction'
        }
    )

    yb_list_unscaled = std_scaler.inverse_transform(yb_list_scaled)

    assert len(y_gid_truth) == len(yb_list_unscaled)
    
    df_gt = pd.DataFrame(
        data = {
            'genome_id': y_gid_truth,
            'encodings': yb_list_unscaled.tolist(),
            'tobe_predicted': 'ground_truth'
        }
    )
    
    df_other = df_encodings[~df_encodings.index.isin(y_gid_truth)].reset_index()[['genome_id', 'encodings']]
    df_other['tobe_predicted'] = 'N'

    df_data = pd.concat([df_pred, df_gt, df_other], ignore_index=True)

    df_data = df_data.merge(
        ordObj.df_labels[['genome_id', 'lineage', 'who_lineage', 'collection_date']].drop_duplicates(), 
        how='left', 
        on='genome_id'
    )

    print(f"Total length of the combined dataframe: {len(df_data)}")
    
    return df_data



def prot_prediction_error(
        df_data,
        save_path=None,
        nbins=500
):
    """
    Plot distribution of prediction error (Euclidean distance)
    and optionally save as an interactive Plotly HTML file.

    Parameters
    ----------
    preds : np.ndarray
        Predicted vectors, shape (N, D)
    gts : np.ndarray
        Ground truth vectors, shape (N, D)
    save_path : str or None
        Path to save HTML file (e.g. "error_dist.html")
    nbins : int
        Number of histogram bins

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    df_pivot = (
        df_data[df_data['label'].isin(['ground_truth', 'prediction'])]
        .pivot_table(
            index='genome_id',
            columns='label',
            values='encodings',
            aggfunc='first'
        )
        .reset_index()
    )

    preds = np.stack(df_pivot['prediction'].to_numpy())
    gts = np.stack(df_pivot['ground_truth'].to_numpy())

    assert preds.shape == gts.shape, "preds and gts must have the same shape"

    # Compute Euclidean distances
    distances = np.linalg.norm(preds - gts, axis=1)

    # Create Plotly histogram
    fig = go.Figure(
        data=[
            go.Histogram(
                x=distances,
                nbinsx=nbins
            )
        ]
    )

    fig.update_layout(
        title="Distribution of Prediction Error (Euclidean Distance)",
        xaxis_title="Distance",
        yaxis_title="Frequency",
        bargap=0.05,
        template="plotly_white"
    )

    # ---- SAVE FIGURE AS HTML ----
    if save_path is not None:
        fig.write_html(
            save_path,
            include_plotlyjs="cdn",
            full_html=True
        )
        print(f"Interactive HTML saved to {save_path}")

    return fig


def plot_prediction_differences(
    df_data, 
    method='umap', 
    color_col = 'tobe_predicted',
    marker_size=6,
    n_bg_samples=1000, 
    fig_width=1000, 
    fig_height=800,
    incl_gt_in_fit=True, 
    seed=42,
    save_path=None
):
    # Set up seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    assert method in ['pca', 'tsne', 'umap'], "Invalid projection method"

    m_pred = df_data['label'].isin(['prediction', 'ground_truth'])

    df_bg = df_data.loc[~m_pred]
    print(f"Data in bg: {len(df_bg)} points.")    
    df_gt = df_data[df_data['label'] == 'ground_truth'].copy()
    print(f"Data in gt: {len(df_gt)} points.")
    df_pred = df_data[df_data['label'] == 'prediction'].copy()
    print(f"Data in pred: {len(df_pred)} points.")
    
    if len(df_bg) > n_bg_samples:     
        df_sampled = df_bg.sample(n=n_bg_samples).copy()
    else:
        df_sampled = df_bg.copy()

    if method == 'pca':
        projector = PCA(n_components=2)
    elif method == 'tsne':
        projector = TSNE(n_components=2, perplexity=30, init='random', random_state=0)
    elif method == 'umap':
        projector = umap.UMAP(n_components=2, init='random')

    if incl_gt_in_fit:
        df_sampled = pd.concat([df_sampled, df_gt])

    print(f"Sampled {len(df_sampled)} points. Calculate projections.")
        
    combined_bg = np.vstack(df_sampled['encodings'].values)    
    proj_combined = projector.fit_transform(combined_bg)
    
    df_sampled['U_0'] = proj_combined[:, 0]
    df_sampled['U_1'] = proj_combined[:, 1]

    combined_pred = np.vstack(df_pred['encodings'].values)
    if method == "umap":
        proj_pred = _umap_transform_rows(projector, combined_pred)
    else:
        proj_pred = projector.transform(combined_pred)
    df_pred['U_0'] = proj_pred[:, 0]
    df_pred['U_1'] = proj_pred[:, 1]
    
    if incl_gt_in_fit:
        df_sampled = pd.concat([df_sampled, df_pred])
    else:
        combined_gt = np.vstack(df_gt['encodings'].values)
        if method == "umap":
            proj_gt = _umap_transform_rows(projector, combined_gt)
        else:
            proj_gt = projector.transform(combined_gt)
        df_gt['U_0'] = proj_gt[:, 0]
        df_gt['U_1'] = proj_gt[:, 1]
        df_sampled = pd.concat([df_sampled, df_gt, df_pred])

    print(f"UMAP projections calculated.")
    
    df_pred = df_sampled[df_sampled['label'] == 'prediction']
    df_gt = df_sampled[df_sampled['label'] == 'ground_truth']
    
    counts = pd.concat([df_gt, df_pred]).groupby(['genome_id', 'label']).size().unstack(fill_value=0)
    valid_gids = counts[(counts['prediction'] == 1) & (counts['ground_truth'] == 1)].index

    df_pred = df_pred[df_pred['genome_id'].isin(valid_gids)]
    df_gt = df_gt[df_gt['genome_id'].isin(valid_gids)]

    merged = pd.merge(df_pred, df_gt, on='genome_id', suffixes=('_pred', '_gt'))

    print(f"Plot the traces.")
    
    # Create line segments
    line_traces = go.Scatter(
        x=sum([[row['U_0_pred'], row['U_0_gt'], None] for _, row in merged.iterrows()], []),
        y=sum([[row['U_1_pred'], row['U_1_gt'], None] for _, row in merged.iterrows()], []),
        mode='lines',
        line=dict(
            #color='silver', 
            color='gray', 
            width=1
        ),
        opacity=0.5,
        hoverinfo='skip',
        showlegend=False
    )

    # Now create scatter points with plotly express
    hover_data = ['genome_id', 'label', 'lineage', 'who_lineage']
    fig_points = px.scatter(
        df_sampled, 
        x="U_0", y="U_1", 
        color=color_col, 
        hover_data=hover_data,
        color_discrete_map={
            "prediction": "red",
            "ground_truth": "green",
            "bg": "gray"
        }
    )
    fig_points.update_traces(marker=dict(size=marker_size))

    # Create figure manually to control trace order
    fig = go.Figure()
    fig.add_trace(line_traces)
    for trace in fig_points.data:
        fig.add_trace(trace)

    fig.update_layout(
        title=f'2D Projection ({method.upper()}): Ground Truth vs Predicted',
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        width=fig_width,
        height=fig_height,
        legend=dict(title='Point Type')
    )

    # ---- SAVE FIGURE ----
    if save_path is not None:
        fig.write_html(
            save_path,
            include_plotlyjs="cdn",
            full_html=True
        )
        print(f"Interactive HTML saved to {save_path}")


    return fig


def plot_training_loss(
    train_loss_history: List[float],
    val_loss_history: List[float],
    save_path: str | Path,
    *,
    train_per_target: dict[str, List[float]] | None = None,
    val_per_target: dict[str, List[float]] | None = None,
    val_select_history: List[float] | None = None,
    select_label: str = "val_select",
) -> None:
    """
    Plot training and validation loss curves and save them as PNG and HTML.

    Args:
        train_loss_history: Primary train MSE (mean head) per epoch.
        val_loss_history: Selection score per epoch (may be joint).
        save_path: Output directory for HTML/PNG.
        train_per_target: Optional ``{target: [mse_ep...]}`` train curves.
        val_per_target: Optional ``{target: [mse_ep...]}`` val curves.
        val_select_history: Optional joint/select score (if different from
            ``val_loss_history`` naming); when set, plotted as ``select_label``.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=train_loss_history,
            mode="lines",
            name="Train MSE (mean)",
        )
    )

    # Prefer explicit per-target val mean over overloaded "Val loss".
    if val_per_target and PRIMARY_TARGET in val_per_target:
        fig.add_trace(
            go.Scatter(
                y=val_per_target[PRIMARY_TARGET],
                mode="lines",
                name="Val MSE (mean)",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                y=val_loss_history,
                mode="lines",
                name="Val loss",
            )
        )

    if train_per_target:
        for tname, ys in train_per_target.items():
            if tname == PRIMARY_TARGET or not ys:
                continue
            fig.add_trace(
                go.Scatter(y=ys, mode="lines", name=f"Train MSE ({tname})")
            )
    if val_per_target:
        for tname, ys in val_per_target.items():
            if tname == PRIMARY_TARGET or not ys:
                continue
            fig.add_trace(
                go.Scatter(y=ys, mode="lines", name=f"Val MSE ({tname})")
            )

    select_ys = val_select_history if val_select_history is not None else None
    if select_ys is None and val_per_target and PRIMARY_TARGET in val_per_target:
        # val_loss_history is the selection score when per-target is available
        select_ys = val_loss_history
    if select_ys is not None and val_per_target:
        fig.add_trace(
            go.Scatter(
                y=select_ys,
                mode="lines",
                name=select_label,
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title="Training vs Validation Loss (per-target)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )

    # Save outputs
    fig.write_html(save_path / "train_val.html")
    #fig.write_image(save_path.with_suffix(".png"), scale=2)

    # ------- Plotting routines -------------------

    def create_train_plots(
            train_losses,
            val_losses,
            plot_dir,
            loss_label="MSE",
            title="Train vs Validation Loss",
            figsize=(10, 6)
    ):
        """Create visualization plots for current epoch.
        """
        plt.figure(figsize=figsize)
        # Plotting losses
        plt.title(title)
        plt.plot(train_losses, label=f"Train {loss_label}")
        plt.plot(val_losses, label=f"Val {loss_label}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        # plt.show()
        # Save
        filepath = os.path.join(plot_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(filepath)
        logger.info(f"Train/Validation plots saved to '{filepath}'")

        plt.close()

# ---- Plot series information ----


def prepare_for_series_plotting(input_json_file, encodings_dir):
    series_gt_pred_list = read_preds_and_gt_from_ndjson(input_json_file)
    # Match encodings to genome ids
    df_encodings = EncodingMatcher.read_from_dir(encodings_dir, encoding_key="encodings").set_index(keys="genome_id")
    encoding_dict = df_encodings["encodings"].to_dict()
    lineage_dict = df_encodings["lineage"].to_dict()

    # Extract sample gids for each series
    series_idx_list = []
    series_element_idx_list = []
    x_gid_list = []
    for series_idx, obj_dict in enumerate(series_gt_pred_list):
        series_idx_list += [series_idx] * len(obj_dict['series_gids'])
        series_element_idx_list += list(range(len(obj_dict['series_gids'])))
        x_gid_list += obj_dict['series_gids']

    df_series = pd.DataFrame(
        data={
            'series_idx': series_idx_list,
            'element_idx': series_element_idx_list,
            'genome_id': x_gid_list
        }
    )

    next_element_idx = df_series.element_idx.max() + 1

    # Extract sample gids for each series
    series_idx_list = []
    series_element_idx_list = []
    y_gid_list = []
    y_gt_encodings_list = []
    y_pred_encodings_list = []
    for series_idx, obj_dict in enumerate(series_gt_pred_list):
        series_idx_list += [series_idx]
        series_element_idx_list += [next_element_idx]
        y_gid_list.append(obj_dict['gt_gid'])
        y_gt_encodings_list.append(obj_dict['gt_encoding'])
        y_pred_encodings_list.append(obj_dict['predicted_encoding'])

    df_prediction = pd.DataFrame(
        data={
            'series_idx': series_idx_list,
            'element_idx': series_element_idx_list,
            'gt_genome_id': y_gid_list,
            'gt_encoding': y_gt_encodings_list,
            'predicted_encoding': y_pred_encodings_list
        }
    )

    df_series.loc[:, 'encoding'] = df_series.genome_id.apply(lambda gid: encoding_dict[gid])

    df_series.loc[:, 'lineage'] = df_series.genome_id.apply(lambda gid: lineage_dict[gid])
    df_prediction.loc[:, 'lineage'] = df_prediction.gt_genome_id.apply(lambda gid: lineage_dict[gid])

    return df_series, df_prediction


def plot_series_w_predictions(
        df_series,
        df_prediction,
        method='umap',
        n_bg_samples=10000,
        marker_size=6,
        fig_width=1000,
        fig_height=800
):
    next_element_idx = df_series.element_idx.max() + 1
    df_sampled = df_series[df_series['element_idx'] == next_element_idx - 1]

    if len(df_series) > n_bg_samples:
        df_sampled = df_series[df_series['element_idx'] == next_element_idx - 1].copy()
        if n_bg_samples > len(df_sampled):
            df_sampled_extra = df_series[df_series['element_idx'] < next_element_idx - 1].sample(
                n=n_bg_samples - len(df_sampled)).copy()
            df_sampled = pd.concat([df_sampled, df_sampled_extra], axis=0)
    else:
        df_sampled = df_series.copy()

    if method == 'pca':
        projector = PCA(n_components=2)
    elif method == 'tsne':
        projector = TSNE(n_components=2, perplexity=30, init='random', random_state=0)
    elif method == 'umap':
        projector = umap.UMAP(n_components=2, init='random')

    print(f"Sampled {len(df_sampled)} points. Calculate projections.")

    combined_bg = np.vstack(df_sampled['encoding'].values)
    proj_combined = projector.fit_transform(combined_bg)
    df_sampled['U_0'] = proj_combined[:, 0]
    df_sampled['U_1'] = proj_combined[:, 1]

    combined_pred = np.vstack(df_prediction['predicted_encoding'].values)
    if method == "umap":
        proj_pred = _umap_transform_rows(projector, combined_pred)
    else:
        proj_pred = projector.transform(combined_pred)
    df_prediction['U_0_pred'] = proj_pred[:, 0]
    df_prediction['U_1_pred'] = proj_pred[:, 1]

    combined_gt = np.vstack(df_prediction['gt_encoding'].values)
    if method == "umap":
        proj_gt = _umap_transform_rows(projector, combined_gt)
    else:
        proj_gt = projector.transform(combined_gt)
    df_prediction['U_0_gt'] = proj_gt[:, 0]
    df_prediction['U_1_gt'] = proj_gt[:, 1]

    # Create line segments for prediction error
    line_traces_pred_gt = go.Scatter(
        x=sum([[row['U_0_pred'], row['U_0_gt'], None] for _, row in df_prediction.iterrows()], []),
        y=sum([[row['U_1_pred'], row['U_1_gt'], None] for _, row in df_prediction.iterrows()], []),
        mode='lines',
        line=dict(
            # color='silver',
            color='red',
            width=1
        ),
        opacity=0.5,
        hoverinfo='skip',
        showlegend=False
    )

    from_element_idx = next_element_idx - 1
    df_sampled_idx = df_sampled[df_sampled['element_idx'] == from_element_idx]

    assert df_sampled_idx['series_idx'].value_counts().sort_index().equals(
        df_prediction['series_idx'].value_counts().sort_index()
    ), "Mismatch number of preditions and number of series in two dataframes"

    df_merged = pd.merge(df_sampled_idx, df_prediction, on='series_idx', how='left')

    # Create line segments series_idx_i to gt
    line_traces_el_gt = go.Scatter(
        x=sum([[row['U_0'], row['U_0_gt'], None] for _, row in df_merged.iterrows()], []),
        y=sum([[row['U_1'], row['U_1_gt'], None] for _, row in df_merged.iterrows()], []),
        mode='lines',
        line=dict(
            # color='silver',
            # color='green',
            color='black',
            width=1
        ),
        opacity=0.5,
        hoverinfo='skip',
        showlegend=False
    )

    # Create line segments series_idx_i to pred
    line_traces_el_pred = go.Scatter(
        x=sum([[row['U_0'], row['U_0_pred'], None] for _, row in df_merged.iterrows()], []),
        y=sum([[row['U_1'], row['U_1_pred'], None] for _, row in df_merged.iterrows()], []),
        mode='lines',
        line=dict(
            # color='silver',
            color='black',
            # color='blue',
            width=1,
            dash='dash'
        ),
        opacity=0.5,
        hoverinfo='skip',
        showlegend=False
    )

    values = range(next_element_idx)
    grays = np.linspace(0, 230, len(values)).astype(int)
    gray_map = {
        v: f"rgb({g},{g},{g})"
        for v, g in zip(values, grays)
    }

    # Now create scatter points with plotly express
    # hover_data = ['genome_id', 'label', 'lineage', 'who_lineage']
    hover_data = ['genome_id', 'lineage']
    fig_points = px.scatter(
        df_sampled,
        x="U_0", y="U_1",
        # color=color_col,
        # color="element_idx",
        hover_data=hover_data,
        # color_discrete_map=gray_map,
        # color_discrete_map={
        #     "prediction": "red",
        #     "ground_truth": "green",
        #     "bg": "gray"
        # }
    )
    fig_points.update_traces(marker=dict(size=marker_size, color='gray'))

    # Now create scatter points with plotly express
    # hover_data = ['genome_id', 'label', 'lineage', 'who_lineage']
    hover_data = ['gt_genome_id', 'lineage']
    fig_points_pred = px.scatter(
        df_prediction,
        x="U_0_pred", y="U_1_pred",
        # color=color_col,
        # color="element_idx",
        hover_data=hover_data,
        # color_discrete_map={
        #     "prediction": "red",
        #     "ground_truth": "green",
        #     "bg": "gray"
        # }
    )
    fig_points_pred.update_traces(marker=dict(size=marker_size, color='red'))

    # Now create scatter points with plotly express
    # hover_data = ['genome_id', 'label', 'lineage', 'who_lineage']
    hover_data = ['gt_genome_id', 'lineage']
    fig_points_gt = px.scatter(
        df_prediction,
        x="U_0_gt", y="U_1_gt",
        # color=color_col,
        # color="element_idx",
        hover_data=hover_data,
        # color_discrete_map={
        #     "prediction": "red",
        #     "ground_truth": "green",
        #     "bg": "gray"
        # }
    )
    fig_points_gt.update_traces(marker=dict(size=marker_size, color='green'))

    # Create figure manually to control trace order
    fig = go.Figure()
    fig.add_trace(line_traces_pred_gt)
    fig.add_trace(line_traces_el_gt)
    fig.add_trace(line_traces_el_pred)
    for trace in fig_points.data:
        fig.add_trace(trace)
    for trace in fig_points_pred.data:
        fig.add_trace(trace)
    for trace in fig_points_gt.data:
        fig.add_trace(trace)

    fig.update_layout(
        title=f'2D Projection ({method.upper()}): Ground Truth vs Predicted',
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        width=fig_width,
        height=fig_height,
        legend=dict(title='Point Type')
    )

    return fig


def _lineage_label_positions(
    proj_bg: np.ndarray,
    bg_lineages: Sequence[str],
    min_cluster_size: int,
    n_bins: int = 60,
    smooth_sigma_bins: float = 1.5,
    label_min_sep_frac: float = 0.04,
    drop_overlapped_smaller: bool = True,
) -> pd.DataFrame:
    """Place each lineage label at the center of its smoothed-density mode.

    For each lineage with ≥ ``min_cluster_size`` points we build a 2D
    histogram of its UMAP coordinates on a *fine* grid (``n_bins`` per axis)
    and convolve it with a Gaussian (``smooth_sigma_bins`` bins). The label
    is placed at the center of mass of the points falling inside the
    densest *smoothed* bin. The smoothing step matters most for small
    clusters: a raw 30-bin histogram of a 35-point lineage is essentially
    sparse (≤1 point per bin) and ``np.argmax`` would break ties in scan
    order, putting the label at a near-random point. Smoothing aggregates
    neighbours, so the mode lands on the actually densest neighbourhood.

    When ``drop_overlapped_smaller=True`` (default) labels are placed
    greedily in **decreasing-size order** and a smaller cluster's label is
    suppressed whenever its position falls within ``label_min_sep_frac``
    of the data-bbox diagonal of any larger label already kept. This stops
    sub-lineages from piling on top of their parent clade label
    (``BA.4 / BA.4.6 / BA.5.2.1`` overlap visible on the right of the
    UMAP). The rejected lineage is logged at INFO level.

    Returns a DataFrame with columns ``lineage, U1, U2, size`` ordered by
    decreasing cluster size.
    """
    df = pd.DataFrame(
        {
            "U1": proj_bg[:, 0],
            "U2": proj_bg[:, 1],
            "lineage": pd.Series(list(bg_lineages)).astype(str),
        }
    )
    df = df[~df["lineage"].isin(["n/a", "nan", "", "None"])]
    sizes = df.groupby("lineage", sort=False).size()
    keep = sizes[sizes >= min_cluster_size].index
    df = df[df["lineage"].isin(keep)]
    if df.empty:
        return pd.DataFrame(columns=["lineage", "U1", "U2", "size"])

    # Global bin edges so bin centers are comparable; per-lineage histogram
    # on top. We use a fine grid (default 60 bins) plus Gaussian smoothing
    # to recover sensible densities even for small lineages.
    u1_min, u1_max = float(df["U1"].min()), float(df["U1"].max())
    u2_min, u2_max = float(df["U2"].min()), float(df["U2"].max())
    if u1_max == u1_min:
        u1_max = u1_min + 1e-6
    if u2_max == u2_min:
        u2_max = u2_min + 1e-6
    edges_x = np.linspace(u1_min, u1_max, n_bins + 1)
    edges_y = np.linspace(u2_min, u2_max, n_bins + 1)
    cx = 0.5 * (edges_x[:-1] + edges_x[1:])
    cy = 0.5 * (edges_y[:-1] + edges_y[1:])
    bin_w_x = (u1_max - u1_min) / n_bins
    bin_w_y = (u2_max - u2_min) / n_bins
    diag = float(np.hypot(u1_max - u1_min, u2_max - u2_min))
    min_sep = float(label_min_sep_frac) * diag if diag > 0 else 0.0

    # Optional Gaussian smoothing of the 2D histogram. Soft-degrade to
    # plain histogram if scipy is unavailable.
    try:
        from scipy.ndimage import gaussian_filter as _gaussian_filter
    except Exception:
        _gaussian_filter = None

    rows = []
    for lin, sub in df.groupby("lineage", sort=False):
        x = sub["U1"].to_numpy()
        y = sub["U2"].to_numpy()
        n = int(len(sub))
        if n < min_cluster_size:
            continue
        hist, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y])
        if _gaussian_filter is not None and smooth_sigma_bins > 0:
            density = _gaussian_filter(
                hist, sigma=float(smooth_sigma_bins), mode="constant"
            )
        else:
            density = hist
        ix, iy = np.unravel_index(np.argmax(density), density.shape)
        if density[ix, iy] <= 0:
            mx, my = float(np.median(x)), float(np.median(y))
        else:
            # Refine with a small neighbourhood (±1 bin) around the mode so
            # we average over actual cluster mass rather than a single bin.
            x_lo = edges_x[max(ix - 1, 0)]
            x_hi = edges_x[min(ix + 2, n_bins)]
            y_lo = edges_y[max(iy - 1, 0)]
            y_hi = edges_y[min(iy + 2, n_bins)]
            in_neigh = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
            if in_neigh.sum() > 0:
                mx = float(np.mean(x[in_neigh]))
                my = float(np.mean(y[in_neigh]))
            else:
                mx, my = float(cx[ix]), float(cy[iy])
        rows.append({"lineage": lin, "U1": mx, "U2": my, "size": n})

    out = pd.DataFrame(rows, columns=["lineage", "U1", "U2", "size"])
    if out.empty or not drop_overlapped_smaller or min_sep <= 0:
        return out.sort_values("size", ascending=False).reset_index(drop=True)

    out = out.sort_values("size", ascending=False).reset_index(drop=True)
    kept_idx: list[int] = []
    dropped: list[tuple[str, str, float]] = []
    for i, row in out.iterrows():
        too_close_to: str | None = None
        for j in kept_idx:
            other = out.iloc[j]
            d = float(np.hypot(row.U1 - other.U1, row.U2 - other.U2))
            if d < min_sep:
                too_close_to = str(other.lineage)
                break
        if too_close_to is None:
            kept_idx.append(int(i))
        else:
            dropped.append((str(row.lineage), too_close_to, d))
    if dropped:
        _logger.info(
            "Lineage labels: dropped %d smaller labels overlapping a larger one "
            "(min_sep=%.3f, ~%.1f%% of bbox diag). Examples: %s",
            len(dropped), min_sep, 100.0 * label_min_sep_frac,
            [(s, o) for s, o, _ in dropped[:5]],
        )
    return out.iloc[kept_idx].reset_index(drop=True)


def plot_umap_inference_predictions(
    background_embeddings: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    copy_emb: Optional[np.ndarray] = None,
    gids_src: Optional[Sequence[str]] = None,
    gids_tgt: Optional[Sequence[str]] = None,
    lineage_src: Optional[Sequence[str]] = None,
    lineage_tgt: Optional[Sequence[str]] = None,
    bg_genome_ids: Optional[Sequence[str]] = None,
    bg_lineages: Optional[Sequence[str]] = None,
    max_points: int = 2000,
    random_state: int = 42,
    marker_size_bg: float = 4.0,
    marker_size_fg: float = 7.0,
    fig_width: int = 1400,
    fig_height: int = 1000,
    umap_metric: str = "cosine",
    title: str = "UMAP: predictions vs ground truth (embedding space)",
    min_cluster_size: int = 20,
    show_lineage_labels: bool = True,
    max_pred_error: Optional[float] = None,
    pred_error_type: str = "l2",
):
    """2D UMAP of spike embeddings: background cloud + predicted vs true next states.

    Matches the pattern of :func:`plot_series_w_predictions`: fit UMAP on **background**
    points only, then ``transform`` prediction / target / copy so they lie in the same
    projection as diverse Spike sequences.

    Parameters
    ----------
    background_embeddings
        Array ``[N_bg, D]`` (e.g. unscaled ESM mean-pooled vectors), same ``D`` as pred.
    pred, target
        ``[N, D]`` predicted and true **absolute** embeddings at the target step.
    copy_emb
        Optional ``[N, D]`` source embedding (last input / copy baseline), same rows as pred.
        When set, draws **true jump** (source→GT), **pred jump** (source→pred), and **pred error**
        (pred→GT) segments in UMAP space.
    gids_src, gids_tgt
        Optional per-row genome IDs for hover text.
    lineage_src, lineage_tgt
        Optional WHO / Pango lineage strings aligned with rows (from metadata lookup).
    bg_genome_ids, bg_lineages
        Optional per-background-point genome ID and lineage (same length as
        ``background_embeddings``). Enables rich hover and lineage centroid labels.
    min_cluster_size
        Minimum number of background points per lineage to draw a text label (GA-style).
    show_lineage_labels
        If True and ``bg_lineages`` is set, plot lineage names at cluster centroids in UMAP space.
    max_pred_error
        If set, keep only rows whose prediction error (see ``pred_error_type``) is **≤** this
        value. Applied before ``max_points`` subsampling.
    pred_error_type
        ``l2`` — :math:`\\|\\hat{x} - x\\|_2` in embedding space (default). ``rmse`` —
        root mean squared difference across dimensions: :math:`\\sqrt{\\frac{1}{D}\\sum_d (\\hat{x}_d-x_d)^2}`.
    max_points
        Subsample rows for pred/target/copy when ``N`` is large (keeps HTML usable).
    umap_metric
        ``cosine`` is typical for ESM geometry; use ``euclidean`` to match some notebooks.
    """
    bg = np.asarray(background_embeddings, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if bg.ndim != 2 or pred.ndim != 2 or target.ndim != 2:
        raise ValueError("background_embeddings, pred, target must be 2D arrays")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    d = pred.shape[1]
    if bg.shape[1] != d:
        raise ValueError(
            f"background dim {bg.shape[1]} != prediction dim {d}"
        )

    if bg_genome_ids is not None and len(bg_genome_ids) != bg.shape[0]:
        raise ValueError("bg_genome_ids length must match background_embeddings")
    if bg_lineages is not None and len(bg_lineages) != bg.shape[0]:
        raise ValueError("bg_lineages length must match background_embeddings")

    if max_pred_error is not None:
        n_before_err = pred.shape[0]
        diff = pred - target
        if pred_error_type == "l2":
            metric = np.linalg.norm(diff, axis=1)
        elif pred_error_type == "rmse":
            metric = np.sqrt(np.mean(diff**2, axis=1))
        else:
            raise ValueError("pred_error_type must be 'l2' or 'rmse'")
        mask = metric <= float(max_pred_error)
        n_kept = int(mask.sum())
        if n_kept == 0:
            raise ValueError(
                f"No samples left after pred error filter (≤{max_pred_error}, "
                f"{pred_error_type}). Relax --max-pred-error."
            )
        _logger.info(
            "UMAP foreground filter: kept %d / %d rows (%s error ≤ %s)",
            n_kept, n_before_err, pred_error_type, max_pred_error,
        )
        pred = pred[mask]
        target = target[mask]
        if copy_emb is not None:
            copy_emb = np.asarray(copy_emb, dtype=np.float64)[mask]
        if gids_src is not None:
            gids_src = np.asarray(gids_src)[mask]
        if gids_tgt is not None:
            gids_tgt = np.asarray(gids_tgt)[mask]
        if lineage_src is not None:
            lineage_src = np.asarray(lineage_src)[mask]
        if lineage_tgt is not None:
            lineage_tgt = np.asarray(lineage_tgt)[mask]

    n = pred.shape[0]
    rng = np.random.default_rng(random_state)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        pred = pred[idx]
        target = target[idx]
        if copy_emb is not None:
            copy_emb = np.asarray(copy_emb, dtype=np.float64)[idx]
        if gids_src is not None:
            gids_src = np.asarray(gids_src)[idx]
        if gids_tgt is not None:
            gids_tgt = np.asarray(gids_tgt)[idx]
        if lineage_src is not None:
            lineage_src = np.asarray(lineage_src)[idx]
        if lineage_tgt is not None:
            lineage_tgt = np.asarray(lineage_tgt)[idx]

    n_bg = bg.shape[0]
    n_neighbors = int(min(15, max(2, n_bg - 1)))

    projector = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric=umap_metric,
        random_state=random_state,
        init="random",
    )
    proj_bg = projector.fit_transform(bg)
    proj_pred = _umap_transform_rows(projector, pred)
    proj_tgt = _umap_transform_rows(projector, target)

    if copy_emb is not None:
        proj_copy = _umap_transform_rows(projector, copy_emb)
    else:
        proj_copy = None

    def _str_lin(x, i):
        if x is None:
            return ""
        v = np.asarray(x)[i]
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    n_fg = len(proj_pred)
    gt_custom = np.empty((n_fg, 2), dtype=object)
    pr_custom = np.empty((n_fg, 2), dtype=object)
    for i in range(n_fg):
        gid_t = str(gids_tgt[i]) if gids_tgt is not None else f"idx={i}"
        lin_t = _str_lin(lineage_tgt, i) or "n/a"
        gt_custom[i] = [gid_t, lin_t]
        pr_custom[i] = [gid_t, lin_t]

    cp_custom = None
    if proj_copy is not None:
        cp_custom = np.empty((n_fg, 2), dtype=object)
        for i in range(n_fg):
            gid_s = str(gids_src[i]) if gids_src is not None else f"idx={i}"
            lin_s = _str_lin(lineage_src, i) or "n/a"
            cp_custom[i] = [gid_s, lin_s]

    bg_custom = None
    if bg_genome_ids is not None and bg_lineages is not None:
        bg_custom = np.stack(
            [np.asarray(bg_genome_ids, dtype=object),
             np.asarray(bg_lineages, dtype=object)],
            axis=1,
        )
    elif bg_genome_ids is not None:
        bg_custom = np.stack(
            [np.asarray(bg_genome_ids, dtype=object),
             np.full(len(bg_genome_ids), "n/a", dtype=object)],
            axis=1,
        )

    line_pred_tgt = go.Scatter(
        x=sum([[proj_pred[i, 0], proj_tgt[i, 0], None] for i in range(len(proj_pred))], []),
        y=sum([[proj_pred[i, 1], proj_tgt[i, 1], None] for i in range(len(proj_pred))], []),
        mode="lines",
        line=dict(color="rgba(200,80,80,0.45)", width=1),
        name="pred error",
        hoverinfo="skip",
        showlegend=True,
    )

    traces_line_copy = []
    if proj_copy is not None:
        line_copy_tgt = go.Scatter(
            x=sum([[proj_copy[i, 0], proj_tgt[i, 0], None] for i in range(len(proj_tgt))], []),
            y=sum([[proj_copy[i, 1], proj_tgt[i, 1], None] for i in range(len(proj_tgt))], []),
            mode="lines",
            line=dict(color="rgba(60,60,60,0.35)", width=1, dash="dash"),
            name="true jump",
            hoverinfo="skip",
            showlegend=True,
        )
        line_copy_pred = go.Scatter(
            x=sum([[proj_copy[i, 0], proj_pred[i, 0], None] for i in range(len(proj_pred))], []),
            y=sum([[proj_copy[i, 1], proj_pred[i, 1], None] for i in range(len(proj_pred))], []),
            mode="lines",
            line=dict(color="rgba(180,90,110,0.5)", width=1, dash="dot"),
            name="pred jump",
            hoverinfo="skip",
            showlegend=True,
        )
        traces_line_copy.append(line_copy_tgt)
        traces_line_copy.append(line_copy_pred)

    hover_fg = (
        "<b>%{fullData.name}</b><br>"
        "genome_id=%{customdata[0]}<br>"
        "lineage=%{customdata[1]}<br>"
        "UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"
    )
    hover_bg = (
        "genome_id=%{customdata[0]}<br>"
        "lineage=%{customdata[1]}<br>"
        "UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"
    )
    hover_bg_plain = "UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"

    fig = go.Figure()
    bg_trace_kwargs = dict(
        x=proj_bg[:, 0],
        y=proj_bg[:, 1],
        mode="markers",
        name="Background (Spike)",
        marker=dict(size=marker_size_bg, color="lightgray"),
        opacity=0.65,
    )
    if bg_custom is not None:
        bg_trace_kwargs["customdata"] = bg_custom
        bg_trace_kwargs["hovertemplate"] = hover_bg
    else:
        bg_trace_kwargs["hovertemplate"] = hover_bg_plain
    fig.add_trace(go.Scatter(**bg_trace_kwargs))

    for t in traces_line_copy:
        fig.add_trace(t)
    fig.add_trace(line_pred_tgt)

    if proj_copy is not None:
        fig.add_trace(
            go.Scatter(
                x=proj_copy[:, 0],
                y=proj_copy[:, 1],
                mode="markers",
                name="Source",
                marker=dict(size=marker_size_fg, color="steelblue", symbol="diamond"),
                customdata=cp_custom,
                hovertemplate=hover_fg,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=proj_pred[:, 0],
            y=proj_pred[:, 1],
            mode="markers",
            name="Predicted",
            marker=dict(size=marker_size_fg, color="crimson"),
            customdata=pr_custom,
            hovertemplate=hover_fg,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=proj_tgt[:, 0],
            y=proj_tgt[:, 1],
            mode="markers",
            name="Ground truth",
            marker=dict(size=marker_size_fg, color="forestgreen"),
            customdata=gt_custom,
            hovertemplate=hover_fg,
        )
    )

    if (
        show_lineage_labels
        and bg_lineages is not None
        and len(bg_lineages) == len(proj_bg)
    ):
        cent = _lineage_label_positions(proj_bg, bg_lineages, min_cluster_size)
        if len(cent) > 0:
            cent_text = [
                f"{row.lineage}<br>n={int(row.size)}"
                for row in cent.itertuples(index=False)
            ]
            fig.add_trace(
                go.Scatter(
                    x=cent["U1"],
                    y=cent["U2"],
                    mode="text",
                    name="Lineage labels",
                    text=cent_text,
                    textposition="middle center",
                    textfont=dict(size=11, color="black"),
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "<extra></extra>"
                    ),
                )
            )

    display_title = title
    if max_pred_error is not None:
        display_title = f"{title} — {pred_error_type} error ≤ {max_pred_error}"

    fig.update_layout(
        **_umap_series_figure_layout(
            title=display_title,
            fig_width=fig_width,
            fig_height=fig_height,
            n_legend_entries=3,
        )
    )
    return fig


def _compute_velocity_field(
    midpoints: np.ndarray,
    vels: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: int = 60,
    sigma: Optional[float] = None,
    density_quantile: float = 0.40,
    weight_velocity_inverse: bool = False,
    direction_coherence_min: float = 0.30,
    clip_to_convex_hull: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smooth per-step displacements onto a regular UMAP grid (Nadaraya-Watson).

    Defaults are tuned to avoid the "global-mean wallpaper" failure mode in which
    sparse data + wide kernels cause every cell to converge to the dataset-wide
    mean velocity. Specifically:

    * ``sigma`` defaults to ``max(2 × grid_step, 1.5 × median nearest-neighbor
      distance between midpoints)`` so the kernel always covers at least a few
      neighbors *and* shrinks where the data is dense.
    * ``direction_coherence_min`` rejects cells whose contributing displacements
      are not directionally consistent (mean-resultant length R below this
      threshold). R = 1 → all vectors point the same way; R = 0 → no consensus.
    * ``clip_to_convex_hull`` masks cells outside the convex hull of midpoints
      so streamlines stay inside the data envelope.
    * ``weight_velocity_inverse`` is **off** by default: down-weighting long
      vectors throws away the few high-information jumps that drive structure.

    Args:
        midpoints: (N, 2) midpoints of consecutive series steps in UMAP space.
        vels: (N, 2) UMAP velocity vector for each midpoint (next - prev).
        x_range, y_range: (lo, hi) plotting extent; grid spans this exactly.
        resolution: cells along each axis.
        sigma: Gaussian smoothing kernel std-dev in UMAP units. ``None`` = auto.
        density_quantile: cells below this percentile of smoothed sample density
            get ``mask=False`` so sparsely sampled regions stay empty.
        weight_velocity_inverse: if True, down-weight large displacements when
            estimating the mean velocity (was the previous default).
        direction_coherence_min: in (0, 1]; reject cells whose contributing
            displacements have mean-resultant length < this. ``0`` disables.
        clip_to_convex_hull: also reject cells outside the convex hull of the
            input midpoints; useful for arbitrary embedding shapes.

    Returns:
        ``(xg, yg, U, V, mask, density)`` — ``xg``/``yg`` are 1D coordinate
        arrays of length ``resolution``; ``U``/``V``/``mask``/``density`` are
        ``(resolution, resolution)`` arrays in row-major order with ``y`` along
        rows and ``x`` along columns (Plotly/matplotlib convention).
    """
    midpoints = np.asarray(midpoints, dtype=np.float64)
    vels = np.asarray(vels, dtype=np.float64)
    if midpoints.ndim != 2 or midpoints.shape[1] != 2:
        raise ValueError(f"midpoints must be (N, 2); got {midpoints.shape}")
    if vels.shape != midpoints.shape:
        raise ValueError(
            f"vels shape {vels.shape} != midpoints shape {midpoints.shape}"
        )
    if midpoints.shape[0] == 0:
        raise ValueError("No midpoints to aggregate")

    xg = np.linspace(x_range[0], x_range[1], int(resolution))
    yg = np.linspace(y_range[0], y_range[1], int(resolution))
    Xg, Yg = np.meshgrid(xg, yg)
    grid_xy = np.column_stack([Xg.ravel(), Yg.ravel()])

    diag = float(np.hypot(x_range[1] - x_range[0], y_range[1] - y_range[0]))
    step = float((x_range[1] - x_range[0]) / max(int(resolution) - 1, 1))
    if sigma is None:
        # Pick a kernel that covers at least a handful of nearest neighbors;
        # this prevents NW from collapsing to the global mean on sparse grids.
        try:
            from sklearn.neighbors import NearestNeighbors
            k = int(min(8, max(2, midpoints.shape[0] // 20)))
            nn = NearestNeighbors(n_neighbors=k).fit(midpoints)
            d, _ = nn.kneighbors(midpoints)
            nn_dist = float(np.median(d[:, -1]))
        except Exception:
            nn_dist = 0.0
        sigma = float(max(2.0 * step, 1.5 * nn_dist))
    sigma = float(max(sigma, 1e-6 * diag))

    if weight_velocity_inverse:
        v_norms = np.linalg.norm(vels, axis=1)
        scale = float(np.median(v_norms[v_norms > 0])) if np.any(v_norms > 0) else 1.0
        sample_w = np.exp(-v_norms / max(scale, 1e-9))
    else:
        sample_w = np.ones(midpoints.shape[0], dtype=np.float64)

    n_cells = grid_xy.shape[0]
    n_pts = midpoints.shape[0]
    block = max(64, min(8192, int(2_000_000 / max(n_pts, 1))))
    U = np.zeros(n_cells, dtype=np.float64)
    V = np.zeros(n_cells, dtype=np.float64)
    Wsum = np.zeros(n_cells, dtype=np.float64)
    Cohx = np.zeros(n_cells, dtype=np.float64)
    Cohy = np.zeros(n_cells, dtype=np.float64)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    v_norms_arr = np.linalg.norm(vels, axis=1)
    safe = v_norms_arr > 0
    unit = np.zeros_like(vels)
    unit[safe] = vels[safe] / v_norms_arr[safe, None]
    for s in range(0, n_cells, block):
        e = min(s + block, n_cells)
        d2 = (
            (grid_xy[s:e, None, 0] - midpoints[None, :, 0]) ** 2
            + (grid_xy[s:e, None, 1] - midpoints[None, :, 1]) ** 2
        )
        w = np.exp(-d2 * inv2s2) * sample_w[None, :]
        wn = w.sum(axis=1)
        Wsum[s:e] = wn
        denom = np.where(wn > 0, wn, 1.0)
        U[s:e] = (w @ vels[:, 0]) / denom
        V[s:e] = (w @ vels[:, 1]) / denom
        Cohx[s:e] = (w @ unit[:, 0]) / denom
        Cohy[s:e] = (w @ unit[:, 1]) / denom

    U = U.reshape(int(resolution), int(resolution))
    V = V.reshape(int(resolution), int(resolution))
    density = Wsum.reshape(int(resolution), int(resolution))
    coherence = np.hypot(Cohx, Cohy).reshape(int(resolution), int(resolution))

    if density_quantile is None:
        mask = np.ones_like(density, dtype=bool)
    else:
        q = float(min(max(density_quantile, 0.0), 0.999))
        thresh = float(np.quantile(density, q))
        mask = density >= thresh

    if direction_coherence_min and direction_coherence_min > 0.0:
        mask &= coherence >= float(direction_coherence_min)

    if clip_to_convex_hull and midpoints.shape[0] >= 3:
        try:
            from scipy.spatial import ConvexHull
            from matplotlib.path import Path as _MplPath

            hull = ConvexHull(midpoints)
            poly = midpoints[hull.vertices]
            inside = _MplPath(poly).contains_points(grid_xy).reshape(mask.shape)
            mask &= inside
        except Exception as exc:  # degenerate midpoints (collinear etc.)
            _logger.warning("Convex hull mask skipped: %s", exc)

    if not mask.any():
        # Always keep at least one cell so downstream rendering does not crash.
        mask[np.unravel_index(int(np.argmax(density)), density.shape)] = True

    return xg, yg, U, V, mask, density


def _integrate_streamlines(
    xg: np.ndarray,
    yg: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    seed_density: float = 2.0,
    max_steps: int = 400,
    step_frac: float = 0.5,
    min_speed_frac: float = 1e-4,
    min_segment_len_cells: float = 1.5,
) -> List[np.ndarray]:
    """Integrate streamlines through a smoothed 2D vector field.

    Custom RK2 integrator that replaces ``plotly.figure_factory.create_streamline``,
    which divides by speed and crashes on zero-velocity cells (e.g. cells with no
    nearby data). This integrator stops cleanly when speed drops below
    ``min_speed_frac × global_max_speed`` or the path leaves the grid.

    Args:
        xg, yg: 1D coordinate arrays (length ``Nx`` and ``Ny``); evenly spaced.
        U, V: ``(Ny, Nx)`` smoothed velocity components.
        mask: ``(Ny, Nx)`` boolean; integration starts only inside masked cells
            and stops as soon as the path leaves the masked region.
        seed_density: roughly ``Nx*seed_density`` × ``Ny*seed_density`` candidate
            seeds; capped to keep runtime bounded.
        max_steps: maximum forward (and backward) integration steps per seed.
        step_frac: integration step length as fraction of grid spacing.
        min_speed_frac: stop integration when local ``|v| < this × max(|v|)``.
        min_segment_len_cells: drop streamlines shorter than this many cells.

    Returns:
        List of ``(L_i, 2)`` polylines in the same coordinate system as ``xg/yg``.
    """
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:  # pragma: no cover - scipy is in requirements
        _logger.warning("scipy not available; streamlines disabled")
        return []

    xg = np.asarray(xg, dtype=np.float64)
    yg = np.asarray(yg, dtype=np.float64)
    U = np.nan_to_num(np.asarray(U, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(np.asarray(V, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Ny, Nx = U.shape
    if xg.size != Nx or yg.size != Ny:
        raise ValueError("xg/yg length must match U/V grid")
    if Nx < 2 or Ny < 2:
        return []

    interp_u = RegularGridInterpolator(
        (yg, xg), U, bounds_error=False, fill_value=0.0
    )
    interp_v = RegularGridInterpolator(
        (yg, xg), V, bounds_error=False, fill_value=0.0
    )

    speeds = np.hypot(U, V)
    max_speed = float(np.max(speeds))
    if max_speed <= 0:
        return []
    speed_floor = float(min_speed_frac) * max_speed

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    h = float(step_frac) * float(min(dx, dy))

    # Seed grid: a coarse grid scaled by seed_density, at most ~64x64.
    sd = float(max(seed_density, 0.5))
    sx = int(np.clip(round(Nx * 0.18 * sd), 8, 64))
    sy = int(np.clip(round(Ny * 0.18 * sd), 8, 64))
    seed_x = np.linspace(xg[0] + dx, xg[-1] - dx, sx)
    seed_y = np.linspace(yg[0] + dy, yg[-1] - dy, sy)
    SX, SY = np.meshgrid(seed_x, seed_y)
    seed_pts = np.column_stack([SX.ravel(), SY.ravel()])

    if mask is not None:
        mask_interp = RegularGridInterpolator(
            (yg, xg), mask.astype(np.float64), bounds_error=False, fill_value=0.0
        )
        keep = mask_interp(seed_pts[:, ::-1]) > 0.5
        seed_pts = seed_pts[keep]
    if seed_pts.size == 0:
        return []

    def vel(p: np.ndarray) -> tuple[float, float]:
        # RegularGridInterpolator wants (y, x); we work in (x, y).
        u = float(interp_u((p[1], p[0])))
        v = float(interp_v((p[1], p[0])))
        return u, v

    x_lo, x_hi = float(xg[0]), float(xg[-1])
    y_lo, y_hi = float(yg[0]), float(yg[-1])

    def integrate_one(seed: np.ndarray, sign: float) -> np.ndarray:
        pts = [seed.copy()]
        p = seed.copy()
        for _ in range(int(max_steps)):
            u1, v1 = vel(p)
            sp1 = float(np.hypot(u1, v1))
            if sp1 < speed_floor:
                break
            # RK2 midpoint
            p_mid = p + sign * (h * 0.5) * np.array([u1, v1]) / sp1
            u2, v2 = vel(p_mid)
            sp2 = float(np.hypot(u2, v2))
            if sp2 < speed_floor:
                break
            p_new = p + sign * h * np.array([u2, v2]) / sp2
            if not (x_lo <= p_new[0] <= x_hi and y_lo <= p_new[1] <= y_hi):
                break
            if mask is not None:
                if mask_interp((p_new[1], p_new[0])) <= 0.5:
                    break
            pts.append(p_new)
            p = p_new
        return np.asarray(pts, dtype=np.float64)

    min_len = float(min_segment_len_cells) * float(min(dx, dy))
    polylines: List[np.ndarray] = []
    for seed in seed_pts:
        forward = integrate_one(seed, +1.0)
        backward = integrate_one(seed, -1.0)
        if forward.shape[0] >= 2 or backward.shape[0] >= 2:
            line = np.vstack([backward[::-1], forward[1:]]) if backward.shape[0] > 1 else forward
            if line.shape[0] >= 2:
                seg_len = float(
                    np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1))
                )
                if seg_len >= min_len:
                    polylines.append(line)
    return polylines


def _add_velocity_field_traces(
    fig: go.Figure,
    xg: np.ndarray,
    yg: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray,
    density: np.ndarray,
    *,
    mode: str = "grid",
    arrow_scale: float = 0.6,
    grid_color: str = "#111111",
    stream_color: str = "#1f4e79",
    grid_opacity: float = 0.85,
    stream_opacity: float = 0.40,
    stream_density: float = 1.0,
    legend_group: str = "field",
) -> None:
    """Render an aggregated UMAP velocity field on top of an existing figure.

    Args:
        mode: ``"grid"`` (one arrow per masked cell), ``"stream"`` (smooth flow
            curves via a custom RK2 integrator), or ``"grid+stream"`` (both).
            Anything else is a no-op.
        arrow_scale: arrow length as fraction of grid spacing for ``"grid"``.
        stream_density: passed to :func:`_integrate_streamlines`
            (larger = more streamlines).
    """
    mode = (mode or "off").lower()
    if mode in ("off", "none", ""):
        return

    U = np.nan_to_num(np.asarray(U, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(np.asarray(V, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Um_grid = np.where(mask, U, 0.0)
    Vm_grid = np.where(mask, V, 0.0)
    dx = float(xg[1] - xg[0]) if xg.size > 1 else 1.0

    if mode in ("grid", "grid+stream"):
        idx = np.flatnonzero(mask.ravel())
        if idx.size > 0:
            Xg, Yg = np.meshgrid(xg, yg)
            x_pts = Xg.ravel()[idx]
            y_pts = Yg.ravel()[idx]
            u_pts = Um_grid.ravel()[idx]
            v_pts = Vm_grid.ravel()[idx]
            speed = np.hypot(u_pts, v_pts)
            max_speed = float(np.max(speed)) if speed.size else 1.0
            if max_speed > 0:
                scale = (arrow_scale * dx) / max_speed
            else:
                scale = arrow_scale * dx
            quiv = ff.create_quiver(
                x_pts, y_pts, u_pts, v_pts,
                scale=scale,
                arrow_scale=0.45,
                line=dict(width=1.4, color=grid_color),
                name="Velocity field (grid)",
            )
            for tr in quiv.data:
                tr.opacity = max(float(grid_opacity), 0.75)
                tr.legendgroup = legend_group
                tr.hoverinfo = "skip"
                fig.add_trace(tr)
            _logger.info(
                "Velocity grid arrows: %d cells, max_speed=%.4g, scale=%.4g",
                int(idx.size),
                max_speed,
                scale,
            )

    if mode in ("stream", "grid+stream"):
        # Custom RK2 integrator handles zero-speed regions cleanly. We pass
        # the *unmasked* smoothed U/V so the field is continuous and non-zero
        # almost everywhere, and use ``mask`` only to constrain seed selection
        # and termination (so we don't draw lines through truly empty regions).
        try:
            polylines = _integrate_streamlines(
                xg, yg, U, V,
                mask=mask,
                seed_density=float(stream_density),
            )
        except Exception as exc:  # defensive; should not happen in practice
            _logger.warning("Streamline rendering skipped: %s", exc)
            polylines = []
        if not polylines:
            _logger.info("No streamlines to draw (degenerate or zero field)")
        else:
            xs: List[float] = []
            ys: List[float] = []
            for poly in polylines:
                xs.extend(list(poly[:, 0]) + [None])
                ys.extend(list(poly[:, 1]) + [None])
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(width=1.0, color=stream_color),
                    opacity=stream_opacity,
                    name="Velocity field (streamlines)",
                    legendgroup=legend_group,
                    hoverinfo="skip",
                )
            )
            # Subtle arrow-tip annotations on the longest streamlines so the
            # global flow direction is unambiguous.
            polylines_sorted = sorted(
                polylines, key=lambda p: float(p.shape[0]), reverse=True
            )
            for poly in polylines_sorted[: min(40, len(polylines_sorted))]:
                if poly.shape[0] < 2:
                    continue
                fig.add_annotation(
                    x=float(poly[-1, 0]), y=float(poly[-1, 1]),
                    ax=float(poly[-2, 0]), ay=float(poly[-2, 1]),
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2,
                    arrowsize=0.9, arrowwidth=1.0,
                    arrowcolor=stream_color, opacity=stream_opacity,
                    standoff=1, text="",
                )


def _add_polyline_arrowheads(
    fig: go.Figure,
    polylines: List[np.ndarray],
    *,
    color_per_line: Optional[List[str]] = None,
    head_size: float = 12.0,
) -> None:
    """Mark the last segment of each polyline with an arrowhead annotation."""
    for i, p in enumerate(polylines):
        if p is None or len(p) < 2:
            continue
        x0, y0 = float(p[-2, 0]), float(p[-2, 1])
        x1, y1 = float(p[-1, 0]), float(p[-1, 1])
        c = color_per_line[i] if color_per_line and i < len(color_per_line) else "#222222"
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.0,
            arrowwidth=1.6, arrowcolor=c,
            standoff=2,
            text="",
        )


def plot_umap_series_trajectories(
    background_embeddings: np.ndarray,
    trajectories: List[np.ndarray],
    trajectory_gids: Optional[List[np.ndarray]] = None,
    trajectory_lineages: Optional[List[np.ndarray]] = None,
    series_labels: Optional[List[str]] = None,
    bg_genome_ids: Optional[Sequence[str]] = None,
    bg_lineages: Optional[Sequence[str]] = None,
    predictions: Optional[List[Optional[dict]]] = None,
    prediction_clouds: Optional[List[Optional[np.ndarray]]] = None,
    prediction_cloud_max_points: int = 0,
    prediction_cloud_idle_white: bool = False,
    prediction_cloud_marker_size: float = 11.0,
    prediction_cloud_idle_marker_size: float = 8.0,
    prediction_cloud_marker_opacity: float = 0.55,
    background_marker_color: Optional[str] = None,
    random_state: int = 42,
    umap_metric: str = "cosine",
    min_cluster_size: int = 20,
    show_lineage_labels: bool = True,
    title: str = "UMAP: embedding trajectories (time order)",
    fig_width: int = 1500,
    fig_height: int = 1050,
    marker_size_start: float = 5.0,
    marker_size_end: float = 13.0,
    line_width: float = 2.0,
    field_trajectories: Optional[List[np.ndarray]] = None,
    field_mode: str = "off",
    field_resolution: int = 60,
    field_sigma: Optional[float] = None,
    field_density_quantile: float = 0.40,
    field_arrow_scale: float = 0.7,
    field_stream_density: float = 1.0,
    field_direction_coherence_min: float = 0.30,
    field_clip_to_hull: bool = True,
    field_weight_velocity_inverse: bool = False,
    polyline_alpha: Optional[float] = None,
    polyline_arrowheads: bool = False,
    polylines_visible_default: bool = True,
    force_polyline_arrowheads: bool = False,
    series_trace_mode: str = "lines+markers",
    label_min_sep_frac: float = 0.04,
    label_smooth_sigma_bins: float = 1.5,
) -> go.Figure:
    """2D UMAP of raw series embeddings: background cloud + polylines in time order.

    Fits UMAP on ``background_embeddings`` only, then transforms each trajectory.
    Each series is drawn as a connected path. When ``series_trace_mode`` is
    ``"lines+markers"`` (default), marker size **increases** along time so jump
    direction is obvious; in ``"lines"`` mode, only the path is drawn (no
    per-step markers) — useful for a clean first view; hover is still
    available at UMAP points when supported by the viewer.

    Background lineage centroids show **two-line** labels (lineage, then ``n=`` count).

    Parameters
    ----------
    predictions
        Optional list aligned with ``trajectories``. Each item is either ``None`` (no
        prediction available for that series) or a ``dict`` with keys:

        - ``vec`` ([D] np.ndarray): predicted next embedding in the same space as
          ``background_embeddings`` and ``trajectories``.
        - ``src_step`` (int): index in the trajectory of the model's source step
          (e.g. ``keep_idx[-2]`` for skip configs).
        - ``tgt_step`` (int, optional): index of the GT step the model targeted
          (defaults to last). Drawn as a **filled circle** (opacity 1, black
          outline) with a solid source→GT segment so it stays visible when the
          polyline is faded.
          (typically ``keep_idx[-1]``); used for hover only.
        - ``gid`` (str, optional): predicted genome id for hover text.
        - ``best_sample_idx`` (int, optional): index into the per-series diffusion
          cloud; that draw is omitted from triangle markers when subsampling clouds.
        - ``pred_star_note`` (str, optional): extra hover line describing the ★
          (e.g. top k-NN density mode vs lowest-MSE draw).

        For each prediction, a dashed segment is drawn from the source-step UMAP
        position to the projected predicted point, plus a star marker at the prediction.
    prediction_clouds
        Optional list aligned with ``trajectories``. Each item is either ``None`` or
        an array ``(n_samples, D)`` of extra draws (e.g. diffusion samples). Points are
        projected with the same UMAP and drawn as semi-transparent **triangle** markers
        (same colour as the series polyline),         except the row matching the star prediction — omitted when
        ``best_sample_idx`` is set on the paired prediction dict, or (for a mean ★
        without that field) the diffusion sample **nearest** to the prediction
        vector in the current (possibly subsampled) cloud. ``prediction_cloud_max_points`` optionally
        subsamples (the best-sample row is kept when ``best_sample_idx`` is set).
    prediction_cloud_max_points
        If > 0, randomly keep at most this many cloud points per series (for large S).
    prediction_cloud_idle_white
        When ``True`` and ``polylines_visible_default`` is ``False`` (legend-only
        series on first load), draw each diffusion cloud twice: a **white**
        marker layer always visible (model samples), and the usual coloured layer
        tied to the legend so it appears on top when the user enables that series.
    prediction_cloud_marker_size
        Plotly marker size for coloured per-series cloud triangles.
    prediction_cloud_idle_marker_size
        Marker size for the always-visible white idle cloud layer (when enabled).
    prediction_cloud_marker_opacity
        Opacity for coloured cloud triangles.
    background_marker_color
        Marker colour for the Spike / background scatter. ``None`` keeps the
        default light grey.
    random_state
        RNG seed for UMAP and optional cloud subsampling.
    field_trajectories
        Optional **larger pool** of series ([T_i, D]) projected through the same UMAP
        and aggregated into a smoothed velocity field. Polylines in ``trajectories``
        are still drawn on top — typically a small, easy-to-read subset highlighted
        against the field. If ``None``, ``trajectories`` themselves are used.
    field_mode
        ``"off"`` (default), ``"grid"`` (one binned mean-velocity arrow per cell),
        ``"stream"`` (continuous streamlines à la Evolocity / scVelo), or
        ``"grid+stream"``. Vector field is computed from per-step UMAP displacements.
    field_resolution / field_sigma / field_density_quantile / field_arrow_scale /
    field_stream_density
        Knobs for :func:`_compute_velocity_field` and :func:`_add_velocity_field_traces`.
    polyline_alpha
        Per-series line/marker opacity. If ``None`` and ``field_mode != "off"`` or
        more than ~12 trajectories are drawn, opacity is auto-scaled to keep the
        canvas readable as the count grows.
    polyline_arrowheads
        Add an arrowhead annotation at the **last** segment of each polyline so
        direction is visible without scanning marker sizes.
    force_polyline_arrowheads
        If ``True``, draw these arrowheads even when ``polylines_visible_default``
        is ``False`` (line traces in ``legendonly``). Used for an "arrowheads
        only" first view: annotations are not legend-toggleable, but the
        polylines stay off until the user enables them in the legend.
    series_trace_mode
        ``"lines+markers"`` (default) or ``"lines"``. The latter draws only
        the polyline for the main trajectory (no per-step size gradient).
    """
    bg = np.asarray(background_embeddings, dtype=np.float64)
    if bg.ndim != 2:
        raise ValueError("background_embeddings must be 2D")
    _stm = str(series_trace_mode)
    if _stm not in ("lines", "lines+markers"):
        raise ValueError(
            f"series_trace_mode must be 'lines' or 'lines+markers', got {_stm!r}"
        )

    clean_trajs: List[np.ndarray] = []
    clean_gids: List[Optional[np.ndarray]] = []
    clean_lins: List[Optional[np.ndarray]] = []
    clean_labels: List[str] = []

    for i, tr in enumerate(trajectories):
        t = np.asarray(tr, dtype=np.float64)
        if t.ndim != 2 or t.shape[0] < 2:
            continue
        if t.shape[1] != bg.shape[1]:
            raise ValueError(
                f"trajectory {i} has dim {t.shape[1]} != background dim {bg.shape[1]}"
            )
        clean_trajs.append(t)
        if trajectory_gids is not None and i < len(trajectory_gids):
            g = np.asarray(trajectory_gids[i], dtype=object)
            if len(g) != t.shape[0]:
                raise ValueError(f"trajectory_gids[{i}] length != trajectory steps")
            clean_gids.append(g)
        else:
            clean_gids.append(None)
        if trajectory_lineages is not None and i < len(trajectory_lineages):
            lin = np.asarray(trajectory_lineages[i], dtype=object)
            if len(lin) != t.shape[0]:
                raise ValueError(f"trajectory_lineages[{i}] length != trajectory steps")
            clean_lins.append(lin)
        else:
            clean_lins.append(None)
        if series_labels is not None and i < len(series_labels):
            clean_labels.append(str(series_labels[i]))
        else:
            clean_labels.append(f"series {i}")

    if not clean_trajs:
        raise ValueError("No trajectories with at least 2 time steps")

    if bg_genome_ids is not None and len(bg_genome_ids) != bg.shape[0]:
        raise ValueError("bg_genome_ids length must match background_embeddings")
    if bg_lineages is not None and len(bg_lineages) != bg.shape[0]:
        raise ValueError("bg_lineages length must match background_embeddings")

    n_bg = bg.shape[0]
    n_neighbors = int(min(15, max(2, n_bg - 1)))
    projector = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric=umap_metric,
        random_state=random_state,
        init="random",
    )
    proj_bg = projector.fit_transform(bg)

    lengths = [t.shape[0] for t in clean_trajs]
    all_pts = np.vstack(clean_trajs)
    proj_flat = _umap_transform_rows(projector, all_pts)
    starts = np.cumsum([0] + lengths[:-1])
    projs: List[np.ndarray] = [
        proj_flat[starts[i] : starts[i] + lengths[i]] for i in range(len(clean_trajs))
    ]

    # Velocity field aggregation (Evolocity-style). Project the larger pool of series,
    # collect per-step (midpoint, velocity) in UMAP space, then smooth onto a grid.
    field_xy = None
    if (field_mode or "off").lower() not in ("off", "none", "") and (
        field_trajectories is not None or len(clean_trajs) > 0
    ):
        pool = field_trajectories if field_trajectories is not None else clean_trajs
        f_clean: List[np.ndarray] = []
        for t in pool:
            arr = np.asarray(t, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] == bg.shape[1]:
                f_clean.append(arr)
        if f_clean:
            f_lengths = [t.shape[0] for t in f_clean]
            f_proj_flat = _umap_transform_rows(projector, np.vstack(f_clean))
            f_starts = np.cumsum([0] + f_lengths[:-1])
            mids: List[np.ndarray] = []
            vels: List[np.ndarray] = []
            for i, T in enumerate(f_lengths):
                seg = f_proj_flat[f_starts[i] : f_starts[i] + T]
                mids.append(0.5 * (seg[:-1] + seg[1:]))
                vels.append(seg[1:] - seg[:-1])
            mids_arr = np.concatenate(mids, axis=0)
            vels_arr = np.concatenate(vels, axis=0)
            x_lo, y_lo = float(np.min(proj_bg[:, 0])), float(np.min(proj_bg[:, 1]))
            x_hi, y_hi = float(np.max(proj_bg[:, 0])), float(np.max(proj_bg[:, 1]))
            # Slight padding so arrows / streamlines are not clipped at the edge.
            pad_x = 0.02 * (x_hi - x_lo) if x_hi > x_lo else 0.5
            pad_y = 0.02 * (y_hi - y_lo) if y_hi > y_lo else 0.5
            field_xy = _compute_velocity_field(
                mids_arr, vels_arr,
                x_range=(x_lo - pad_x, x_hi + pad_x),
                y_range=(y_lo - pad_y, y_hi + pad_y),
                resolution=int(field_resolution),
                sigma=field_sigma,
                density_quantile=float(field_density_quantile),
                weight_velocity_inverse=bool(field_weight_velocity_inverse),
                direction_coherence_min=float(field_direction_coherence_min),
                clip_to_convex_hull=bool(field_clip_to_hull),
            )
            xg_, yg_, _U_, _V_, mask_, _dens_ = field_xy
            n_total_cells = int(mask_.size)
            n_kept_cells = int(mask_.sum())
            _logger.info(
                "Velocity field: mode=%s pool=%d series, %d displacements, "
                "grid=%dx%d, sigma=%s, density-q=%.2f, coherence-min=%.2f, "
                "hull-clip=%s, kept %d/%d cells (%.1f%%)",
                field_mode, len(f_clean), mids_arr.shape[0],
                int(field_resolution), int(field_resolution),
                "auto" if field_sigma is None else f"{float(field_sigma):.4g}",
                float(field_density_quantile),
                float(field_direction_coherence_min),
                bool(field_clip_to_hull),
                n_kept_cells, n_total_cells,
                100.0 * n_kept_cells / max(n_total_cells, 1),
            )

    # Align optional single predictions + sample clouds with cleaned trajectories.
    clean_pos = 0
    orig_to_clean: dict[int, int] = {}
    for i, tr in enumerate(trajectories):
        t = np.asarray(tr)
        if t.ndim != 2 or t.shape[0] < 2:
            continue
        orig_to_clean[i] = clean_pos
        clean_pos += 1

    clean_preds: List[Optional[dict]] = [None] * len(clean_trajs)
    if predictions is not None:
        if len(predictions) != len(trajectories):
            raise ValueError("predictions length must match trajectories length")
        pred_vecs: List[np.ndarray] = []
        pred_clean_idx: List[int] = []
        for orig_i, p in enumerate(predictions):
            if p is None or orig_i not in orig_to_clean:
                continue
            v = np.asarray(p["vec"], dtype=np.float64).reshape(-1)
            if v.shape[0] != bg.shape[1]:
                raise ValueError(
                    f"prediction[{orig_i}] dim {v.shape[0]} != background dim {bg.shape[1]}"
                )
            pred_vecs.append(v)
            pred_clean_idx.append(orig_to_clean[orig_i])
        if pred_vecs:
            proj_pred_all = _umap_transform_rows(
                projector, np.stack(pred_vecs, axis=0)
            )
            for k_idx, ci in enumerate(pred_clean_idx):
                p_orig = predictions[
                    next(o for o, c in orig_to_clean.items() if c == ci)
                ]
                clean_preds[ci] = {
                    "vec": pred_vecs[k_idx],
                    "proj": proj_pred_all[k_idx],
                    "src_step": int(p_orig.get("src_step", clean_trajs[ci].shape[0] - 2)),
                    "tgt_step": int(p_orig.get("tgt_step", clean_trajs[ci].shape[0] - 1)),
                    "gid": str(p_orig.get("gid", "n/a")),
                    "best_sample_idx": p_orig.get("best_sample_idx"),
                }

    clean_clouds: List[Optional[np.ndarray]] = [None] * len(clean_trajs)
    clean_cloud_exclude_ix: List[Optional[int]] = [None] * len(clean_trajs)
    if prediction_clouds is not None:
        if len(prediction_clouds) != len(trajectories):
            raise ValueError(
                "prediction_clouds length must match trajectories length"
            )
        for orig_i, cloud in enumerate(prediction_clouds):
            if cloud is None or orig_i not in orig_to_clean:
                continue
            arr = np.asarray(cloud, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    f"prediction_clouds[{orig_i}] must be 2D (n_samples, D); "
                    f"got shape {arr.shape}"
                )
            if arr.shape[1] != bg.shape[1]:
                raise ValueError(
                    f"prediction_clouds[{orig_i}] dim {arr.shape[1]} != "
                    f"background dim {bg.shape[1]}"
                )
            preserve_ix: Optional[int] = None
            if predictions is not None and orig_i < len(predictions):
                pm = predictions[orig_i]
                if pm is not None and pm.get("best_sample_idx") is not None:
                    _bi = int(pm["best_sample_idx"])
                    if 0 <= _bi < arr.shape[0]:
                        preserve_ix = _bi
            row_finite = np.isfinite(arr).all(axis=1)
            if not np.any(row_finite):
                continue
            if not np.all(row_finite):
                orig_indices = np.flatnonzero(row_finite)
                arr = arr[row_finite]
                if preserve_ix is not None:
                    remap = np.flatnonzero(orig_indices == preserve_ix)
                    preserve_ix = int(remap[0]) if remap.size else None
            n_full = int(arr.shape[0])

            arr_work = arr
            exclude_cloud_row: Optional[int] = None
            mpc = int(prediction_cloud_max_points)
            if mpc > 0 and n_full > mpc:
                rng = np.random.default_rng(int(random_state) + orig_i)
                if preserve_ix is not None:
                    others = np.array(
                        [j for j in range(n_full) if j != preserve_ix], dtype=int
                    )
                    need_extra = mpc - 1
                    if len(others) == 0:
                        pick = np.array([preserve_ix], dtype=int)
                    elif need_extra <= 0:
                        pick = np.array([preserve_ix], dtype=int)
                    else:
                        take = min(need_extra, len(others))
                        extra = rng.choice(others, size=take, replace=False)
                        pick = np.concatenate(
                            [np.array([preserve_ix], dtype=int), extra.astype(int)]
                        )
                    arr_work = arr[pick]
                    exclude_cloud_row = int(np.flatnonzero(pick == preserve_ix)[0])
                else:
                    pick = rng.choice(n_full, size=mpc, replace=False)
                    arr_work = arr[pick]
            elif preserve_ix is not None:
                exclude_cloud_row = preserve_ix

            if exclude_cloud_row is None and predictions is not None and orig_i < len(
                predictions
            ):
                pm = predictions[orig_i]
                if (
                    pm is not None
                    and pm.get("best_sample_idx") is None
                    and pm.get("vec") is not None
                    and arr_work.shape[0] > 0
                ):
                    pv = np.asarray(pm["vec"], dtype=np.float64).reshape(-1)
                    d = np.linalg.norm(
                        np.asarray(arr_work, dtype=np.float64) - pv.reshape(1, -1),
                        axis=1,
                    )
                    exclude_cloud_row = int(np.argmin(d))

            ci = orig_to_clean[orig_i]
            clean_clouds[ci] = arr_work
            clean_cloud_exclude_ix[ci] = exclude_cloud_row
    palette = list(px.colors.qualitative.Safe) + list(px.colors.qualitative.Dark24)

    field_on = (field_mode or "off").lower() not in ("off", "none", "")
    if polyline_alpha is None:
        n_lines_eff = max(1, len(clean_trajs))
        if field_on:
            # The field carries global structure; keep the few highlighted polylines
            # readable on top of it (white outline below also helps).
            polyline_alpha_eff = 0.85 if n_lines_eff <= 12 else 0.65
        elif n_lines_eff > 12:
            polyline_alpha_eff = float(min(1.0, max(0.25, 12.0 / n_lines_eff)))
        else:
            polyline_alpha_eff = 1.0
    else:
        polyline_alpha_eff = float(min(1.0, max(0.05, polyline_alpha)))
    # Width / outline scale up when a field is drawn underneath so polylines
    # don't get visually buried by streamlines or quiver arrows.
    polyline_line_width_eff = float(line_width) + (1.0 if field_on else 0.0)
    polyline_marker_outline_w = 1.6 if field_on else 0.5

    hover_fg = (
        "<b>%{fullData.name}</b><br>"
        "step %{customdata[0]} / %{customdata[1]} (time order)<br>"
        "genome_id=%{customdata[2]}<br>"
        "lineage=%{customdata[3]}<br>"
        "‖Δembed‖₂ from prev=%{customdata[4]:.3f}<extra></extra>"
    )
    hover_bg = (
        "genome_id=%{customdata[0]}<br>"
        "lineage=%{customdata[1]}<br>"
        "UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"
    )
    hover_bg_plain = "UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"

    fig = go.Figure()
    _bg_mc = background_marker_color if background_marker_color else "lightgray"
    bg_trace_kwargs = dict(
        x=proj_bg[:, 0],
        y=proj_bg[:, 1],
        mode="markers",
        name="Background (Spike)",
        marker=dict(size=4.0, color=_bg_mc),
        opacity=0.55,
    )
    if bg_genome_ids is not None and bg_lineages is not None:
        bg_trace_kwargs["customdata"] = np.stack(
            [np.asarray(bg_genome_ids, dtype=object), np.asarray(bg_lineages, dtype=object)],
            axis=1,
        )
        bg_trace_kwargs["hovertemplate"] = hover_bg
    elif bg_genome_ids is not None:
        bg_trace_kwargs["customdata"] = np.stack(
            [np.asarray(bg_genome_ids, dtype=object),
             np.full(len(bg_genome_ids), "n/a", dtype=object)],
            axis=1,
        )
        bg_trace_kwargs["hovertemplate"] = hover_bg
    else:
        bg_trace_kwargs["hovertemplate"] = hover_bg_plain
    fig.add_trace(go.Scatter(**bg_trace_kwargs))

    if field_xy is not None:
        xg_, yg_, U_, V_, mask_, density_ = field_xy
        _add_velocity_field_traces(
            fig, xg_, yg_, U_, V_, mask_, density_,
            mode=field_mode,
            arrow_scale=float(field_arrow_scale),
            stream_density=float(field_stream_density),
        )

    has_any_pred = any(cp is not None for cp in clean_preds)
    if has_any_pred:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers",
                line=dict(color="black", dash="dash", width=1.5),
                marker=dict(symbol="star", size=12, color="black",
                            line=dict(width=1.2, color="black")),
                name="Prediction (★ at predicted next; dashed from source)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers",
                line=dict(color="black", width=2.0),
                marker=dict(
                    symbol="circle",
                    size=12,
                    color="black",
                    line=dict(width=2.2, color="black"),
                ),
                name="Ground truth (● filled circle; solid from source)",
                hoverinfo="skip",
            )
        )

    has_any_cloud = any(cc is not None for cc in clean_clouds)
    if has_any_cloud:
        _idle_white = bool(prediction_cloud_idle_white) and not bool(
            polylines_visible_default
        )
        _legend_cloud = (
            "Diffusion samples: white triangles until series selected; colour when toggled on"
            if _idle_white
            else "Diffusion samples (triangles; ★ marks chosen draw)"
        )
        _legend_cloud_size = (
            float(prediction_cloud_idle_marker_size)
            if _idle_white
            else float(prediction_cloud_marker_size)
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=_legend_cloud_size,
                    color=("lightgray" if _idle_white else "gray"),
                    opacity=min(1.0, float(prediction_cloud_marker_opacity) + 0.1),
                    symbol="triangle-up",
                    line=dict(width=0, color="rgba(255,255,255,0)"),
                ),
                name=_legend_cloud,
                hoverinfo="skip",
            )
        )

    # Per-trace visibility for series polylines + their pred overlays. When
    # `polylines_visible_default=False` we hand the user a clean field-only
    # canvas; individual series light up by clicking their entry in the legend.
    series_visible = True if polylines_visible_default else "legendonly"
    _cloud_idle_white = (
        bool(prediction_cloud_idle_white)
        and not bool(polylines_visible_default)
        and has_any_cloud
    )

    tri_marker_white = dict(
        size=float(prediction_cloud_idle_marker_size),
        symbol="triangle-up",
        color="white",
        opacity=0.72,
        line=dict(width=0, color="rgba(255,255,255,0)"),
    )
    cloud_hover_idle = (
        "<b>%{fullData.name}</b><br>"
        "sample %{customdata[0]}<br>"
        "model ref step %{customdata[1]} → GT step %{customdata[2]} "
        "/ %{customdata[3]}<br>"
        "‖sample − GT‖₂=%{customdata[4]:.3f}<br>"
        "<i>(toggle series in legend for colour)</i><extra></extra>"
    )
    cloud_hover_colour = (
        "<b>%{fullData.name}</b><br>"
        "sample %{customdata[0]}<br>"
        "model ref step %{customdata[1]} → GT step %{customdata[2]} "
        "/ %{customdata[3]}<br>"
        "‖sample − GT‖₂=%{customdata[4]:.3f}<extra></extra>"
    )

    cloud_bundles: list[dict | None] = []
    for i, p in enumerate(projs):
        T = p.shape[0]
        tr_raw = clean_trajs[i]
        cloud_bundles.append(
            _prepare_diffusion_cloud_proj_bundle(
                cloud_arr=clean_clouds[i],
                projector=projector,
                pred_for_steps=clean_preds[i],
                tr_raw=tr_raw,
                T=T,
                exc_ix=clean_cloud_exclude_ix[i],
            )
        )

    # Z-order pass 1: idle white diffusion clouds (always visible hints) sit above
    # the Spike background but below every toggled-on trajectory / prediction.
    if _cloud_idle_white:
        for i, bundle in enumerate(cloud_bundles):
            if bundle is None:
                continue
            proj_plot = bundle["proj_plot"]
            cd_plot = bundle["cd_plot"]
            fig.add_trace(
                go.Scatter(
                    x=proj_plot[:, 0],
                    y=proj_plot[:, 1],
                    mode="markers",
                    name=f"{clean_labels[i]} (diffusion samples, idle)",
                    marker=tri_marker_white,
                    legendgroup=f"g{i}",
                    showlegend=False,
                    customdata=cd_plot,
                    hovertemplate=cloud_hover_idle,
                    visible=True,
                )
            )

    # Z-order pass 2: coloured clouds, polylines, and predictions (legend toggled).
    for i, p in enumerate(projs):
        T = p.shape[0]
        tr_raw = clean_trajs[i]
        color = palette[i % len(palette)]
        dists = np.zeros(T, dtype=np.float64)
        for t in range(1, T):
            dists[t] = float(np.linalg.norm(tr_raw[t] - tr_raw[t - 1]))
        gids_i = clean_gids[i]
        lins_i = clean_lins[i]
        frac = np.linspace(0.0, 1.0, T)
        msize = marker_size_start + frac * (marker_size_end - marker_size_start)
        cd = np.empty((T, 5), dtype=object)
        for t in range(T):
            cd[t, 0] = int(t + 1)
            cd[t, 1] = int(T)
            cd[t, 2] = str(gids_i[t]) if gids_i is not None else "n/a"
            cd[t, 3] = str(lins_i[t]) if lins_i is not None else "n/a"
            cd[t, 4] = float(dists[t])

        tri_marker_colour = dict(
            size=float(prediction_cloud_marker_size),
            symbol="triangle-up",
            color=color,
            opacity=float(prediction_cloud_marker_opacity),
            line=dict(width=0.5, color="white"),
        )
        bundle = cloud_bundles[i]
        if bundle is not None:
            proj_plot = bundle["proj_plot"]
            cd_plot = bundle["cd_plot"]
            fig.add_trace(
                go.Scatter(
                    x=proj_plot[:, 0],
                    y=proj_plot[:, 1],
                    mode="markers",
                    name=f"{clean_labels[i]} (diffusion samples)",
                    marker=tri_marker_colour,
                    legendgroup=f"g{i}",
                    showlegend=False,
                    customdata=cd_plot,
                    hovertemplate=cloud_hover_colour,
                    visible=series_visible,
                )
            )

        if _stm == "lines+markers":
            fig.add_trace(
                go.Scatter(
                    x=p[:, 0],
                    y=p[:, 1],
                    mode="lines+markers",
                    name=clean_labels[i],
                    line=dict(color=color, width=polyline_line_width_eff),
                    marker=dict(
                        size=msize,
                        color=color,
                        line=dict(width=polyline_marker_outline_w, color="white"),
                    ),
                    opacity=polyline_alpha_eff,
                    customdata=cd,
                    hovertemplate=hover_fg,
                    legendgroup=f"g{i}",
                    visible=series_visible,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=p[:, 0],
                    y=p[:, 1],
                    mode="lines",
                    name=clean_labels[i],
                    line=dict(color=color, width=polyline_line_width_eff),
                    opacity=polyline_alpha_eff,
                    customdata=cd,
                    hovertemplate=hover_fg,
                    legendgroup=f"g{i}",
                    visible=series_visible,
                )
            )

        pred_info = clean_preds[i]
        if pred_info is not None:
            src_step = pred_info["src_step"]
            tgt_step = pred_info["tgt_step"]
            if src_step < 0 or src_step >= T:
                _logger.warning(
                    "Series %d: pred src_step=%d out of range [0,%d), skipping prediction",
                    i, src_step, T,
                )
            else:
                src_xy = p[src_step]
                pred_xy = pred_info["proj"]
                tr_raw_pred = float(
                    np.linalg.norm(pred_info["vec"] - tr_raw[tgt_step])
                    if 0 <= tgt_step < T else np.nan
                )
                fig.add_trace(
                    go.Scatter(
                        x=[src_xy[0], pred_xy[0]],
                        y=[src_xy[1], pred_xy[1]],
                        mode="lines",
                        line=dict(color=color, width=line_width, dash="dash"),
                        name=f"{clean_labels[i]} (pred)",
                        legendgroup=f"g{i}",
                        showlegend=False,
                        hoverinfo="skip",
                        visible=series_visible,
                    )
                )
                star_note = pred_info.get("pred_star_note")
                if star_note is None:
                    bsi = pred_info.get("best_sample_idx")
                    star_note = (
                        ""
                        if bsi is None
                        else (
                            f"best diffusion draw (sample #{int(bsi)}, "
                            f"lowest MSE vs GT)<br>"
                        )
                    )
                elif star_note and not str(star_note).endswith("<br>"):
                    star_note = f"{star_note}<br>"
                fig.add_trace(
                    go.Scatter(
                        x=[pred_xy[0]],
                        y=[pred_xy[1]],
                        mode="markers",
                        name=f"{clean_labels[i]} (pred)",
                        marker=dict(
                            size=marker_size_end + 2,
                            color=color,
                            symbol="star",
                            line=dict(width=1.2, color="black"),
                        ),
                        legendgroup=f"g{i}",
                        showlegend=False,
                        customdata=np.array(
                            [[
                                int(src_step + 1),
                                int(tgt_step + 1),
                                int(T),
                                pred_info["gid"],
                                tr_raw_pred,
                                star_note,
                            ]],
                            dtype=object,
                        ),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "%{customdata[5]}"
                            "predicted from step %{customdata[0]} → target step %{customdata[1]} / %{customdata[2]}<br>"
                            "pred genome_id=%{customdata[3]}<br>"
                            "‖pred − GT‖₂=%{customdata[4]:.3f}<extra></extra>"
                        ),
                        visible=series_visible,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[src_xy[0]],
                        y=[src_xy[1]],
                        mode="markers",
                        name=f"{clean_labels[i]} (src)",
                        marker=dict(
                            size=marker_size_end + 1,
                            color=color,
                            symbol="circle-open",
                            line=dict(width=2.0, color=color),
                        ),
                        legendgroup=f"g{i}",
                        showlegend=False,
                        hovertemplate=(
                            "<b>source step</b><br>step %{x:.3f}, %{y:.3f}<extra></extra>"
                        ),
                        visible=series_visible,
                    )
                )
                # GT target step: dedicated full-opacity marker (polyline
                # endpoints inherit series opacity and are easy to miss).
                if 0 <= tgt_step < T:
                    gt_xy = p[tgt_step]
                    fig.add_trace(
                        go.Scatter(
                            x=[src_xy[0], gt_xy[0]],
                            y=[src_xy[1], gt_xy[1]],
                            mode="lines",
                            line=dict(color=color, width=line_width + 0.5, dash="solid"),
                            opacity=1.0,
                            name=f"{clean_labels[i]} (gt)",
                            legendgroup=f"g{i}",
                            showlegend=False,
                            hoverinfo="skip",
                            visible=series_visible,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[gt_xy[0]],
                            y=[gt_xy[1]],
                            mode="markers",
                            name=f"{clean_labels[i]} (gt)",
                            marker=dict(
                                size=marker_size_end + 3,
                                color=color,
                                symbol="circle",
                                opacity=1.0,
                                line=dict(width=2.2, color="black"),
                            ),
                            opacity=1.0,
                            legendgroup=f"g{i}",
                            showlegend=False,
                            customdata=np.array(
                                [[
                                    int(tgt_step + 1),
                                    int(T),
                                    str(
                                        clean_gids[i][tgt_step]
                                        if clean_gids[i] is not None
                                        else "n/a"
                                    ),
                                ]],
                                dtype=object,
                            ),
                            hovertemplate=(
                                "<b>%{fullData.name}</b><br>"
                                "ground-truth step %{customdata[0]} / %{customdata[1]}<br>"
                                "genome_id=%{customdata[2]}<extra></extra>"
                            ),
                            visible=series_visible,
                        )
                    )

    if polyline_arrowheads and (polylines_visible_default or force_polyline_arrowheads):
        # Usually skip arrowheads when polylines are legend-only (orphans). The
        # `force_polyline_arrowheads` path is intentional: e.g. arrowheads-only
        # on first load while full polylines stay in `legendonly`.
        traj_colors = [palette[i % len(palette)] for i in range(len(projs))]
        _add_polyline_arrowheads(fig, projs, color_per_line=traj_colors)
    elif (
        polyline_arrowheads
        and not polylines_visible_default
        and not force_polyline_arrowheads
    ):
        _logger.info(
            "polyline_arrowheads disabled because polylines start in legend-only mode "
            "(use force_polyline_arrowheads to draw them anyway)"
        )

    if (
        show_lineage_labels
        and bg_lineages is not None
        and len(bg_lineages) == len(proj_bg)
    ):
        cent = _lineage_label_positions(
            proj_bg,
            bg_lineages,
            min_cluster_size,
            label_min_sep_frac=label_min_sep_frac,
            smooth_sigma_bins=label_smooth_sigma_bins,
        )
        if len(cent) > 0:
            for row in cent.itertuples(index=False):
                # Plain text directly on the plot — no background box, so the
                # underlying density / velocity field stays readable. We keep a
                # subtle dark-grey colour and slight bold so the labels remain
                # legible against the field arrows without occluding them.
                fig.add_annotation(
                    x=float(row.U1),
                    y=float(row.U2),
                    text=f"<b>{row.lineage}</b><br>n={int(row.size)}",
                    showarrow=False,
                    font=dict(size=11, color="rgb(40,40,40)"),
                    align="center",
                    xanchor="center",
                    yanchor="middle",
                )

    n_legend_entries = len(projs) + 1
    if has_any_pred:
        n_legend_entries += 1
    if has_any_cloud:
        n_legend_entries += 1
    fig.update_layout(
        **_umap_series_figure_layout(
            title=title,
            fig_width=fig_width,
            fig_height=fig_height,
            n_legend_entries=n_legend_entries,
        )
    )
    return fig

