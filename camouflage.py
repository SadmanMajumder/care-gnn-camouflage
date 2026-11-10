"""
camouflage.py
=============
Construct a camouflaged version of the YelpChi dataset for stress-testing
CARE-GNN (Dou et al., CIKM 2020).

The CARE-GNN paper claims its model resists two kinds of fraudster
"camouflage":
  * feature camouflage  -- a fraud node's features are made to look normal;
  * relation camouflage -- a fraud node is connected to many genuine nodes.

This script injects both kinds of camouflage into YelpChi so we can measure
how much the model's performance degrades, which directly tests that claim.

Usage
-----
    python camouflage.py --in  data/YelpChi.mat \
                         --out data/YelpChi_camo.mat

Input  : the original YelpChi.mat shipped with the CARE-GNN repo.
Output : YelpChi_camo.mat in the identical format (same keys), ready to be
         fed through data_process.py and train.py.
"""

import argparse
import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def add_camouflage_edges(mat, fraud_idx, genuine_idx, edges_per_fraud, rng):
    """Add undirected fraud->genuine edges to one relation graph."""
    mat = mat.tolil()
    for node in fraud_idx:
        neighbours = rng.choice(genuine_idx, size=edges_per_fraud, replace=False)
        for g in neighbours:
            mat[node, g] = 1
            mat[g, node] = 1
    return mat.tocoo()


def main():
    parser = argparse.ArgumentParser(description="Inject camouflage into YelpChi.")
    parser.add_argument("--in", dest="infile", default="data/YelpChi.mat")
    parser.add_argument("--out", dest="outfile", default="data/YelpChi_camo.mat")
    parser.add_argument("--feat-fraction", type=float, default=0.5,
                        help="fraction of fraud nodes to feature-camouflage")
    parser.add_argument("--feat-alpha", type=float, default=0.7,
                        help="how strongly to blend fraud features toward genuine")
    parser.add_argument("--edges-per-fraud", type=int, default=5,
                        help="fraud->genuine edges added per fraud node")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data = sio.loadmat(args.infile)

    features = data["features"].tocsr().astype(float)   # 45954 x 32
    labels = data["label"].ravel()                      # 0 genuine, 1 fraud
    fraud_idx = np.where(labels == 1)[0]
    genuine_idx = np.where(labels == 0)[0]
    print(f"Fraud nodes: {len(fraud_idx)}, Genuine nodes: {len(genuine_idx)}")

    # ---- 1. Feature camouflage ---------------------------------------
    genuine_mean = np.asarray(features[genuine_idx].mean(axis=0)).ravel()
    features_camo = features.tolil()
    n_feat = int(len(fraud_idx) * args.feat_fraction)
    targets = rng.choice(fraud_idx, size=n_feat, replace=False)
    for node in targets:
        original = features_camo[node].toarray().ravel()
        features_camo[node] = (1 - args.feat_alpha) * original + args.feat_alpha * genuine_mean
    features_camo = features_camo.tocoo()
    print(f"Feature camouflage applied to {n_feat} fraud nodes.")

    # ---- 2. Relation camouflage --------------------------------------
    net_rur = add_camouflage_edges(data["net_rur"], fraud_idx, genuine_idx, args.edges_per_fraud, rng)
    net_rtr = add_camouflage_edges(data["net_rtr"], fraud_idx, genuine_idx, args.edges_per_fraud, rng)
    net_rsr = add_camouflage_edges(data["net_rsr"], fraud_idx, genuine_idx, args.edges_per_fraud, rng)
    homo = add_camouflage_edges(data["homo"], fraud_idx, genuine_idx, args.edges_per_fraud, rng)
    print(f"Relation camouflage: +{args.edges_per_fraud} fraud->genuine edges per fraud node.")

    # ---- 3. Save in the identical format -----------------------------
    sio.savemat(args.outfile, {
        "homo": homo,
        "net_rur": net_rur,
        "net_rtr": net_rtr,
        "net_rsr": net_rsr,
        "features": features_camo,
        "label": labels.reshape(1, -1),
    })
    print(f"Saved camouflaged dataset to {args.outfile}")


if __name__ == "__main__":
    main()
