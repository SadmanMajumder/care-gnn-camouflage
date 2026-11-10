"""
llm_judge.py
============
LLM-as-judge evaluation for CARE-GNN on YelpChi.

CARE-GNN classifies review nodes using 32 engineered numeric features.
Here we test whether a small, locally-run Large Language Model, given a
plain-language summary of the SAME features, can act as an independent
"judge" of whether a node is fraudulent. We then measure how often the
LLM agrees with the ground-truth labels.

Because the YelpChi .mat file contains only engineered numeric features
(not raw review text), each node is described to the LLM by how its
features compare to the average genuine and average fraudulent node.
This is an honest use of the available data.

The model used (Qwen2.5-0.5B-Instruct) runs locally in Google Colab with
no API key and no cost.

Usage
-----
    pip install transformers torch
    python llm_judge.py --data data/YelpChi.mat --n-each 15
"""

import argparse
import numpy as np
import scipy.io as sio
from transformers import pipeline


def describe_node(f, fraud_mean, genuine_mean):
    """Turn a node's 32 features into a short, LLM-readable description."""
    closer_to_fraud = np.abs(f - fraud_mean) < np.abs(f - genuine_mean)
    pct = 100.0 * closer_to_fraud.mean()
    return (f"A review has {pct:.0f}% of its 32 numeric behaviour features "
            f"closer to the average fraudulent review than to the average "
            f"genuine review. Is this review FRAUD or GENUINE?")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge for CARE-GNN / YelpChi.")
    parser.add_argument("--data", default="data/YelpChi.mat")
    parser.add_argument("--n-each", type=int, default=15,
                        help="number of fraud and genuine nodes each to judge")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # ---- 1. Load features + labels -----------------------------------
    data = sio.loadmat(args.data)
    features = np.asarray(data["features"].todense(), dtype=float)
    labels = data["label"].ravel()
    fraud_idx = np.where(labels == 1)[0]
    genuine_idx = np.where(labels == 0)[0]
    fraud_mean = features[fraud_idx].mean(axis=0)
    genuine_mean = features[genuine_idx].mean(axis=0)

    # ---- 2. Balanced sample ------------------------------------------
    rng = np.random.default_rng(args.seed)
    sample = np.concatenate([
        rng.choice(fraud_idx, args.n_each, replace=False),
        rng.choice(genuine_idx, args.n_each, replace=False),
    ])
    rng.shuffle(sample)

    # ---- 3. Load the local LLM ---------------------------------------
    judge = pipeline("text-generation", model=args.model, device_map="auto")
    system = ("You are a fraud-detection analyst. Reply with exactly one "
              "word: FRAUD or GENUINE.")

    llm_preds, true_labels = [], []
    for i in sample:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": describe_node(features[i], fraud_mean, genuine_mean)}]
        out = judge(messages, max_new_tokens=5, do_sample=False)
        verdict = out[0]["generated_text"][-1]["content"].strip().upper()
        llm_preds.append(1 if "FRAUD" in verdict else 0)
        true_labels.append(int(labels[i]))

    llm_preds = np.array(llm_preds)
    true_labels = np.array(true_labels)

    # ---- 4. Report ---------------------------------------------------
    acc = (llm_preds == true_labels).mean()
    fraud_mask = true_labels == 1
    recall = (llm_preds[fraud_mask] == 1).mean()
    print(f"Nodes judged:              {len(sample)}")
    print(f"LLM accuracy vs truth:     {acc:.2f}")
    print(f"LLM recall on fraud nodes: {recall:.2f}")


if __name__ == "__main__":
    main()
