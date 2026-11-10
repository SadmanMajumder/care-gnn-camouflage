# Replicating CARE-GNN for Camouflage-Resistant Fraud Detection

COMP8240 Applications of Data Science — major project.

This repository replicates **CARE-GNN** (Dou et al., *Enhancing Graph Neural
Network-based Fraud Detectors against Camouflaged Fraudsters*, CIKM 2020) on its
original YelpChi data, and extends it to new data: a **camouflaged YelpChi**
dataset constructed to stress-test the paper's central claim that the model
resists fraudster camouflage.

Original paper: https://arxiv.org/abs/2008.08692
Original code:  https://github.com/YingtongDou/CARE-GNN

## What's here

| File | Purpose |
|------|---------|
| `camouflage.py` | Builds a camouflaged version of YelpChi (feature + relation camouflage). |
| `llm_judge.py`  | LLM-as-judge evaluation: a small local LLM independently judges nodes from the same features CARE-GNN uses. |
| `README.md`     | This file. |

## Setup

These scripts run on top of the original CARE-GNN repo.

```bash
# 1. Clone the original CARE-GNN code and unzip its data
git clone https://github.com/YingtongDou/CARE-GNN.git
cd CARE-GNN
pip install -r requirements.txt
unzip -o data/YelpChi.zip -d data/

# 2. Copy the scripts from this repo into the CARE-GNN folder
#    (camouflage.py and llm_judge.py)
```

## 1. Replicating the original work

```bash
python data_process.py      # build adjacency lists
python train.py             # train CARE-GNN on original YelpChi
```

Environment: Google Colab, Python 3.x, PyTorch (CPU). The model trains in a few
minutes and reaches a test AUC of roughly **0.77** on YelpChi, in line with the
original paper.

## 2. Constructing the camouflaged dataset

```bash
python camouflage.py --in data/YelpChi.mat --out data/YelpChi_camo.mat
```

This injects:
* **feature camouflage** — 50% of fraud nodes blended 70% toward the genuine
  feature average;
* **relation camouflage** — 5 extra fraud→genuine edges per fraud node.

To evaluate CARE-GNN on it, regenerate the adjacency lists from
`YelpChi_camo.mat`, swap it in as `YelpChi.mat`, and re-run `train.py`.
(See the project report for the exact steps.)

## 3. LLM-as-judge evaluation

```bash
pip install transformers torch
python llm_judge.py --data data/YelpChi.mat
```

Uses `Qwen/Qwen2.5-0.5B-Instruct`, which runs locally in Colab with no API key.
Prints the LLM's accuracy and fraud recall against the ground-truth labels.

## Results summary

| Setting | Final test AUC | Final AP |
|---------|---------------|----------|
| Original YelpChi | 0.77 | 0.38 |
| Camouflaged YelpChi | 0.75 | 0.27 |

Injected camouflage measurably degraded detection — most clearly a ~31% drop in
Average Precision — supporting the paper's thesis that camouflage makes fraud
harder to detect. Full analysis is in the project report.
