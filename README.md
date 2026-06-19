# 🔍 Fraud Detection with Graph Neural Networks

> Detecting fraudulent transactions by modeling account relationships as a graph.
> Built with GraphSAGE on the IEEE-CIS dataset (590k real transactions).

---

## 💡 Key Insight

Fraudsters don't operate in isolation — they reuse the same email domains,
billing addresses, and merchants across multiple stolen cards.

**Tabular models (XGBoost, Random Forest) see each transaction independently.**
**GNNs see the entire network — and catch fraud rings that flat models miss.**

---

## 🏗️ Architecture
---

## 📊 Results

| Model | Fraud Recall | Fraud Precision | Fraud F1 | AUC-ROC |
|---|---|---|---|---|
| **GraphSAGE GNN** | **0.70** | 0.54 | **0.61** | 0.89 |
| XGBoost | 0.58 | 0.28 | 0.38 | 0.89 |
| Random Forest | 0.47 | 0.91 | 0.62 | 0.94 |

**GNN catches 49% more fraudsters than Random Forest** by modeling
account-merchant relationship networks — at the same AUC as XGBoost
but with 93% better fraud precision.

---

## 🕸️ Fraud Subgraph Visualization

![Fraud Subgraph](subgraph.png)

*Red nodes = fraudulent accounts, Blue nodes = legitimate accounts.
Notice how fraud nodes cluster together — this is what the GNN detects.*

---

## 📈 Model Comparison

![Model Comparison](model_comparison.png)

---

## 📁 Project Structure

| Notebook | Description |
|---|---|
| `01_data_loading` | Load and clean 590k IEEE-CIS transactions, EDA |
| `02_graph_construction` | Build account graph (13,553 nodes, 2.3M edges) |
| `03_gnn_model` | Train GraphSAGE for node-level fraud classification |
| `04_baselines` | XGBoost and Random Forest comparison |

---

## 🛠️ Tech Stack

`Python` `PyTorch Geometric` `XGBoost` `scikit-learn` `NetworkX` `pandas` `matplotlib`

---

## 📦 Dataset

**IEEE-CIS Fraud Detection** — Kaggle (Vesta Corporation)
- 590,540 real online transactions
- 3.5% fraud rate
- 434 features (reduced to 10 engineered graph features)

---

## 🚀 How to Run

1. Clone the repo
2. Open notebooks in Google Colab in order (01 → 04)
3. Upload `ieee-fraud-detection.zip` from Kaggle when prompted
4. Run all cells top to bottom

