# Fraud Detection with Graph Neural Networks

Detecting fraudulent transactions using GraphSAGE GNN on the IEEE-CIS dataset.
The key insight: fraudsters form networks — they reuse the same email domains,
devices, and merchants. GNNs capture these patterns; tabular models cannot.

## Results

| Model | Fraud Recall | Fraud Precision | Fraud F1 | AUC-ROC |
|---|---|---|---|---|
| GraphSAGE GNN | 0.58 | 0.44 | 0.50 | 0.84 |
| XGBoost | 0.58 | 0.28 | 0.38 | 0.89 |
| Random Forest | 0.47 | 0.91 | 0.62 | 0.94 |

GNN achieves 57% better fraud precision than XGBoost by modeling
account-merchant relationships as a graph.

## Project Structure

| Notebook | Description |
|---|---|
| 01_data_loading | Load and clean 590k IEEE-CIS transactions |
| 02_graph_construction | Build account graph (13,553 nodes, 2.3M edges) |
| 03_gnn_model | Train GraphSAGE for node-level fraud classification |
| 04_baselines | XGBoost and Random Forest comparison |

## Tech Stack
Python · PyTorch Geometric · XGBoost · scikit-learn · NetworkX · pandas

## Dataset
IEEE-CIS Fraud Detection — Kaggle (Vesta Corporation)
590,540 transactions · 3.5% fraud rate
