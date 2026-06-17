# Fraud Detection with Graph Neural Networks

Detecting fraudulent transactions using GraphSAGE GNN on the IEEE-CIS dataset.

## Key Insight
GNNs can leverage relational information between accounts, devices, and merchants
that is difficult to capture with purely tabular features. Fraudsters form networks —
they reuse the same email domains, devices, and merchants. GNNs capture these
patterns; tabular models cannot.

## Pipeline
Transaction Data → Graph Construction → GraphSAGE → Fraud Prediction

## Graph Construction
Transactions were converted into a graph where:
- Nodes represent customer accounts
- Edges connect accounts that share behavioral signals such as email domains, merchants, or transaction patterns

The resulting graph contains:
- 13,553 nodes
- 2.3 million edges

This graph structure enables GraphSAGE to learn from relationships between
connected accounts rather than treating each transaction independently.

## Results

| Model | Fraud Recall | Fraud Precision | Fraud F1 | AUC-ROC |
|---|---|---|---|---|
| GraphSAGE GNN | 0.58 | 0.44 | 0.50 | 0.84 |
| XGBoost | 0.58 | 0.28 | 0.38 | 0.89 |
| Random Forest | 0.47 | 0.91 | 0.62 | 0.94 |

While Random Forest achieved the highest overall AUC, GraphSAGE significantly
improved fraud precision relative to XGBoost, reducing false positive investigations.

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

