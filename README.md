# Fraud Detection with Graph Neural Networks

> Detecting fraudulent transactions by modeling account relationships as a graph — built with GraphSAGE on the IEEE-CIS dataset (590K real transactions).

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch Geometric](https://img.shields.io/badge/built%20with-PyTorch%20Geometric-ee4c2c)
![Notebooks](https://img.shields.io/badge/format-Jupyter%20Notebook-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

## Table of Contents

- [Highlights](#-highlights)
- [Key Insight](#-key-insight)
- [How It Works](#-how-it-works)
- [Graph Construction](#graph-construction)
- [Class Imbalance Handling](#-class-imbalance-handling)
- [Results](#-results)
- [Fraud Subgraph Visualization](#-fraud-subgraph-visualization)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)

---

## Highlights

- Built a 13,553-node, 2.3M-edge transaction graph from 590K raw IEEE-CIS fraud records
- Engineered graph edges from behavioral signals — shared email domains, merchants, and transaction patterns
- Trained GraphSAGE for node-level fraud classification, benchmarked against XGBoost and Random Forest
- Achieved 0.89 ROC-AUC with 0.70 fraud recall — catching ~49% more fraud than Random Forest at matched AUC
  
---

## Key Insight

Fraudsters don't operate in isolation — they reuse the same email domains, billing addresses, and merchants across multiple stolen cards.

| | |
|---|---|
| **Tabular models** (XGBoost, Random Forest) | see each transaction in isolation |
| **Graph Neural Networks** | see the entire network, and catch fraud rings that flat models miss |

By representing accounts, devices, and merchants as a connected graph instead of independent rows, a GNN can pick up on relational fraud signals that simply don't exist in a tabular feature vector.

---

## How It Works

```mermaid
flowchart LR
    A["Raw Transactions
    590K rows · 434 features"] --> B["Cleaning & EDA"]
    B --> C["Graph Construction
    13,553 nodes · 2.3M edges"]
    C --> D["GraphSAGE
    node embeddings"]
    D --> E["Fraud Prediction"]
    C -. tabular features .-> F["XGBoost /
    Random Forest"]
    F --> E
```
---
## Graph Construction

The original IEEE-CIS dataset is purely tabular, with each row representing an individual transaction. To enable graph learning, transactions were transformed into an account-level graph.

- **Nodes** represent customer accounts (`card1`)
- **Edges** connect accounts sharing behavioral signals such as email domains, billing regions, merchants, or suspicious transaction patterns
- **Node features** include 10 engineered transaction and identity attributes aggregated at the account level

This converts 590K independent transaction rows into a relational graph with **13,553 nodes** and **2.3M edges**.

The motivation is that fraud often appears as coordinated behavior rather than isolated anomalies. Fraudsters frequently reuse devices, merchants, email domains, and billing information across multiple stolen cards, forming localized fraud rings that GNNs can detect through neighborhood structure.

---

## Class Imbalance Handling
The dataset is highly imbalanced, with fraud representing **3.5% of transactions**
and **12.8% of graph-level account nodes** after graph construction.
To improve minority-class detection, GraphSAGE was trained using weighted
CrossEntropyLoss with a fraud class weight of **4.75×** relative to the
legitimate class. This was tuned down from the inverse-frequency ratio
(**6.79×**) to better balance fraud precision and recall.
Evaluation emphasized fraud recall, F1-score, and ROC-AUC rather than raw
accuracy, since accuracy can be misleading under heavy class imbalance. 

---

## Results

| Model | Fraud Recall | Fraud Precision | Fraud F1 | AUC-ROC |
|---|---|---|---|---|
| **GraphSAGE GNN** | **0.70** | 0.54 | **0.61** | 0.89 |
| XGBoost | 0.58 | 0.28 | 0.38 | 0.89 |
| Random Forest | 0.47 | 0.91 | 0.62 | 0.94 |

**Takeaway:** the GNN catches roughly 49% more fraudulent transactions than Random Forest by modeling account–merchant relationships, while nearly doubling fraud precision relative to XGBoost at matched ROC-AUC, meaning far fewer false-positive investigations for the same detection power.

Random Forest still wins on raw AUC and precision, but at the cost of missing nearly half of all fraud cases (recall of 0.47) — a tradeoff that matters more in some fraud-ops contexts than others.

<img width="594" height="293" alt="image" src="https://github.com/user-attachments/assets/e84695bc-ee95-455b-bd46-23628b0f0b64" />

---

## Fraud Subgraph Visualization

<img width="1408" height="1208" alt="subgraph" src="https://github.com/user-attachments/assets/b289a490-10f0-4d15-a556-5cc561096d53" />

*Red nodes = fraudulent accounts, blue nodes = legitimate accounts. Fraud nodes visibly cluster together — this is the structural signal GraphSAGE learns to exploit.*

---

## Project Structure

| Notebook | Description |
|---|---|
| [`01_data_loading.ipynb`](01_data_loading.ipynb) | Load and clean 590K IEEE-CIS transactions, exploratory data analysis |
| [`02_graph_construction.ipynb`](02_graph_construction.ipynb) | Build the account graph (13,553 nodes, 2.3M edges) |
| [`03_gnn_model.ipynb`](03_gnn_model.ipynb) | Train GraphSAGE for node-level fraud classification |
| [`04_baselines.ipynb`](04_baselines.ipynb) | XGBoost and Random Forest baseline comparison |

---

## Dataset

**[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)** — Kaggle, provided by Vesta Corporation

- 590,540 real online transactions
- 3.5% fraud rate (highly imbalanced)
- 434 raw features, reduced to 10 engineered graph-relevant features

---

## Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/sruthi-kurra/fraud-detection-gnn.git
   ```
2. **Get the data** — download `ieee-fraud-detection.zip` from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) (requires a free Kaggle account)
3. **Install dependencies**
```bash
   pip install -r requirements.txt
```
4. **Open the notebooks in order**, either locally in Jupyter or in [Google Colab](https://colab.research.google.com/):
   - `01_data_loading.ipynb` → `02_graph_construction.ipynb` → `03_gnn_model.ipynb` → `04_baselines.ipynb`
5. **Upload the dataset** when prompted in the first notebook
6. **Run all cells top to bottom**

> The graph construction and GNN training steps are the most memory-intensive — a GPU runtime (e.g. Colab's free tier) is recommended for `03_gnn_model.ipynb`.

---

## Tech Stack

`Python` · `PyTorch Geometric` · `XGBoost` · `scikit-learn` · `NetworkX` · `pandas` · `matplotlib`

---

## Future Improvements

- [ ] Tune the classification threshold per-model to directly compare precision/recall at matched operating points
- [ ] Package the trained GraphSAGE model behind a simple inference script for scoring new transactions
