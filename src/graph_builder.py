"""
graph_builder.py
Builds a PyTorch Geometric graph from cleaned transaction data.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder


def build_node_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data to account-level node features."""
    le = LabelEncoder()
    df['card4_enc']         = le.fit_transform(df['card4'])
    df['ProductCD_enc']     = le.fit_transform(df['ProductCD'])
    df['P_emaildomain_enc'] = le.fit_transform(df['P_emaildomain'])

    node_features = df.groupby('card1').agg(
        total_transactions = ('TransactionID', 'count'),
        avg_time           = ('TransactionDT', 'mean'),
        total_amount       = ('TransactionAmt', 'sum'),
        avg_amount         = ('TransactionAmt', 'mean'),
        max_amount         = ('TransactionAmt', 'max'),
        std_amount         = ('TransactionAmt', 'std'),
        unique_cards       = ('card4_enc', 'nunique'),
        unique_products    = ('ProductCD_enc', 'nunique'),
        unique_emails      = ('P_emaildomain_enc', 'nunique'),
        fraud_transactions = ('isFraud', 'sum'),
        fraud_label        = ('isFraud', 'max')
    ).reset_index()

    node_features['std_amount']         = node_features['std_amount'].fillna(0)
    node_features['avg_time']           = node_features['avg_time'].fillna(0)
    node_features['fraud_transactions'] = node_features['fraud_transactions'].fillna(0)

    return node_features


def build_edges(df: pd.DataFrame, card1_to_idx: dict) -> tuple:
    """Connect accounts sharing email domain or billing address."""
    src_nodes, dst_nodes = [], []

    for groupby_col in ['P_emaildomain', 'addr1']:
        edges = df.groupby(groupby_col)['card1'].apply(list).reset_index()
        for _, row in edges.iterrows():
            accounts = row['card1']
            if len(accounts) > 1:
                for i in range(len(accounts)):
                    for j in range(i+1, min(i+5, len(accounts))):
                        src = card1_to_idx.get(accounts[i])
                        dst = card1_to_idx.get(accounts[j])
                        if src is not None and dst is not None:
                            src_nodes.append(src)
                            dst_nodes.append(dst)

    return src_nodes, dst_nodes


def build_graph(df: pd.DataFrame) -> Data:
    """Full pipeline: dataframe → PyG graph object."""
    node_features = build_node_features(df)
    card1_to_idx  = {c: i for i, c in enumerate(node_features['card1'])}
    src_nodes, dst_nodes = build_edges(df, card1_to_idx)

    feature_cols = [
        'total_transactions', 'avg_time', 'total_amount',
        'avg_amount', 'max_amount', 'std_amount',
        'unique_cards', 'unique_products', 'unique_emails',
        'fraud_transactions'
    ]

    x          = torch.tensor(node_features[feature_cols].values, dtype=torch.float)
    y          = torch.tensor(node_features['fraud_label'].values, dtype=torch.long)
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y), node_features, feature_cols
