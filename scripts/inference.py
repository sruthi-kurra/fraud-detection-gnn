"""
inference.py
Score new transactions using saved GraphSAGE model.
Usage: python scripts/inference.py --data path/to/transactions.csv
"""

import torch
import argparse
import pandas as pd
from src.graph_builder import build_graph
from sklearn.preprocessing import StandardScaler

def load_model(checkpoint_path: str, in_channels: int):
    from torch_geometric.nn import SAGEConv
    import torch.nn as nn
    import torch.nn.functional as F

    class FraudGNN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, out_channels)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            return self.conv3(x, edge_index)

    checkpoint = torch.load(checkpoint_path)
    model = FraudGNN(in_channels, 64, 2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def score(data_path: str, checkpoint_path: str = 'models/graphsage_fraud.pt'):
    df  = pd.read_csv(data_path)
    data, node_features, _ = build_graph(df)

    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)

    model = load_model(checkpoint_path, data.x.shape[1])
    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        prob = torch.softmax(out, dim=1)[:, 1]

    node_features['fraud_prediction'] = pred.numpy()
    node_features['fraud_probability'] = prob.numpy()
    print(node_features[['card1', 'fraud_prediction', 'fraud_probability']].head(20))
    return node_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', default='models/graphsage_fraud.pt')
    args = parser.parse_args()
    score(args.data, args.checkpoint)
