import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout

from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGE(torch.nn.Module):

    def __init__(
        self,
        input_dim=140,
        hidden_dim=128,
        num_classes=10,
        dropout=0.3
    ):
        super().__init__()

        
        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        
        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        
        self.dropout = Dropout(dropout)

        
        self.classifier = Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x, edge_index, batch):

        
        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)
        x = self.dropout(x)

    
        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        
        x = self.classifier(x)

        return x