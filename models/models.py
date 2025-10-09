import pytorch_lightning as L
import pyro.distributions as dist
import pyro.distributions.transforms as T
from torch import nn

class MLP(nn.Sequential):
    """MLP model"""

    def __init__(self, n_in, n_out, n_hidden=(16, 16, 16), act=None, dropout=0):
        if act is None:
            act = [
                nn.LeakyReLU(),
            ] * (len(n_hidden) + 1)
        assert len(act) == len(n_hidden) + 1

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_) - 2):
            layer.append(nn.Linear(n_[i], n_[i + 1]))
            layer.append(act[i])
            layer.append(nn.Dropout(p=dropout))
        layer.append(nn.Linear(n_[-2], n_[-1]))
        super(MLP, self).__init__(*layer)

class ResidualBlock(nn.Module):
    def __init__(self, n_units, activation=nn.ReLU, dropout=0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.BatchNorm1d(n_units),  # BatchNorm before activation
            activation(),
            nn.Dropout(dropout),
            nn.Linear(n_units, n_units),
            nn.BatchNorm1d(n_units),  # BatchNorm after the second linear layer
        )

    def forward(self, x):
        return x + self.layer(x)

class ResidualMLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=(64, 64, 64), act=nn.ReLU, dropout=0.1):
        super().__init__()
        
        # Initial input layer
        layers = [
            nn.Linear(n_in, n_hidden[0]),
            nn.BatchNorm1d(n_hidden[0]),
            act(),
        ]
        
        # Process each hidden layer
        for i in range(len(n_hidden)):
            # Add residual block for current layer size
            layers.append(ResidualBlock(n_hidden[i], act, dropout))
            
            # Add dimension-changing layer if not the last hidden layer
            if i < len(n_hidden) - 1:
                layers.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
                layers.append(nn.BatchNorm1d(n_hidden[i+1]))
                layers.append(act())
        
        # Final output layer
        layers.append(nn.Linear(n_hidden[-1], n_out))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SelfAttentionMLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=(256, 128), num_heads=8, dropout=0.1):
        super().__init__()
        self.linear_in = nn.Sequential(
            nn.Linear(n_in, n_hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Reshape input into a sequence of smaller vectors for meaningful attention
        self.seq_length = 16  # Number of vectors in sequence
        self.hidden_per_seq = n_hidden[0] // self.seq_length  # Size of each vector
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_per_seq, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden[0], n_hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden[1], n_out),
        )

    def forward(self, x):
        # Initial dimension reduction
        x = self.linear_in(x)
        
        # Reshape to (seq_length, batch_size, hidden_per_seq)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_length, self.hidden_per_seq)
        x = x.transpose(0, 1)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        
        # Reshape and process through final MLP
        x = attn_out.transpose(0, 1).reshape(batch_size, -1)
        return self.mlp(x)


class ConditionalFlowStack(dist.conditional.ConditionalComposeTransformModule):
    """Normalizing flow stack for conditional distribution"""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dims: int,
        num_flows: int,
        device: str = "cuda",
    ):
        coupling_transforms = [
            T.conditional_spline(
                input_dim,
                context_dim,
                count_bins=8,
                hidden_dims=hidden_dims,
                order="quadratic",
            ).to(device)
            for _ in range(num_flows)
        ]

        super().__init__(coupling_transforms, cache_size=1)
