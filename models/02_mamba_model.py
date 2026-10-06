import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaRULModel(nn.Module):
    """
    Mamba-based regression model for Remaining Useful Life (RUL) prediction.

    Input shape:
        (batch_size, sequence_length, input_features)

    Output shape:
        (batch_size,)
    """

    def __init__(
        self,
        input_size=25,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        # Project input features to the Mamba hidden dimension
        self.input_projection = nn.Linear(input_size, d_model)

        # Stacked Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])

        # Layer normalization for each residual block
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Regression head for RUL prediction
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # Input: (batch, sequence_length, 25)
        x = self.input_projection(x)

        # Residual Mamba blocks
        for mamba, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = mamba(x)
            x = self.dropout(x)
            x = norm(x + residual)

        # Representation of the final operating cycle
        x = x[:, -1, :]

        # Predict one RUL value per sequence
        x = self.regression_head(x)

        return x.squeeze(-1)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MambaRULModel().to(device)

    # Matches the actual dataset:
    # batch=32, sequence_length=30, features=25
    x = torch.randn(
        32, 30, 25,
        device=device
    )

    with torch.no_grad():
        y = model(x)

    print("Device:", device)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )
    print("Mamba RUL model forward pass successful")