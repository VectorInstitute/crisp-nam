"""PyTorch implementation of DeepHit for competing risks survival analysis."""

from typing import Callable, Optional

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class FCLayer(nn.Module):
    """Fully connected layer with optional batch norm, dropout, and activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Optional[torch.Tensor] = None,
        batch_norm: bool =False,
        dropout_rate:float =0.0,
        init_fn: Optional[Callable[[torch.Tensor]]|None] = nn.init.xavier_normal_,
    ) -> None:
        """Initialize the fully connected layer."""
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.activation = activation if activation else nn.ReLU()

        # Initialize weights
        if init_fn:
            init_fn(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        Args:
            x: Tensor of shape (batch_size, in_dim)
        Returns
        -------
            out: Tensor of shape (batch_size, out_dim)
        """

        x = self.fc(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class FCNet(nn.Module):
    """Multi-layer fully connected network."""

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        h_dim: int,
        activation: Optional[nn.module]=None,
        out_dim: Optional[int|None] = None,
        out_activation: Optional[int|None] = None,
        batch_norm: bool = False,
        dropout_rate: float = 0.0,
        init_fn: Optional[Callable[[torch.Tensor]]|None] = nn.init.xavier_normal_,
    ) -> None:
        """Initialize the fully connected network."""
        super(FCNet, self).__init__()

        layers = []
        prev_dim = in_dim
        activation = activation if activation else nn.ReLU()

        # Hidden layers
        for i in range(num_layers):
            curr_dim = out_dim if (i == num_layers - 1 and out_dim) else h_dim
            curr_act = (
                out_activation
                if (i == num_layers - 1 and out_activation)
                else activation
            )

            layers.append(
                FCLayer(
                    prev_dim,
                    curr_dim,
                    activation=curr_act,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    init_fn=init_fn,
                )
            )
            prev_dim = curr_dim

        self.network = nn.Sequential(*layers)

    def forward(self,
        x:torch.Tensor) -> Optional[nn.Module]:
        """Forward pass through the network.

        Args:
            x: Tensor of shape (batch_size, in_dim)

        Returns:
        -------
            out: Tensor of shape (batch_size, out_dim)
        """

        return self.network(x)


class DeepHit(nn.Module):
    """PyTorch implementation of DeepHit for competing risks survival analysis."""

    def __init__(self,
        input_dims: dict,
        network_settings: dict):
        """Initialize the DeepHit model."""
        super(DeepHit, self).__init__()

        # Input dimensions
        self.x_dim = input_dims["x_dim"]
        self.num_Event = input_dims["num_Event"]
        self.num_Category = input_dims["num_Category"]

        # Network settings
        self.h_dim_shared = network_settings["h_dim_shared"]
        self.h_dim_CS = network_settings["h_dim_CS"]
        self.num_layers_shared = network_settings["num_layers_shared"]
        self.num_layers_CS = network_settings["num_layers_CS"]

        # Activation function
        if network_settings["active_fn"] == "relu":
            self.active_fn = nn.ReLU()
        elif network_settings["active_fn"] == "elu":
            self.active_fn = nn.ELU()
        elif network_settings["active_fn"] == "tanh":
            self.active_fn = nn.Tanh()
        else:
            self.active_fn = nn.ReLU()

        # Regularization
        self.keep_prob = network_settings.get("keep_prob", 0.5)
        self.dropout_rate = 1.0 - self.keep_prob

        # Initialize networks
        self._build_network()

    def _build_network(self):
        """Build the shared and cause-specific networks.

        Args:
            None

        Returns:
        -------
            None
        """

        # Shared network
        self.shared_net = FCNet(
            in_dim=self.x_dim,
            num_layers=self.num_layers_shared,
            h_dim=self.h_dim_shared,
            activation=self.active_fn,
            dropout_rate=self.dropout_rate,
        )

        # Cause-specific networks
        self.cs_nets = nn.ModuleList(
            [
                FCNet(
                    in_dim=self.x_dim
                    + self.h_dim_shared,  # Concatenate input and shared output
                    num_layers=self.num_layers_CS,
                    h_dim=self.h_dim_CS,
                    activation=self.active_fn,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(self.num_Event)
            ]
        )

        # Output layer
        self.output_layer = nn.Linear(
            self.num_Event * self.h_dim_CS, self.num_Event * self.num_Category
        )

    def forward(self,
        x:torch.Tensor) -> tuple[torch.Tensor, None]:
        """Forward pass through the network.

        Args:
            x: Tensor of shape (batch_size, num_Event, num_Category)

        Returns
        -------
            risk_scores: List of (batch_size, 1) Tensors
            feature_outputs: None
        """

        # Shared network
        shared_out = self.shared_net(x)

        # Concatenate input with shared output
        h = torch.cat([x, shared_out], dim=1)

        # Cause-specific networks
        cs_outputs = []
        for cs_net in self.cs_nets:
            cs_out = cs_net(h)
            cs_outputs.append(cs_out)

        # Stack outputs
        stacked_out = torch.stack(
            cs_outputs, dim=1
        )  # [batch_size, num_Event, h_dim_CS]
        reshaped_out = stacked_out.view(
            -1, self.num_Event * self.h_dim_CS
        )  # [batch_size, num_Event * h_dim_CS]

        # Final output layer
        logits = self.output_layer(
            F.dropout(reshaped_out, self.dropout_rate, self.training)
        )
        out = F.softmax(logits.view(-1, self.num_Event * self.num_Category), dim=1)

        # Reshape to [batch_size, num_Event, num_Category]
        out = out.view(-1, self.num_Event, self.num_Category)

        # For compatibility with the training script, return both
        # raw risks and shape functions
        # In this model, we don't have separate shape functions, so just return None
        return out, None

    def log_likelihood_loss(self,
        out: torch.Tensor,
        t: Optional[torch.Tensor|np.ndarray],
        k: Optional[torch.Tensor|np.ndarray],
        mask1: torch.Tensor,
        mask2: torch.Tensor):
        """Log-likelihood loss (including log-likelihood of censored subjects).

        Args:
            out: Torch.tensor
            t: Torch.tensor or numpy array
            k: Torch.tensor or numpy array
            mask1: Torch.tensor
            mask2: Torch.tensor

        Returns
        -------
            loss: Torch.tensor
        """

        # Convert to PyTorch tensors if necessary
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=out.device)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=out.device)

        # Indicator for uncensored subjects
        i_1 = (k > 0).float().view(-1, 1)

        # For uncensored: log P(T=t, K=k|x)
        tmp1 = torch.sum(torch.sum(mask1 * out, dim=2), dim=1, keepdim=True)
        tmp1 = i_1 * torch.log(tmp1 + 1e-8)

        # For censored: log ∑ P(T>t|x)
        tmp2 = torch.sum(
            torch.sum(mask2.unsqueeze(1) * out, dim=2), dim=1, keepdim=True
        )
        tmp2 = (1.0 - i_1) * torch.log(tmp2 + 1e-8)

        return -torch.mean(tmp1 + tmp2)

    def ranking_loss(self,
        out: torch.Tensor,
        t: Optional[torch.Tensor|np.ndarray],
        k: Optional[torch.Tensor|np.ndarray],
        mask2: torch.Tensor):
        """Ranking loss (calculated only for acceptable pairs).

        Args:
            out: Torch.tensor
            t: Torch.tensor or numpy array
            k: Torch.tensor or numpy array
            mask2: Torch.tensor

        Returns
        -------
            loss: Torch.tensor
        """

        sigma1 = 0.1
        eta = []

        # Convert to PyTorch tensors if necessary
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=out.device)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=out.device)

        one_vector = torch.ones_like(t)

        for e in range(self.num_Event):
            # Indicator for current event type
            i_2 = (k == e + 1).float()
            i_2_diag = torch.diag(i_2.squeeze())

            # Extract event-specific probabilities
            tmp_e = out[:, e, :]  # [batch_size, num_Category]

            # Calculate risk scores
            r = torch.matmul(tmp_e, mask2.transpose(0, 1))  # [batch_size, batch_size]
            diag_r = torch.diag(r).unsqueeze(1)  # [batch_size, 1]
            r = (
                torch.matmul(one_vector, diag_r.transpose(0, 1)) - r
            )  # [batch_size, batch_size]
            r = r.transpose(0, 1)  # Now R_ij = r_i(T_i) - r_j(T_i)

            # Time comparison matrix
            t = F.relu(
                torch.sign(
                    torch.matmul(one_vector, t.transpose(0, 1))
                    - torch.matmul(t, one_vector.transpose(0, 1))
                )
            )

            # Filter by event occurrence
            t = torch.matmul(i_2_diag, t)

            # Calculate ranking loss for current event
            tmp_eta = torch.mean(t * torch.exp(-r / sigma1), dim=1, keepdim=True)
            eta.append(tmp_eta)

        eta = torch.stack(eta, dim=1)  # [batch_size, num_Event]
        eta = torch.mean(eta.reshape(-1, self.num_Event), dim=1, keepdim=True)

        return torch.sum(eta)

    def calibration_loss(self,
        out: torch.Tensor,
        t: Optional[torch.Tensor|np.ndarray],
        k: Optional[torch.Tensor|np.ndarray],
        mask2: torch.Tensor) -> torch.Tensor:
        """Calibration loss.

        Args:
            out: Torch.tensor
            t: Torch.tensor or numpy array
            k: Torch.tensor or numpy array
            mask2: Torch.tensor

        Returns
        -------
            loss: Torch.tensor
        """

        eta = []

        # Convert to PyTorch tensors if necessary
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=out.device)

        for e in range(self.num_Event):
            # Indicator for current event type
            i_2 = (k == e + 1).float()

            # Extract event-specific probabilities
            tmp_e = out[:, e, :]  # [batch_size, num_Category]

            # Calculate calibration loss
            r = torch.sum(tmp_e * mask2, dim=1)
            tmp_eta = torch.mean((r - i_2) ** 2, dim=0, keepdim=True)
            eta.append(tmp_eta)

        eta = torch.stack(eta, dim=1)  # [1, num_Event]
        eta = torch.mean(eta.reshape(-1, self.num_Event), dim=1, keepdim=True)

        return torch.sum(eta)

    def compute_loss(self,
        out: torch.Tensor,
        t: Optional[torch.Tensor|np.ndarray],
        k: Optional[torch.Tensor|np.ndarray],
        mask1: Optional[torch.Tensor|np.ndarray],
        mask2: torch.Tensor,
        alpha: float = 1.0,
        beta:float = 1.0,
        gamma: float = 1.0):
        """Compute total loss.

        Args:
            out: Torch.tensor
            t: Torch.tensor or numpy array
            k: Torch.tensor or numpy array
            mask1: Torch.tensor
            mask2: Torch.tensor
            alpha: float, weight for log-likelihood loss
            beta: float, weight for ranking loss
            gamma: float, weight for calibration loss

        Returns
        -------
            total_loss: Torch.tensor
        """

        loss1 = self.log_likelihood_loss(out, t, k, mask1, mask2)
        loss2 = self.ranking_loss(out, t, k, mask2)
        loss3 = self.calibration_loss(out, t, k, mask2)

        # L2 regularization is handled by optimizer (weight_decay)
        return alpha * loss1 + beta * loss2 + gamma * loss3

    def predict(self,
        x: torch.Tensor) -> torch.Tensor:
        """Predict risk scores for input x.

        Args:
            x: Tensor of shape (batch_size, num_Event, num_Category)
        Returns
        -------
            out: Tensor of shape (batch_size, num_Event, num_Category)
        """

        self.eval()
        with torch.no_grad():
            out, _ = self.forward(x)
        return out