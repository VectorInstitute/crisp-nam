import torch
import torch.nn as nn
import torch.nn.functional as F

class FCLayer(nn.Module):
    """Fully connected layer with optional batch norm, dropout, and activation."""
    def __init__(self, in_dim, out_dim, activation=nn.ReLU(), batch_norm=False, dropout_rate=0.0, 
                 init_fn=nn.init.xavier_normal_):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.activation = activation
        
        # Initialize weights
        if init_fn:
            init_fn(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
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
    def __init__(self, in_dim, num_layers, h_dim, activation=nn.ReLU(), 
                 out_dim=None, out_activation=None, batch_norm=False, 
                 dropout_rate=0.0, init_fn=nn.init.xavier_normal_):
        super(FCNet, self).__init__()
        
        layers = []
        prev_dim = in_dim
        
        # Hidden layers
        for i in range(num_layers):
            curr_dim = out_dim if (i == num_layers - 1 and out_dim) else h_dim
            curr_act = out_activation if (i == num_layers - 1 and out_activation) else activation
            
            layers.append(FCLayer(
                prev_dim, curr_dim, activation=curr_act, 
                batch_norm=batch_norm, dropout_rate=dropout_rate, init_fn=init_fn
            ))
            prev_dim = curr_dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DeepHit(nn.Module):
    """PyTorch implementation of DeepHit for competing risks survival analysis."""
    
    def __init__(self, input_dims, network_settings):
        super(DeepHit, self).__init__()
        
        # Input dimensions
        self.x_dim = input_dims['x_dim']
        self.num_Event = input_dims['num_Event']
        self.num_Category = input_dims['num_Category']
        
        # Network settings
        self.h_dim_shared = network_settings['h_dim_shared']
        self.h_dim_CS = network_settings['h_dim_CS']
        self.num_layers_shared = network_settings['num_layers_shared']
        self.num_layers_CS = network_settings['num_layers_CS']
        
        # Activation function
        if network_settings['active_fn'] == 'relu':
            self.active_fn = nn.ReLU()
        elif network_settings['active_fn'] == 'elu':
            self.active_fn = nn.ELU()
        elif network_settings['active_fn'] == 'tanh':
            self.active_fn = nn.Tanh()
        else:
            self.active_fn = nn.ReLU()
        
        # Regularization
        self.keep_prob = network_settings.get('keep_prob', 0.5)
        self.dropout_rate = 1.0 - self.keep_prob
        
        # Initialize networks
        self._build_network()
    
    def _build_network(self):
        # Shared network
        self.shared_net = FCNet(
            in_dim=self.x_dim,
            num_layers=self.num_layers_shared,
            h_dim=self.h_dim_shared,
            activation=self.active_fn,
            dropout_rate=self.dropout_rate
        )
        
        # Cause-specific networks
        self.cs_nets = nn.ModuleList([
            FCNet(
                in_dim=self.x_dim + self.h_dim_shared,  # Concatenate input and shared output
                num_layers=self.num_layers_CS,
                h_dim=self.h_dim_CS,
                activation=self.active_fn,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.num_Event)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(self.num_Event * self.h_dim_CS, self.num_Event * self.num_Category)
    
    def forward(self, x):
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
        stacked_out = torch.stack(cs_outputs, dim=1)  # [batch_size, num_Event, h_dim_CS]
        reshaped_out = stacked_out.view(-1, self.num_Event * self.h_dim_CS)  # [batch_size, num_Event * h_dim_CS]
        
        # Final output layer
        logits = self.output_layer(F.dropout(reshaped_out, self.dropout_rate, self.training))
        out = F.softmax(logits.view(-1, self.num_Event * self.num_Category), dim=1)
        
        # Reshape to [batch_size, num_Event, num_Category]
        out = out.view(-1, self.num_Event, self.num_Category)
        
        # For compatibility with the training script, return both raw risks and shape functions
        # In this model, we don't have separate shape functions, so just return None
        return out, None
    
    def log_likelihood_loss(self, out, t, k, mask1, mask2):
        """Log-likelihood loss (including log-likelihood of censored subjects)"""
        batch_size = out.size(0)
        
        # Convert to PyTorch tensors if necessary
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=out.device)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=out.device)
        
        # Indicator for uncensored subjects
        I_1 = (k > 0).float().view(-1, 1)
        
        # For uncensored: log P(T=t, K=k|x)
        tmp1 = torch.sum(torch.sum(mask1 * out, dim=2), dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1 + 1e-8)
        
        # For censored: log ∑ P(T>t|x)
        tmp2 = torch.sum(torch.sum(mask2.unsqueeze(1) * out, dim=2), dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * torch.log(tmp2 + 1e-8)
        
        return -torch.mean(tmp1 + tmp2)
    
    def ranking_loss(self, out, t, k, mask2):
        """Ranking loss (calculated only for acceptable pairs)"""
        batch_size = out.size(0)
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
            I_2 = (k == e+1).float()
            I_2_diag = torch.diag(I_2.squeeze())
            
            # Extract event-specific probabilities
            tmp_e = out[:, e, :]  # [batch_size, num_Category]
            
            # Calculate risk scores
            R = torch.matmul(tmp_e, mask2.transpose(0, 1))  # [batch_size, batch_size]
            diag_R = torch.diag(R).unsqueeze(1)  # [batch_size, 1]
            R = torch.matmul(one_vector, diag_R.transpose(0, 1)) - R  # [batch_size, batch_size]
            R = R.transpose(0, 1)  # Now R_ij = r_i(T_i) - r_j(T_i)
            
            # Time comparison matrix
            T = F.relu(torch.sign(torch.matmul(one_vector, t.transpose(0, 1)) - 
                                 torch.matmul(t, one_vector.transpose(0, 1))))
            
            # Filter by event occurrence
            T = torch.matmul(I_2_diag, T)
            
            # Calculate ranking loss for current event
            tmp_eta = torch.mean(T * torch.exp(-R / sigma1), dim=1, keepdim=True)
            eta.append(tmp_eta)
        
        eta = torch.stack(eta, dim=1)  # [batch_size, num_Event]
        eta = torch.mean(eta.reshape(-1, self.num_Event), dim=1, keepdim=True)
        
        return torch.sum(eta)
    
    def calibration_loss(self, out, t, k, mask2):
        """Calibration loss"""
        batch_size = out.size(0)
        eta = []
        
        # Convert to PyTorch tensors if necessary
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=out.device)
        
        for e in range(self.num_Event):
            # Indicator for current event type
            I_2 = (k == e+1).float()
            
            # Extract event-specific probabilities
            tmp_e = out[:, e, :]  # [batch_size, num_Category]
            
            # Calculate calibration loss
            r = torch.sum(tmp_e * mask2, dim=1)
            tmp_eta = torch.mean((r - I_2) ** 2, dim=0, keepdim=True)
            eta.append(tmp_eta)
            
        eta = torch.stack(eta, dim=1)  # [1, num_Event]
        eta = torch.mean(eta.reshape(-1, self.num_Event), dim=1, keepdim=True)
        
        return torch.sum(eta)
    
    def compute_loss(self, out, t, k, mask1, mask2, alpha=1.0, beta=1.0, gamma=1.0):
        """Compute total loss"""
        loss1 = self.log_likelihood_loss(out, t, k, mask1, mask2)
        loss2 = self.ranking_loss(out, t, k, mask2)
        loss3 = self.calibration_loss(out, t, k, mask2)
        
        # L2 regularization is handled by optimizer (weight_decay)
        return alpha * loss1 + beta * loss2 + gamma * loss3
    
    def predict(self, x):
        """Predict risk scores for input x"""
        self.eval()
        with torch.no_grad():
            out, _ = self.forward(x)
        return out


# Utility functions to create masks for DeepHit
def create_fc_mask1(k, t, num_Event, num_Category, device=None):
    """Create mask1 for loss calculation - for uncensored loss"""
    N = len(k)
    mask = torch.zeros((N, num_Event, num_Category), device=device)
    
    for i in range(N):
        if k[i] > 0:  # Not censored
            event_idx = int(k[i] - 1)
            time_idx = int(t[i])
            if time_idx < num_Category:
                mask[i, event_idx, time_idx] = 1.0
    
    return mask

def create_fc_mask2(t, num_Category, device=None):
    """Create mask2 for loss calculation - for censored loss"""
    N = len(t)
    mask = torch.zeros((N, num_Category), device=device)
    
    for i in range(N):
        time_idx = int(t[i])
        for j in range(time_idx, num_Category):
            mask[i, j] = 1.0
    
    return mask