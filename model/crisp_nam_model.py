import torch
import torch.nn as nn
import numpy as np


class FeatureNet(nn.Module):
    """
    Neural network to model the effect of a single feature on hazard.
    This is the building block for NAM with optional batch normalization.
    """
    def __init__(self, hidden_sizes=[64, 64], dropout_rate=0.1, feature_dropout=0.0, 
                 batch_norm=False):
        super(FeatureNet, self).__init__()
        self.batch_norm = batch_norm
        layers = []
        
        # Input layer
        layers.append(nn.Linear(1, hidden_sizes[0]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
            
        # Final representation layer
        layers.append(nn.Linear(hidden_sizes[-1], hidden_sizes[-1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[-1]))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.feature_dropout = feature_dropout
        
    def forward(self, x):
        x = x.to(dtype=torch.float32)
        # Apply feature dropout during training if specified
        if self.training and self.feature_dropout > 0:
            mask = torch.rand_like(x) > self.feature_dropout
            x = x * mask.float()
        
        # Handle BatchNorm with single sample
        if self.batch_norm and x.size(0) == 1:
            return self._forward_singleton(x)
            
        return self.network(x)
    
    def _forward_singleton(self, x):
        """
        Handle the case of a single sample (batch_size=1)
        where BatchNorm1d would fail
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            result = self.network(x)
        if was_training:
            self.train()
        return result


class CrispNamModel(nn.Module):
    """
    Competing risks CoxNAM that models multiple event types.
    Each feature contributes to each risk through a separate shape function.
    """
    def __init__(self, num_features, num_competing_risks, hidden_sizes=[64, 64],
                 dropout_rate=0.1, feature_dropout=0.1, batch_norm=False):
        super(CrispNamModel, self).__init__()
        self.num_features = num_features
        self.num_competing_risks = num_competing_risks
        self.batch_norm = batch_norm
        self.feature_dropout = feature_dropout
        
        # Create a FeatureNet for each input feature
        self.feature_nets = nn.ModuleList([
            FeatureNet(hidden_sizes, dropout_rate, feature_dropout, batch_norm) 
            for _ in range(num_features)
        ])
        
        # For each feature and risk type, create a projection layer
        self.risk_projections = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_sizes[-1], 1, bias=False) 
                for _ in range(num_competing_risks)
            ])
            for _ in range(num_features)
        ])

    def forward(self, x):
        """
        Forward pass to compute risk scores for all competing risks
        
        Args:
            x: Tensor of shape (batch_size, num_features)
        Returns:
            risk_scores: List of (batch_size, 1) Tensors
            feature_outputs: List of (batch_size, hidden) Tensors
        """
        # ensure float32
        x = x.to(dtype=torch.float32)
        batch_size, _ = x.shape
        device = x.device

        # one-shot feature dropout
        if self.training and self.feature_dropout > 0:
            # bernoulli_ is in-place, fast
            mask = torch.empty_like(x).bernoulli_(1.0 - self.feature_dropout)
            x = x * mask

        # pre-allocate combined scores [batch, num_risks]
        combined = torch.zeros(batch_size, self.num_competing_risks, device=device)
        feature_outputs = []

        # loop features (unchanged architecture)
        for feat_idx, fnet in enumerate(self.feature_nets):
            # take one column and get repr
            col = x[:, feat_idx].unsqueeze(1)            # [batch,1]
            repr = fnet(col)                              # [batch, hidden]
            feature_outputs.append(repr)

            # project into each risk channel
            for risk_idx, proj in enumerate(self.risk_projections[feat_idx]):
                # proj(repr) -> [batch,1], .view(-1) -> [batch]
                combined[:, risk_idx] += proj(repr).view(-1)

        # split back into list of [batch,1]
        risk_scores = [
            combined[:, r].unsqueeze(1) 
            for r in range(self.num_competing_risks)
        ]

        return risk_scores, feature_outputs
        
    
        
    def get_shape_functions(self, x_values, feature_idx, risk_idx=None, normalize=True):
        """
        Extract shape functions for a specific feature across all risks or a specific risk
        
        Args:
            x_values: Feature values to evaluate (numpy array or tensor)
            feature_idx: Index of the feature to get shape functions for
            risk_idx: Optional; if provided, only returns the shape function for this risk
            normalize: Whether to center the shape functions
            
        Returns:
            Dictionary mapping risk names to shape function values
        """
        self.eval()  
        
        
        if not isinstance(x_values, torch.Tensor):
            x_values = torch.FloatTensor(x_values)
        
        x_vals = x_values.view(-1, 1)
        
        with torch.no_grad():
            
            feature_repr = self.feature_nets[feature_idx](x_vals)
            
            shape_funcs = {}
            
            # If risk_idx is specified, only compute for that risk
            risk_indices = [risk_idx] if risk_idx is not None else range(self.num_competing_risks)
            
            for j in risk_indices:
                # Apply the projection to get shape function values
                values = self.risk_projections[feature_idx][j](feature_repr).cpu().numpy().flatten()
                
                # Normalize if requested
                if normalize:
                    values = values - np.mean(values)
                
                shape_funcs[f'risk_{j+1}'] = values
                
        return shape_funcs
    
    def calculate_feature_importance(self, x_data, feature_idx=None):
        """
        Calculate feature importance based on the magnitude of risk-specific projection weights
        and shape function outputs
        
        Args:
            x_data: Input data tensor or numpy array
            feature_idx: Optional; if provided, only calculate importance for this feature
            
        Returns:
            Dictionary of feature importances by risk type
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Convert to tensor if needed
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.FloatTensor(x_data)
        x_data = x_data.to(device)
            
        feature_indices = [feature_idx] if feature_idx is not None else range(self.num_features)
        importance = {f'risk_{j+1}': {} for j in range(self.num_competing_risks)}
        
        for i in feature_indices:
            # Get feature values
            feature_values = x_data[:, i].view(-1, 1)
            
            with torch.no_grad():
                # Get the feature representation
                feature_repr = self.feature_nets[i](feature_values)
                
                # Calculate importance for each risk (mean absolute value)
            for j in range(self.num_competing_risks):
                # Apply the risk-specific projection
                risk_specific_output = self.risk_projections[i][j](feature_values)
                abs_values = torch.abs(risk_specific_output).cpu().numpy()
                importance[f'risk_{j+1}'][f'feature_{i}'] = float(np.mean(abs_values))
                                    
        return importance
    


    
    def predict_risk(self, x, baseline_hazards=None):
        """
        Predict survival probability or cumulative incidence
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            baseline_hazards: Optional dict of baseline hazards for each risk
            
        Returns:
            Dictionary of predictions for each competing risk
        """
        self.eval()
        
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            
        with torch.no_grad():
            risk_scores, _ = self(x)
            
            # Convert scores to hazard ratios
            hazard_ratios = [torch.exp(score).cpu().numpy() for score in risk_scores]
            
            # If baseline hazards are provided, compute absolute risks
            if baseline_hazards is not None:
                predictions = {}
                
                for j in range(self.num_competing_risks):
                    risk_name = f'risk_{j+1}'
                    
                    # Baseline survival and hazard
                    baseline_surv = baseline_hazards.get(risk_name, {}).get('survival', None)
                    baseline_haz = baseline_hazards.get(risk_name, {}).get('hazard', None)
                    
                    if baseline_surv is not None:
                        # Compute survival probability: S(t|x) = S0(t)^exp(f(x))
                        predictions[f'{risk_name}_survival'] = np.power(
                            baseline_surv.reshape(1, -1), 
                            hazard_ratios[j].reshape(-1, 1)
                        )
                    
                    if baseline_haz is not None:
                        # Compute cumulative hazard: H(t|x) = H0(t) * exp(f(x))
                        predictions[f'{risk_name}_cumhazard'] = baseline_haz.reshape(1, -1) * hazard_ratios[j].reshape(-1, 1)
                        
                return predictions
            else:
                # Without baseline hazards, just return hazard ratios
                return {f'risk_{j+1}_hazard_ratio': hazard_ratios[j] for j in range(self.num_competing_risks)}