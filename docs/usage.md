Example code to show how to use two classes in the package.

## Use of `CrispNamModel` model class

```python
import torch

from crisp_nam.models import CrispNamModel

# Example usage and testing
if __name__ == "__main__":

    # Generate some test data
    torch.manual_seed(42)
    test_data = torch.randn(100, 5)

    # Create model with L2 normalized projections
    model = CrispNamModel(
        num_features=5,
        num_competing_risks=3,
        hidden_sizes=[32, 32],
        normalize_projections=True,
    )

    # Forward pass
    risk_predictions = model(test_data)

    print("First 5 risk predictions:", risk_predictions[:5])

    #Calculate feature importance
    feature_importance = model.calculate_feature_importance(test_data)
    print("First 5 feature importance values:", feature_importance)

    # Analyze projection weights
    print('Analyzing projection weights...')
    projection_weights = model.analyze_projection_weights()
```

## Use of `DeepHIT` model class

```python
import torch

from crisp_nam.models.deephit_model import DeepHit

# Example usage and testing
if __name__ == "__main__":

    # Generate some test data
    torch.manual_seed(42)
    test_data = torch.randn(100, 5)

    input_dims = {
        'x_dim': test_data.shape[1],
        'num_Event': 2,
        'num_Category': 100
    }
        
    network_settings = {
        'h_dim_shared': 128,
        'h_dim_CS': 32,
        'num_layers_shared': 1,
        'num_layers_CS': 2,
        'active_fn': 'tanh',
        'keep_prob': 1.0 - 0.3 #1.0 - dropout_rate
    }
    
    model = DeepHit(input_dims, network_settings)

    out = model.predict(test_data)
    print("Output shape:", out.shape)
```

## Training script options

`training_scripts/train.py` exposes a `--risk_model` CLI option (`cause_specific`,
the default, or `fine_gray`) that selects the competing-risks loss formulation
used during training: cause-specific hazards treat each cause's risk set
independently, while Fine-Gray subdistribution hazards keep subjects who
experience a competing event in every other cause's risk set.
