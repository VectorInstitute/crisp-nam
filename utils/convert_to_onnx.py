"""
Script to load trained model weights and convert to ONNX format.

This script provides functionality to:
1. Load a trained CrispNamModel from saved PyTorch weights (.pth)
2. Convert the model to ONNX format for deployment
3. Validate the converted ONNX model

Usage:
    python convert_to_onnx.py --model_path saved_models/framingham_model.pth \
                              --output_path onnx_models/framingham_model.onnx \
                              --num_features 32 \
                              --num_competing_risks 2 \
                              --hidden_dimensions 64,128
"""

import argparse
import os
import torch
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crisp_nam.models import CrispNamModel


class CrispNamModelONNXWrapper(torch.nn.Module):
    """
    Wrapper for CrispNamModel to make it ONNX-compatible.
    Converts list output to concatenated tensor.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        risk_scores, feature_outputs = self.model(x)
        # Concatenate list of risk scores into single tensor
        risk_scores_tensor = torch.cat(risk_scores, dim=1)
        return risk_scores_tensor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert trained CrispNamModel weights to ONNX format"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved PyTorch model weights (.pth file)"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path where the ONNX model will be saved (default: same name with .onnx extension)"
    )

    parser.add_argument(
        "--num_features",
        type=int,
        required=True,
        help="Number of input features"
    )

    parser.add_argument(
        "--num_competing_risks",
        type=int,
        required=True,
        help="Number of competing risks"
    )

    parser.add_argument(
        "--hidden_dimensions",
        type=str,
        default="64,64",
        help="Hidden layer dimensions (comma-separated, e.g., '64,128')"
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate used during training (for model initialization)"
    )

    parser.add_argument(
        "--feature_dropout",
        type=float,
        default=0.1,
        help="Feature dropout rate used during training (for model initialization)"
    )

    parser.add_argument(
        "--batch_norm",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Whether batch normalization was used"
    )

    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the ONNX model after conversion"
    )

    return parser.parse_args()


def load_model(model_path, num_features, num_competing_risks, hidden_sizes,
               dropout_rate, feature_dropout, batch_norm):
    """
    Load a trained CrispNamModel from saved weights.

    Args:
        model_path: Path to the .pth file containing model weights
        num_features: Number of input features
        num_competing_risks: Number of competing risks
        hidden_sizes: List of hidden layer dimensions
        dropout_rate: Dropout rate (used for initialization)
        feature_dropout: Feature dropout rate (used for initialization)
        batch_norm: Whether to use batch normalization

    Returns:
        Loaded PyTorch model in evaluation mode
    """
    # Initialize model with the same architecture
    model = CrispNamModel(
        num_features=num_features,
        num_competing_risks=num_competing_risks,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        feature_dropout=feature_dropout,
        batch_norm=batch_norm
    )

    # Load saved weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    # Set to evaluation mode (disables dropout)
    model.eval()

    print(f"Successfully loaded model from: {model_path}")
    print(f"Model architecture: {num_features} features, {num_competing_risks} competing risks")
    print(f"Hidden layers: {hidden_sizes}")

    return model


def convert_to_onnx(model, output_path, num_features, opset_version=14):
    """
    Convert PyTorch model to ONNX format.

    Args:
        model: PyTorch model in evaluation mode
        output_path: Path where ONNX model will be saved
        num_features: Number of input features
        opset_version: ONNX opset version

    Returns:
        Path to saved ONNX model
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Wrap the model to make it ONNX-compatible
    wrapped_model = CrispNamModelONNXWrapper(model)
    wrapped_model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, num_features, dtype=torch.float32)

    # Export to ONNX
    print(f"\nConverting model to ONNX format...")
    print(f"Input shape: (batch_size, {num_features})")

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['risk_scores'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'risk_scores': {0: 'batch_size'}
        }
    )

    print(f"Successfully exported model to: {output_path}")
    return output_path


def validate_onnx_model(onnx_path, pytorch_model, num_features):
    """
    Validate the ONNX model by comparing outputs with PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        num_features: Number of input features
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("\nWarning: onnx and onnxruntime are required for validation.")
        print("Install with: pip install onnx onnxruntime")
        return

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("\nONNX model is valid!")

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Create test input
    test_input = torch.randn(5, num_features, dtype=torch.float32)

    # Get PyTorch output using the wrapper
    wrapped_model = CrispNamModelONNXWrapper(pytorch_model)
    wrapped_model.eval()

    with torch.no_grad():
        pytorch_output = wrapped_model(test_input)

    # Get ONNX output
    onnx_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, onnx_input)
    onnx_risk_scores = onnx_output[0]

    pytorch_np = pytorch_output.numpy()

    # Check if shapes match
    print(f"\nPyTorch output shape: {pytorch_np.shape}")
    print(f"ONNX output shape: {onnx_risk_scores.shape}")

    # Check if values are close
    max_diff = np.max(np.abs(pytorch_np - onnx_risk_scores))
    mean_diff = np.mean(np.abs(pytorch_np - onnx_risk_scores))

    print(f"\nValidation Results:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    if max_diff < 1e-5:
        print("  ✓ ONNX model outputs match PyTorch model (within tolerance)")
    else:
        print("  ⚠ Warning: Outputs differ more than expected")


def main():
    """Main function to convert PyTorch model to ONNX."""
    args = parse_args()

    # Parse hidden dimensions
    hidden_sizes = [int(dim) for dim in args.hidden_dimensions.split(",")]

    # Convert batch_norm string to boolean
    batch_norm = args.batch_norm.lower() == "true"

    # Determine output path
    if args.output_path is None:
        # Use same name as input but with .onnx extension
        base_name = os.path.splitext(args.model_path)[0]
        output_path = base_name + ".onnx"
    else:
        output_path = args.output_path

    print("=" * 70)
    print("PyTorch to ONNX Model Conversion")
    print("=" * 70)

    # Load PyTorch model
    model = load_model(
        model_path=args.model_path,
        num_features=args.num_features,
        num_competing_risks=args.num_competing_risks,
        hidden_sizes=hidden_sizes,
        dropout_rate=args.dropout_rate,
        feature_dropout=args.feature_dropout,
        batch_norm=batch_norm
    )

    # Convert to ONNX
    onnx_path = convert_to_onnx(
        model=model,
        output_path=output_path,
        num_features=args.num_features,
        opset_version=args.opset_version
    )

    # Validate if requested
    if args.validate:
        validate_onnx_model(onnx_path, model, args.num_features)

    print("\n" + "=" * 70)
    print("Conversion complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
