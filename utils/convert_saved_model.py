"""
Convenience wrapper script for converting saved models to ONNX.

This script automatically loads the corresponding config file to extract
model parameters, making conversion easier.

Usage:
    # Convert framingham model
    python convert_saved_model.py --dataset framingham

    # Convert support model with validation
    python convert_saved_model.py --dataset support --validate

    # Specify custom paths
    python convert_saved_model.py --dataset pbc \
                                  --model_dir saved_models \
                                  --output_dir onnx_models
"""

import argparse
import os
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convert_to_onnx import load_model, convert_to_onnx, validate_onnx_model


# Dataset configurations (default values if config file not found)
DATASET_CONFIGS = {
    "framingham": {
        "num_features": 18,
        "num_competing_risks": 2,
    },
    "support": {
        "num_features": 31,
        "num_competing_risks": 2,
    },
    "pbc": {
        "num_features": 24,
        "num_competing_risks": 2,
    },
    "synthetic": {
        "num_features": 10,
        "num_competing_risks": 2,
    }
}


def load_config_from_yaml(dataset, config_dir="results/best_params"):
    """
    Load model configuration from YAML file.

    Args:
        dataset: Dataset name
        config_dir: Directory containing config files

    Returns:
        Dictionary of configuration parameters
    """
    config_path = os.path.join(config_dir, f"best_params_{dataset}.yaml")

    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        print("Using default parameters...")
        return None

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert saved model to ONNX format (convenience wrapper)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["framingham", "support", "pbc", "synthetic"],
        help="Dataset name (determines model filename and parameters)"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models",
        help="Directory containing saved model weights (default: saved_models)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="onnx_models",
        help="Directory to save ONNX models (default: onnx_models)"
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="results/best_params",
        help="Directory containing config YAML files (default: results/best_params)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the ONNX model after conversion"
    )

    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )

    return parser.parse_args()


def main():
    """Main conversion function."""
    args = parse_args()

    # Construct model path
    model_path = os.path.join(args.model_dir, f"{args.dataset}_model.pth")

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        print(f"Please train the model first using train.py")
        sys.exit(1)

    # Construct output path
    output_path = os.path.join(args.output_dir, f"{args.dataset}_model.onnx")

    # Load configuration
    config = load_config_from_yaml(args.dataset, args.config_dir)

    # Get parameters from config or use defaults
    if config:
        hidden_dimensions_str = config.get("hidden_dimensions", "64,64")
        hidden_sizes = [int(dim) for dim in hidden_dimensions_str.split(",")]
        dropout_rate = config.get("dropout_rate", 0.5)
        feature_dropout = config.get("feature_dropout", 0.1)
        batch_norm = str(config.get("batch_norm", "False")).lower() == "true"
        print(f"Loaded configuration from {args.config_dir}/best_params_{args.dataset}.yaml")
    else:
        hidden_sizes = [64, 64]
        dropout_rate = 0.5
        feature_dropout = 0.1
        batch_norm = False
        print("Using default parameters")

    # Get dataset-specific parameters
    dataset_config = DATASET_CONFIGS.get(args.dataset, {})
    num_features = dataset_config.get("num_features", 32)
    num_competing_risks = dataset_config.get("num_competing_risks", 2)

    print("\n" + "=" * 70)
    print(f"Converting {args.dataset} Model to ONNX")
    print("=" * 70)
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Configuration:")
    print(f"  - Number of features: {num_features}")
    print(f"  - Number of competing risks: {num_competing_risks}")
    print(f"  - Hidden dimensions: {hidden_sizes}")
    print(f"  - Dropout rate: {dropout_rate}")
    print(f"  - Feature dropout: {feature_dropout}")
    print(f"  - Batch normalization: {batch_norm}")
    print("=" * 70)

    # Load PyTorch model
    model = load_model(
        model_path=model_path,
        num_features=num_features,
        num_competing_risks=num_competing_risks,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        feature_dropout=feature_dropout,
        batch_norm=batch_norm
    )

    # Convert to ONNX
    onnx_path = convert_to_onnx(
        model=model,
        output_path=output_path,
        num_features=num_features,
        opset_version=args.opset_version
    )

    # Validate if requested
    if args.validate:
        validate_onnx_model(onnx_path, model, num_features)

    print("\n" + "=" * 70)
    print("Conversion complete!")
    print(f"ONNX model saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
