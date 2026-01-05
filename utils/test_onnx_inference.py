"""
Simple script to test ONNX model inference.

This demonstrates how to load and use the converted ONNX model for predictions.

Usage:
    python test_onnx_inference.py --model onnx_models/framingham_model.onnx
"""

import argparse
import numpy as np
import onnxruntime as ort


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test ONNX model inference"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="onnx_models/framingham_model.onnx",
        help="Path to ONNX model"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of random samples to generate for testing"
    )

    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    print("=" * 70)
    print("ONNX Model Inference Test")
    print("=" * 70)

    # Load ONNX model
    print(f"\nLoading ONNX model from: {args.model}")
    session = ort.InferenceSession(args.model)

    # Get model info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"\nModel Information:")
    print(f"  Input name: {input_info.name}")
    print(f"  Input shape: {input_info.shape}")
    print(f"  Input type: {input_info.type}")
    print(f"\n  Output name: {output_info.name}")
    print(f"  Output shape: {output_info.shape}")
    print(f"  Output type: {output_info.type}")

    # Generate random test data
    num_features = input_info.shape[1]
    test_data = np.random.randn(args.num_samples, num_features).astype(np.float32)

    print(f"\nGenerating {args.num_samples} random test samples with {num_features} features...")

    # Run inference
    print("\nRunning inference...")
    input_name = input_info.name
    outputs = session.run(None, {input_name: test_data})

    # Get risk scores
    risk_scores = outputs[0]

    print(f"\nInference Results:")
    print(f"  Output shape: {risk_scores.shape}")
    print(f"  Risk scores (first 5 samples):")
    print(f"  {'Sample':<10} {'Risk 1':>12} {'Risk 2':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12}")

    for i in range(min(5, args.num_samples)):
        print(f"  {i:<10} {risk_scores[i, 0]:>12.6f} {risk_scores[i, 1]:>12.6f}")

    print("\n" + "=" * 70)
    print("Inference test completed successfully!")
    print("=" * 70)

    # Show basic statistics
    print(f"\nRisk Score Statistics:")
    print(f"  Risk 1 - Mean: {np.mean(risk_scores[:, 0]):.6f}, Std: {np.std(risk_scores[:, 0]):.6f}")
    print(f"  Risk 2 - Mean: {np.mean(risk_scores[:, 1]):.6f}, Std: {np.std(risk_scores[:, 1]):.6f}")


if __name__ == "__main__":
    main()
