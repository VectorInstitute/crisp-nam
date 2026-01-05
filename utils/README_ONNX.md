# ONNX Model Conversion Guide

This directory contains utilities for converting trained PyTorch models to ONNX format for deployment.

## Overview

Two scripts are provided:

1. **`convert_to_onnx.py`**: Generic conversion script with full parameter control
2. **`convert_saved_model.py`**: Convenience wrapper that auto-detects parameters from config files

## Prerequisites

Install required packages:

```bash
pip install onnx onnxruntime
```

## Quick Start

### Method 1: Using the Convenience Wrapper (Recommended)

The easiest way to convert a trained model:

```bash
# Convert framingham model
python utils/convert_saved_model.py --dataset framingham

# Convert with validation
python utils/convert_saved_model.py --dataset framingham --validate

# Convert support model
python utils/convert_saved_model.py --dataset support --validate

# Convert pbc model
python utils/convert_saved_model.py --dataset pbc --validate

# Specify custom directories
python utils/convert_saved_model.py \
    --dataset framingham \
    --model_dir saved_models \
    --output_dir onnx_models \
    --config_dir results/best_params \
    --validate
```

This script automatically:
- Loads the model weights from `saved_models/{dataset}_model.pth`
- Reads configuration from `results/best_params/best_params_{dataset}.yaml`
- Saves ONNX model to `onnx_models/{dataset}_model.onnx`

### Method 2: Using the Generic Script

For more control over parameters:

```bash
python utils/convert_to_onnx.py \
    --model_path saved_models/framingham_model.pth \
    --output_path onnx_models/framingham_model.onnx \
    --num_features 32 \
    --num_competing_risks 2 \
    --hidden_dimensions 64,128 \
    --dropout_rate 0.5 \
    --feature_dropout 0.1 \
    --batch_norm False \
    --validate
```

## Command-Line Arguments

### convert_saved_model.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset` | Yes | - | Dataset name (framingham, support, pbc, synthetic) |
| `--model_dir` | No | `saved_models` | Directory containing model weights |
| `--output_dir` | No | `onnx_models` | Directory to save ONNX models |
| `--config_dir` | No | `results/best_params` | Directory containing config files |
| `--validate` | No | False | Validate ONNX model after conversion |
| `--opset_version` | No | 14 | ONNX opset version |

### convert_to_onnx.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_path` | Yes | - | Path to .pth model weights |
| `--output_path` | No | Same as input with .onnx | Output path for ONNX model |
| `--num_features` | Yes | - | Number of input features |
| `--num_competing_risks` | Yes | - | Number of competing risks |
| `--hidden_dimensions` | No | `64,64` | Hidden layer sizes (comma-separated) |
| `--dropout_rate` | No | 0.5 | Dropout rate |
| `--feature_dropout` | No | 0.1 | Feature dropout rate |
| `--batch_norm` | No | False | Whether batch normalization was used |
| `--validate` | No | False | Validate ONNX model after conversion |
| `--opset_version` | No | 14 | ONNX opset version |

## Dataset Configurations

Default configurations for each dataset:

| Dataset | Features | Competing Risks |
|---------|----------|-----------------|
| framingham | 32 | 2 |
| support | 31 | 2 |
| pbc | 24 | 2 |
| synthetic | 10 | 2 |

## Validation

The `--validate` flag performs the following checks:

1. **Model Integrity**: Verifies the ONNX model structure is valid
2. **Output Comparison**: Compares PyTorch and ONNX outputs on random test data
3. **Numerical Accuracy**: Reports max and mean absolute differences

Example output:
```
ONNX model is valid!

PyTorch output shape: (5, 2)
ONNX output shape: (5, 2)

Validation Results:
  Max absolute difference: 1.234e-06
  Mean absolute difference: 5.678e-07
  ✓ ONNX model outputs match PyTorch model (within tolerance)
```

## Using ONNX Models

### Python Example

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("onnx_models/framingham_model.onnx")

# Prepare input (batch_size, num_features)
input_data = np.random.randn(10, 32).astype(np.float32)

# Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_data})

# Get risk scores
risk_scores = outputs[0]
print(f"Risk scores shape: {risk_scores.shape}")
```

### Deployment

ONNX models can be deployed to:
- **ONNX Runtime** (Python, C++, C#, Java)
- **TensorFlow** (via onnx-tf)
- **TensorRT** (NVIDIA GPUs)
- **OpenVINO** (Intel hardware)
- **CoreML** (Apple devices)
- **Web browsers** (via ONNX.js)

## Troubleshooting

### Issue: "Model weights not found"

**Solution**: Ensure you've trained the model first:
```bash
python training_scripts/train.py --dataset framingham -c results/best_params/best_params_framingham.yaml
```

### Issue: "Config file not found"

**Solution**: Either:
1. Ensure config file exists at `results/best_params/best_params_{dataset}.yaml`
2. Use the generic script with manual parameters
3. The script will use default parameters if config is missing

### Issue: "onnx and onnxruntime are required"

**Solution**: Install the required packages:
```bash
pip install onnx onnxruntime
```

### Issue: Validation shows large differences

**Solution**: This may indicate:
- Incorrect model parameters during initialization
- Version mismatch in PyTorch/ONNX
- Try updating packages: `pip install --upgrade torch onnx onnxruntime`

## Notes

1. **Batch Normalization**: Models with batch normalization require careful handling. The conversion script automatically sets the model to evaluation mode.

2. **Dropout**: Dropout layers are disabled during ONNX conversion (model is in eval mode).

3. **Dynamic Batch Size**: The converted ONNX models support dynamic batch sizes for inference.

4. **Opset Version**: Default is opset 14. Use higher versions for newer ONNX features, lower for broader compatibility.

## Example Workflow

```bash
# 1. Train the model
python training_scripts/train.py \
    --dataset framingham \
    -c results/best_params/best_params_framingham.yaml

# 2. Convert to ONNX with validation
python utils/convert_saved_model.py \
    --dataset framingham \
    --validate

# 3. Use in production
python your_inference_script.py --model onnx_models/framingham_model.onnx
```

## Support

For issues or questions:
1. Check this README
2. Verify model parameters match training configuration
3. Ensure all dependencies are installed
4. Try validation to identify conversion issues
