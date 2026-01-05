# Model Conversion Summary

## Successfully Converted Model

**Model:** Framingham Competing Risks Model
**Date:** January 5, 2026
**Status:** ✅ Successfully converted and validated

## Conversion Details

### Input Model
- **Format:** PyTorch (.pth)
- **Path:** `saved_models/framingham_model.pth`
- **Size:** 6.3 MB

### Output Model
- **Format:** ONNX
- **Path:** `onnx_models/framingham_model.onnx`
- **Size:** 6.3 MB
- **ONNX Opset Version:** 14

### Model Configuration
- **Number of features:** 18
- **Number of competing risks:** 2
- **Hidden dimensions:** [64, 256]
- **Dropout rate:** 0.645
- **Feature dropout:** 0.334
- **Batch normalization:** True
- **Scaling method:** MinMax

### Validation Results
✅ **PASSED** - Model outputs match between PyTorch and ONNX

- **Max absolute difference:** 2.38e-06
- **Mean absolute difference:** 8.46e-07
- **Status:** Within acceptable tolerance (< 1e-05)

## Model Specifications

### Input
- **Name:** `input`
- **Shape:** `[batch_size, 18]`
- **Type:** `float32`
- **Dynamic Axes:** Batch size is dynamic (supports any batch size)

### Output
- **Name:** `risk_scores`
- **Shape:** `[batch_size, 2]`
- **Type:** `float32`
- **Description:** Risk scores for 2 competing events
  - Column 0: Risk score for event type 1 (CVD)
  - Column 1: Risk score for event type 2 (Death)

## Usage Example

### Python with ONNX Runtime

```python
import numpy as np
import onnxruntime as ort

# Load the model
session = ort.InferenceSession("onnx_models/framingham_model.onnx")

# Prepare input data (batch_size, 18 features)
input_data = np.random.randn(10, 18).astype(np.float32)

# Run inference
outputs = session.run(None, {"input": input_data})
risk_scores = outputs[0]  # Shape: (10, 2)

print(f"Risk scores: {risk_scores}")
```

### Testing the Model

A test script is provided to verify the ONNX model works correctly:

```bash
python utils/test_onnx_inference.py --model onnx_models/framingham_model.onnx
```

## Deployment Options

The ONNX model can now be deployed to various platforms:

1. **ONNX Runtime** (Python, C++, C#, Java, JavaScript)
2. **TensorFlow** (via onnx-tf converter)
3. **TensorRT** (NVIDIA GPU acceleration)
4. **OpenVINO** (Intel CPU/GPU optimization)
5. **CoreML** (Apple devices)
6. **Mobile Devices** (via ONNX Mobile)
7. **Web Browsers** (via ONNX.js)

## Files Created

1. `utils/convert_to_onnx.py` - Generic ONNX conversion script
2. `utils/convert_saved_model.py` - Convenience wrapper for saved models
3. `utils/test_onnx_inference.py` - Test script for ONNX inference
4. `utils/README_ONNX.md` - Comprehensive documentation
5. `onnx_models/framingham_model.onnx` - Converted ONNX model

## Next Steps

### To convert other models:

```bash
# Support dataset
python utils/convert_saved_model.py --dataset support --validate

# PBC dataset
python utils/convert_saved_model.py --dataset pbc --validate

# Synthetic dataset
python utils/convert_saved_model.py --dataset synthetic --validate
```

### To use the model in production:

1. Load the ONNX model using ONNX Runtime
2. Ensure input features are properly scaled (MinMax scaling)
3. Input shape must be `[batch_size, 18]`
4. Output will be `[batch_size, 2]` risk scores

## Notes

- The ONNX model includes all trained weights and architecture
- Dropout layers are disabled (model is in evaluation mode)
- Batch normalization uses learned statistics
- The model supports dynamic batch sizes
- Input features should be preprocessed the same way as during training

## Performance Characteristics

- **Inference Speed:** Fast (optimized for production)
- **Memory Usage:** 6.3 MB
- **Precision:** FP32 (can be quantized to FP16 or INT8 for faster inference)
- **Batch Processing:** Supported (dynamic batch size)

## Warnings

During conversion, a TracerWarning was generated regarding batch normalization with batch_size=1. This is expected and does not affect model functionality. The ONNX model handles all batch sizes correctly.
