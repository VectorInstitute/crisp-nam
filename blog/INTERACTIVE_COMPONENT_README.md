# Interactive CRISP-NAM Model Component

## Overview

An interactive browser-based component has been added to the CRISP-NAM blog that allows users to:
1. Select a dataset from a dropdown menu
2. Load the corresponding trained ONNX model
3. Perform real-time inference directly in the browser using ONNX.js
4. View prediction results with visualizations

## Files Created/Modified

### New Files
1. **`src/model-inference.js`** - Main JavaScript module for model inference
   - ModelInference class for managing ONNX models
   - Functions for loading models, running predictions, and displaying results
   - Dataset configurations with feature names and ranges

2. **`static/models/framingham_model.onnx`** - ONNX model file (6.3 MB)
   - Converted from PyTorch using the conversion scripts
   - Loaded directly in the browser via ONNX Runtime Web

### Modified Files
1. **`src/index.ejs`** - Added HTML for the interactive component
   - Dropdown for dataset selection
   - Buttons for loading model and running inference
   - Containers for displaying results and status

2. **`src/index.js`** - Updated to initialize the component
   - Import and call `initModelInference()`

3. **`src/style.css`** - Added comprehensive styling
   - Interactive component layout
   - Buttons, dropdowns, and controls
   - Result cards with progress bars
   - Status indicators and loading spinners

4. **`package.json`** - Added dependency
   - `onnxruntime-web` for browser-based inference

## Features

### 1. Dataset Selection
- Dropdown menu to choose between available datasets
- Currently supports: Framingham Heart Study
- Easy to extend for other datasets (Support2, PBC, Synthetic)

### 2. Model Loading
- Asynchronous model loading with progress indication
- Status messages for success/error states
- Model metadata display (features, risks, description)

### 3. Inference
- Generates random sample data based on dataset configuration
- Displays input features in a grid layout
- Runs inference using ONNX Runtime Web
- Shows raw risk scores and normalized percentages

### 4. Results Visualization
- Two risk cards showing predictions for each competing risk
- Horizontal progress bars for visual comparison
- Percentage distribution for easier interpretation
- Color-coded risk indicators

## Technical Details

### ONNX Runtime Web
- Uses WebAssembly (WASM) for fast browser-based inference
- Supports dynamic batch sizes
- No server-side processing required
- Model runs entirely in the user's browser

### Model Format
- **Input**: `[batch_size, 18]` float32 tensor
- **Output**: `[batch_size, 2]` float32 tensor (risk scores for 2 competing risks)
- **Size**: 6.3 MB (includes all weights and architecture)

### Dataset Configuration
```javascript
{
  framingham: {
    name: 'Framingham Heart Study',
    numFeatures: 18,
    features: ['SEX_1.0', 'CURSMOKE_1.0', ... 'BMI'],
    featureRanges: { /* min-max ranges for each feature */ },
    riskNames: ['CVD Risk', 'Death Risk'],
    description: '...'
  }
}
```

## Usage

### Building the Blog
```bash
cd blog
npm install
npm run build
```

### Development Mode
```bash
npm run dev
```
This starts a webpack dev server at `http://localhost:8080`

### Production Build
```bash
npm run build
```
Outputs to `public/` directory

## User Interaction Flow

1. **Select Dataset**: User chooses "Framingham Heart Study" from dropdown
2. **Load Model**: Click "Load Model" button
   - Model loads from `models/framingham_model.onnx`
   - Status updates to show success/failure
   - Dataset info card appears with metadata
3. **Run Inference**: Click "Run Inference" button
   - Random sample data is generated
   - Input features displayed in grid
   - Model prediction runs in browser
   - Results shown with risk scores and visualizations

## Extending to Other Datasets

To add more datasets (e.g., Support2, PBC):

1. **Convert model to ONNX**:
   ```bash
   python utils/convert_saved_model.py --dataset support --validate
   ```

2. **Copy to blog**:
   ```bash
   cp onnx_models/support_model.onnx blog/static/models/
   ```

3. **Update `model-inference.js`**:
   ```javascript
   DATASET_CONFIGS.support = {
     name: 'SUPPORT2',
     numFeatures: 31,
     features: [...],
     featureRanges: {...},
     riskNames: ['Cancer Death', 'Other Death'],
     description: '...'
   };
   ```

4. **Update dropdown in `index.ejs`**:
   ```html
   <select id="dataset-select">
     <option value="framingham">Framingham Heart Study</option>
     <option value="support">SUPPORT2 Dataset</option>
   </select>
   ```

5. **Rebuild**:
   ```bash
   npm run build
   ```

## Performance

- **Initial Load**: ~6.3 MB model download + ~23 MB WASM runtime
- **Model Loading**: ~1-2 seconds (one-time per dataset)
- **Inference Time**: <100ms for single prediction
- **Browser Compatibility**: Modern browsers with WebAssembly support

## Browser Requirements

- Modern browser (Chrome 87+, Firefox 79+, Safari 14+, Edge 87+)
- WebAssembly support
- JavaScript enabled
- Minimum 30 MB available memory

## Styling

The component uses a modern, professional design:
- **Colors**: Blue (#3498db) for primary actions, Green (#27ae60) for inference
- **Layout**: Responsive grid system
- **Animations**: Smooth transitions and hover effects
- **Feedback**: Clear status indicators and loading spinners

## Security & Privacy

- **No data transmission**: All inference happens locally in browser
- **No tracking**: No analytics or user data collection
- **Client-side only**: No server-side processing
- **Privacy-preserving**: Random samples generated, no real patient data

## Future Enhancements

Potential improvements:
1. **Custom Input**: Allow users to input their own feature values
2. **Batch Processing**: Support multiple predictions at once
3. **Export Results**: Download predictions as CSV/JSON
4. **Model Comparison**: Compare different models side-by-side
5. **Feature Importance**: Show which features contributed most
6. **Time-to-Event**: Add survival curve visualization
7. **Educational Mode**: Explain each feature and prediction

## Troubleshooting

### Model not loading
- Check browser console for errors
- Verify model file exists at `models/framingham_model.onnx`
- Ensure WASM support in browser
- Clear cache and reload

### Inference fails
- Check if model loaded successfully
- Verify input data format matches expected shape
- Check browser console for ONNX runtime errors

### Build fails
- Run `npm install` to ensure dependencies installed
- Check for JavaScript syntax errors
- Verify webpack configuration

## Files Summary

```
blog/
├── src/
│   ├── index.ejs              # HTML template with component
│   ├── index.js               # Main entry point
│   ├── model-inference.js     # NEW: Inference logic
│   └── style.css              # Updated with component styles
├── static/
│   └── models/
│       └── framingham_model.onnx  # NEW: ONNX model
├── public/                    # Build output
│   ├── index.html
│   ├── index.bundle.js
│   ├── models/
│   │   └── framingham_model.onnx
│   └── *.wasm                 # ONNX Runtime WebAssembly
└── package.json               # Updated dependencies
```

## Credits

- **ONNX Runtime Web**: Microsoft
- **Webpack**: Build tooling
- **Distill Template**: Blog framework
- **CRISP-NAM**: Model architecture

## License

Same as parent project.
