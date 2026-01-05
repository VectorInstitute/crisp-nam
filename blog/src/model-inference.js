/**
 * Interactive ONNX Model Inference Component
 * Allows users to select a dataset and perform inference using the trained model
 */

import * as ort from 'onnxruntime-web';

// Dataset configurations
const DATASET_CONFIGS = {
  framingham: {
    name: 'Framingham Heart Study',
    numFeatures: 18,
    features: [
      'SEX_1.0', 'CURSMOKE_1.0', 'DIABETES_1.0', 'BPMEDS_1.0',
      'PREVCHD_1.0', 'PREVAP_1.0', 'PREVMI_1.0', 'PREVSTRK_1.0',
      'PREVHYP_1.0', 'educ_2.0', 'educ_3.0', 'educ_4.0',
      'TOTCHOL', 'AGE', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI'
    ],
    featureRanges: {
      // Categorical features (0-1)
      'SEX_1.0': [0, 1],
      'CURSMOKE_1.0': [0, 1],
      'DIABETES_1.0': [0, 1],
      'BPMEDS_1.0': [0, 1],
      'PREVCHD_1.0': [0, 1],
      'PREVAP_1.0': [0, 1],
      'PREVMI_1.0': [0, 1],
      'PREVSTRK_1.0': [0, 1],
      'PREVHYP_1.0': [0, 1],
      'educ_2.0': [0, 1],
      'educ_3.0': [0, 1],
      'educ_4.0': [0, 1],
      // Continuous features (approximate ranges for MinMax scaling [0,1])
      'TOTCHOL': [0, 1],
      'AGE': [0, 1],
      'SYSBP': [0, 1],
      'DIABP': [0, 1],
      'CIGPDAY': [0, 1],
      'BMI': [0, 1]
    },
    riskNames: ['CVD Risk', 'Death Risk'],
    description: 'Predicts cardiovascular disease and death risk based on patient data from the Framingham Heart Study.',
    // Baseline cumulative incidence computed from actual Framingham training data
    // Values represent the proportion of the population experiencing each event by the given time
    baselineCIF: {
      times: [182.5, 365, 547.5, 730],
      timeLabels: ["6 months", "12 months", "18 months", "24 months"],
      risk1: [0.002808, 0.009595, 0.013574, 0.015212],  // CVD baseline CIF
      risk2: [0.001638, 0.003276, 0.005617, 0.007255],  // Death baseline CIF
    }
  }
};

class ModelInference {
  constructor() {
    this.session = null;
    this.currentDataset = 'framingham';
  }

  /**
   * Load ONNX model
   */
  async loadModel(datasetName) {
    try {
      const modelPath = `models/${datasetName}_model.onnx`;
      this.session = await ort.InferenceSession.create(modelPath);
      console.log('Model loaded successfully:', modelPath);
      return true;
    } catch (error) {
      console.error('Error loading model:', error);
      throw error;
    }
  }

  /**
   * Run inference on input data
   */
  async predict(inputData) {
    if (!this.session) {
      throw new Error('Model not loaded');
    }

    try {
      // Convert input to Float32Array
      const inputArray = new Float32Array(inputData);

      // Create tensor
      const tensor = new ort.Tensor('float32', inputArray, [1, inputData.length]);

      // Run inference
      const feeds = { input: tensor };
      const results = await this.session.run(feeds);

      // Get output tensor (log-hazard scores)
      const output = results.risk_scores.data;

      return {
        risk1: output[0],
        risk2: output[1]
      };
    } catch (error) {
      console.error('Error during inference:', error);
      throw error;
    }
  }

  /**
   * Compute absolute risk predictions at multiple time points
   * Uses Fine-Gray model: CIF(t) = 1 - (1 - baseline_CIF(t))^exp(log_hazard)
   */
  computeTimeBasedRisks(logHazards, datasetName) {
    const config = DATASET_CONFIGS[datasetName];
    const baselineCIF = config.baselineCIF;

    const timeBasedRisks = {
      times: baselineCIF.times,
      timeLabels: baselineCIF.timeLabels,
      risk1: [],
      risk2: []
    };

    // Compute absolute risk for Risk 1 (CVD) at each time point
    for (let i = 0; i < baselineCIF.times.length; i++) {
      const baseline = baselineCIF.risk1[i];
      const hazardRatio = Math.exp(logHazards.risk1);
      const absoluteRisk = 1 - Math.pow(1 - baseline, hazardRatio);
      timeBasedRisks.risk1.push(absoluteRisk);
    }

    // Compute absolute risk for Risk 2 (Death) at each time point
    for (let i = 0; i < baselineCIF.times.length; i++) {
      const baseline = baselineCIF.risk2[i];
      const hazardRatio = Math.exp(logHazards.risk2);
      const absoluteRisk = 1 - Math.pow(1 - baseline, hazardRatio);
      timeBasedRisks.risk2.push(absoluteRisk);
    }

    return timeBasedRisks;
  }

  /**
   * Generate random sample data based on dataset configuration
   */
  generateRandomSample(datasetName) {
    const config = DATASET_CONFIGS[datasetName];
    const sample = [];

    config.features.forEach((featureName) => {
      const range = config.featureRanges[featureName];
      // Generate random value within range
      const value = Math.random() * (range[1] - range[0]) + range[0];
      sample.push(value);
    });

    return sample;
  }
}

// Initialize the model inference system
const modelInference = new ModelInference();

/**
 * Initialize the interactive component
 */
export async function initModelInference() {
  const datasetSelect = document.getElementById('dataset-select');
  const loadModelBtn = document.getElementById('load-model-btn');
  const runInferenceBtn = document.getElementById('run-inference-btn');
  const statusDiv = document.getElementById('model-status');
  const resultsDiv = document.getElementById('inference-results');
  const loadingSpinner = document.getElementById('loading-spinner');
  const inputFeaturesDiv = document.getElementById('input-features');

  // Load model button click handler
  loadModelBtn.addEventListener('click', async () => {
    const selectedDataset = datasetSelect.value;

    try {
      statusDiv.textContent = 'Loading model...';
      statusDiv.className = 'status loading';
      loadingSpinner.style.display = 'inline-block';
      loadModelBtn.disabled = true;

      await modelInference.loadModel(selectedDataset);
      modelInference.currentDataset = selectedDataset;

      statusDiv.textContent = `✓ Model loaded: ${DATASET_CONFIGS[selectedDataset].name}`;
      statusDiv.className = 'status success';
      loadingSpinner.style.display = 'none';
      runInferenceBtn.disabled = false;

      // Display dataset info
      displayDatasetInfo(selectedDataset);

    } catch (error) {
      statusDiv.textContent = `✗ Error loading model: ${error.message}`;
      statusDiv.className = 'status error';
      loadingSpinner.style.display = 'none';
      loadModelBtn.disabled = false;
    }
  });

  // Run inference button click handler
  runInferenceBtn.addEventListener('click', async () => {
    const selectedDataset = modelInference.currentDataset;

    try {
      runInferenceBtn.disabled = true;
      resultsDiv.innerHTML = '<div class="loading-text">Running inference...</div>';

      // Generate random sample
      const inputData = modelInference.generateRandomSample(selectedDataset);

      // Display input features
      displayInputFeatures(selectedDataset, inputData);

      // Run prediction (get log-hazard scores)
      const predictions = await modelInference.predict(inputData);

      // Compute time-based absolute risks
      const timeBasedRisks = modelInference.computeTimeBasedRisks(predictions, selectedDataset);

      // Display results with graph
      displayResults(selectedDataset, predictions, timeBasedRisks);

      runInferenceBtn.disabled = false;

    } catch (error) {
      resultsDiv.innerHTML = `<div class="error">Error during inference: ${error.message}</div>`;
      runInferenceBtn.disabled = false;
    }
  });
}

/**
 * Display dataset information
 */
function displayDatasetInfo(datasetName) {
  const config = DATASET_CONFIGS[datasetName];
  const infoDiv = document.getElementById('dataset-info');

  infoDiv.innerHTML = `
    <div class="dataset-info-card">
      <h4>${config.name}</h4>
      <p>${config.description}</p>
      <p><strong>Features:</strong> ${config.numFeatures}</p>
      <p><strong>Risks:</strong> ${config.riskNames.join(', ')}</p>
    </div>
  `;
}

/**
 * Display input features
 */
function displayInputFeatures(datasetName, inputData) {
  const config = DATASET_CONFIGS[datasetName];
  const inputFeaturesDiv = document.getElementById('input-features');

  let html = '<h4>Input Features (Random Sample)</h4>';
  html += '<div class="features-grid">';

  config.features.forEach((featureName, index) => {
    const value = inputData[index].toFixed(4);
    html += `
      <div class="feature-item">
        <span class="feature-name">${featureName}:</span>
        <span class="feature-value">${value}</span>
      </div>
    `;
  });

  html += '</div>';
  inputFeaturesDiv.innerHTML = html;
}

/**
 * Display inference results
 */
function displayResults(datasetName, predictions, timeBasedRisks) {
  const config = DATASET_CONFIGS[datasetName];
  const resultsDiv = document.getElementById('inference-results');

  // Calculate relative risks (softmax-like normalization for visualization)
  const total = Math.abs(predictions.risk1) + Math.abs(predictions.risk2);
  const risk1Percent = (Math.abs(predictions.risk1) / total * 100).toFixed(1);
  const risk2Percent = (Math.abs(predictions.risk2) / total * 100).toFixed(1);

  resultsDiv.innerHTML = `
    <h4>Prediction Results</h4>

    <div class="log-hazard-section">
      <h5>Log-Hazard Scores</h5>
      <div class="results-container">
        <div class="risk-card">
          <div class="risk-header">${config.riskNames[0]}</div>
          <div class="risk-score">${predictions.risk1.toFixed(6)}</div>
          <div class="risk-bar-container">
            <div class="risk-bar" style="width: ${risk1Percent}%"></div>
          </div>
          <div class="risk-percentage">Relative: ${risk1Percent}%</div>
        </div>

        <div class="risk-card">
          <div class="risk-header">${config.riskNames[1]}</div>
          <div class="risk-score">${predictions.risk2.toFixed(6)}</div>
          <div class="risk-bar-container">
            <div class="risk-bar risk-bar-secondary" style="width: ${risk2Percent}%"></div>
          </div>
          <div class="risk-percentage">Relative: ${risk2Percent}%</div>
        </div>
      </div>
    </div>

    <div class="time-based-section">
      <h5>Time-Based Risk Predictions</h5>
      <div class="risk-chart-container">
        <canvas id="risk-chart" width="800" height="400"></canvas>
      </div>

      <div class="time-predictions-table">
        <table class="predictions-table">
          <thead>
            <tr>
              <th>Time Point</th>
              <th>${config.riskNames[0]}</th>
              <th>${config.riskNames[1]}</th>
            </tr>
          </thead>
          <tbody>
            ${timeBasedRisks.timeLabels.map((label, i) => `
              <tr>
                <td class="time-label">${label}</td>
                <td class="risk-value risk-1">${(timeBasedRisks.risk1[i] * 100).toFixed(2)}%</td>
                <td class="risk-value risk-2">${(timeBasedRisks.risk2[i] * 100).toFixed(2)}%</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>

    <div class="results-note">
      <strong>Note:</strong> Log-hazard scores are the raw model outputs. Higher (less negative) values indicate higher baseline risk.
      Time-based predictions show the absolute probability of experiencing each event by the specified time point, accounting for the patient's specific risk factors.
    </div>
  `;

  // Draw the chart after the DOM is updated
  setTimeout(() => drawRiskChart(timeBasedRisks, config), 100);
}

/**
 * Draw risk progression chart using Canvas
 */
function drawRiskChart(timeBasedRisks, config) {
  const canvas = document.getElementById('risk-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  // Chart dimensions
  const padding = { top: 40, right: 100, bottom: 60, left: 70 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Find max risk for scaling - use actual data with slight padding
  const maxRisk1 = Math.max(...timeBasedRisks.risk1);
  const maxRisk2 = Math.max(...timeBasedRisks.risk2);
  const dataMax = Math.max(maxRisk1, maxRisk2);

  // Add 20% padding to top of scale for better visualization
  const maxRisk = dataMax * 1.2;

  // Draw axes
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 2;
  ctx.beginPath();
  // Y-axis
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  // X-axis
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  // Draw grid lines and Y-axis labels
  ctx.strokeStyle = '#e0e0e0';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#666';
  ctx.font = '12px Arial';
  ctx.textAlign = 'right';

  for (let i = 0; i <= 5; i++) {
    const y = height - padding.bottom - (i / 5) * chartHeight;
    const value = (maxRisk * i / 5 * 100).toFixed(1);

    // Grid line
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();

    // Label
    ctx.fillText(`${value}%`, padding.left - 10, y + 4);
  }

  // Draw X-axis labels
  ctx.textAlign = 'center';
  timeBasedRisks.timeLabels.forEach((label, i) => {
    const x = padding.left + (i / (timeBasedRisks.timeLabels.length - 1)) * chartWidth;
    ctx.fillText(label, x, height - padding.bottom + 25);
  });

  // Helper function to plot line
  function plotLine(data, color, label) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();

    data.forEach((risk, i) => {
      const x = padding.left + (i / (data.length - 1)) * chartWidth;
      const y = height - padding.bottom - (risk / maxRisk) * chartHeight;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw points
    ctx.fillStyle = color;
    data.forEach((risk, i) => {
      const x = padding.left + (i / (data.length - 1)) * chartWidth;
      const y = height - padding.bottom - (risk / maxRisk) * chartHeight;

      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  // Plot both risk lines
  plotLine(timeBasedRisks.risk1, '#3498db', config.riskNames[0]);
  plotLine(timeBasedRisks.risk2, '#e74c3c', config.riskNames[1]);

  // Draw legend
  const legendX = width - padding.right + 20;
  const legendY = padding.top + 20;

  // Risk 1 legend
  ctx.fillStyle = '#3498db';
  ctx.fillRect(legendX, legendY, 15, 15);
  ctx.fillStyle = '#333';
  ctx.font = '14px Arial';
  ctx.textAlign = 'left';
  ctx.fillText(config.riskNames[0], legendX + 20, legendY + 12);

  // Risk 2 legend
  ctx.fillStyle = '#e74c3c';
  ctx.fillRect(legendX, legendY + 25, 15, 15);
  ctx.fillStyle = '#333';
  ctx.fillText(config.riskNames[1], legendX + 20, legendY + 37);

  // Chart title
  ctx.fillStyle = '#2c3e50';
  ctx.font = 'bold 16px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Risk Progression Over Time', width / 2, 25);

  // Y-axis label
  ctx.save();
  ctx.translate(20, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#666';
  ctx.font = '14px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Absolute Risk (%)', 0, 0);
  ctx.restore();
}

// Export for use in main app
export default modelInference;
