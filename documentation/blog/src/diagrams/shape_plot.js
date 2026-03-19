let DATA;
let currentDataset = 'support';
let currentRisk = 'risk1';
let currentFeature = 'sps'
let features = [];
let risks = ['risk1', 'risk2'].map(r => ({ id: r }));

/* ---------- Load JSON ---------- */
fetch("../final_results.json")
  .then(res => res.json())
  .then(json => {
    DATA = json;
    currentDataset = Object.keys(DATA)[0];

    // Infer risks from dataset
    risks = Object.keys(DATA[currentDataset]);
    currentRisk = risks[0];

    // Infer features from risk
    features = Object.keys(DATA[currentDataset][currentRisk]);
    currentFeature = features[0];
    populateDatasetDropdown();
    populateRiskDropdown();
    populateFeatureDropdown();
    drawPlot();
  });

/* ---------- Populate UI ---------- */
function populateDatasetDropdown() {
    const select = document.getElementById("datasetSelect");
    select.innerHTML = "";

    Object.keys(DATA).forEach(name => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
    });

    select.value = currentDataset;

    select.onchange = e => {
        currentDataset = e.target.value;

        risks = Object.keys(DATA[currentDataset]);
        currentRisk = risks[0];

        features = Object.keys(DATA[currentDataset][currentRisk]);
        currentFeature = features[0];

        populateRiskDropdown();
        populateFeatureDropdown();
        drawPlot();
    };
}

function populateRiskDropdown() {
    const select = document.getElementById("riskSelect");
    select.innerHTML = "";
  
    risks.forEach(risk => {
      const opt = document.createElement("option");
      opt.value = risk;
      opt.textContent = risk;
      select.appendChild(opt);
    });
  
    select.value = currentRisk;
  
    select.onchange = e => {
      currentRisk = e.target.value;
  
      features = Object.keys(DATA[currentDataset][currentRisk]);
      currentFeature = features[0];
      populateDatasetDropdown();
      populateFeatureDropdown();
      drawPlot();
    };
}

function populateFeatureDropdown() {
    const select = document.getElementById("featureSelect");
    select.innerHTML = "";
  
    features.forEach(feature => {
      const opt = document.createElement("option");
      opt.value = feature;
      opt.textContent = feature;
      select.appendChild(opt);
    });
  
    select.value = currentFeature;
  
    select.onchange = e => {
        currentFeature = e.target.value;
        populateDatasetDropdown();
        populateRiskDropdown();
      drawPlot();
    };
}

/* ----- Shape function data ----- */
function drawPlot() {

    const entry = DATA[currentDataset][currentRisk][currentFeature];

    if (!entry || !entry.shape_plots?.pts?.length) {
        Plotly.purge("plot");
        return;
      }

    /* ----- Traces ----- */
    const curve = {
    x: entry.shape_plots.pts,
    y: entry.shape_plots.shp,
    type: "scatter",
    mode: "lines",
    line: { color: "#1f77b4", width: 6 },
    name : "Shape Function",
    };

    const zeroLine = {
    x: [Math.min(...entry.shape_plots.pts), Math.max(...entry.shape_plots.pts)],
    y: [0, 0],
    type: "scatter",
    mode: "lines",
    line: { color: "#1f77b4", width: 3, dash: "dash" },
    hoverinfo: "skip",
    showlegend: false
    };

    const rugBars = {
        x: entry.values,
        y: Array(entry.values.length).fill(-0.08),
        type: "scatter",
        mode: "markers",
        marker: {
            symbol: "line-ns-open",
            size: 14,
            color: "rgba(255,127,14,0.45)",
            opacity: 1.0
        },
        cliponaxis: false,
        hoverinfo: "skip",
        showlegend: false
    };

    /* ----- Render ----- */
    Plotly.react(
        "plot",
        [curve, zeroLine, rugBars],
        {
            title: `${currentDataset} — ${currentRisk} — ${currentFeature}`,
            xaxis: { title: "Feature Value" },
            yaxis: { title: "Contribution", range: [[Math.min(...entry.shape_plots.shp), Math.max(...entry.shape_plots.shp)]] },
            margin: { l: 90, r: 30, t: 80, b: 80 },
            paper_bgcolor: "white",
            plot_bgcolor: "white",
            showlegend: false
      },
      { displayModeBar: false }
    );
}

