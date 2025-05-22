# CRISP-NAM: Competing Risks Interpretable Survival Prediction with Neural Additive Models

CRISP-NAM (Competing Risks Interpretable Survival Prediction with Neural Additive Models), an interpretable neural additive model for 
competing risks survival analysis which extends the neural additive architecture to model cause-specific hazards while preserving feature-level interpretability.

## Project Directory Structure

- Feature 1
- Feature 2
- Feature 3

## Getting Started

```bash
git clone https://github.com/yourusername/crisp-nam.git

cd crisp-nam
# Add setup or installation instructions here
## Setting Up Python Dependencies

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management:

```bash
# Install uv if you don't have it
pip install uv

# Create a virtual environment
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv pip install -r pyproject.toml
```

## Usage

To run examples:

```
python -m train.train --dataset framingham

```


## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.