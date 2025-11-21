# Quick Start Guide

This guide will help you get started with the Variational Inference and Stochastic Variational Inference examples.

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r scripts/requirements.txt
   ```

   Or install manually:
   ```bash
   pip install numpy scipy matplotlib
   ```

## Running the Examples

### Example 1: Variational Inference for Gaussian Mixture Model

```bash
python scripts/examples.py
```

This will run all three examples. To run a specific example, you can modify the `if __name__ == "__main__"` section in `scripts/examples.py`.

The GMM example will:
- Generate synthetic data from a mixture of 3 Gaussians
- Fit a Variational Inference model
- Display learned parameters
- Save a visualization to `gmm_vi_example.png`

### Example 2: Stochastic Variational Inference for LDA

The SVI-LDA example demonstrates:
- How to process documents in mini-batches
- Natural gradient updates
- Topic discovery from documents

### Example 3: Variational Bayesian Regression

The Bayesian Regression example shows:
- Uncertainty quantification in predictions
- Variational inference for regression
- Predictive distributions

## Understanding the Code

### Key Concepts Demonstrated

1. **Variational Inference (VI)**
   - See `VariationalGMM` class
   - Coordinate ascent updates
   - ELBO computation

2. **Stochastic Variational Inference (SVI)**
   - See `StochasticVI_LDA` class
   - Mini-batch processing
   - Natural gradients
   - Step size scheduling

3. **Mean Field Approximation**
   - See `VariationalBayesianRegression` class
   - Factorized posterior distributions
   - Coordinate ascent updates

## Customizing the Examples

### Changing Model Parameters

In `scripts/examples.py`, you can modify:

- **GMM**: Number of components, convergence tolerance
- **SVI-LDA**: Number of topics, batch size, step size parameters
- **Bayesian Regression**: Prior parameters, number of iterations

### Using Your Own Data

1. **For GMM**: Replace the synthetic data generation with your own data array
2. **For SVI-LDA**: Provide your documents as lists of word indices
3. **For Bayesian Regression**: Provide your feature matrix `X` and target vector `y`

## Visualizations

This repository includes 7 high-quality visualizations in the `images/` directory:

- **vi_concept.png**: Core concept of VI approximating complex distributions
- **vi_algorithm_flowchart.png**: Step-by-step algorithm visualization
- **vi_vs_svi_comparison.png**: Comparison between VI and SVI approaches
- **elbo_convergence.png**: Convergence behavior plots
- **mean_field_approximation.png**: Mean field independence assumption
- **applications_overview.png**: Real-world applications diagram
- **kl_divergence_visualization.png**: KL divergence behaviors

See [ILLUSTRATIONS.md](ILLUSTRATIONS.md) for detailed descriptions of each visualization.

## Next Steps

1. Read the main [README.md](README.md) for detailed explanations
2. Explore the [visualizations](images/) to understand concepts visually
3. Experiment with different hyperparameters in the examples
4. Try applying VI/SVI to your own problems
5. Explore the references in the README for deeper understanding

## Troubleshooting

### Import Errors
- Make sure all dependencies are installed: `pip install -r scripts/requirements.txt`

### Convergence Issues
- Try adjusting the number of iterations
- Adjust the convergence tolerance
- Check that your data is properly normalized

### Memory Issues (SVI)
- Reduce batch size
- Process documents in smaller chunks
- Use sparse representations for large vocabularies

## Questions?

Refer to the main README.md for detailed explanations of the theory and algorithms.

