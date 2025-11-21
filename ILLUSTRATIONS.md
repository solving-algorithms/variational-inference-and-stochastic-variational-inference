# Image Descriptions

This document provides detailed descriptions of all visualizations in the `images/` directory.

## 1. vi_concept.png

**Title:** Variational Inference: Approximating Complex Distributions

**Description:**
This diagram illustrates the core concept of Variational Inference. It shows:
- **Blue curve**: The true, complex posterior distribution `p(z|x)` which is often intractable to compute exactly
- **Red dashed curve**: A simpler, tractable approximate distribution `q(z)` that VI uses to approximate the true distribution
- **Green annotation**: Highlights that KL Divergence is used to measure the "distance" between the two distributions

**Key Takeaway:** VI finds a simpler distribution that is "close enough" to the complex true distribution, making inference computationally feasible.

---

## 2. vi_algorithm_flowchart.png

**Title:** Variational Inference Algorithm Flowchart

**Description:**
This flowchart shows the step-by-step process of the Variational Inference algorithm:
1. **Start**: Begin the algorithm
2. **Choose variational family Q**: Select a family of simpler distributions (e.g., Gaussian distributions)
3. **Initialize q(z)**: Set initial values for the approximate distribution
4. **Compute ELBO**: Calculate the Evidence Lower BOund
5. **Optimize q(z)**: Update the approximate distribution to maximize ELBO
6. **Converged?**: Check if the algorithm has converged
   - If **No**: Return to step 4 (optimization continues)
   - If **Yes**: Return the optimal approximate distribution `q*(z)`

**Key Takeaway:** VI is an iterative optimization algorithm that maximizes ELBO to find the best approximation.

---

## 3. vi_vs_svi_comparison.png

**Title:** Variational Inference vs Stochastic Variational Inference

**Description:**
This side-by-side comparison illustrates the key differences between VI and SVI:

**Left Panel (VI):**
- Processes the **entire dataset** (all N samples) in each iteration
- Memory requirement: O(N)
- Time complexity: O(N) per iteration
- Deterministic updates

**Right Panel (SVI):**
- Processes data in **mini-batches** (B samples at a time)
- Memory requirement: O(B) where B << N
- Time complexity: O(B) per iteration
- Stochastic updates with natural gradients
- Scalable to millions of data points

**Key Takeaway:** SVI enables scalable inference on large datasets by processing data incrementally in mini-batches.

---

## 4. elbo_convergence.png

**Title:** ELBO Convergence: VI vs SVI

**Description:**
This plot shows how the Evidence Lower BOund (ELBO) changes during optimization:

- **Blue solid line (VI)**: Shows smooth, deterministic convergence. The ELBO increases monotonically and converges to a stable value.

- **Red dashed line (SVI)**: Shows noisy but trending upward convergence. The stochastic nature of mini-batch updates causes variance, but the overall trend is upward.

- **Red dotted line**: Smoothed version of SVI showing the underlying trend.

**Key Takeaway:** 
- VI provides smooth, deterministic convergence
- SVI has noisy updates but converges faster and can handle larger datasets
- Both methods maximize ELBO, which is equivalent to minimizing KL divergence

---

## 5. mean_field_approximation.png

**Title:** Mean Field Approximation

**Description:**
This side-by-side visualization demonstrates the mean field assumption:

**Left Panel:**
- Shows the **true posterior** `p(z₁, z₂|x)` where the two latent variables are **correlated**
- The distribution has an elliptical shape, indicating dependence between variables

**Right Panel:**
- Shows the **mean field approximation** `q(z₁)q(z₂)` where variables are assumed **independent**
- The distribution is circular, reflecting the independence assumption
- Each variable has its own marginal distribution

**Key Takeaway:** Mean field VI assumes independence between latent variables, which simplifies computation but may miss important correlations. The approximation `q(z) = ∏ᵢ qᵢ(zᵢ)` factorizes the joint distribution into independent factors.

---

## 6. applications_overview.png

**Title:** Applications of Variational Inference

**Description:**
This diagram shows various real-world applications where VI and SVI are used:

1. **Topic Modeling (LDA)**: Discovering topics in document collections
2. **Variational Autoencoders**: Learning generative models and representations
3. **Bayesian Neural Networks**: Quantifying uncertainty in deep learning
4. **Gaussian Processes**: Non-parametric regression with uncertainty
5. **Recommendation Systems**: Collaborative filtering and matrix factorization

All applications connect to the central VI/SVI methodology, demonstrating its versatility across different domains.

**Key Takeaway:** VI and SVI are widely applicable across many machine learning and statistical modeling problems.

---

## 7. kl_divergence_visualization.png

**Title:** Kullback-Leibler Divergence: Different Behaviors

**Description:**
This four-panel visualization demonstrates different aspects of KL divergence:

**Top Left - Small KL Divergence:**
- Shows a good approximation where `q(z)` closely matches `p(z)`
- KL ≈ 0.02 indicates the distributions are very similar
- Green shaded area shows the overlap

**Top Right - Large KL Divergence:**
- Shows a poor approximation where `q(z)` is far from `p(z)`
- KL ≈ 1.8 indicates significant difference
- Red shaded area shows limited overlap

**Bottom Left - Mode-Seeking (KL(q||p)):**
- When minimizing KL(q||p), the approximation tends to capture one mode
- Here, `q(z)` (unimodal) captures one mode of the bimodal `p(z)`
- This is the standard VI objective

**Bottom Right - Mean-Seeking (KL(p||q)):**
- When minimizing KL(p||q), the approximation tends to cover all modes
- Here, `q(z)` (bimodal) spreads to cover the unimodal `p(z)`
- This is used in some alternative formulations

**Key Takeaway:** 
- KL divergence is asymmetric: KL(q||p) ≠ KL(p||q)
- The direction matters: KL(q||p) is mode-seeking, KL(p||q) is mean-seeking
- VI typically uses KL(q||p) which is mode-seeking

---

## How to Use These Images

1. **In Presentations**: Use these images to explain VI and SVI concepts visually
2. **In Documentation**: Reference these images in your README or papers
3. **For Learning**: Study the visualizations alongside the mathematical explanations
4. **For Teaching**: Use these as teaching aids to explain complex concepts

## Regenerating Images

To regenerate all images, run:
```bash
python scripts/generate_visualizations.py
```

Make sure you have the required dependencies installed:
```bash
pip install -r scripts/requirements.txt
```

## Image Specifications

- **Format**: PNG
- **Resolution**: 300 DPI (suitable for printing and presentations)
- **Color Scheme**: Colorful with clear distinctions between elements
- **Font Size**: Optimized for readability at various sizes

