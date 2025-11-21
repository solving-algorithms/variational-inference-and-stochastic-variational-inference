"""
Practical Examples of Variational Inference and Stochastic Variational Inference

This file contains working implementations of VI and SVI algorithms
for educational purposes.
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


# ============================================================================
# Example 1: Simple Gaussian Mixture Model with Variational Inference
# ============================================================================

class VariationalGMM:
    """
    Variational Inference for Gaussian Mixture Model
    
    This is a simplified implementation demonstrating the core concepts of VI.
    """
    
    def __init__(self, n_components: int = 3, max_iter: int = 100, tol: float = 1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X: np.ndarray, alpha_0: float = 1.0):
        """
        Fit the model using Variational Inference
        
        Parameters:
        -----------
        X : array-like, shape (n_samples,)
            Input data
        alpha_0 : float
            Prior concentration parameter for Dirichlet distribution
        """
        n_samples = len(X)
        
        # Initialize variational parameters
        # q(pi) ~ Dirichlet(alpha)
        alpha = np.ones(self.n_components) * alpha_0
        
        # q(mu_k) ~ Normal(m_k, s_k^2)
        m = np.random.randn(self.n_components)
        s_sq = np.ones(self.n_components)
        
        # q(tau_k) ~ Gamma(a_k, b_k) where tau = 1/sigma^2
        a = np.ones(self.n_components)
        b = np.ones(self.n_components)
        
        # Responsibilities (variational parameters for latent assignments)
        r = np.random.dirichlet(np.ones(self.n_components), size=n_samples)
        
        prev_elbo = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: Update responsibilities
            for k in range(self.n_components):
                E_log_pi_k = digamma(alpha[k]) - digamma(alpha.sum())
                E_log_tau_k = digamma(a[k]) - np.log(b[k])
                E_tau_k = a[k] / b[k]
                E_mu_k = m[k]
                
                # Compute log probability
                log_prob = (E_log_pi_k + 
                           0.5 * E_log_tau_k - 
                           0.5 * E_tau_k * ((X - E_mu_k)**2 + s_sq[k]))
                r[:, k] = np.exp(log_prob)
            
            # Normalize responsibilities
            r = r / r.sum(axis=1, keepdims=True)
            
            # M-step: Update variational parameters
            N_k = r.sum(axis=0)
            
            # Update alpha (Dirichlet parameters)
            alpha = alpha_0 + N_k
            
            # Update m_k and s_k^2 (Gaussian parameters for means)
            for k in range(self.n_components):
                E_tau_k = a[k] / b[k]
                m[k] = (E_tau_k * (r[:, k] * X).sum()) / (E_tau_k * N_k[k] + 1e-10)
                s_sq[k] = 1.0 / (E_tau_k * N_k[k] + 1e-10)
            
            # Update a_k and b_k (Gamma parameters for precisions)
            for k in range(self.n_components):
                a[k] = 1.0 + 0.5 * N_k[k]
                b[k] = (1.0 + 
                        0.5 * (r[:, k] * ((X - m[k])**2 + s_sq[k])).sum())
            
            # Compute ELBO
            elbo = self._compute_elbo(X, r, alpha, m, s_sq, a, b, alpha_0)
            
            # Check convergence
            if abs(elbo - prev_elbo) < self.tol:
                print(f"Converged at iteration {iteration}")
                break
            
            prev_elbo = elbo
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, ELBO: {elbo:.4f}")
        
        self.alpha = alpha
        self.m = m
        self.s_sq = s_sq
        self.a = a
        self.b = b
        self.r = r
        
        return self
    
    def _compute_elbo(self, X, r, alpha, m, s_sq, a, b, alpha_0):
        """Compute Evidence Lower Bound"""
        n_samples = len(X)
        n_components = len(alpha)
        
        # Reconstruction term
        recon = 0.0
        for k in range(n_components):
            E_tau_k = a[k] / b[k]
            E_log_tau_k = digamma(a[k]) - np.log(b[k])
            E_mu_k = m[k]
            
            for n in range(n_samples):
                recon += r[n, k] * (
                    digamma(alpha[k]) - digamma(alpha.sum()) +
                    0.5 * E_log_tau_k -
                    0.5 * E_tau_k * ((X[n] - E_mu_k)**2 + s_sq[k]) -
                    0.5 * np.log(2 * np.pi)
                )
        
        # KL divergence terms (simplified)
        kl = 0.0
        # KL for pi
        kl += gammaln(alpha_0 * n_components) - n_components * gammaln(alpha_0)
        kl -= gammaln(alpha.sum()) - gammaln(alpha).sum()
        kl += ((alpha - alpha_0) * (digamma(alpha) - digamma(alpha.sum()))).sum()
        
        # Entropy of responsibilities
        entropy = -(r * np.log(r + 1e-10)).sum()
        
        return recon - kl + entropy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict component assignments"""
        n_samples = len(X)
        r = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            E_log_pi_k = digamma(self.alpha[k]) - digamma(self.alpha.sum())
            E_tau_k = self.a[k] / self.b[k]
            E_mu_k = self.m[k]
            
            log_prob = (E_log_pi_k - 
                       0.5 * E_tau_k * (X - E_mu_k)**2)
            r[:, k] = np.exp(log_prob)
        
        r = r / r.sum(axis=1, keepdims=True)
        return r.argmax(axis=1)


# ============================================================================
# Example 2: Stochastic Variational Inference for Latent Dirichlet Allocation
# ============================================================================

class StochasticVI_LDA:
    """
    Stochastic Variational Inference for Latent Dirichlet Allocation
    
    This implements the SVI algorithm from Hoffman et al. (2013)
    """
    
    def __init__(self, n_topics: int, vocab_size: int, 
                 alpha: float = 0.1, eta: float = 0.01,
                 batch_size: int = 100, tau: float = 1.0, kappa: float = 0.7):
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.eta = eta
        self.batch_size = batch_size
        self.tau = tau
        self.kappa = kappa
        
        # Global parameters: lambda (topic-word distributions)
        # Initialize with prior
        self.lambda_ = np.random.gamma(100, 1/100, (n_topics, vocab_size))
        
        # Track iteration for step size
        self.iteration = 0
        
    def fit(self, documents: List[List[int]], n_iter: int = 100):
        """
        Fit the model using Stochastic Variational Inference
        
        Parameters:
        -----------
        documents : list of lists
            Each document is a list of word indices
        n_iter : int
            Number of iterations
        """
        n_docs = len(documents)
        
        for iteration in range(n_iter):
            # Sample a mini-batch
            batch_indices = np.random.choice(n_docs, size=min(self.batch_size, n_docs), 
                                            replace=False)
            batch = [documents[i] for i in batch_indices]
            
            # Update using mini-batch
            self.update_minibatch(batch)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{n_iter}")
    
    def update_minibatch(self, batch: List[List[int]]):
        """Update global parameters using a mini-batch"""
        n_docs = len(batch)
        
        # Compute E[log beta] (expected log topic-word probabilities)
        E_log_beta = self._compute_E_log_beta()
        
        # Sufficient statistics accumulator
        lambda_update = np.zeros_like(self.lambda_)
        
        # Process each document in the batch
        for doc in batch:
            # Optimize local parameters (document-topic distribution)
            gamma = self._optimize_local_params(doc, E_log_beta)
            
            # Compute document-level sufficient statistics
            phi = self._compute_phi(doc, gamma, E_log_beta)
            
            # Accumulate statistics
            for word_id in doc:
                lambda_update[:, word_id] += phi[:, word_id]
        
        # Compute step size
        self.iteration += 1
        step_size = (self.iteration + self.tau) ** (-self.kappa)
        
        # Update global parameters using natural gradient
        # Natural gradient = prior + data - current
        lambda_prior = self.eta * np.ones((self.n_topics, self.vocab_size))
        lambda_data = (len(batch) / self.batch_size) * lambda_update
        
        natural_grad = lambda_prior + lambda_data - self.lambda_
        
        # Update
        self.lambda_ = self.lambda_ + step_size * natural_grad
        
        # Ensure positivity
        self.lambda_ = np.maximum(self.lambda_, 1e-10)
    
    def _compute_E_log_beta(self) -> np.ndarray:
        """Compute E[log beta] where beta ~ Dirichlet(lambda)"""
        lambda_sum = self.lambda_.sum(axis=1, keepdims=True)
        return digamma(self.lambda_) - digamma(lambda_sum)
    
    def _optimize_local_params(self, doc: List[int], E_log_beta: np.ndarray) -> np.ndarray:
        """
        Optimize local variational parameters gamma (document-topic distribution)
        using coordinate ascent
        """
        # Initialize gamma
        gamma = np.ones(self.n_topics) * self.alpha
        
        # Fixed-point iteration
        for _ in range(10):
            E_log_theta = digamma(gamma) - digamma(gamma.sum())
            
            # Update gamma
            gamma_new = self.alpha.copy()
            for word_id in doc:
                gamma_new += np.exp(E_log_theta + E_log_beta[:, word_id])
            
            gamma = gamma_new
        
        return gamma
    
    def _compute_phi(self, doc: List[int], gamma: np.ndarray, 
                    E_log_beta: np.ndarray) -> np.ndarray:
        """
        Compute word-topic assignment probabilities phi
        """
        phi = np.zeros((self.n_topics, self.vocab_size))
        
        E_log_theta = digamma(gamma) - digamma(gamma.sum())
        
        for word_id in doc:
            phi[:, word_id] = np.exp(E_log_theta + E_log_beta[:, word_id])
            phi[:, word_id] /= phi[:, word_id].sum()
        
        return phi
    
    def get_topics(self, top_n: int = 10) -> List[List[int]]:
        """Get top words for each topic"""
        topics = []
        for k in range(self.n_topics):
            top_words = np.argsort(self.lambda_[k, :])[-top_n:][::-1]
            topics.append(top_words.tolist())
        return topics


# ============================================================================
# Example 3: Simple Variational Inference for Bayesian Linear Regression
# ============================================================================

class VariationalBayesianRegression:
    """
    Variational Inference for Bayesian Linear Regression
    
    Approximates the posterior over weights using a Gaussian distribution
    """
    
    def __init__(self, n_features: int, alpha_0: float = 1.0, beta_0: float = 1.0):
        self.n_features = n_features
        self.alpha_0 = alpha_0  # Prior precision for weights
        self.beta_0 = beta_0    # Prior precision for noise
        
        # Variational parameters
        # q(w) ~ N(m, S)
        self.m = np.zeros(n_features)
        self.S = np.eye(n_features)
        
        # q(alpha) ~ Gamma(a_alpha, b_alpha) where alpha is weight precision
        self.a_alpha = alpha_0
        self.b_alpha = alpha_0
        
        # q(beta) ~ Gamma(a_beta, b_beta) where beta is noise precision
        self.a_beta = beta_0
        self.b_beta = beta_0
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_iter: int = 50):
        """
        Fit the model using Variational Inference
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Target values
        """
        n_samples = n_features = X.shape[1]
        
        for iteration in range(n_iter):
            # Update q(w)
            E_alpha = self.a_alpha / self.b_alpha
            E_beta = self.a_beta / self.b_beta
            
            self.S = np.linalg.inv(E_alpha * np.eye(n_features) + E_beta * X.T @ X)
            self.m = E_beta * self.S @ X.T @ y
            
            # Update q(alpha)
            self.a_alpha = self.alpha_0 + 0.5 * n_features
            self.b_alpha = (self.alpha_0 + 
                           0.5 * (self.m.T @ self.m + np.trace(self.S)))
            
            # Update q(beta)
            self.a_beta = self.beta_0 + 0.5 * n_samples
            y_pred = X @ self.m
            self.b_beta = (self.beta_0 + 
                          0.5 * ((y - y_pred)**2).sum() + 
                          0.5 * np.trace(X.T @ X @ self.S))
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Make predictions
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        return_std : bool
            Whether to return predictive standard deviation
        
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted means
        y_std : array-like, shape (n_samples,), optional
            Predictive standard deviations
        """
        y_pred = X @ self.m
        
        if return_std:
            E_beta = self.a_beta / self.b_beta
            y_var = 1.0 / E_beta + np.diag(X @ self.S @ X.T)
            y_std = np.sqrt(y_var)
            return y_pred, y_std
        
        return y_pred


# ============================================================================
# Example Usage and Visualization
# ============================================================================

def example_gmm():
    """Example: Variational Inference for Gaussian Mixture Model"""
    print("=" * 60)
    print("Example 1: Variational Inference for GMM")
    print("=" * 60)
    
    # Generate synthetic data from a mixture of 3 Gaussians
    np.random.seed(42)
    n_samples = 200
    true_means = [-2, 0, 2]
    true_stds = [0.5, 0.8, 0.6]
    true_weights = [0.3, 0.4, 0.3]
    
    data = []
    for mean, std, weight in zip(true_means, true_stds, true_weights):
        n = int(n_samples * weight)
        data.append(np.random.normal(mean, std, n))
    data = np.concatenate(data)
    np.random.shuffle(data)
    
    # Fit model
    model = VariationalGMM(n_components=3, max_iter=100)
    model.fit(data)
    
    # Predict
    assignments = model.predict(data)
    
    print(f"\nLearned means: {model.m}")
    print(f"Learned component weights: {model.alpha / model.alpha.sum()}")
    print(f"\nTrue means: {true_means}")
    print(f"True weights: {true_weights}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.5, label='Data', density=True)
    x = np.linspace(data.min(), data.max(), 100)
    for k in range(model.n_components):
        mean_k = model.m[k]
        std_k = np.sqrt(model.b[k] / model.a[k])
        weight_k = model.alpha[k] / model.alpha.sum()
        plt.plot(x, weight_k * stats.norm.pdf(x, mean_k, std_k), 
                label=f'Component {k+1}', linewidth=2)
    plt.legend()
    plt.title('Variational Inference for Gaussian Mixture Model')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.savefig('gmm_vi_example.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to 'gmm_vi_example.png'")
    plt.close()


def example_svi_lda():
    """Example: Stochastic Variational Inference for LDA"""
    print("\n" + "=" * 60)
    print("Example 2: Stochastic Variational Inference for LDA")
    print("=" * 60)
    
    # Generate synthetic documents
    np.random.seed(42)
    n_docs = 1000
    doc_length = 50
    vocab_size = 100
    n_topics = 5
    
    # Create documents with random word assignments
    documents = []
    for _ in range(n_docs):
        doc = np.random.randint(0, vocab_size, size=doc_length).tolist()
        documents.append(doc)
    
    # Fit model
    model = StochasticVI_LDA(n_topics=n_topics, vocab_size=vocab_size,
                            batch_size=100, n_iter=50)
    model.fit(documents, n_iter=50)
    
    # Get topics
    topics = model.get_topics(top_n=5)
    print(f"\nTop 5 words for each of {n_topics} topics:")
    for k, topic in enumerate(topics):
        print(f"Topic {k+1}: {topic}")


def example_bayesian_regression():
    """Example: Variational Inference for Bayesian Linear Regression"""
    print("\n" + "=" * 60)
    print("Example 3: Variational Inference for Bayesian Regression")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.0, -0.5, 0.3, 0.8, -0.2])
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Fit model
    model = VariationalBayesianRegression(n_features=n_features)
    model.fit(X, y, n_iter=50)
    
    # Predict
    y_pred, y_std = model.predict(X, return_std=True)
    
    print(f"\nTrue weights: {true_weights}")
    print(f"Learned weights (mean): {model.m}")
    print(f"Learned weights (std): {np.sqrt(np.diag(model.S))}")
    print(f"\nRMSE: {np.sqrt(((y - y_pred)**2).mean()):.4f}")
    
    # Visualize predictions with uncertainty
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.errorbar(y, y_pred, yerr=y_std, fmt='none', alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect prediction')
    plt.xlabel('True y')
    plt.ylabel('Predicted y')
    plt.title('Bayesian Regression Predictions with Uncertainty')
    plt.legend()
    plt.savefig('bayesian_regression_vi_example.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to 'bayesian_regression_vi_example.png'")
    plt.close()


if __name__ == "__main__":
    # Run examples
    try:
        example_gmm()
    except Exception as e:
        print(f"Error in GMM example: {e}")
    
    try:
        example_svi_lda()
    except Exception as e:
        print(f"Error in SVI-LDA example: {e}")
    
    try:
        example_bayesian_regression()
    except Exception as e:
        print(f"Error in Bayesian Regression example: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

