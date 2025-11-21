"""
Generate visualizations and diagrams for Variational Inference and SVI

This script creates various diagrams and plots to illustrate key concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
from matplotlib.patches import Rectangle, Arrow
import matplotlib.patheffects as path_effects

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


# ============================================================================
# Figure 1: VI Concept - Approximating Complex Distribution
# ============================================================================

def create_vi_concept_diagram():
    """Create diagram showing VI approximating complex distribution"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    x = np.linspace(-4, 4, 1000)
    
    # True complex distribution (mixture of Gaussians)
    true_dist = (0.3 * np.exp(-0.5 * ((x + 1.5) / 0.6)**2) / (0.6 * np.sqrt(2 * np.pi)) +
                 0.5 * np.exp(-0.5 * ((x - 0.5) / 0.8)**2) / (0.8 * np.sqrt(2 * np.pi)) +
                 0.2 * np.exp(-0.5 * ((x - 2) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi)))
    
    # Approximate simple distribution (single Gaussian)
    approx_dist = np.exp(-0.5 * ((x - 0.3) / 1.2)**2) / (1.2 * np.sqrt(2 * np.pi))
    
    # Plot
    ax.plot(x, true_dist, 'b-', linewidth=3, label='True Distribution $p(z|x)$', alpha=0.8)
    ax.plot(x, approx_dist, 'r--', linewidth=3, label='Approximate Distribution $q(z)$', alpha=0.8)
    ax.fill_between(x, true_dist, alpha=0.2, color='blue')
    ax.fill_between(x, approx_dist, alpha=0.2, color='red')
    
    # Add annotations
    ax.annotate('Complex, intractable', xy=(-1.5, 0.25), xytext=(-3, 0.35),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=14, color='blue', weight='bold')
    
    ax.annotate('Simple, tractable', xy=(0.3, 0.3), xytext=(2, 0.35),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=14, color='red', weight='bold')
    
    ax.annotate('KL Divergence\nmeasures distance', xy=(1, 0.15), xytext=(2.5, 0.1),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, color='green',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('$z$ (latent variable)', fontsize=16)
    ax.set_ylabel('Probability Density', fontsize=16)
    ax.set_title('Variational Inference: Approximating Complex Distributions', 
                 fontsize=18, weight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.4)
    
    plt.tight_layout()
    plt.savefig('images/vi_concept.png', dpi=300, bbox_inches='tight')
    print("Created: images/vi_concept.png")
    plt.close()


# ============================================================================
# Figure 2: VI Algorithm Flowchart
# ============================================================================

def create_vi_algorithm_flowchart():
    """Create flowchart showing VI algorithm"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define box style
    box_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                     edgecolor='navy', linewidth=2)
    decision_style = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                          edgecolor='orange', linewidth=2)
    
    # Start
    start_box = FancyBboxPatch((4, 11), 2, 0.8, boxstyle='round,pad=0.3',
                               facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(start_box)
    ax.text(5, 11.4, 'Start', ha='center', va='center', fontsize=14, weight='bold')
    
    # Choose variational family
    box1 = FancyBboxPatch((3, 9.5), 4, 0.8, boxstyle='round,pad=0.5',
                          facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax.add_patch(box1)
    ax.text(5, 9.9, 'Choose variational family $Q$', ha='center', va='center', 
            fontsize=13, weight='bold')
    
    # Initialize
    box2 = FancyBboxPatch((3, 8), 4, 0.8, boxstyle='round,pad=0.5',
                          facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax.add_patch(box2)
    ax.text(5, 8.4, 'Initialize $q(z)$', ha='center', va='center', 
            fontsize=13, weight='bold')
    
    # Compute ELBO
    box3 = FancyBboxPatch((3, 6.5), 4, 0.8, boxstyle='round,pad=0.5',
                          facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax.add_patch(box3)
    ax.text(5, 6.9, 'Compute ELBO', ha='center', va='center', 
            fontsize=13, weight='bold')
    
    # Optimize
    box4 = FancyBboxPatch((3, 5), 4, 0.8, boxstyle='round,pad=0.5',
                          facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax.add_patch(box4)
    ax.text(5, 5.4, 'Optimize $q(z)$ to maximize ELBO', ha='center', va='center', 
            fontsize=13, weight='bold')
    
    # Decision
    decision = FancyBboxPatch((3, 3.2), 4, 0.8, boxstyle='round,pad=0.5',
                              facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(decision)
    ax.text(5, 3.6, 'Converged?', ha='center', va='center', 
            fontsize=13, weight='bold')
    
    # End
    end_box = FancyBboxPatch((3, 1.5), 4, 0.8, boxstyle='round,pad=0.3',
                             facecolor='lightcoral', edgecolor='darkred', linewidth=2)
    ax.add_patch(end_box)
    ax.text(5, 1.9, 'Return $q^*(z)$', ha='center', va='center', 
            fontsize=13, weight='bold')
    
    # Arrows
    arrows = [
        ((5, 11), (5, 10.3)),
        ((5, 9.5), (5, 8.8)),
        ((5, 8), (5, 7.3)),
        ((5, 6.5), (5, 5.8)),
        ((5, 5), (5, 4)),
        ((5, 3.2), (5, 2.3)),  # Yes path
        ((3, 3.6), (1.5, 3.6)),  # No path (left)
        ((1.5, 3.6), (1.5, 6.5)),  # No path (up)
        ((1.5, 6.5), (3, 6.5)),  # No path (right)
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', lw=2, color='black',
                               mutation_scale=20)
        ax.add_patch(arrow)
    
    # Labels
    ax.text(0.5, 5, 'No', ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(5, 2.75, 'Yes', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Variational Inference Algorithm Flowchart', 
                 fontsize=18, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('images/vi_algorithm_flowchart.png', dpi=300, bbox_inches='tight')
    print("Created: images/vi_algorithm_flowchart.png")
    plt.close()


# ============================================================================
# Figure 3: VI vs SVI Comparison
# ============================================================================

def create_vi_vs_svi_comparison():
    """Create comparison diagram between VI and SVI"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # VI Diagram
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Variational Inference (VI)', fontsize=16, weight='bold', pad=20)
    
    # Data box (full dataset)
    data_box = FancyBboxPatch((2, 7), 6, 1.5, boxstyle='round,pad=0.5',
                              facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax1.add_patch(data_box)
    ax1.text(5, 7.75, 'Full Dataset\n(All N samples)', ha='center', va='center',
             fontsize=12, weight='bold')
    
    # Arrow
    arrow1 = FancyArrowPatch((5, 7), (5, 5.5), arrowstyle='->', lw=3, color='black')
    ax1.add_patch(arrow1)
    
    # Process box
    process_box = FancyBboxPatch((2, 3.5), 6, 1.5, boxstyle='round,pad=0.5',
                                 facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax1.add_patch(process_box)
    ax1.text(5, 4.25, 'Process All Data\nUpdate Parameters', ha='center', va='center',
             fontsize=12, weight='bold')
    
    # Arrow
    arrow2 = FancyArrowPatch((5, 3.5), (5, 2), arrowstyle='->', lw=3, color='black')
    ax1.add_patch(arrow2)
    
    # Result
    result_box = FancyBboxPatch((2, 0.5), 6, 1, boxstyle='round,pad=0.5',
                                facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax1.add_patch(result_box)
    ax1.text(5, 1, 'Updated $q(z)$', ha='center', va='center',
             fontsize=12, weight='bold')
    
    # Characteristics
    ax1.text(5, 9.5, '• Processes all data each iteration\n• Memory: O(N)\n• Time: O(N) per iteration',
             ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # SVI Diagram
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Stochastic Variational Inference (SVI)', fontsize=16, weight='bold', pad=20)
    
    # Data boxes (mini-batches)
    batch1 = FancyBboxPatch((1, 7), 2.5, 1.5, boxstyle='round,pad=0.5',
                            facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax2.add_patch(batch1)
    ax2.text(2.25, 7.75, 'Mini-batch 1\n(B samples)', ha='center', va='center',
             fontsize=11, weight='bold')
    
    batch2 = FancyBboxPatch((4, 7), 2.5, 1.5, boxstyle='round,pad=0.5',
                            facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax2.add_patch(batch2)
    ax2.text(5.25, 7.75, 'Mini-batch 2\n(B samples)', ha='center', va='center',
             fontsize=11, weight='bold')
    
    batch3 = FancyBboxPatch((7, 7), 2.5, 1.5, boxstyle='round,pad=0.5',
                            facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax2.add_patch(batch3)
    ax2.text(8.25, 7.75, '...', ha='center', va='center',
             fontsize=11, weight='bold')
    
    # Arrow
    arrow3 = FancyArrowPatch((5, 7), (5, 5.5), arrowstyle='->', lw=3, color='black')
    ax2.add_patch(arrow3)
    
    # Process box
    process_box2 = FancyBboxPatch((2, 3.5), 6, 1.5, boxstyle='round,pad=0.5',
                                  facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax2.add_patch(process_box2)
    ax2.text(5, 4.25, 'Process Mini-batch\nStochastic Update', ha='center', va='center',
             fontsize=12, weight='bold')
    
    # Arrow
    arrow4 = FancyArrowPatch((5, 3.5), (5, 2), arrowstyle='->', lw=3, color='black')
    ax2.add_patch(arrow4)
    
    # Result
    result_box2 = FancyBboxPatch((2, 0.5), 6, 1, boxstyle='round,pad=0.5',
                                 facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax2.add_patch(result_box2)
    ax2.text(5, 1, 'Incrementally Updated $q(z)$', ha='center', va='center',
             fontsize=12, weight='bold')
    
    # Characteristics
    ax2.text(5, 9.5, '• Processes mini-batches\n• Memory: O(B)\n• Time: O(B) per iteration\n• Scalable to millions',
             ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('images/vi_vs_svi_comparison.png', dpi=300, bbox_inches='tight')
    print("Created: images/vi_vs_svi_comparison.png")
    plt.close()


# ============================================================================
# Figure 4: ELBO Convergence
# ============================================================================

def create_elbo_convergence():
    """Create plot showing ELBO convergence"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Simulate ELBO convergence
    np.random.seed(42)
    iterations = np.arange(1, 101)
    
    # VI convergence (smooth, deterministic)
    elbo_vi = -100 + 80 * (1 - np.exp(-iterations / 20)) + np.random.normal(0, 0.5, 100)
    elbo_vi = np.maximum.accumulate(elbo_vi)  # Monotonic increase
    
    # SVI convergence (noisy but trending up)
    elbo_svi = -100 + 75 * (1 - np.exp(-iterations / 25)) + np.random.normal(0, 2, 100)
    # Make it generally increasing but with noise
    for i in range(1, len(elbo_svi)):
        elbo_svi[i] = max(elbo_svi[i-1] - 1, elbo_svi[i])
    
    ax.plot(iterations, elbo_vi, 'b-', linewidth=2.5, label='VI (Deterministic)', alpha=0.8)
    ax.plot(iterations, elbo_svi, 'r--', linewidth=2.5, label='SVI (Stochastic)', alpha=0.8)
    
    # Add smoothed SVI line
    from scipy.ndimage import uniform_filter1d
    elbo_svi_smooth = uniform_filter1d(elbo_svi, size=5)
    ax.plot(iterations, elbo_svi_smooth, 'r:', linewidth=2, label='SVI (Smoothed)', alpha=0.6)
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('ELBO Value', fontsize=14)
    ax.set_title('ELBO Convergence: VI vs SVI', fontsize=16, weight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('VI: Smooth, deterministic\nconvergence', 
                xy=(30, elbo_vi[29]), xytext=(50, elbo_vi[29] + 10),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=11, color='blue',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.annotate('SVI: Noisy but trending\nupward (faster)', 
                xy=(40, elbo_svi[39]), xytext=(60, elbo_svi[39] - 15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('images/elbo_convergence.png', dpi=300, bbox_inches='tight')
    print("Created: images/elbo_convergence.png")
    plt.close()


# ============================================================================
# Figure 5: Mean Field Approximation
# ============================================================================

def create_mean_field_diagram():
    """Create diagram explaining mean field approximation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # True posterior (correlated)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    
    # Generate correlated data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]
    x, y = np.random.multivariate_normal(mean, cov, 1000).T
    
    # Plot
    ax1.scatter(x, y, alpha=0.3, s=10, color='blue')
    
    # Add contour
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    pos = np.dstack((xx, yy))
    from scipy.stats import multivariate_normal
    rv = multivariate_normal(mean, cov)
    ax1.contour(xx, yy, rv.pdf(pos), levels=5, colors='darkblue', linewidths=2, alpha=0.7)
    
    ax1.set_xlabel('$z_1$', fontsize=14)
    ax1.set_ylabel('$z_2$', fontsize=14)
    ax1.set_title('True Posterior $p(z_1, z_2|x)$\n(Correlated)', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Mean field approximation (independent)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    
    # Generate independent data
    x_ind = np.random.normal(0, 1, 1000)
    y_ind = np.random.normal(0, 1, 1000)
    
    ax2.scatter(x_ind, y_ind, alpha=0.3, s=10, color='red')
    
    # Add contour
    rv_ind = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    ax2.contour(xx, yy, rv_ind.pdf(pos), levels=5, colors='darkred', linewidths=2, alpha=0.7)
    
    ax2.set_xlabel('$z_1$', fontsize=14)
    ax2.set_ylabel('$z_2$', fontsize=14)
    ax2.set_title('Mean Field Approximation $q(z_1)q(z_2)$\n(Independent)', 
                  fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add formula
    fig.text(0.5, 0.02, 'Mean Field: $q(z) = \\prod_{i=1}^{n} q_i(z_i)$ (assumes independence)',
             ha='center', fontsize=14, weight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('images/mean_field_approximation.png', dpi=300, bbox_inches='tight')
    print("Created: images/mean_field_approximation.png")
    plt.close()


# ============================================================================
# Figure 6: Applications Overview
# ============================================================================

def create_applications_diagram():
    """Create diagram showing various applications"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Applications of Variational Inference', fontsize=18, weight='bold', pad=20)
    
    applications = [
        {'name': 'Topic Modeling\n(LDA)', 'pos': (2, 8.5), 'color': 'lightblue'},
        {'name': 'Variational\nAutoencoders', 'pos': (8, 8.5), 'color': 'lightgreen'},
        {'name': 'Bayesian\nNeural Networks', 'pos': (2, 6), 'color': 'lightyellow'},
        {'name': 'Gaussian\nProcesses', 'pos': (8, 6), 'color': 'lightcoral'},
        {'name': 'Recommendation\nSystems', 'pos': (5, 3.5), 'color': 'lavender'},
    ]
    
    # Center VI/SVI box
    center_box = FancyBboxPatch((3.5, 4.5), 3, 1.5, boxstyle='round,pad=0.5',
                                facecolor='gold', edgecolor='darkorange', linewidth=3)
    ax.add_patch(center_box)
    ax.text(5, 5.25, 'VI / SVI', ha='center', va='center',
            fontsize=16, weight='bold')
    
    # Application boxes
    for app in applications:
        box = FancyBboxPatch((app['pos'][0]-1, app['pos'][1]-0.75), 2, 1.5,
                            boxstyle='round,pad=0.5', facecolor=app['color'],
                            edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(app['pos'][0], app['pos'][1], app['name'], ha='center', va='center',
                fontsize=11, weight='bold')
        
        # Arrow to center
        arrow = FancyArrowPatch(app['pos'], (5, 5.25),
                               arrowstyle='->', lw=2, color='gray', alpha=0.6,
                               connectionstyle='arc3,rad=0.2')
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('images/applications_overview.png', dpi=300, bbox_inches='tight')
    print("Created: images/applications_overview.png")
    plt.close()


# ============================================================================
# Figure 7: KL Divergence Visualization
# ============================================================================

def create_kl_divergence_visualization():
    """Create visualization of KL divergence"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.linspace(-3, 3, 1000)
    
    # Case 1: Good approximation
    p1 = np.exp(-0.5 * ((x - 0) / 1)**2) / (1 * np.sqrt(2 * np.pi))
    q1 = np.exp(-0.5 * ((x - 0.2) / 1.1)**2) / (1.1 * np.sqrt(2 * np.pi))
    
    axes[0, 0].plot(x, p1, 'b-', linewidth=2.5, label='$p(z)$', alpha=0.8)
    axes[0, 0].plot(x, q1, 'r--', linewidth=2.5, label='$q(z)$', alpha=0.8)
    axes[0, 0].fill_between(x, np.minimum(p1, q1), alpha=0.3, color='green')
    axes[0, 0].set_title('Small KL Divergence\n(Good Approximation)', fontsize=12, weight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.02, 0.95, 'KL ≈ 0.02', transform=axes[0, 0].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Case 2: Poor approximation
    p2 = np.exp(-0.5 * ((x - 0) / 1)**2) / (1 * np.sqrt(2 * np.pi))
    q2 = np.exp(-0.5 * ((x - 1.5) / 0.8)**2) / (0.8 * np.sqrt(2 * np.pi))
    
    axes[0, 1].plot(x, p2, 'b-', linewidth=2.5, label='$p(z)$', alpha=0.8)
    axes[0, 1].plot(x, q2, 'r--', linewidth=2.5, label='$q(z)$', alpha=0.8)
    axes[0, 1].fill_between(x, np.minimum(p2, q2), alpha=0.3, color='red')
    axes[0, 1].set_title('Large KL Divergence\n(Poor Approximation)', fontsize=12, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.02, 0.95, 'KL ≈ 1.8', transform=axes[0, 1].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Case 3: Mode-seeking behavior
    p3 = (0.5 * np.exp(-0.5 * ((x + 1) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi)) +
          0.5 * np.exp(-0.5 * ((x - 1) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi)))
    q3 = np.exp(-0.5 * ((x - 1) / 0.6)**2) / (0.6 * np.sqrt(2 * np.pi))
    
    axes[1, 0].plot(x, p3, 'b-', linewidth=2.5, label='$p(z)$ (bimodal)', alpha=0.8)
    axes[1, 0].plot(x, q3, 'r--', linewidth=2.5, label='$q(z)$ (unimodal)', alpha=0.8)
    axes[1, 0].fill_between(x, np.minimum(p3, q3), alpha=0.3, color='orange')
    axes[1, 0].set_title('KL$(q||p)$: Mode-Seeking\n(Captures one mode)', fontsize=12, weight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Case 4: Mean-seeking behavior (reverse KL)
    p4 = np.exp(-0.5 * ((x - 0) / 1)**2) / (1 * np.sqrt(2 * np.pi))
    q4 = (0.5 * np.exp(-0.5 * ((x + 0.5) / 0.4)**2) / (0.4 * np.sqrt(2 * np.pi)) +
          0.5 * np.exp(-0.5 * ((x - 0.5) / 0.4)**2) / (0.4 * np.sqrt(2 * np.pi)))
    
    axes[1, 1].plot(x, p4, 'b-', linewidth=2.5, label='$p(z)$ (unimodal)', alpha=0.8)
    axes[1, 1].plot(x, q4, 'r--', linewidth=2.5, label='$q(z)$ (bimodal)', alpha=0.8)
    axes[1, 1].fill_between(x, np.minimum(p4, q4), alpha=0.3, color='purple')
    axes[1, 1].set_title('KL$(p||q)$: Mean-Seeking\n(Covers all modes)', fontsize=12, weight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Kullback-Leibler Divergence: Different Behaviors', 
                 fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('images/kl_divergence_visualization.png', dpi=300, bbox_inches='tight')
    print("Created: images/kl_divergence_visualization.png")
    plt.close()


if __name__ == "__main__":
    import os
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    print("Generating visualizations...")
    print("=" * 60)
    
    create_vi_concept_diagram()
    create_vi_algorithm_flowchart()
    create_vi_vs_svi_comparison()
    create_elbo_convergence()
    create_mean_field_diagram()
    create_applications_diagram()
    create_kl_divergence_visualization()
    
    print("=" * 60)
    print("All visualizations created successfully!")
    print("Images saved in the 'images/' directory")

