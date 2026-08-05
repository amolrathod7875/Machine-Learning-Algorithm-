import matplotlib.pyplot as plt
import pandas as pd

def plot_pca_results(results_df, mse_baseline, r2_baseline):
    """
    Create a 2x2 subplot visualization for PCA analysis results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Must contain columns: n_components, variance_explained, mse, r2_score
    mse_baseline : float
        Baseline MSE without PCA
    r2_baseline : float
        Baseline R² score without PCA
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.plot(results_df['n_components'], results_df['mse'], 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=mse_baseline, color='r', linestyle='--', label=f'Baseline MSE: {mse_baseline:.4f}')
    ax1.set_xlabel('Number of PCA Components')
    ax1.set_ylabel('MSE (Mean Squared Error)')
    ax1.set_title('Model Performance: MSE vs Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(results_df['n_components'], results_df['r2_score'], 'g-o', linewidth=2, markersize=6)
    ax2.axhline(y=r2_baseline, color='r', linestyle='--', label=f'Baseline R²: {r2_baseline:.4f}')
    ax2.set_xlabel('Number of PCA Components')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Model Performance: R² vs Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.bar(results_df['n_components'], results_df['variance_explained'], color='steelblue', alpha=0.7, edgecolor='white')
    ax3.set_xlabel('Number of PCA Components')
    ax3.set_ylabel('Explained Variance Ratio')
    ax3.set_title('Individual Variance Explained per Component')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(results_df['n_components'], results_df['variance_explained'], 'purple', marker='o', linewidth=2, markersize=6)
    ax4.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% variance')
    ax4.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% variance')
    ax4.set_xlabel('Number of PCA Components')
    ax4.set_ylabel('Cumulative Variance Explained')
    ax4.set_title('Information Retention: Variance Explained')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig