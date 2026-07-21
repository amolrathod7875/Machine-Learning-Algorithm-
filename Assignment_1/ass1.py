import numpy as np
import pandas as pd
#import matplotlib.subplots as plt
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sk

# 1. Load Data
df = pd.read_csv("HousingData.csv")

# ============================================================
# NEW: DATA ANALYSIS SECTION
# ============================================================
print("\n" + "="*60)
print("DATA ANALYSIS (EDA)")
print("="*60)

print(f"Dataset Total Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("1. NULL VALUES:")
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("✅ No null values found in any column.")
else:
    print("⚠️ Null values detected:")
    print(null_counts[null_counts > 0])

print("\n2. UNIQUE VALUES PER COLUMN:")
print(df.nunique())

print("\n3. DATA TYPES:")
print(df.dtypes.value_counts())

print("\n4. BASIC STATISTICS (Mean, Std, Min, Max):")
print(df.describe().T[['mean', 'std', 'min', 'max']])

print("\n" + "="*60)
# ============================================================

# 2. Prepare Data
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

print(f"\nFeatures: {list(X.columns)}")
print(f"Target: medv (median home value in $1000s)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train_scaled.shape}")
print(f"Test size: {X_test_scaled.shape}")

# 3. Baseline Model
print("\n" + "="*60)
print("BASELINE: Training on ORIGINAL features (13 features)")
print("="*60)

model_baseline = LinearRegression()

start_time = time.time()
model_baseline.fit(X_train_scaled, y_train)
train_time_baseline = time.time() - start_time

y_pred_baseline = model_baseline.predict(X_test_scaled)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

print(f"Training time: {train_time_baseline:.6f} seconds")
print(f"MSE: {mse_baseline:.4f}")
print(f"R² Score: {r2_baseline:.4f}")

# 4. PCA Experiments
print("\n" + "="*60)
print("PCA EXPERIMENTS: Varying number of components")
print("="*60)

n_components_list = list(range(1, X_train_scaled.shape[1] + 1))
results = []

for n in n_components_list:
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    model_pca = LinearRegression()
    
    start_time = time.time()
    model_pca.fit(X_train_pca, y_train)
    train_time_pca = time.time() - start_time
    
    y_pred_pca = model_pca.predict(X_test_pca)
    mse_pca = mean_squared_error(y_test, y_pred_pca)
    r2_pca = r2_score(y_test, y_pred_pca)
    
    variance_explained = np.sum(pca.explained_variance_ratio_)
    
    speedup_val = train_time_baseline / train_time_pca if train_time_pca > 1e-9 else float('inf')
    
    results.append({
        'n_components': n,
        'variance_explained': variance_explained,
        'mse': mse_pca,
        'r2': r2_pca,
        'train_time': train_time_pca,
        'speedup': speedup_val
    })
    
    print(f"Components: {n:2d} | Var Explained: {variance_explained:.3f} | "
          f"MSE: {mse_pca:.4f} | R²: {r2_pca:.4f} | "
          f"Time: {train_time_pca:.6f}s | Speedup: {speedup_val:.2f}x")

results_df = pd.DataFrame(results)

# 5. Visualization
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
ax2.plot(results_df['n_components'], results_df['r2'], 'g-o', linewidth=2, markersize=6)
ax2.axhline(y=r2_baseline, color='r', linestyle='--', label=f'Baseline R²: {r2_baseline:.4f}')
ax2.set_xlabel('Number of PCA Components')
ax2.set_ylabel('R² Score')
ax2.set_title('Model Performance: R² vs Components')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.plot(results_df['n_components'], results_df['train_time'], 'orange', marker='o', linewidth=2, markersize=6)
ax3.axhline(y=train_time_baseline, color='r', linestyle='--', label=f'Baseline: {train_time_baseline:.6f}s')
ax3.set_xlabel('Number of PCA Components')
ax3.set_ylabel('Training Time (seconds)')
ax3.set_title('Computational Efficiency: Training Time vs Components')
ax3.legend()
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
plt.show()

# 6. Summary and Findings
print("\n" + "="*60)
print("SUMMARY: Comparison of All Configurations")
print("="*60)
summary = results_df[['n_components', 'variance_explained', 'mse', 'r2', 'train_time', 'speedup']].copy()
summary.columns = ['Components', 'Var Explained', 'MSE', 'R²', 'Time (s)', 'Speedup']
print(summary.to_string(index=False))

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

best_idx = results_df['mse'].idxmin()
best = results_df.loc[best_idx]

print(f"\n1. BASELINE (No PCA):")
print(f"   - Features: 13")
print(f"   - MSE: {mse_baseline:.4f}")
print(f"   - R²: {r2_baseline:.4f}")
print(f"   - Training time: {train_time_baseline:.6f}s")

print(f"\n2. BEST PCA CONFIGURATION:")
print(f"   - Components: {int(best['n_components'])}")
print(f"   - Variance explained: {best['variance_explained']:.3f}")
print(f"   - MSE: {best['mse']:.4f}")
print(f"   - R²: {best['r2']:.4f}")
print(f"   - Training time: {best['train_time']:.6f}s")
print(f"   - Speedup: {best['speedup']:.2f}x")