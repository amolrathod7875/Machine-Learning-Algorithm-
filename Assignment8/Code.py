import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train_scaled, y_train)
y_pred_base = baseline.predict(X_test_scaled)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{name} Performance:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    return acc, prec, rec, f1

base_acc, base_prec, base_rec, base_f1 = evaluate("Baseline (Logistic Regression)", y_test, y_pred_base)
rf_acc, rf_prec, rf_rec, rf_f1 = evaluate("Random Forest Classifier", y_test, y_pred_rf)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_base = confusion_matrix(y_test, y_pred_base)
disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base, display_labels=data.target_names)
disp_base.plot(ax=axes[0], colorbar=False)
axes[0].set_title("Baseline: Logistic Regression")

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=data.target_names)
disp_rf.plot(ax=axes[1], colorbar=False)
axes[1].set_title("Random Forest Classifier")

plt.tight_layout()
plt.savefig("comparison_confusion_matrix.png", dpi=150)
print("\nSaved comparison_confusion_matrix.png")

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
base_scores = [base_acc, base_prec, base_rec, base_f1]
rf_scores = [rf_acc, rf_prec, rf_rec, rf_f1]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, base_scores, width, label="Logistic Regression")
plt.bar(x + width/2, rf_scores, width, label="Random Forest")
plt.xticks(x, metrics)
plt.ylim(0.9, 1.0)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("performance_comparison.png", dpi=150)
print("Saved performance_comparison.png")

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Feature Importances (Random Forest):")
print(importances.head(10).to_string())
