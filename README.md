# Machine Learning Algorithm — Assignments Workspace

A collection of eight self-contained machine learning assignments that demonstrate core techniques in dimensionality reduction, regression, classification, clustering, ensemble learning, and computer vision using `scikit-learn`, `pandas`, `matplotlib`, and `OpenCV`. Each assignment is independently documented and includes a detailed theoretical write-up, a flowchart, and a set of likely exam/faculty questions with answers.

## Table of Contents

- [Project Description](#project-description)
- [Assignments Overview](#assignments-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Architecture](#project-architecture)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [Dependencies](#dependencies)

## Project Description

This repository is a learning-oriented workspace that walks through foundational machine learning algorithms applied to well-known public datasets:

1. **PCA (Principal Component Analysis)** — Unsupervised dimensionality reduction on the Boston Housing dataset, studying the trade-off between feature reduction and model performance.
2. **LDA vs PCA** — A supervised/unsupervised comparison of Linear Discriminant Analysis against PCA on the same Boston Housing dataset.
3. **Regression Pipeline** — A full cab-price prediction pipeline comparing Linear, Ridge, and Lasso regression on Uber/Lyft ride data.
4. **Logistic Regression** — Binary classification of diabetes risk using the Pima Indians Diabetes dataset, evaluated with a confusion matrix, precision, recall, and F1-score.
5. **SVM Classification** — Multi-class digit recognition using a Support Vector Machine with RBF kernel on the scikit-learn Digits dataset.
6. **K-Means Clustering** — Unsupervised clustering on the Iris dataset using the Elbow Method to determine the optimal number of clusters.
7. **OpenCV + AI Inference** — Image reading and annotation with OpenCV, plus AI-powered face detection using YuNet via OpenCV's DNN module.
8. **Random Forest Classifier** — Ensemble classification on the Breast Cancer Wisconsin dataset, comparing Random Forest against a Logistic Regression baseline using accuracy, precision, recall, F1-score, and feature importance.

The assignments emphasize hands-on experimentation, clear theory, and reproducible results rather than production deployment.

## Assignments Overview

| Assignment | Topic | Primary Algorithm(s) | Dataset | Interface |
| --- | --- | --- | --- | --- |
| 1 | Dimensionality Reduction | PCA | Boston Housing (`HousingData.csv`) | Jupyter Notebook + `visualization.py` |
| 2 | PCA vs LDA | PCA, LDA | Boston Housing | Jupyter Notebook |
| 3 | Regression Pipeline | Linear, Ridge, Lasso | Uber/Lyft (`cab_rides.csv`, `weather.csv`) | Python script (`assignment3.py`) |
| 4 | Classification | Logistic Regression | Pima Indians Diabetes (`diabetes.csv`) | Jupyter Notebook |
| 5 | SVM Classification | SVM (RBF kernel) | Digits (built-in) | Python script (`Code.py`) |
| 6 | Clustering | K-Means, Elbow Method | Iris (built-in) | Python script (`Code.py`) |
| 7 | Computer Vision / AI Inference | OpenCV, YuNet Face Detector | `test_preview.jpg` + OpenCV Zoo model | Python script (`Code.py`) |
| 8 | Ensemble Classification | Random Forest vs Logistic Regression | Breast Cancer Wisconsin (built-in) | Python script (`Code.py`) |

## Prerequisites

- **Python** 3.8 or newer
- **pip** (Python package manager)
- **Git** (to clone the repository)
- A terminal / command prompt (PowerShell, Bash, or similar)
- For notebook-based assignments: **Jupyter Notebook** or **JupyterLab** (or Google Colab)

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Machine-Learning-Algorithm-
   ```

2. **(Recommended) Create and activate a virtual environment**

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirement.txt
   ```

   This installs: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `kaggle`.

   For the notebook-based assignments (1, 2, and 4) you will also need Jupyter:

   ```bash
   pip install jupyter
   ```

## Usage

Each assignment is run independently. Choose the appropriate command for the assignment you want to execute.

### Assignment 1 — PCA Dimensionality Reduction

```bash
jupyter notebook Assignment1/Assignment_1.ipynb
```

The notebook loads `Assignment1/HousingData.csv`, standardizes the features, trains a baseline Linear Regression model, applies PCA with varying component counts, and produces `Assignment1/output.png`. Visualization helpers live in `Assignment1/visualization.py`.

### Assignment 2 — LDA vs PCA

```bash
jupyter notebook Assignment2/Assignment_2.ipynb
```

Compares PCA (unsupervised) and LDA (supervised, using a discretized MEDV target) on the Boston Housing dataset, generating the `ass2.*.png` figures.

### Assignment 3 — Regression Pipeline

```bash
cd Assignment3
python assignment3.py
```

Trains and compares Linear, Ridge, and Lasso regression models on `cab_rides.csv` joined with `weather.csv`. Produces:

- `outliers_boxplot.png` — Boxplot of ride prices showing outliers
- `correlation_matrix.png` — Heatmap of numerical feature correlations

### Assignment 4 — Logistic Regression

Open `Assignment4/assignment4.ipynb` in Jupyter or Google Colab and run all cells sequentially. The notebook reads `Assignment4/diabetes.csv` and outputs the confusion matrix, precision, recall, and F1-score, plus a heatmap visualization (`output.png`).

> **Note:** Assignment 4's notebook references `google.colab` for Drive mounting. When running locally, ensure `diabetes.csv` is present in the working directory and remove or adapt the Colab-specific cells as needed.

### Assignment 5 — SVM Classification

```bash
python Assignment5/Code.py
```

Trains an SVM classifier with an RBF kernel on the built-in scikit-learn Digits dataset (1797 samples of 8x8 handwritten digits). Prints a classification report and accuracy score.

### Assignment 6 — K-Means Clustering

```bash
python Assignment6/Code.py
```

Applies K-Means clustering on the built-in Iris dataset, determines the optimal number of clusters using the Elbow Method and Silhouette Score, and produces `elbow_and_clusters.png` with the elbow curve, silhouette scores, and final cluster visualization.

### Assignment 7 — OpenCV Image Reading and AI Inference

```bash
python Assignment7/Code.py
```

Reads `Assignment7/test_preview.jpg` with OpenCV, downloads and loads the YuNet face detection model (`face_detection_yunet_2023mar.onnx`), runs inference using OpenCV's DNN module, and saves `Assignment7/output_with_faces.jpg` with detected faces annotated.

### Assignment 8 — Random Forest Classifier vs Baseline Model

```bash
python Assignment8/Code.py
```

Compares a Random Forest Classifier against a Logistic Regression baseline on the built-in Breast Cancer Wisconsin dataset. Prints accuracy, precision, recall, and F1-score for both models, and generates `comparison_confusion_matrix.png` and `performance_comparison.png`.

## Project Architecture

The workspace is organized as a flat collection of per-assignment folders, each self-contained with its own data, code, outputs, and documentation.

```
Machine-Learning-Algorithm-/
├── README.md                                         # This file — workspace-level documentation
├── requirement.txt                                   # Shared Python dependencies
├── .gitignore                                        # Ignored build/artifact files
├── Assignment1/                                      # PCA on Boston Housing
│   ├── Assignment_1.ipynb                            #   Main analysis notebook
│   ├── HousingData.csv                               #   Dataset
│   ├── visualization.py                              #   PCA plotting helpers
│   ├── output.png                                    #   Generated visualization
│   └── README.md                                     #   Assignment documentation
├── Assignment2/                                      # LDA vs PCA comparison
│   ├── Assignment_2.ipynb                            #   Main analysis notebook
│   ├── HousingData.csv                               #   Dataset
│   ├── ass2.2.png … ass2.5.png, output.png
│   └── README.md
├── Assignment3/                                      # Cab price regression pipeline
│   ├── assignment3.py                                #   End-to-end Python script
│   ├── cab_rides.csv                                 #   Primary dataset
│   ├── weather.csv                                   #   Auxiliary dataset
│   ├── correlation_matrix.png
│   ├── outliers_boxplot.png
│   └── README.md
├── Assignment4/                                      # Logistic Regression classification
│   ├── assignment4.ipynb                             #   Main analysis notebook
│   ├── diabetes.csv                                  #   Dataset
│   ├── output.png                                    #   Confusion matrix heatmap
│   └── README.md
├── Assignment5/                                      # SVM digit classification
│   ├── Code.py                                       #   Main Python script
│   └── README.md
├── Assignment6/                                      # K-Means clustering
│   ├── Code.py                                       #   Main Python script
│   ├── elbow_and_clusters.png                        #   Generated visualization
│   └── README.md
└── Assignment7/                                      # OpenCV image I/O + AI inference
    ├── Code.py                                       #   Main Python script
    ├── test_preview.jpg                              #   Input image
    ├── models/                                       #   Auto-downloaded AI models
    │   └── face_detection_yunet_2023mar.onnx
    ├── output_with_faces.jpg                         #   Generated output image
    └── README.md
├── Assignment8/                                      # Random Forest vs baseline classification
│   ├── Code.py                                       #   Main Python script
│   ├── comparison_confusion_matrix.png               #   Confusion matrices
│   ├── performance_comparison.png                    #   Metrics comparison chart
│   └── README.md
```

### Data flow

Each assignment follows a common ML pipeline:

1. **Load** the dataset (CSV, built-in dataset, or image file).
2. **Preprocess** — handle missing values, standardize/scale features, encode categorical variables, or convert color spaces.
3. **Split** into train/test sets (typically 80/20, `random_state=42`) **when applicable**.
4. **Train** one or more models or load a pre-trained model.
5. **Evaluate** using task-appropriate metrics (MSE/R² for regression, precision/recall/F1 for classification, silhouette/ARI for clustering).
6. **Visualize** results (variance explained, performance curves, correlation heatmaps, confusion matrices, annotated images).

## Configuration

There is no external configuration file; hyperparameters are set inline within each script/notebook:

- `StandardScaler` for feature standardization (scale-sensitive algorithms such as PCA, LDA, and regularized regression).
- `train_test_split(..., test_size=0.2, random_state=42)` for reproducible splits.
- `PCA(n_components=...)` / `LinearDiscriminantAnalysis(n_components=...)` for the number of retained components.
- `Ridge(alpha=...)` / `Lasso(alpha=...)` for regularization strength.
- `LogisticRegression(solver='liblinear', random_state=42)` for the classification solver.
- `cv2.FaceDetectorYN.create(model, None, input_size, score_threshold, nms_threshold, top_k)` for OpenCV DNN face detection.
- `RandomForestClassifier(n_estimators=200, random_state=42)` for ensemble classification in Assignment 8.

Adjust these values directly in the source files to experiment.

## Running Tests

This repository does not ship an automated test suite. To validate an assignment manually:

- **Assignments 1, 2, 4 (notebooks):** Run all cells and confirm metrics/outputs print without errors and figures are generated.
- **Assignments 3, 5, 6, 7, 8 (scripts):** Run the corresponding `python Code.py` or `python assignment3.py` from the assignment directory and confirm the expected outputs are produced.

Optional sanity check that dependencies import correctly:

```bash
python -c "import pandas, numpy, matplotlib, sklearn, cv2; print('dependencies OK')"
```

## Contributing

Contributions are welcome for educational improvement. To contribute:

1. Fork the repository and create a feature branch (`git checkout -b assignment-8-feature`).
2. Add or modify an assignment, keeping its folder self-contained (data, code, outputs, `README.md`).
3. Follow the existing documentation style — include an Objective/Theory section, a Mermaid flowchart, and a Q&A section where appropriate.
4. Ensure code runs cleanly with the shared `requirement.txt` and uses `random_state=42` for reproducibility where applicable.
5. Open a pull request describing the change and the dataset/model used.

Please keep each assignment independently runnable and document any new dependencies.

## Dependencies

All assignments rely on the packages listed in [`requirement.txt`](./requirement.txt):

| Package | Purpose |
| --- | --- |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Plotting and visualization |
| `scikit-learn` | ML models, preprocessing, and metrics |
| `kaggle` | (Optional) dataset download via the Kaggle API |
| `opencv-python-headless` | Image I/O, drawing, and DNN inference (Assignment 7) |
| `requests` | Model downloading for Assignment 7 |

For notebook execution, also install `jupyter`. Assignment 4's notebook additionally uses `seaborn` for its confusion-matrix heatmap.

## License

This is an educational workspace; refer to individual dataset licenses (UCI ML Repository, Kaggle) for data usage terms.
