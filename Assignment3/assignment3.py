import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
def main():
    print("=== Uber and Lyft Cab Prices Prediction Pipeline ===")
    # ---------------------------------------------------------
    # a. Load and Pre-process the dataset
    # ---------------------------------------------------------
    print("\n--- Loading Dataset ---")
    try:
        df = pd.read_csv('cab_rides.csv')
        print("Successfully loaded 'cab_rides.csv'.")
    except FileNotFoundError:
        print("Dataset not found. Using fallback dummy data for demonstration.")
        np.random.seed(42)
        df = pd.DataFrame({
            'distance': np.random.uniform(0.5, 5.0, 500),
            'source': np.random.choice(['North Station', 'Haymarket Square', 'Back Bay'], 500),
            'destination': np.random.choice(['Haymarket Square', 'North Station', 'Fenway'], 500),
            'cab_type': np.random.choice(['Uber', 'Lyft'], 500),
            'name': np.random.choice(['Shared', 'Lux', 'XL'], 500),
            'price': np.random.uniform(5.0, 40.0, 500)
        })
        # Inject missing values and outliers
        df.loc[0:5, 'price'] = np.nan
        df.loc[6:10, 'price'] = [100, 150, 120, 130, 140]
    # Handle missing values
    initial_shape = df.shape
    df = df.dropna(subset=['price'])
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows with missing prices.")
    # Select relevant features and target
    features = ['distance', 'source', 'destination', 'cab_type', 'name']
    X = df[features]

    y = df['price']
    # Encode categorical variables and scale numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['distance']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['source', 'destination', 'cab_type',
                                                             'name'])
        ])
    X_processed = preprocessor.fit_transform(X)
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2,
                                                        random_state=42)
    print("Data pre-processed and split successfully.")
    # ---------------------------------------------------------
    # b. Record outliers
    # ---------------------------------------------------------
    print("\n--- Outlier Detection ---")
    # Calculate IQR to mathematically record outliers
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Record outliers
    outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
    print(f"Number of outliers detected based on IQR: {len(outliers)}")
    # Visualize outliers using a Boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df['price'])
    plt.title("Boxplot of Ride Prices")
    plt.savefig('outliers_boxplot.png')
    print("Saved outlier boxplot to 'outliers_boxplot.png'.")
    # ---------------------------------------------------------
    # c. Check the correlation
    # ---------------------------------------------------------

    print("\n--- Correlation Analysis ---")
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()
    print("Correlation with 'price':")
    print(corr_matrix['price'].sort_values(ascending=False))
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig('correlation_matrix.png')
    print("Saved correlation heatmap to 'correlation_matrix.png'.")
    # ---------------------------------------------------------
    # d. Implement linear regression and ridge, Lasso regression models
    # ---------------------------------------------------------
    print("\n--- Training Models ---")
    lr_model = LinearRegression()
    ridge_model = Ridge(alpha=1.0)
    lasso_model = Lasso(alpha=0.1)
    lr_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    lasso_model.fit(X_train, y_train)
    print("Linear Regression, Ridge, and Lasso models trained.")
    # Make predictions
    y_pred_lr = lr_model.predict(X_test)
    y_pred_ridge = ridge_model.predict(X_test)
    y_pred_lasso = lasso_model.predict(X_test)
    # ---------------------------------------------------------
    # e. Compute and compare their respective scores like R2, RMSE
    # ---------------------------------------------------------
    print("\n--- Model Evaluation ---")
    def evaluate_model(name, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {"Model": name, "R2 Score": r2, "RMSE": rmse}
    results = [
        evaluate_model("Linear Regression", y_test, y_pred_lr),
        evaluate_model("Ridge Regression", y_test, y_pred_ridge),

        evaluate_model("Lasso Regression", y_test, y_pred_lasso)
    ]
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()