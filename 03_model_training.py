import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import joblib
import os


def main():
    print("=" * 60)
    print("URBAN MOBILITY ML - MODEL TRAINING PIPELINE")
    print("=" * 60)

    csv_path = "data/processed_traffic.csv"
    print(f"\n[STEP 1] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    print(f"\n[STEP 2] Defining Features (X) and Target (y)...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    weather_encoded = encoder.fit_transform(df[["weather_main"]])
    weather_feature_names = encoder.categories_[0].tolist()
    print(f"Weather one-hot encoded features: {weather_feature_names}")

    numeric_features = ["temp", "humidity", "visibility"]
    X_numeric = df[numeric_features].values
    X = np.hstack([X_numeric, weather_encoded])
    feature_names = numeric_features + weather_feature_names
    y = df["traffic_state"].values

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Target distribution: Clear(0)={sum(y == 0)}, Congested(1)={sum(y == 1)}")

    print(f"\n[STEP 3] Splitting data (80/20) and scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {len(X_train)} rows")
    print(f"Testing set: {len(X_test)} rows")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled using StandardScaler")

    print("\n" + "=" * 60)
    print("LO3: LINEAR MODEL - LOGISTIC REGRESSION")
    print("=" * 60)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f} ({lr_accuracy * 100:.2f}%)")

    print("\n" + "=" * 60)
    print("LO4: CLUSTERING - GAUSSIAN MIXTURE MODEL")
    print("=" * 60)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_train_scaled)
    gmm_labels = gmm.predict(X_train_scaled)
    print(f"Gaussian Mixture Model trained with 2 components")
    print(f"Cluster 0: {sum(gmm_labels == 0)} samples")
    print(f"Cluster 1: {sum(gmm_labels == 1)} samples")
    print("Hidden traffic patterns discovered successfully!")

    print("\n" + "=" * 60)
    print("LO5: NEURAL NETWORK - SINGLE LAYER PERCEPTRON")
    print("=" * 60)
    mlp = MLPClassifier(
        hidden_layer_sizes=(), max_iter=2000, random_state=42, early_stopping=True
    )
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)
    mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
    print(f"MLP Perceptron Accuracy: {mlp_accuracy:.4f} ({mlp_accuracy * 100:.2f}%)")

    print("\n" + "=" * 60)
    print("LO6: DIMENSIONALITY REDUCTION - PCA")
    print("=" * 60)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    print(
        f"PCA applied: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]} components"
    )
    print(f"Explained Variance Ratio:")
    print(
        f"  PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0] * 100:.2f}%)"
    )
    print(
        f"  PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1] * 100:.2f}%)"
    )
    print(
        f"  Total: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_) * 100:.2f}%)"
    )

    print("\n" + "=" * 60)
    print("EXPORTING BEST MODEL")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)

    if lr_accuracy >= mlp_accuracy:
        best_model = lr_model
        best_name = "Logistic Regression"
    else:
        best_model = mlp
        best_name = "MLP Perceptron"

    joblib.dump(best_model, "models/traffic_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"Best model: {best_name} (Accuracy: {max(lr_accuracy, mlp_accuracy):.4f})")
    print("Saved to:")
    print("  - models/traffic_model.pkl")
    print("  - models/scaler.pkl")

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
