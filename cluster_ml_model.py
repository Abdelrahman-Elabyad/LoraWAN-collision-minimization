import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from clustering_channel_assignment import random_select
from simulation import run_simulation, generate_devices, generate_arrivals


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

def load_training_data(csv_path="GA_dataset_master.csv"):
    print("[ML] Loading dataset...")
    df = pd.read_csv(csv_path)

    # Features
    X = df[["d", "RX", "C"]]

    # Labels
    y = df["cluster_optimal"]

    print("[ML] Dataset loaded:", df.shape)
    return X, y, df


# ============================================================
# 2. TRAIN LIGHTGBM MODEL
# ============================================================

def train_lightgbm_model(X, y, model_path="cluster_model_lgbm.bin"):
    print("[ML] Preparing LightGBM Dataset...")

    train_data = lgb.Dataset(X, label=y)

    params = {
        "objective": "multiclass",
        "num_class": y.nunique(),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "verbosity": -1
    }

    print("[ML] Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500
    )

    print("[ML] Training complete. Saving model...")
    joblib.dump(model, model_path)
    print(f"[ML] Model saved to {model_path}")

    return model


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

def load_trained_model(model_path="cluster_model_lgbm.bin"):
    print("[ML] Loading trained model...")
    return joblib.load(model_path)


# ============================================================
# 4. ML-BASED CLUSTER ASSIGNMENT
# ============================================================

def assign_clusters_ml(model, d, RX, C):
    N = len(d)
    X_test = np.column_stack([d, RX, np.full(N, C)])

    preds = model.predict(X_test)

    K = 8 // C  # number of valid clusters
    clusters = preds.argmax(axis=1)

    # clamp all predictions to valid range
    clusters = np.clip(clusters, 0, K - 1)

    return clusters



if __name__ == "__main__":
    # 1) Load dataset
    X, y, df = load_training_data("GA_dataset_master.csv")

    # 2) Train model
    model = train_lightgbm_model(X, y, model_path="cluster_model_lgbm.bin")

    print("\n[ML] Training finished successfully!")
