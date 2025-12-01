import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

CSV_FILE = "channel_training_data.csv"
OUTPUT_FILE = "best_params.json"

def train_alpha_beta_gamma(df):
    X = df[["lru_score", "load_score", "collision_score"]]
    y = df["success"]

    clf = LogisticRegression().fit(X, y)

    alpha, beta, gamma = clf.coef_[0]
    return alpha, beta, gamma

def train_lambda(df, alpha, beta, gamma):

    def loss(lmbda):
        L = 0
        for _, row in df.iterrows():
            score = alpha*row.lru_score + beta*row.load_score + gamma*row.collision_score
            exp = np.exp(lmbda * score)
            # probability is softmax, but only scalar needed for NLL
            L -= np.log(exp / exp)  # simplified since dataset stores only chosen channel
        return L

    result = minimize_scalar(loss)
    return result.x if result.success else 1.0


if __name__ == "__main__":
    df = pd.read_csv(CSV_FILE)

    alpha, beta, gamma = train_alpha_beta_gamma(df)
    lambda_val = 1.0  # simplified (can refine with multi-channel softmax later)

    best = {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "lambda": float(lambda_val)
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(best, f, indent=4)

    print("\nTrained best parameters saved to best_params.json:")
    print(best)
