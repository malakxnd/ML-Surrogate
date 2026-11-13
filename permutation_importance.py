import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import json
 
SEED = 12345
FIELD_NAMES = ["u", "w", "phi", "T", "C", "sxx", "szz", "sxz", "Dx", "Dz", "mu"]
PARAM_NAMES = ["H0", "omega", "a", "tau_th", "tau_q", "tau_st"]
MODEL_FEATURES = PARAM_NAMES + ["c_ratio"]
 
def log_transform(df):
    df["c_ratio"] = df["omega"] / df["a"]
    for f in FIELD_NAMES:
        df[f"log_amp_{f}"] = np.log10(df[f'peak_{f}']+1e-300)
    return [f"log_amp_{f}" for f in FIELD_NAMES] + [f"depth_{f}" for f in FIELD_NAMES]

with open("best_model_params.json", "r") as f:
    best_configs = json.load(f)

best_gb_params = best_configs["GB"]

def main():
    df = pd.read_csv("dataset_summary_2000.csv")
    targets = log_transform(df)

    degenerate = [t for t in targets if df[t].std() < 1e-12]
    targets = [t for t in targets if t not in degenerate]
 
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
    X_train, X_test = df_train[MODEL_FEATURES].values, df_test[MODEL_FEATURES].values
    Y_train, Y_test = df_train[targets].values, df_test[targets].values

    model = MultiOutputRegressor(
        GradientBoostingRegressor(
        **best_gb_params,
        random_state=SEED
        ),
    n_jobs=-1
    )

    model.fit(X_train, Y_train)

    # we want to see how much each value contributes to the model's performance based on R2
    result = permutation_importance(model, X_test, Y_test, n_repeats=20,
                                      random_state=SEED, scoring="r2", n_jobs=-1)
 
    imp = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
 
    imp.to_csv("permutation_importance.csv", index=False)
    
    print("=== Permutation importance (drop in mean R2 across all 21 targets when this feature is shuffled) ===")
    print(imp.to_string(index=False))
 
    top = imp.iloc[0]["feature"]
    print(f"\nMost important feature: {top}")

    h0_row = imp[imp.feature == "H0"].iloc[0] # we want to see how important the magnatic field is
    print(f"H0's importance: {h0_row['importance_mean']:.4f} (rank {1+list(imp['feature']).index('H0')} of {len(MODEL_FEATURES)})")
 
if __name__ == "__main__":
    main()
 

