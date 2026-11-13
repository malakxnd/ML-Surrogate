import time
import numpy as np
import pandas as pd
import mpmath as mp
 
import solver as S
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import json
 
PARAM_NAMES = ["H0", "omega", "a", "tau_th", "tau_q", "tau_st"]
MODEL_FEATURES = PARAM_NAMES + ["c_ratio"]
FIELD_NAMES = ["u", "w", "phi", "T", "C", "sxx", "szz", "sxz", "Dx", "Dz", "mu"]
SEEDS = [12345, 777, 2024, 31415, 90210]  # 12345 is training_nn.py's original seed
 
# Load the best configurations selected by training_nn.py
with open("best_model_params.json", "r") as f:
    best_configs = json.load(f)

BEST_NN = best_configs["NN"]
BEST_NN["hidden_layer_sizes"] = tuple(BEST_NN["hidden_layer_sizes"])

BEST_RF = best_configs["RF"]
BEST_GB = best_configs["GB"]
 
N_Z = 100
Z_MAX = 2.0
 
def log_transform(df):
    df["c_ratio"] = df["omega"] / df["a"]
    for f in FIELD_NAMES:
        df[f"log_amp_{f}"] = np.log10(df[f'peak_{f}']+1e-300)
    return [f"log_amp_{f}" for f in FIELD_NAMES] + [f"depth_{f}" for f in FIELD_NAMES]
 
def rel_L2(y_true, y_pred):
    den = np.linalg.norm(y_true)
    return np.linalg.norm(y_true - y_pred) / den if den > 0 else np.nan
 
def build_metric_rows(targets, y_true, y_pred, model_name, seed):
    rows = []
    for j, target in enumerate(targets):
        true, pred = y_true[:, j], y_pred[:, j]
        # only depth_* targets get a meaningful rel_L2 -- see module docstring
        rl2 = rel_L2(true, pred) if target.startswith("depth_") else np.nan
        rows.append(dict(seed=seed, target=target, model=model_name,
                          MAE=mean_absolute_error(true, pred),
                          RMSE=np.sqrt(mean_squared_error(true, pred)),
                          R2=r2_score(true, pred),
                          rel_L2=rl2))
    return rows
 
def run_one_seed(df, targets, seed):

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)

    X_train, X_test = df_train[MODEL_FEATURES].values, df_test[MODEL_FEATURES].values
    Y_train, Y_test = df_train[targets].values, df_test[targets].values
 
    x_scaler, y_scaler = StandardScaler().fit(X_train), StandardScaler().fit(Y_train)
    X_train_s, X_test_s = x_scaler.transform(X_train), x_scaler.transform(X_test)
    Y_train_s = y_scaler.transform(Y_train)
 
    rows = []
 
    nn = MLPRegressor(**BEST_NN, activation="relu", max_iter=6000, early_stopping=True,
                       validation_fraction=0.15, n_iter_no_change=40, random_state=seed)
    
    nn.fit(X_train_s, Y_train_s)
    pred_nn = y_scaler.inverse_transform(nn.predict(X_test_s))
    rows += build_metric_rows(targets, Y_test, pred_nn, "multitask_nn", seed)
 
    rf = RandomForestRegressor(**BEST_RF, random_state=seed, n_jobs=1)
    rf.fit(X_train, Y_train)
    pred_rf = rf.predict(X_test)
    rows += build_metric_rows(targets, Y_test, pred_rf, "random_forest", seed)
 
    gb = MultiOutputRegressor(GradientBoostingRegressor(**BEST_GB, random_state=seed), n_jobs=-1)
    gb.fit(X_train, Y_train)
    pred_gb = gb.predict(X_test)
    rows += build_metric_rows(targets, Y_test, pred_gb, "gradient_boosting", seed)
 
    return rows
 
def time_exact_solve(df, n_samples=20):
    # we select a small number of random rows from the dataset and time how long it takes to solve the full z-profile for each one.  This is a rough estimate of the time it would take to do a full exact solve for all 2000 rows in the dataset.
    rng = np.random.RandomState(0)
    sample = df.iloc[rng.choice(len(df), n_samples, replace=False)]
    z_grid = np.linspace(0, Z_MAX, N_Z)
 
    mp.mp.dps = 50  # same precision generate_data.py actually used
    t0 = time.time()
    for _, r in sample.iterrows():
        m = S.MaterialConstants(H0=mp.mpf(r.H0), omega=mp.mpf(r.omega), a=mp.mpf(r.a),
                                 tau_th=mp.mpf(r.tau_th), tau_q=mp.mpf(r.tau_q), tau_st=mp.mpf(r.tau_st))
        sol = S.solve(m)
        for z in z_grid:
            S.mapTo_envelope(sol, mp.mpf(z))

    return (time.time() - t0) / n_samples
 
def main():
    df = pd.read_csv("dataset_summary_2000.csv")
    targets = log_transform(df)
    degenerate = [t for t in targets if df[t].std() < 1e-12]
    targets = [t for t in targets if t not in degenerate]
 
    all_rows = []
    for seed in SEEDS:
        print(f"--- seed {seed} ---")
        all_rows += run_one_seed(df, targets, seed)
 
    res = pd.DataFrame(all_rows)
    res.to_csv("test_metrics_multiseed.csv", index=False)
 
    print(f"\n=== Mean R^2 across {len(targets)} targets, mean +- std across {len(SEEDS)} seeds ===")
    per_seed_mean = res.groupby(["model", "seed"])["R2"].mean().reset_index()
    summary = per_seed_mean.groupby("model")["R2"].agg(["mean", "std"])
    print(summary.round(4))
 
    # ---- exact-solve time vs surrogate inference time ----
    t_exact = time_exact_solve(df)
    print(f"\nExact solve time (one parameter combo, full {N_Z}-point z-profile): {1000*t_exact:.1f} ms")
 
    X_all = df[MODEL_FEATURES].values
    Y_all = df[targets].values
    x_scaler_all = StandardScaler().fit(X_all)
    X_all_s = x_scaler_all.transform(X_all)
    bench = X_all[:1]          # unscaled, for the tree models
    bench_s = X_all_s[:1]      # scaled, for the NN

    timing_rows = []
    for name, model, X_fit, bench_row in [
            ("multitask_nn", MLPRegressor(**BEST_NN, max_iter=200, random_state=0), X_all_s, bench_s),
            ("random_forest", RandomForestRegressor(**BEST_RF, random_state=0, n_jobs=-1), X_all, bench),
            ("gradient_boosting", MultiOutputRegressor(GradientBoostingRegressor(**BEST_GB, random_state=0), n_jobs=-1), X_all, bench)]:
        
        model.fit(X_fit, Y_all if name != "multitask_nn" else StandardScaler().fit_transform(Y_all))
        
        t0 = time.time()
        for _ in range(200):
            model.predict(bench_row)
        t_infer = (time.time() - t0) / 200
        speedup = t_exact / t_infer
        
        timing_rows.append(dict(model=name, exact_solve_ms=1000*t_exact,
                                 surrogate_infer_ms=1000*t_infer, speedup_factor=speedup))
 
    timing = pd.DataFrame(timing_rows)
    timing.to_csv("exact_vs_surrogate_timing.csv", index=False)
    print("\n=== Exact solve vs surrogate inference, one parameter combo at a time ===")
    print(timing.to_string(index=False))
 
if __name__ == "__main__":
    main()