import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import json
 
PARAM_NAMES = ["H0", "omega", "a", "tau_th", "tau_q", "tau_st"]
MODEL_FEATURES = PARAM_NAMES + ["c_ratio"]  

FIELD_NAMES = ["u", "w", "phi", "T", "C", "sxx", "szz", "sxz", "Dx", "Dz", "mu"]
SEED = 12345
N_SPLITS = 5

# we will log the numeric data, to make it eaiser for the neural network to learn
def log_transform(df):
    df["c_ratio"] = df["omega"] / df["a"]
    for f in FIELD_NAMES:
        df[f"log_amp_{f}"] = np.log10(df[f'peak_{f}']+1e-300) # avoid log(0)
    return [f"log_amp_{f}" for f in FIELD_NAMES] + [f"depth_{f}" for f in FIELD_NAMES]

def rel_L2(y_true, y_pred): # relative L2 norm
    den = np.linalg.norm(y_true)
    return np.linalg.norm(y_true - y_pred) / den if den>0 else np.nan

def build_metric_rows(targets, y_true, y_pred, model_name):
    rows = []
    for j, target in enumerate(targets):
        true = y_true[:, j]
        pred = y_pred[:, j]
        rows.append(dict(target=target, model=model_name,
                          MAE=mean_absolute_error(true, pred),
                          RMSE=np.sqrt(mean_squared_error(true, pred)),
                          R2=r2_score(true, pred),
                          rel_L2=rel_L2(true, pred)))
    return rows

def main():
    df = pd.read_csv("dataset_summary_2000.csv")
    targets = log_transform(df)

    # Remove zero-variance targets because they contain no variation to learn
    # and can make the overall R² comparison misleading.
    degenerate = [t for t in targets if df[t].std() < 1e-12]
    if degenerate:
        print(f"dropping degenerate (zero-variance) targets: {degenerate}")
        targets = [t for t in targets if t not in degenerate]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
    X_train, X_test = df_train[MODEL_FEATURES].values, df_test[MODEL_FEATURES].values
    y_train, y_test = df_train[targets].values, df_test[targets].values

    x_scaler, y_scaler = StandardScaler().fit(X_train), StandardScaler().fit(y_train)

    x_train_scaled, x_test_scaled = x_scaler.transform(X_train), x_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train)

    KF = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_rows = []        # metric rows from every model go in here
    timing_rows = []      # fit time / inference time per model, for comparison

    # =====================================================================
    # MODEL 1: multi-task neural network
    # =====================================================================
    candidate_models = [
        dict(hidden_layer_sizes=(64, 32), alpha=1e-4),
        dict(hidden_layer_sizes=(64, 32), alpha=1e-3),
        dict(hidden_layer_sizes=(128, 64), alpha=1e-3),
        dict(hidden_layer_sizes=(128, 64, 32), alpha=1e-3),
        dict(hidden_layer_sizes=(128, 64, 32), alpha=1e-2),
    ]

    search_results = []
    for model_params in candidate_models:
        folds_scores=[]
        for tr, va in KF.split(x_train_scaled):
            model = MLPRegressor(**model_params, activation="relu", max_iter=4000, early_stopping=True,
                                validation_fraction=0.15, n_iter_no_change=30,
                                random_state=SEED)
            model.fit(x_train_scaled[tr], y_train_scaled[tr])
            pred_s = model.predict(x_train_scaled[va])

            pred = y_scaler.inverse_transform(pred_s)
            true = y_train[va]

            folds_scores.append([r2_score(true[:, i], pred[:, i]) for i in range(true.shape[1])])

        r2_mean = float(np.mean(folds_scores))
        search_results.append((model_params, r2_mean))
        print(f"Model: {model_params}, R2 mean: {r2_mean:.4f}")

    # Select the best model based on R2 score
    best_model_params, best_r2 = max(search_results, key=lambda x: x[1])
    print(f"Best model: {best_model_params}, R2 mean: {best_r2:.4f}")

    final_net = MLPRegressor(activation="relu", max_iter=6000, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=40,
                              random_state=SEED, **best_model_params)
    t0 = time.time()
    final_net.fit(x_train_scaled, y_train_scaled)
    t_fit_nn = time.time() - t0
    pred_final_s = final_net.predict(x_test_scaled)
    pred_final = y_scaler.inverse_transform(pred_final_s)

    rows = []
    for j, target in enumerate(targets):
        true = y_test[:, j]
        pred = pred_final[:, j]
        rows.append(dict(target=target, model="multitask_nn",
                          MAE=mean_absolute_error(true, pred),
                          RMSE=np.sqrt(mean_squared_error(true, pred)),
                          R2=r2_score(true, pred),
                          rel_L2=rel_L2(true, pred)))
    res = pd.DataFrame(rows)
    res.to_csv("test_metrics_multitask_nn_2000.csv", index=False)
 
    print("\n=== Holdout R^2: multi-task NN (2000-case dataset) ===")
    print(res.set_index("target")["R2"].round(4))
    print(f"\nMean R^2 across 22 targets  -  multi-task NN: {res['R2'].mean():.4f}")
    print(f"\nTraining time  -  multi-task NN (1 model): {t_fit_nn:.1f}s")

    # ---- inference speed: one multi-task NN call over a benchmark batch ----
    n_bench = 200
    bench = X_test[:min(n_bench, len(X_test))]
    bench_scaled = x_scaler.transform(bench)
    t0 = time.time()
    for _ in range(50):
        _ = final_net.predict(bench_scaled)
    t_nn_infer = (time.time() - t0) / 50
    timing_rows.append(dict(model="multitask_nn", fit_time_s=t_fit_nn, infer_ms_per_batch=1000*t_nn_infer))

    print(f"\nInference (batch of {len(bench)})  -  multi-task NN (1 model): {1000*t_nn_infer:.2f} ms")

    # =====================================================================
    # MODEL 2: random forest
    # trees don't care what scale a feature or target is on -- a split is
    # just "is this value <= some threshold", so unlike the NN we fit this
    # one straight on the raw (unscaled) X_train/y_train, no x_scaler or
    # y_scaler needed here
    # =====================================================================
    rf_candidates = [
        dict(n_estimators=100, max_depth=None),
        dict(n_estimators=200, max_depth=None),
        dict(n_estimators=250, max_depth=None),
        dict(n_estimators=300, max_depth=12),
        dict(n_estimators=500, max_depth=20),
        ]

 
    rf_search_results = []

    for model_params in rf_candidates:
        folds_scores = []
        for tr, va in KF.split(X_train):
            model = RandomForestRegressor(**model_params, random_state=SEED, n_jobs=-1)
            model.fit(X_train[tr], y_train[tr])
            pred = model.predict(X_train[va])
            true = y_train[va]
            folds_scores.append([r2_score(true[:, i], pred[:, i]) for i in range(true.shape[1])])
 
        r2_mean = float(np.mean(folds_scores))
        rf_search_results.append((model_params, r2_mean))
        print(f"RF model: {model_params}, R2 mean: {r2_mean:.4f}")
 
    best_rf_params, best_rf_r2 = max(rf_search_results, key=lambda x: x[1])
    print(f"Best RF model: {best_rf_params}, R2 mean: {best_rf_r2:.4f}")
 
    final_rf = RandomForestRegressor(**best_rf_params, random_state=SEED, n_jobs=-1)

    t0 = time.time()
    final_rf.fit(X_train, y_train)
    t_fit_rf = time.time() - t0
    pred_rf = final_rf.predict(X_test)
 
    all_rows += build_metric_rows(targets, y_test, pred_rf, "random_forest")
 
    t0 = time.time()
    for _ in range(50):
        _ = final_rf.predict(bench)
    t_rf_infer = (time.time() - t0) / 50
    timing_rows.append(dict(model="random_forest", fit_time_s=t_fit_rf, infer_ms_per_batch=1000*t_rf_infer))

    # =====================================================================
    # MODEL 3: gradient boosting
    # sklearn's GradientBoostingRegressor only predicts ONE target at a
    # time (no built-in multi-output like RandomForestRegressor has), so
    # MultiOutputRegressor just fits one GradientBoostingRegressor per
    # target for us (22 of them) and stacks the predictions back together
    # =====================================================================
    gb_candidates = [
        dict(n_estimators=100, max_depth=2, learning_rate=0.1),
        dict(n_estimators=200, max_depth=2, learning_rate=0.05),
        dict(n_estimators=300, max_depth=2, learning_rate=0.03),
        dict(n_estimators=200, max_depth=3, learning_rate=0.05),
        dict(n_estimators=300, max_depth=3, learning_rate=0.03),
        ]
 
    gb_search_results = []
    for model_params in gb_candidates:
        folds_scores = []
        for tr, va in KF.split(X_train):
            base = GradientBoostingRegressor(**model_params, random_state=SEED)
            model = MultiOutputRegressor(base, n_jobs=-1)
            model.fit(X_train[tr], y_train[tr])
            pred = model.predict(X_train[va])
            true = y_train[va]
            folds_scores.append([r2_score(true[:, i], pred[:, i]) for i in range(true.shape[1])])
 
        r2_mean = float(np.mean(folds_scores))
        gb_search_results.append((model_params, r2_mean))
        print(f"GB model: {model_params}, R2 mean: {r2_mean:.4f}")
 
    best_gb_params, best_gb_r2 = max(gb_search_results, key=lambda x: x[1])
    print(f"Best GB model: {best_gb_params}, R2 mean: {best_gb_r2:.4f}")
 
    final_gb = MultiOutputRegressor(GradientBoostingRegressor(**best_gb_params, random_state=SEED), n_jobs=-1)
    t0 = time.time()
    final_gb.fit(X_train, y_train)
    t_fit_gb = time.time() - t0
    pred_gb = final_gb.predict(X_test)
 
    all_rows += build_metric_rows(targets, y_test, pred_gb, "gradient_boosting")
 
    t0 = time.time()
    for _ in range(50):
        _ = final_gb.predict(bench)
    t_gb_infer = (time.time() - t0) / 50
    timing_rows.append(dict(model="gradient_boosting", fit_time_s=t_fit_gb, infer_ms_per_batch=1000*t_gb_infer))

    # =====================================================================
    # compare all 3 models
    # =====================================================================
    res = pd.DataFrame(all_rows)
    res.to_csv("test_metrics_2000.csv", index=False)
 
    timing = pd.DataFrame(timing_rows)
    timing.to_csv("timing_2000.csv", index=False)
 
    print(f"\n=== Mean R^2 across {len(targets)} targets, by model ===")
    print(res.groupby("model")["R2"].mean().round(4))
 
    print("\n=== Fit / inference time, by model ===")
    print(timing.to_string(index=False))

        # ---- overfitting check: CV vs train vs test (mean R2 over all targets) ----
    nn_train_pred = y_scaler.inverse_transform(final_net.predict(x_train_scaled))
    summary = pd.DataFrame([
        dict(model="multitask_nn",
             CV=best_r2,
             train=r2_score(y_train, nn_train_pred),
             test=r2_score(y_test, pred_final)),
        dict(model="random_forest",
             CV=best_rf_r2,
             train=r2_score(y_train, final_rf.predict(X_train)),
             test=r2_score(y_test, pred_rf)),
        dict(model="gradient_boosting",
             CV=best_gb_r2,
             train=r2_score(y_train, final_gb.predict(X_train)),
             test=r2_score(y_test, pred_gb)),
    ])
    summary["train_minus_test"] = summary["train"] - summary["test"]
    print("\n=== Overfitting check (mean R2 across targets) ===")
    print(summary.round(4).to_string(index=False))
    summary.to_csv("overfitting_check.csv", index=False)
    
    # =====================================================================
    # Save the best configurations for reuse in the multi-seed experiment
    best_configs = {
        "NN": best_model_params,
        "RF": best_rf_params,
        "GB": best_gb_params
    }
    
    with open("best_model_params.json", "w") as f:
        json.dump(best_configs, f, indent=4)
        
    # # ---- parity plot for a representative spread of targets ----
    # show = ["log_amp_szz", "log_amp_T", "log_amp_u", "log_amp_w", "depth_mu", "depth_sxx"]
    # fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    # for ax, target in zip(axes.flat, show):
    #     j = targets.index(target)
    #     true = y_test[:, j]
    #     lo, hi = true.min(), true.max()
    #     ax.scatter(true, pred_final[:, j], s=12, alpha=0.5, label="multitask_nn")
    #     ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    #     ax.set_title(target)
    #     ax.set_xlabel("exact"); ax.set_ylabel("predicted")
    #     ax.legend(fontsize=8)

    # fig.tight_layout()
    # fig.savefig("parity_plots_multitask_nn_2000.png", dpi=150)
    # print("\nSaved parity_plots_multitask_nn_2000.png")

if __name__ == "__main__":
    main()