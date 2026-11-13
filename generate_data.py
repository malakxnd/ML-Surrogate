import argparse
import time
import numpy as np
import pandas as pd
import mpmath as mp
from scipy.stats import qmc
import solver as S
 
mp.mp.dps = 50


PARAM_NAMES = ["H0", "omega", "a", "tau_th", "tau_q", "tau_st"]
PARAM_LOW   = np.array([0.0,   1.5, 5,  0, 0.40, 0.05])
PARAM_HIGH  = np.array([1.2e8, 3.5, 15.0, 0.40, 5.0, 0.20])
 
FIELD_NAMES = ["u", "w", "phi", "T", "C", "sxx", "szz", "sxz", "Dx", "Dz", "mu"]
 
N_TOTAL = 2000
N_Z = 100
Z_MAX = 2.0
# X_FIXED, T_FIXED = 0.1, 0.9
SEED = 12345 

def generate_parameters():
    u = qmc.LatinHypercube(d=len(PARAM_LOW), seed=SEED).random(N_TOTAL)
    return PARAM_LOW + u * (PARAM_HIGH - PARAM_LOW)

'''We do not save all of the information or numbers. We only save summaries of it. 
   So that would be more efficient, and we do so by monitoring the max value and whether 
   or not that effect reached has disappeared/ dicreased'''
def summarize(field_vals, z_grid):
    a = np.abs(field_vals)
    peak_idx = int(np.argmax(a))
    peak_val = a[peak_idx]

    if peak_val< 1e-300:
        return 0.0, Z_MAX, True

    thresh = 0.05 * peak_val # did the peak value decrease by 5% of the peak value?
    depth, censored = Z_MAX, True

    for i in range(peak_idx, len(z_grid)):
        if a[i] <= thresh:
            depth = z_grid[i] - z_grid[peak_idx]
            censored = False
            break

    return peak_val, depth, censored

# -------------------------------------------------------
def main():
    # ex for a command parse: python generate.py --start 0 --end 400 --out chunk1.csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    command_args = ap.parse_args()

    t0 = time.time()

    all_params = generate_parameters()
    selected_params = all_params[command_args.start:command_args.end]

    z_grid = np.linspace(0, Z_MAX, N_Z)
    rows=[]
    n_fails = 0

    for i, p in enumerate(selected_params):
        H0, omega, a_, tau_th, tau_q, tau_st = p

        m = S.MaterialConstants(H0=mp.mpf(H0), omega=mp.mpf(omega), a=mp.mpf(a_),
                  tau_th=mp.mpf(tau_th), tau_q=mp.mpf(tau_q), tau_st=mp.mpf(tau_st))

        try:
            sol = S.solve(m)
        except Exception:
            n_fails += 1
            continue

        series = {k: np.empty(N_Z) for k in FIELD_NAMES}
        for j, z in enumerate(z_grid):
            # f = S.mapTo_ct(sol, m, z=mp.mpf(z), x=mp.mpf(X_FIXED), t=mp.mpf(T_FIXED))
            f = S.mapTo_envelope(sol, z=mp.mpf(z))
            for k in FIELD_NAMES:
                series[k][j] = float(f[k])

        row = dict(zip(PARAM_NAMES, p))
        row["global_id"] = command_args.start + i

        for k in FIELD_NAMES:
            peak_val, depth, censored = summarize(series[k], z_grid)
            row[f"peak_{k}"], row[f"depth_{k}"], row[f"censored_{k}"] = peak_val, depth, censored
        rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(command_args.out, index=False)
        print(f"[{command_args.start}:{command_args.end}] kept {len(df)} with ({n_fails} fails), {time.time()-t0:.1f}s -> {command_args.out}")

if __name__ == "__main__":
    main()