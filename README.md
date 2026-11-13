# Surrogate models for magneto-piezothermoelastic diffusion in CdSe

This repo is the code and data behind the surrogate part of my paper on normal-mode analysis of magneto-piezothermoelastic diffusion in CdSe (dual-phase-lag vs Lord–Shulman). It has the exact solver, the 2,000-case dataset, and the scripts that train and compare the three surrogates (gradient boosting, random forest, neural network).

The MATLAB code that made the figures in Section 9 is not included. Everything here is Python.

## First, install the dependencies

I used Python 3.13.3 on Windows. From the repo folder:

```
pip install -r requirements.txt
```

Do this before anything else, whichever option you pick below.

## What's in here

- `solver.py`: the exact normal-mode solver (mpmath, arbitrary precision)
- `generate_data.py`: builds the dataset by solving 2,000 parameter sets
- `training_nn.py`: picks hyperparameters by 5-fold CV, then trains and tests all three models on the first split
- `training_nn-multiSeed.py`: repeats the training on five random splits and times the surrogates against an exact solve
- `permutation_importance.py`: permutation importance for the gradient boosting model
- `dataset_summary_2000.csv`: the dataset (peak and penetration depth of every field, 2,000 cases)
- `best_model_params.json`: the best hyperparameters I found (it's here because two scripts read it; a reference copy is also in `outputs/`)
- `requirements.txt`: the library versions I used
- `outputs/`: all the result files I got, so you can compare them with yours

## Option 1: use what's already here (the quick one)

I'm giving you the generated dataset and the best parameters so you don't have to spend the time generating the data yourself. With those, just run:

```
python training_nn-multiSeed.py
python permutation_importance.py
```

These use the saved `best_model_params.json` directly, so there's no hyperparameter search. The first one gives the five-seed results and the timing comparison, and the second gives the permutation importances.

If you also want the first-split table and the overfitting check, run `python training_nn.py`. Heads up: it redoes the hyperparameter search and rewrites `best_model_params.json` in the top folder. That's fine, my copy is safe in `outputs/`.

## Option 2: do everything from scratch

If you want to test it all yourself:

1. Generate the data. You pick how many cases with `--start` and `--end` (case numbers, counting from 0). For the full 2,000 I used:

```
python generate_data.py --start 0 --end 2000 --out dataset_summary_2000.csv
```

This overwrites the included `dataset_summary_2000.csv` with your regenerated version, which should be identical to mine.

Want fewer? Just change the numbers, e.g. `--start 0 --end 500 --out my_dataset.csv` gives you the first 500 cases. Each solve takes roughly 0.3 s on my laptop, so the full 2,000 take a while (my estimate is around ten minutes). The parameter sets come from a Latin hypercube with a fixed seed, so you get the same cases I did.

Want more than 2,000? Open `generate_data.py` and change `N_TOTAL = 2000` to the number you want, then run it with `--end` set to that number. Be aware that a Latin hypercube with a different size is a new sample, not an extension of mine, so the first 2,000 cases won't match my dataset.

The training scripts read `dataset_summary_2000.csv`. If you use a different output name, change the filename at the top of each script (or just keep my name).

2. Run the scripts in this order (the second and third need the JSON that the first one writes):

```
python training_nn.py
python training_nn-multiSeed.py
python permutation_importance.py
```

## Where the results go, and how to compare

When you run the scripts, the files they generate (CSVs and the JSON) are written in the same folder as the code. My own results are in `outputs/`, so you can put yours next to mine and compare.

## About the numbers (they depend on the machine)

Results, especially timings, depend on the machine. Mine was a Dell laptop with an Intel Core i5-1135G7 @ 2.40 GHz, 8 GB RAM (7.73 GB usable), 64-bit Windows. All the timings in the paper (the exact solve of about 0.3 s and the single-prediction times) come from that laptop, so yours will be different. The R² values should be very close but might differ in the last decimals, depending on your library versions (that's why the versions are pinned in `requirements.txt`).

## Limits

The surrogate only works inside the parameter ranges I sampled. The exact solver is the reference, and the surrogate is a proof of concept.
