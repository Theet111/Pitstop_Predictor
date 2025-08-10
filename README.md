# Pitstop Predictor – F1 Strategy Windows

Predict pit windows (1/2/3-stop) using FastF1 race data + weather. Models blend CatBoost and RandomForest, then convert to **compound-aware windows** (e.g. C5/C4/C3) using configurable tyre-life baselines that get nudged toward the model’s pit-lap anchors.

> **Important note on MAE:**  
> The pit-lap MAE **varies by circuit and season** due to different layouts, degradation, safety cars, tyres, and weather. It is **not guaranteed to be less than 3 laps** on every circuit. Use the included knobs (more seasons, per-track priors, pace features) to improve stability.

## Features
- Historical build per track & year (sectors, tyre life, stint-decay slopes, SC/VSC priors).
- Weather-aware scaling via Open-Meteo (forecast → archive fallback).
- First/second pit prediction with auto-blended CatBoost/RF and **integer-lap MAE** reporting.
- Compound windows from baseline tyre lives, scaled for temperature, rain, and track abrasiveness.
- Quick ablation + permutation importance for `CtorRating`.

## Install
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Run (CLI)
python predict_strategies.py
 

Example output:
Stop-count acc: 0.933
FirstPit: CV integer-lap MAE = 8.02 (blend w=0.85)
SecondPit: CV integer-lap MAE = 7.30 (blend w=0.55)

Overall possible strategies for Australia on 2025-03-16:
 Stops │ Compounds            │ Window 1 │ Window 2 │ Window 3
--------------------------------------------------------------
1-stop │ C4 → C3 → C3         │ 7–14     │ 30–45
2-stop │ C3 → C4 → C3         │ 16–24    │ 30–45
3-stop │ C5 → C3 → C3         │ 6–12     │ 29–43

Data & Caching
FastF1 cache: cache_folder/ (Create in Route Directory).
Weather cache: .weather_cache/ (via requests_cache).

Tips to Improve MAE
Train on more seasons (mind regulation changes).
Add/keep pace proxies: qualifying best, early/late median race pace.
Calibrate COMPOUND_LIFE_BASE and TRACK_TYRE_MULT per circuit.
Use fractional pit targets (pit_lap / race_laps) with GroupKFold by year when you expand to multi-track training.
Track-level priors for SC/VSC help.

Contributions welcome
If you can improve feature engineering, priors, or modeling, feel free to open a PR or issue. Ideas:
Better pit-loss estimation per track, including in/out delta modeling.
Safety car / VSC sequence features (first 10 laps vs late).
Driver/constructor pace embeddings instead of a static rating.

Troubleshooting
ValueError: could not convert string to float: 'SOFT'
Ensure categoricals are encoded for RF (the repo encodes StartCompound & Track; CatBoost can consume strings).
Open-Meteo “out of allowed range”
The code falls back to the archive endpoint automatically.
Session name mismatch
FastF1 sometimes changes event names; verify the track string matches FastF1.