# predict_strategies.py

import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, accuracy_score
import requests_cache
from retry_requests import retry
import openmeteo_requests
from copy import deepcopy
from typing import Tuple, List, Dict
from stint_features import compute_avg_stint_decay

fastf1.Cache.enable_cache("cache_folder")

########################################
# 1) TRACK & PRIOR DATA
########################################
TRACK_LOOKUP = {
    'Australia':      {'length_km':5.303, 'lat':-37.8497, 'lon':144.9680},
    'Bahrain':        {'length_km':5.412, 'lat':26.0325,  'lon':50.5106},
    'Saudi Arabia':   {'length_km':6.174, 'lat':21.6308,  'lon':39.1044},
    'Miami':          {'length_km':5.412, 'lat':25.9581,  'lon':-80.2389},
    'Imola':          {'length_km':4.909, 'lat':44.3392,  'lon':11.7165},
    'Monaco':         {'length_km':3.337, 'lat':43.7384,  'lon':7.4246},
    'Barcelona':      {'length_km':4.675, 'lat':41.5700,  'lon':2.2610},
    'Montreal':       {'length_km':4.361, 'lat':45.5000,  'lon':-73.5220},
    'Austria':        {'length_km':4.318, 'lat':47.2197,  'lon':14.7647},
    'Silverstone':    {'length_km':5.891, 'lat':52.0786,  'lon':-1.0169},
    'Belgium':        {'length_km':7.004, 'lat':50.4372,  'lon':5.9714},
    'Hungaroring':    {'length_km':4.381, 'lat':47.5780,  'lon':19.2486},
    'Zandvoort':      {'length_km':4.259, 'lat':52.3888,  'lon':4.5410},
    'Monza':          {'length_km':5.793, 'lat':45.6160,  'lon':9.2811},
    'Baku':           {'length_km':6.003, 'lat':40.3725,  'lon':49.8533},
    'Singapore':      {'length_km':5.063, 'lat':1.2910,   'lon':103.8640},
    'Austin':         {'length_km':5.513, 'lat':30.1328,  'lon':-97.6413},
    'Mexico City':    {'length_km':4.304, 'lat':19.4040,  'lon':-99.0900},
    'Sao Paulo':      {'length_km':4.309, 'lat':-23.7036, 'lon':-46.6997},
    'Las Vegas':      {'length_km':6.120, 'lat':36.1169,  'lon':-115.1552},
    'Lusail':         {'length_km':5.807, 'lat':25.3374,  'lon':51.4430},
    'Yas Marina':     {'length_km':5.554, 'lat':24.4672,  'lon':54.6031},
    'Japan':          {'length_km':5.807, 'lat':34.8431,  'lon':136.5410},
    'China':          {'length_km':5.451, 'lat':31.3389,  'lon':121.2230},
}

# Example priors (trim to keep concise)
PRIORS = {
    "Australia": {
        1: {"compounds":["C4","C3","C3"], "windows":[(14,20),(33,39)]},
        2: {"compounds":["C3","C4","C3"], "windows":[(19,25),(33,39)]},
        3: {"compounds":["C5","C3","C3"], "windows":[(9,15),(31,37)]},
    },
    "Belgium": {
        1: {"compounds":["C3","C2"],      "windows":[(27,33)]},
        2: {"compounds":["C3","C3","C2"], "windows":[(12,18),(31,41)]},
        3: {"compounds":["C3","C3","C3","C2"], "windows":[(10,14),(28,32),(44,48)]},
    },
    "Hungary": {
        1: {"compounds":["C4","C3","C4"], "windows":[(16,21),(45,50)]},
        2: {"compounds":["C4","C3","C3"], "windows":[(13,18),(41,47)]},
        3: {"compounds":["C5","C3","C4"], "windows":[(9,14),(43,49)]},
    },
    "Silverstone": {
        1: {"compounds":["C3","C2"],      "windows":[(19,25)]},
        2: {"compounds":["C3","C3","C2"], "windows":[(12,18),(32,38)]},
        3: {"compounds":["C3","C3","C3","C2"], "windows":[(10,14),(26,30),(42,46)]},
    },
}

TEAM_MAPPING = {
    "VER":"Red Bull","HAM":"Ferrari","LEC":"Ferrari",
    "NOR":"McLaren","PIA":"McLaren","RUS":"Mercedes","ANT":"Mercedes",
    "ALO":"Aston Martin","STR":"Aston Martin",
    "OCO":"Haas","BEA":"Haas","GAS":"Alpine","DOO":"Alpine",
    "HUL":"Sauber","BOR":"Sauber","ALB":"Williams","SAI":"Williams"
}

# ---- Compound life baselines (laps) at reference conditions ----
# C5 softest … C1 hardest.
COMPOUND_LIFE_BASE = {
    "C5": (8, 16),   # Soft
    "C4": (10, 20),  # Soft
    "C3": (30, 40),  # Medium
    "C2": (40, 50),  # Hard
    "C1": (45, 55),  # Hard+
}

# Optional per-track scalar (abrasiveness). 1.0 = neutral.
TRACK_TYRE_MULT = {
    # tune per track if you want; examples:
    "Australia": 1.00, "Bahrain": 0.90, "Hungaroring": 0.95, "Monza": 1.05
}

def _scale_range(base_range, deg_factor, temp_c, rain_prob, track):
    """
    Scale a (min,max) stint range by track/race conditions.
    - deg_factor: laps per km proxy; higher → more laps for same kilometers → slightly longer stints in laps
    - temp: hotter shortens stints; ~1% per °C over 20 (cap ~15%)
    - rain_prob: shrinks usable dry-life; up to -30% at 100%
    - track scalar: tune per track (abrasiveness)
    """
    rmin, rmax = base_range
    # base scaling
    scale = (deg_factor / 10.0) * TRACK_TYRE_MULT.get(track, 1.0)

    # temperature penalty (above 20°C)
    hot_pen = 1.0 - min(0.15, max(0.0, (temp_c - 20.0)) * 0.01)

    # rain penalty (linear up to 30% reduction)
    rain_pen = 1.0 - 0.30 * max(0.0, min(1.0, rain_prob))

    s = max(0.5, scale * hot_pen * rain_pen)
    return (int(round(rmin * s)), int(round(rmax * s)))

def _compound_windows(compounds, total_laps, temp, rain, deg_factor, track, anchors=None, pull=0.4):
    """
    Build pit windows from compound life ranges.
    - compounds: e.g. ["C4","C3","C3"] → 2 stops → 2 windows
    """
    wins = []
    cum_min = 0
    cum_max = 0
    n_stops = max(0, len(compounds) - 1)
    for i in range(n_stops):
        base_rng = COMPOUND_LIFE_BASE.get(compounds[i], (20, 30))
        rmin, rmax = _scale_range(base_rng, deg_factor, temp, rain, track)

        smin = max(1, cum_min + rmin)
        smax = min(total_laps - 1, cum_max + rmax)

        # pull toward model anchor (if provided)
        if anchors and i < len(anchors):
            target = int(anchors[i])
            mid = (smin + smax) / 2.0
            shift = int(round(pull * (target - mid)))
            smin = max(1, smin + shift)
            smax = min(total_laps - 1, smax + shift)

        wins.append((smin, smax))
        cum_min += rmin
        cum_max += rmax
    return wins


# --- Race Control parsing for SC/VSC laps ---
def _extract_sc_vsc_laps(rcm: pd.DataFrame, total_laps: int) -> Tuple[set, set]:
    """
    Parse session.race_control_messages (if available) and return sets
    of lap numbers run under SC and VSC. Falls back to empty sets if not usable.
    """
    if rcm is None or rcm.empty or ("Message" not in rcm.columns) or ("Lap" not in rcm.columns):
        return set(), set()

    r = rcm.copy().sort_values("Lap")
    sc_on, vsc_on = None, None
    sc_laps, vsc_laps = set(), set()

    for _, row in r.iterrows():
        msg = str(row.get("Message", "")).upper()
        lap = int(row.get("Lap") or 0)
        lap = max(1, min(total_laps, lap))
        if ("VIRTUAL SAFETY CAR" in msg) or ("VSC" in msg):
            if ("DEPLOY" in msg) or ("ACTIVATED" in msg) or ("DEPLOYED" in msg) or ("START" in msg):
                if vsc_on is None:
                    vsc_on = lap
            elif ("ENDING" in msg) or ("END" in msg) or ("DISABLED" in msg) or ("FINISH" in msg):
                if vsc_on is not None:
                    for L in range(vsc_on, lap + 1):
                        vsc_laps.add(L)
                    vsc_on = None
            continue  # don't double-count as SC
        if "SAFETY CAR" in msg:
            if ("DEPLOY" in msg) or ("ACTIVATED" in msg) or ("DEPLOYED" in msg) or ("START" in msg):
                if sc_on is None:
                    sc_on = lap
            elif ("IN THIS LAP" in msg) or ("ENDING" in msg) or ("END" in msg) or ("FINISH" in msg):
                if sc_on is not None:
                    for L in range(sc_on, lap + 1):
                        sc_laps.add(L)
                    sc_on = None
    if vsc_on is not None:
        for L in range(vsc_on, total_laps + 1):
            vsc_laps.add(L)
    if sc_on is not None:
        for L in range(sc_on, total_laps + 1):
            sc_laps.add(L)

    return sc_laps, vsc_laps

########################################
# 2) WEATHER FETCH (forecast → archive)
########################################
def fetch_weather(lat: float, lon: float, date: str) -> Tuple[float, float]:
    sess   = requests_cache.CachedSession('.weather_cache', expire_after=3600)
    sess   = retry(sess, retries=3, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=sess)

    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date, "end_date": date,
        "hourly": ["temperature_2m","precipitation_probability"]
    }

    try:
        resp = client.weather_api("https://api.open-meteo.com/v1/forecast", params)[0]
    except Exception as e:
        if "out of allowed range" in str(e):
            resp = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params)[0]
        else:
            raise

    hr    = resp.Hourly()
    temps = hr.Variables(0).ValuesAsNumpy()
    probs = hr.Variables(1).ValuesAsNumpy()/100.0
    idx   = min(6, len(temps)-1)  # around early race time
    return float(temps[idx]), float(probs[idx])

########################################
# 3) BUILD HISTORICAL DATA FOR ONE TRACK
########################################
def _start_compounds(laps: pd.DataFrame) -> pd.DataFrame:
    """Get starting compound per Driver from lap 1, fallback to driver's mode or 'Medium'."""
    comp = (laps.sort_values(["Driver","LapNumber"])
                 .groupby("Driver").first()[["Compound"]].reset_index())
    # mode fallback if NaN
    mode_map = (laps.groupby("Driver")["Compound"]
                    .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "Medium"))
    comp["Compound"] = comp.apply(
        lambda r: r["Compound"] if pd.notna(r["Compound"]) else mode_map.get(r["Driver"], "Medium"),
        axis=1
    )
    comp = comp.rename(columns={"Compound":"StartCompound"})
    return comp

def _pace_summaries(laps: pd.DataFrame, race_laps: int) -> pd.DataFrame:
    """Early and late pace summaries to capture fuel burn + tyre fade tendencies."""
    df = laps.copy()
    df["LapTimeSec"] = (
        df["LapTime"].dt.total_seconds()
        if pd.api.types.is_timedelta64_dtype(df["LapTime"])
        else df["LapTime"]
    )
    early = (df[df.LapNumber <= min(10, race_laps//3)]
             .groupby("Driver")["LapTimeSec"].median()
             .rename("EarlyPace").reset_index())
    late_cut = max(1, race_laps - 10)
    late = (df[df.LapNumber >= late_cut]
            .groupby("Driver")["LapTimeSec"].median()
            .rename("LatePace").reset_index())
    return early.merge(late, on="Driver", how="outer")

def build_historical(track: str, years: List[int], race_dates: Dict[int,str]) -> pd.DataFrame:
    info   = TRACK_LOOKUP[track]
    frames = []
    yr_sc_frac = {}  # year -> race SC fraction
    yr_vsc_frac = {}

    for yr in years:
        ses  = fastf1.get_session(yr, track, 'R')
        ses.load(messages=True)

        laps = (ses.laps.reset_index()
                    .dropna(subset=["Sector1Time","Sector2Time","Sector3Time"])
                    .copy())

        # sector times
        laps["S1"] = laps.Sector1Time.dt.total_seconds()
        laps["S2"] = laps.Sector2Time.dt.total_seconds()
        laps["S3"] = laps.Sector3Time.dt.total_seconds()

        # overall lap (ensure seconds)
        if "LapTime" in laps.columns and pd.api.types.is_timedelta64_dtype(laps["LapTime"]):
            laps["LapTime"] = laps["LapTime"].dt.total_seconds()
        else:
            laps["LapTime"] = (laps.Sector1Time + laps.Sector2Time + laps.Sector3Time).dt.total_seconds()

        race_laps = int(laps.LapNumber.max())

        # per-stint decay
        decay_df = compute_avg_stint_decay(laps) 

        # pit summary
        pits = (laps[laps.PitInTime.notna()]
                .groupby("Driver")["LapNumber"]
                .apply(list)
                .reset_index(name="PitLaps"))
        pits["StopCount"]  = pits.PitLaps.map(len)
        pits["FirstPit"]   = pits.PitLaps.map(lambda L: L[0] if L else np.nan)
        pits["SecondPit"]  = pits.PitLaps.map(lambda L: L[1] if len(L)>1 else np.nan)

        # average sectors & tyre life
        sector = laps.groupby("Driver")[["S1","S2","S3"]].mean().reset_index()
        tyre   = (laps.groupby("Driver")["TyreLife"]
                      .mean()
                      .reset_index()
                      .rename(columns={"TyreLife":"AvgTyreLife"}))

        # start compound + pace summaries
        startc = _start_compounds(laps)
        pace   = _pace_summaries(laps, race_laps)

        # Grid position from results
        try:
            res = ses.results[["Abbreviation","GridPosition"]].rename(columns={"Abbreviation":"Driver"})
        except Exception:
            res = pd.DataFrame({"Driver": laps["Driver"].unique(), "GridPosition": np.nan})

        rcm = getattr(ses, "race_control_messages", None)
        sc_laps, vsc_laps = _extract_sc_vsc_laps(rcm, race_laps)
        race_sc_frac = (len(sc_laps) / race_laps) if race_laps else 0.0
        race_vsc_frac = (len(vsc_laps) / race_laps) if race_laps else 0.0
        yr_sc_frac[yr]  = race_sc_frac
        yr_vsc_frac[yr] = race_vsc_frac

        # merge
        df = (sector
              .merge(tyre, on="Driver", how="left")
              .merge(pits.drop(columns="PitLaps"), on="Driver", how="left")
              .merge(decay_df, on="Driver", how="left")
              .merge(startc, on="Driver", how="left")
              .merge(pace, on="Driver", how="left")
              .merge(res, on="Driver", how="left"))

        # fill meta
        df.loc[:, "Year"]            = yr
        df.loc[:, "YearNorm"]        = yr
        df.loc[:, "Track"]           = track
        df.loc[:, "TrackLenKm"]      = info["length_km"]
        df.loc[:, "RaceLaps"]        = race_laps
        df.loc[:, "DegFactor"]       = race_laps / info["length_km"]

        # weather per year
        try:
            t, r = fetch_weather(info["lat"], info["lon"], race_dates[yr])
        except Exception:
            t, r = 0.0, 0.0
        df.loc[:, "Temperature"]     = t
        df.loc[:, "RainProbability"] = r
        df.loc[:, "Constructor"]     = df.Driver.map(TEAM_MAPPING).fillna("Other")
        df.loc[:, "StartTyreLife"]   = 35

        # NA fills
        df.loc[:, "AvgStintDecay"]   = df.AvgStintDecay.fillna(0.0)
        df.loc[:, "AvgTyreLife"]     = df.AvgTyreLife.fillna(df.AvgTyreLife.median())
        df.loc[:, "EarlyPace"]       = df.EarlyPace.fillna(df.EarlyPace.median())
        df.loc[:, "LatePace"]        = df.LatePace.fillna(df.LatePace.median())
        df.loc[:, "StartCompound"]   = df.StartCompound.fillna("Medium")
        df.loc[:, "GridPosition"]    = df.GridPosition.fillna(df.GridPosition.median())

        # keep per-race realized fractions for later aggregation (NOT used directly as features!)
        df.loc[:, "RaceSCFrac"]      = race_sc_frac
        df.loc[:, "RaceVSCFrac"]     = race_vsc_frac

        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    # --- Build prior-only SC/VSC propensity ---
    years_sorted = sorted(full["Year"].unique())
    def _prior_mean(d: Dict[int,float], y: int) -> float:
        prev = [v for yy, v in d.items() if yy < y]
        if prev:
            return float(np.mean(prev))
        # fallback: overall mean over ALL years (constant prior if first year)
        return float(np.mean(list(d.values()))) if d else 0.0

    full["SCFracHist"]  = full["Year"].map(lambda y: _prior_mean(yr_sc_frac, y))
    full["VSCFracHist"] = full["Year"].map(lambda y: _prior_mean(yr_vsc_frac, y))

    return full


########################################
# 4) TRAIN MODELS (de-leaked + auto-blended)
########################################
def train_models(hist: pd.DataFrame, verbose: bool = True):
    # normalize constructor strength
    rating = {"Red Bull":450,"Mercedes":430,"Ferrari":440,
              "McLaren":420,"Alpine":200,"Haas":100,"Williams":80,
              "Aston Martin":350,"Sauber":150,"Other":250}
    hist["CtorRating"] = hist.Constructor.map(rating).fillna(250) / max(rating.values())

    # categorical encodings
    for c in ("StartCompound","Track"):
        hist[c] = hist[c].astype("category").cat.codes

    # IMPORTANT: remove leaked features (Phase, StintLength) — Phase used FirstPit, StintLength used First/SecondPit.
    base_feats = [
        "S1","S2","S3","AvgTyreLife","AvgStintDecay",
        "TrackLenKm","DegFactor","Temperature","RainProbability",
        "CtorRating","StartTyreLife","YearNorm","StartCompound","Track",
        "RaceLaps","EarlyPace","LatePace","GridPosition",           
        "SCFracHist","VSCFracHist"
    ]

    # ---- Stop-count classifier (no leakage features) ----
    Xc = hist[base_feats]
    yc = (hist.StopCount.fillna(0) >= 2).astype(int)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.25, random_state=42, stratify=yc)
    clf = RandomForestClassifier(n_estimators=400, max_depth=14,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    clf.fit(Xc_tr, yc_tr)
    if verbose:
        print("Stop-count acc:", round(accuracy_score(yc_te, clf.predict(Xc_te)), 3))

    # helper to train a pit regressor with K-Fold + auto-blend + integer rounding eval
    def _fit_pit_regressor(target_col: str):
        dfp = hist.dropna(subset=[target_col]).copy()
        Xp, yp = dfp[base_feats], np.log1p(dfp[target_col])

        kf = KFold(n_splits=min(5, max(2, len(dfp)//4)), shuffle=True, random_state=42)
        val_preds_cb, val_preds_rf, val_true = [], [], []

        cbs, rfs = [], []
        for tr_idx, te_idx in kf.split(Xp):
            X_tr, X_te = Xp.iloc[tr_idx], Xp.iloc[te_idx]
            y_tr, y_te = yp.iloc[tr_idx], yp.iloc[te_idx]

            cb = CatBoostRegressor(
                iterations=2500, learning_rate=0.035, depth=10,
                loss_function="MAE", eval_metric="MAE",
                early_stopping_rounds=60, random_seed=42, verbose=False
            )
            cb.fit(X_tr, y_tr, eval_set=(X_te, y_te))
            rf = RandomForestRegressor(n_estimators=500, max_depth=28,
                                       min_samples_leaf=4, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)

            cbs.append(cb); rfs.append(rf)
            val_preds_cb.append(np.expm1(cb.predict(X_te)))
            val_preds_rf.append(np.expm1(rf.predict(X_te)))
            val_true.append(np.expm1(y_te))

        val_preds_cb = np.concatenate(val_preds_cb)
        val_preds_rf = np.concatenate(val_preds_rf)
        val_true     = np.concatenate(val_true)

        # auto-blend weight search on validation folds (round to integer laps for MAE)
        best_w, best_mae = 0.5, 1e9
        for w in np.linspace(0.0, 1.0, 21):
            p = np.rint(w*val_preds_cb + (1.0-w)*val_preds_rf)
            mae = mean_absolute_error(val_true, p)
            if mae < best_mae:
                best_mae, best_w = mae, w

        if verbose:
            print(f"{target_col}: CV integer-lap MAE = {best_mae:.2f} (blend w={best_w:.2f})")

        # refit on all data once weight is chosen
        cb_full = CatBoostRegressor(
            iterations=2500, learning_rate=0.035, depth=10,
            loss_function="MAE", eval_metric="MAE",
            early_stopping_rounds=60, random_seed=42, verbose=False
        )
        cb_full.fit(Xp, yp, verbose=False)
        rf_full = RandomForestRegressor(n_estimators=500, max_depth=28,
                                        min_samples_leaf=4, random_state=42, n_jobs=-1)
        rf_full.fit(Xp, yp)
        return (cb_full, rf_full, best_w, base_feats, (val_true, val_preds_cb, val_preds_rf, best_mae))

    pit1_models = _fit_pit_regressor("FirstPit")
    pit2_models = _fit_pit_regressor("SecondPit")

    # quick feature ablation for CtorRating (retrain light models without it)
    if verbose:
        def _ablation_without_ctor(target_col: str):
            dfp = hist.dropna(subset=[target_col]).copy()
            feats_wo = [f for f in base_feats if f != "CtorRating"]
            Xp, yp = dfp[feats_wo], np.log1p(dfp[target_col])
            X_tr, X_te, y_tr, y_te = train_test_split(Xp, yp, test_size=0.25, random_state=42)
            cb = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=8,
                                   loss_function="MAE", early_stopping_rounds=40,
                                   random_seed=42, verbose=False)
            cb.fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=False)
            rf = RandomForestRegressor(n_estimators=300, max_depth=24,
                                       min_samples_leaf=4, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            # integer-lap MAE
            preds = np.rint(0.5*np.expm1(cb.predict(X_te)) + 0.5*np.expm1(rf.predict(X_te)))
            mae = mean_absolute_error(np.expm1(y_te), preds)
            print(f"[Ablation] {target_col} MAE without CtorRating: {mae:.2f}")
        _ablation_without_ctor("FirstPit")
        _ablation_without_ctor("SecondPit")

    return clf, pit1_models, pit2_models

########################################
# 5) ENUMERATE STRATEGIES FOR ONE RACE
########################################
def enumerate_strategies(clf, pit1_models, pit2_models, hist, track: str, date: str):
    # weather & degradation
    temp, rain = fetch_weather(
        TRACK_LOOKUP[track]['lat'],
        TRACK_LOOKUP[track]['lon'],
        date
    )
    total_laps = int(round(305 / TRACK_LOOKUP[track]['length_km'])) 
    deg_factor = total_laps / TRACK_LOOKUP[track]['length_km']

    # build “average” driver row from hist means
    feats = pit1_models[3] 
    stats = hist[feats].mean().to_dict()
    stats.update({
        "Temperature":     temp,
        "RainProbability": rain,
        "DegFactor":       deg_factor,
        "RaceLaps":        total_laps,
        "StartTyreLife":   35,
        "YearNorm":        stats.get("YearNorm", 2024),
        "Track":           stats.get("Track", hist["Track"].iloc[0] if "Track" in hist else 0),
    })
    Xnew = pd.DataFrame([stats])[feats]

    # model pit anchors (rounded to laps)
    cb1, rf1, w1, _, _ = pit1_models
    cb2, rf2, w2, _, _ = pit2_models
    pit1_anchor = int(np.rint(np.expm1(w1*cb1.predict(Xnew) + (1.0-w1)*rf1.predict(Xnew)))[0])
    pit2_anchor = int(np.rint(np.expm1(w2*cb2.predict(Xnew) + (1.0-w2)*rf2.predict(Xnew)))[0])

    out = []
    # If you have PRIORS for the track, use the compound lists from there.
    if track in PRIORS:
        for stops, prior in PRIORS[track].items():
            comps = prior["compounds"]
            anchors = [pit1_anchor, pit2_anchor]
            windows = _compound_windows(
                comps, total_laps, temp, rain, deg_factor, track,
                anchors=anchors, pull=0.4
            )
            out.append({"strategy": f"{stops}-stop", "compounds": comps, "windows": windows})
    else:
        # Fallback: assume a generic S/M/H 2-stop
        comps = ["C4","C3","C2"]
        windows = _compound_windows(
            comps, total_laps, temp, rain, deg_factor, track,
            anchors=[pit1_anchor, pit2_anchor], pull=0.4
        )
        out.append({"strategy":"Model-only","compounds":comps,"windows":windows})
    return out


########################################
# 6) PERMUTATION IMPORTANCE FOR CtorRating
########################################
def permutation_importance_ctor(models_tuple, hist, target_col: str, n_rounds=10):
    # Support both (5) and (6) element tuples
    if len(models_tuple) == 6:
        cb, rf, w, feats, _cv, _cat_idx = models_tuple
    elif len(models_tuple) == 5:
        cb, rf, w, feats, _cv = models_tuple
    else:
        raise ValueError(f"Unexpected model tuple length: {len(models_tuple)}")

    if "CtorRating" not in feats:
        print(f"Permutation importance skipped for {target_col} (CtorRating not in features).")
        return

    # Build X exactly like in training (you trained on log(target) with already-encoded features)
    dfp = hist.dropna(subset=[target_col]).copy()
    X = dfp[feats].copy()
    y_true_laps = dfp[target_col].values

    # Baseline predictions (blend, then round to integer laps)
    base_pred = np.rint(
        np.expm1(w * cb.predict(X) + (1.0 - w) * rf.predict(X))
    )
    base_mae = mean_absolute_error(y_true_laps, base_pred)

    # Permute CtorRating and re-evaluate
    deltas = []
    for _ in range(n_rounds):
        Xp = X.copy()
        Xp["CtorRating"] = np.random.permutation(Xp["CtorRating"].values)
        pred = np.rint(
            np.expm1(w * cb.predict(Xp) + (1.0 - w) * rf.predict(Xp))
        )
        deltas.append(mean_absolute_error(y_true_laps, pred) - base_mae)

    print(f"[Permutation] {target_col}: ΔMAE when permuting CtorRating = "
          f"{np.mean(deltas):.2f} ± {np.std(deltas):.2f}")

########################################
# 7) SCRIPT USAGE
########################################
if __name__=="__main__":
    # >>> Train on multiple years for accuracy <<<
    YEARS       = [2022, 2023, 2024]
    # Fill with actual GP dates you care about (local race dates)
    RACE_DATES  = {
        2022: "2022-04-10",
        2023: "2023-04-02",
        2024: "2024-03-24",
    }
    track = "Australia"
    date  = "2025-03-16"

    hist = build_historical(track, YEARS, RACE_DATES)
    clf, pit1_models, pit2_models = train_models(hist, verbose=True)

    # Constructor rating importance check
    permutation_importance_ctor(pit1_models, hist, "FirstPit", n_rounds=20)
    permutation_importance_ctor(pit2_models, hist, "SecondPit", n_rounds=20)

    # Display strategies
    print(f"\nOverall possible strategies for {track} on {date}:")
    print(" Stops │ Compounds            │ Window 1 │ Window 2 │ Window 3")
    print("-"*62)
    for s in enumerate_strategies(clf, pit1_models, pit2_models, hist, track, date):
        comps = " → ".join(s["compounds"]).ljust(20)
        w = s["windows"]
        w += [("", "")] * (3 - len(w))
        (w1s, w1e), (w2s, w2e), (w3s, w3e) = w
        def __f(a,b):
            return f"{a}-{b:<6}" if a != "" else " " * 8
        print(f"{s['strategy']:>6} │ {comps} │ {__f(w1s,w1e)} │ {__f(w2s,w2e)} │ {__f(w3s,w3e)}")
