import numpy as np
import pandas as pd

def compute_avg_stint_decay(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Given a FastF1 laps DataFrame (with columns Driver, LapNumber, Sector*Times),
    returns a DataFrame with columns [Driver, AvgStintDecay] where AvgStintDecay
    is the average slope (seconds lost per lap) across each stint for that driver.
    A stint is defined by consecutive laps between pit-ins.
    """
    # make a working copy
    laps = laps.copy()

    # 1) If LapTime exists as a timedelta, convert it to float seconds.
    # 2) If LapTime is missing entirely, compute it from the three sector times.
    if "LapTime" in laps.columns:
        if pd.api.types.is_timedelta64_dtype(laps["LapTime"]):
            laps["LapTime"] = laps["LapTime"].dt.total_seconds()
    else:
        laps["LapTime"] = (
            laps.Sector1Time.dt.total_seconds()
            + laps.Sector2Time.dt.total_seconds()
            + laps.Sector3Time.dt.total_seconds()
        )

    # build a StintID per driver by counting pit-ins
    laps["StintID"] = (
        laps
          .groupby("Driver")["PitInTime"]
          .transform(lambda s: s.notna().cumsum())
    )

    decay_slopes = []
    for (drv, stint), grp in laps.groupby(["Driver", "StintID"]):
        if len(grp) >= 3:
            # now LapTime is pure float, so polyfit will work
            m, _ = np.polyfit(grp.LapNumber, grp.LapTime, 1)
            decay_slopes.append((drv, m))

    decay_df = pd.DataFrame(decay_slopes, columns=["Driver", "DecaySlope"])
    return (
        decay_df
          .groupby("Driver")["DecaySlope"]
          .mean()
          .reset_index(name="AvgStintDecay")
    )
