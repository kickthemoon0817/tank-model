import pathlib

import numpy as np
import pandas as pd

from src.model import make_models
from src.opt import GAOptimizer
from src.vis import visualize_runoff, visualize_tuning

TANK_LEVEL = 4

TIMESTEPS = 60 * 60 * 24
AREA = 601.61

DATA_PATH = pathlib.Path("./data")
DATA_NAME = pathlib.Path("3009680_p4.csv")

STORAGE_RANGE = [
    (0, 100),
    (0, 100),
    (0, 100),
    (0, 100)
]
RUNOFF_RANGE = [
    (0.1, 0.5),
    (0.1, 0.5),
    (0.03, 0.1),
    (0.005, 0.01),
    (0.0005, 0.01),
]
SIDE_RANGE = [
    (5, 60),
    (5, 60),
    (0, 50),
    (0, 30),
    (0, 0),
]
INFIL_RANGE = [
    (0.1, 0.5),
    (0.01, 0.1),
    (0.005, 0.01),
    (0, 0),
]

GENERATION = 200
POPULATION = 100


base_range = {
    "storage_range" : STORAGE_RANGE,
    "side_outlet_height_range": SIDE_RANGE,
    "runoff_coef_range": RUNOFF_RANGE,
    "infiltration_coef_range": INFIL_RANGE
}

df_total = pd.read_csv(DATA_PATH / DATA_NAME)

df_total["date"] = pd.to_datetime(df_total["date"])

years = range(2009, 2013)
periods = []
for y in years:
    periods.extend([
        (f"{y}-02-01", f"{y}-06-14"),
        (f"{y}-06-15", f"{y}-09-30"),
        (f"{y}-10-01", f"{y+1}-01-31"),
    ])

def rmse_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean = np.mean(y_true)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rrmse = rmse / y_mean if y_mean != 0 else np.nan

    return rrmse

final_storage = None    # Set this None, since initial storage
                        # for the first period is not decided.

for start, end in periods:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    mask = (df_total["date"] >= start) & (df_total["date"] <= end)
    df_target = df_total.loc[mask]

    df_len = len(df_target)

    dates  = df_target["date"].to_numpy()
    precip_target = df_target["P"].to_numpy()
    runoff_target = df_target["Q"].to_numpy()
    aet_target = df_target["AET"].to_numpy()

    optimizer = GAOptimizer(
        tank_num=TANK_LEVEL,
        area=AREA,
        timesteps=TIMESTEPS,
        precip=precip_target,
        AET=aet_target,
        observed_runoff=runoff_target,
        objective=rmse_score,
        direction="minimize",
        fixed_storage=final_storage,
        generations=GENERATION,
        population_size=POPULATION,
        **base_range
    )
    result = optimizer.run()
    params_history = optimizer.get_history()

    best_coefs, best_value = result

    best_coefs.print()

    best_model = make_models(
        tank_num=TANK_LEVEL,
        coef=best_coefs,
        area=AREA,
        timesteps=TIMESTEPS
    )
    for i in range(df_len):
        best_model.update(precip_target[i], aet_target[i])
    tank_history = best_model.get_history()

    total_runoff, total_storage = tank_history

    final_storage = total_storage[-1]

    visualize_runoff(dates, runoff_target, total_runoff, show=True, save_path=None)
    visualize_tuning(params_history, show=True, save_path=None)
