# part_1_data_simulation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    "slope": np.random.uniform(0, 45, n_samples),  # in degrees
    "elevation": np.random.uniform(100, 1000, n_samples),  # in meters
    "soil_permeability": np.random.choice([1, 2, 3], n_samples),  # 1: low, 2: med, 3: high
    "land_use": np.random.choice(["urban", "forest", "agriculture", "barren"], n_samples),
    "rainfall": np.random.normal(800, 200, n_samples),  # mm/year
    "drainage_density": np.random.uniform(0.1, 2.5, n_samples),  # km/km²
    "depth_to_water_table": np.random.uniform(5, 100, n_samples)  # meters
})

def classify_recharge(row):
    score = 0

    if row["slope"] < 10:
        score = score + 1
    if row["soil_permeability"] == 3: 
        score = score + 1
    if row["land_use"] in ["agriculture", "forest"]: 
        score = score + 1
    if row["rainfall"] > 800: 
        score = score + 1
    if row["drainage_density"] < 1: 
        score = score + 1
    if row["depth_to_water_table"] < 30: 
        score = score + 1

    if score >= 5:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"

data["recharge_potential"] = data.apply(classify_recharge, axis = 1)

print(data.head())
data.to_csv("synthetic_groundwater_data.csv", index = False)