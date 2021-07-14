# -*- coding: utf-8 -*-
"""
download stable elasticity data (task_id, formula, and K_VRH) from Materials Project API and export train/val/test.csv files.

Created on Mon Jul 12 21:22:35 2021

@author: sterg
"""
from pymatgen.ext.matproj import MPRester
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join
from pathlib import Path

# data directory
data_dir = join("data", "materials_data", "stable_elasticity")
Path(data_dir).mkdir(parents=True, exist_ok=True)

# MPRester
props = ["task_id", "pretty_formula", "elasticity"]
with MPRester() as m:
    results = m.query(
        {"e_above_hull": {"$lt": 0.5}, "elasticity": {"$exists": True}},
        properties=props,
    )

# separate mpids, formulas, etc.
mpids = [d["task_id"] for d in results]
formulas = [d["pretty_formula"] for d in results]
elasticity = [d["elasticity"] for d in results]
K_VRH = [d["K_VRH"] for d in elasticity]

# combine into DataFrame
df = pd.DataFrame({"formula": formulas, "task_id": mpids, "target": K_VRH})

# train, val, test splits
test_size = 0.1
train_size = 0.7
dummy_size = 1 - train_size / (1 - test_size)
train_df, test_df = train_test_split(df, test_size=test_size, random_state=1)
train_df, val_df = train_test_split(train_df, test_size=dummy_size, random_state=1)

# export
train_df.to_csv(join(data_dir, "train.csv"), index=False)
val_df.to_csv(join(data_dir, "val.csv"), index=False)
test_df.to_csv(join(data_dir, "test.csv"), index=False)
