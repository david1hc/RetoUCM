import pandas as pd
from utils import chargers_prop

def create_centers():

    df = pd.read_csv("utils/data/points_centers.csv")
    df["chargers_type"] = df["id_centro"].map(chargers_prop)

    return df

def create_m30():

    df = pd.read_csv("utils/data/points_m30.csv").reset_index(drop=True)
    df["id_point"] = pd.Series(df.index).apply(lambda x: f"P{x}")
    
    return df

df_m30 = create_m30()
df_centers = create_centers()