import pandas as pd


class Solver:

    def __init__(self, n_vans=4, verbose=False) -> None:

        self.n_vans = n_vans
        self.df_centers = self.create_centers()

    def create_centers(self):

        df = pd.read_csv("data/points_centers.csv")
        df["id_point"] = df.index

        return df