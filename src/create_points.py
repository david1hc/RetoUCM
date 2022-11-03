import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Points:

    def __init__(self, n_points, verbose=False) -> None:
        
        
        self.df_m30 = self.create_m30()
        self.polygon_m30 = self.create_m30_polygon()
        self.df_centers = self.create_centers()
        self.df_points = self.create_points(n_points)
        if verbose:
            self.plot_points()

    def create_centers(self):

        df = pd.read_csv("data/points_centers.csv")
        df["id_point"] = df.index

        return df
    
    def create_m30_polygon(self):

        return Polygon(zip(self.df_m30["coord_x"], self.df_m30["coord_y"]))
    
    def point_in_m30(self, p):
        
        x, y = p["coord_x"],p["coord_y"]
        return self.polygon_m30.contains(Point(x,y))

    def create_points(self, n_points):

        df = pd.DataFrame()

        while len(df) < n_points:
            df["coord_x"] = self.df_m30.coord_x.min() + \
                np.random.random(n_points)*(self.df_m30.coord_x.max() - self.df_m30.coord_x.min())
            df["coord_y"] = self.df_m30.coord_y.min() + \
                np.random.random(n_points)*(self.df_m30.coord_y.max() - self.df_m30.coord_y.min())
            df["in"] = df.apply(lambda p: self.point_in_m30(p),axis=1)
            
            df = pd.concat([
                df[df["in"]],
                self.create_points(sum(~df["in"]))
            ])

        return df.reset_index(drop=True)

    def create_m30(self):

        df = pd.read_csv("data/points_m30.csv").reset_index(drop=True)
        df["id_point"] = df.index
        return df

    def plot_points(self):
        
        figure = plt.figure(figsize=(10, 10))

        plt.scatter(
            self.df_points["coord_x"],
            self.df_points["coord_y"],
            s=15,
            c='green',
            marker="x")

        plt.scatter(
            self.df_m30["coord_x"],
            self.df_m30["coord_y"],
            s=15,
            c='blue')

        plt.scatter(
            self.df_centers["coord_x"],
            self.df_centers["coord_y"],
            s=15,
            marker="o",
            c='red')
        
        plt.show()
