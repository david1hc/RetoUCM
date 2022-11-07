from utils import df_centers, df_m30
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Points:

    def __init__(self, n_points, verbose=False) -> None:
        
        self.df_m30 = df_m30
        self.polygon_m30 = self.create_m30_polygon()
        self.df_centers = df_centers
        self.polygon_m30 = self.create_m30_polygon()
        self.df_points = self.create_points(n_points).drop(columns=["in"])
        if verbose:
            self.plot_points()
    
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
            ]).reset_index(drop=True)
        
        df["id_point"] = pd.Series(df.index).apply(lambda x: f"P{x}")

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
