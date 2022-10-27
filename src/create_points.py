import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Points:

    def __init__(self,n_points, verbose=False) -> None:
   
        self.df_m30 = self.create_m30()
        self.df_centers = self.create_centers()
        self.df_points = self.create_points(n_points)
        
        if verbose:
            self.plot_points()
    
    def create_centers(self):

        df = pd.read_csv("data/points_centers.csv")

        return df
    
    def create_points(self,n_points):

        df = pd.DataFrame(columns=["coord_x","coord_y"])

        for i in range(n_points):
            df = self.add_point(df)
        
        return df.reset_index(drop=True)
    
    def create_m30(self):

        df = pd.read_csv("data/points_m30.csv")

        return df
    
    def add_point(self,df):

        df_aux = pd.concat([df,self.df_m30,self.df_centers])

        rand1, rand2 = np.random.randint(len(df_aux)), np.random.randint(len(df_aux))
        p1, p2 = df_aux.iloc[rand1], df_aux.iloc[rand2]

        t = np.random.random(1)
        pn = p2.coord_x + (p1.coord_x - p2.coord_x)*t, p2.coord_y + (p1.coord_y - p2.coord_y)*t

        df_point = pd.DataFrame([pn],columns=["coord_x","coord_y"])
  
        return pd.concat([df,df_point])
    
    def plot_points(self):
        figure = plt.figure(figsize=(10,12))
        plt.scatter(self.df_points.coord_x, self.df_points.coord_y, s=15, c='green',marker="x")
        plt.scatter(self.df_m30.coord_x, self.df_m30.coord_y, s=15, c='blue')
        plt.scatter(self.df_centers.coord_x, self.df_centers.coord_y, s=15, c='red')
        plt.show()
        
points = Points(150,verbose=True)
