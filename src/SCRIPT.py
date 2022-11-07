import pandas as pd
import os
from datetime import datetime as dt

from utils import Points
from utils import Package

class GetInput:

    def __init__(self, num_packg, num_points):

        self.num_packg = num_packg
        self.num_points = num_points
        today = dt.today()
        self.path_base = f"data/{self.num_packg}/{self.num_points}/{today.strftime('%Y%m%d')}_{today.strftime('%H%M%S')}_0"

        points = Points(num_points)
        package = Package(num_packg, num_points)

        self.df_centers = points.df_centers
        self.df_points = points.df_points
        self.df_packg = package.df_packg

    def save(self):
    
        if not(os.path.exists(f"data/{self.num_packg}")):
            os.mkdir(f"data/{self.num_packg}")
        if not(os.path.exists(f"data/{self.num_packg}/{self.num_points}")):
            os.mkdir(f"data/{self.num_packg}/{self.num_points}")
        k = 1
        pbase = self.path_base[:-1]
        while os.path.exists(self.path_base):

            self.path_base = f"{pbase}{k}"
            k+=1
        
        os.mkdir(self.path_base)

        self.df_centers.to_csv(f"{self.path_base}/centers.csv",index=False,sep=";")
        self.df_points.to_csv(f"{self.path_base}/points.csv",index=False,sep=";")
        self.df_packg.to_csv(f"{self.path_base}/packages.csv",index=False,sep=";")