import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


class Van:

    def __init__(self, van_id):
        self.packages = []
        self.battery = float(100)
        self.position = (0, 0)
        self.van_id = van_id

    def add_packages(self, new_packages:list=[]):
        self.packages.append(new_packages)

    def deliver_packages(self, pks_to_deliver):
        self.packages.remove(pks_to_deliver)

    def set_position(self, new_pos):
        self.position = new_pos


class LogisticCenter:

    def __init__(self, center_id, position, packages=pd.DataFrame(), charger1_type='slow', charger2_type='slow'):
        self.center_id = center_id
        self.position = position
        self.df_packages = packages
        self.chargers_type = [charger1_type, charger2_type]
        self.chargers_occupied = [False, False]
        self.charge_time = self.set_charge_time()

    def set_packages(self, packages):
        self.packages = packages

    def set_charge_time(self):
        times = []
        for ch in self.chargers_type:
            if ch == 'slow':
                times.append(1)
            if ch == 'fast':
                times.append(3)
        return times


class Solver:

    def __init__(self, verbose=False) -> None:

        self.df_centers = pd.read_csv("../data/Centros-RetoAccenture.csv", sep=';')
        self.df_chargers = pd.read_csv("../properties/chargers.csv")
        self.df_packages = pd.read_csv("../data/Paquetes-RetoAccenture.csv", sep=';')
        self.logistic_centers = self.set_centers()
        self.df_points = pd.read_csv("../data/Posiciones-RetoAccenture.csv", sep=';')
        self.vans_list = [Van(van_id='1'), Van(van_id='2'), Van(van_id='3'), Van(van_id='4')]

    def split_pckgs_by_center(self, center_id):
        packages = self.df_packages.loc[self.df_packages['id_centro'] == center_id]
        return packages

    def set_centers(self):
        centers_list = []
        for i in range(len(self.df_centers)):
            packages_c = self.split_pckgs_by_center(list(self.df_centers.id_centro.iloc[[i]])[0])
            centers_list.append(LogisticCenter(center_id=list(self.df_centers.id_centro.iloc[[i]])[0], position=(self.df_centers.iloc[[i]].coord_x, self.df_centers.iloc[[i]].coord_y), packages=packages_c))
        return centers_list

    def _calculate_distance_matrix(self):
        dist_array = pdist(self.df_points[['coord_x', 'coord_y']])
        dist_matrix = squareform(dist_array)
        return dist_matrix

    def solve_problem(self):
        dist_matrix = self._calculate_distance_matrix()
        return True


if __name__ == "__main__":
    test_solver = Solver()
    solution = test_solver.solve_problem()

