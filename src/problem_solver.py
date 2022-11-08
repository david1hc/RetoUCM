import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np


class Van:

    def __init__(self, van_id, max_dist):
        self.packages = []
        self.battery = float(100)
        self.position = (0, 0)
        self.van_id = van_id
        self.max_distance = max_dist
        self.remaining_distance = max_dist
        self.velocity = 50

    def add_packages(self, new_packages: []):
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

    def set_charge_time(self):
        times = []
        for ch in self.chargers_type:
            if ch == 'slow':
                times.append(1)
            if ch == 'fast':
                times.append(3)
        return times


class Solver:

    def __init__(self, weight_maxdist=False, verbose=False) -> None:

        self.df_centers = pd.read_csv("../data/Centros-RetoAccenture.csv", sep=';')
        self.df_chargers = pd.read_csv("../properties/chargers.csv")
        self.df_packages = pd.read_csv("../data/Paquetes-RetoAccenture.csv", sep=';')
        self.logistic_centers = self.set_centers()
        self.df_points = pd.read_csv("../data/Posiciones-RetoAccenture.csv", sep=';')
        self.max_dist = self.set_max_dist(weight_maxdist)
        self.vans_list = [Van(van_id='1', max_dist=self.max_dist), Van(van_id='2', max_dist=self.max_dist), Van(van_id='3', max_dist=self.max_dist), Van(van_id='4', max_dist=self.max_dist)]
        # self.dist_matrix = self.calculate_distance_matrix(pd.concat([self.df_points[['coord_x', 'coord_y']], self.df_centers[['coord_x', 'coord_y']]]))
        self.velocity = 50
        self.feasible_solution = self.check_feasible_solution()

    def _split_pckgs_by_center(self, center_id):
        packages = self.df_packages.loc[self.df_packages['id_centro'] == center_id]
        return packages

    def set_centers(self):
        centers_list = []
        for i in range(len(self.df_centers)):
            packages_c = self._split_pckgs_by_center(list(self.df_centers.id_centro.iloc[[i]])[0])
            centers_list.append(LogisticCenter(center_id=list(self.df_centers.id_centro.iloc[[i]])[0], position=self.df_centers[['coord_x', 'coord_y']].iloc[[i]], packages=packages_c))
        return centers_list

    def set_max_dist(self, bool_weight=False):
        if bool_weight:
            return 300
        else:
            return 0.15

    def calculate_distance_matrix(self, df):
        dist_array = pdist(df)
        dist_matrix = squareform(dist_array)
        return dist_matrix

    def _add_pos_to_pckgs(self, packages):
        new_packages = packages.merge(self.df_points, how='left', on='id_pos')
        return new_packages

    def _calculate_dist_packages(self, df_packages, distances):
        # Función que ordena los paquetes de menor distancia a mayor.
        # El primer paquete es el más cercano al CL, el segundo paquete es el que tenga el punto de entrega más cercano
        # a la primera entrega, etc

        list_order = [0]
        list_distances = []
        dist_tmp = distances
        min_dist = np.min([n for n in dist_tmp[0, :] if np.where(dist_tmp[0, :] == n)[0][0] not in list_order])
        min_index = np.where(dist_tmp[0, :] == min_dist)[0]
        min_index = min_index.tolist()
        list_order = list_order + min_index
        if len(min_index) > 1:
            list_distances = list_distances + min_dist + [0]*(len(min_index)-1)
        else:
            list_distances.append(min_dist)

        while len(list_order) < np.shape(distances)[0]:
            row = dist_tmp[list_order[-1], :]
            min_dist = np.min([n for n in row if n > 0 and np.where(row == n)[0][0] not in list_order])
            min_index = np.where(row == min_dist)[0]
            min_index = min_index.tolist()
            list_order = list_order + min_index
            min_dist = [min_dist]
            if len(min_index) > 1:
                list_distances = list_distances + min_dist + [0] * (len(min_index) - 1)
            else:
                list_distances = list_distances + min_dist
#             if len(list_order) == np.shape(distances)[0]:
#                 break

        list_order = [n-1 for n in list_order if n > 0]
        ordered_packages = df_packages #.reindex(list_order)
        # Se crea una columna 'distancia_cl' con la distancia de cada punto de entrega al centro logistico
        ordered_packages['distancia_cl'] = distances[0, 1:]

        array_dist_p = np.asarray(list_distances)
        ordered_packages = ordered_packages.reindex(list_order)
        # Se crea una columna 'distancia_p' con la distancia de entrega a entrega
        ordered_packages['distancia_p'] = array_dist_p
        ordered_packages = ordered_packages.reset_index(drop=True)
        return ordered_packages

    def _calculate_groups_pckgs(self, df_packages_dist):
        list_groups = []
        i = 0
        while i < df_packages_dist.shape[0]:
            group_tmp = []
            group_tmp.append(i)
            dist_tmp = df_packages_dist.distancia_cl[i] + df_packages_dist.distancia_p[i]

            while True:
                if i+1 < df_packages_dist.shape[0]:
                    # Se comprueba si el paquete siguiente se entrega en la misma posición antes de sumar las distancias
                    if df_packages_dist.distancia_p[i+1] > 0:
                        dist_tmp1 = df_packages_dist.distancia_p[i+1] + df_packages_dist.distancia_cl[i+1]
                    else:
                        dist_tmp1 = 0
                    # Se comprueba que entregando el siguiente paquete no se supera la distancia maxima que puede
                    # recorrer el vehiculo
                    if dist_tmp + dist_tmp1 >= self.max_dist:
                        list_groups.append(group_tmp)
                        i = i+1
                        break
                    else:
                        i = i+1
                        group_tmp.append(i)
                        dist_tmp = dist_tmp + dist_tmp1
                else:
                    list_groups.append(group_tmp)
                    i = i+1
                    break

        list_df = []
        for group in list_groups:
            list_df.append(df_packages_dist.iloc[group])

        return list_df

    def group_packages(self, center):
        # Funcion que dado un objeto LogisticCenter calcula rutas de entrega de paquetes en funcion de la distancia
        # maxima que puede recorrer una furgoneta.

        df_packages = self._add_pos_to_pckgs(center.df_packages)
        df_pos_center = center.position
        distances = self.calculate_distance_matrix(pd.concat([df_pos_center, df_packages[['coord_x', 'coord_y']]]))
        df_packages_dist = self._calculate_dist_packages(df_packages, distances)
        list_groups_pckgs = self._calculate_groups_pckgs(df_packages_dist)
        return list_groups_pckgs

    def check_feasible_solution(self):
        for center in self.logistic_centers:
            groups_pckgs = self.group_packages(center)
            # TODO: separar rutas de entrega en furgonetas
            # TODO: calcular tiempo de entrega total

        return False



if __name__ == "__main__":
    test_solver = Solver()

