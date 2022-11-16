import os
os.chdir('../..')

import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
from datetime import time, timedelta, datetime, date
from utils import GetInput


class Van:
    """
    Clase para represenatar una furgoneta
    ...

    Attributos
    ----------
    TODO

    Methods
    -------
    TODO
    """
    def __init__(self, van_id, max_dist):
        """
        Parameters
        ----------
        van_id : str
            El id de la furgoneta
        max_dist : float
            La distancia maxima que la furgoneta puede recorrer con la bateria cargada al maximo
        """
        self.packages = []
        self.battery = float(100)
        self.position = pd.DataFrame(columns=['coord_x', 'coord_y'])
        self.van_id = van_id
        self.max_distance = max_dist
        self.remaining_distance = max_dist
        self.velocity = 50
        self.delivered_packages = pd.DataFrame(columns=['id_pos', 'paquetes', 'distancia'])

    def add_packages(self, new_packages: list = []):
        self.packages.append(new_packages)

    def deliver_packages(self, df_pckgs2deliver, total_deliver_distance, last_pos):
        self.delivered_packages = pd.concat(self.delivered_packages, df_pckgs2deliver)
        self.remaining_distance = self.remaining_distance - total_deliver_distance
        self.position = last_pos

    def deliver_packages(self, pks_to_deliver):
        self.packages.remove(pks_to_deliver)

    def set_position(self, new_pos):
        self.position = new_pos


class LogisticCenter:
    """
    Clase para represenatar un centro logistico
    ...

    Attributos
    ----------
    TODO

    Methods
    -------
    TODO
    """
    def __init__(self, center_id, position, packages=pd.DataFrame(), charger1_type='slow', charger2_type='slow'):
        """
        Parameters
        ----------
        center_id : str
            El id del centro logistico
        position : Pandas DataFrame.
            Coordenadas del centro logistico
            Columnas: ['coord_x', 'coord_y']

        packages: Pandas DataFrame.
            Paquetes almacenados inicialmente en el centro logistico
            Columnas: ['peso', 'id_centro', 'id_pos', 'id_paquete']
                -'peso': peso del paquete.
                -'id_centro': id del centro logistico en el que inicialmente esta almacenado el paquete.
                -'id_pos': id de la posicion a la que debe ser entregado el paquete.
                -'id_paquete': id del paquete.
        charger1_type: str
            Tipo de carga del cargador 1. ('slow' o 'fast')
        """
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
    """
    Clase para solucionar el problema de optimizacion de rutas con vehiculos electricos.
    ...

    Attributos
    ----------
    TODO

    Methods
    -------
    TODO
    """
    def __init__(self, bool_weight_maxdist=False, path_input=None, save_output=False, verbose=False) -> None:
        """
        Parameters
        ----------
        bool_weight_maxdist : bool
            Flag que toma el valor:
                -True si distancia maxima a recorrer = distancia maxima sin peso transportado.
                -False si distancia maxima a recorrer = distancia maxima con peso maximo transportado.

        path input : str
            ruta a directorio con los 3 archivos csv de entrada:
                'centers.csv'
                'packages.csv'
                'points.csv'
        """
        self.df_centers = pd.read_csv(path_input+"/centers.csv", sep=';')
        # self.df_chargers = pd.read_csv("../properties/chargers.csv")
        self.df_packages = pd.read_csv(path_input+"/packages.csv", sep=';')
        self.logistic_centers = self.set_centers()
        self.df_points = pd.read_csv(path_input+"/points.csv", sep=';')
        self.max_dist = self.set_max_dist(bool_weight_maxdist)
        self.vans_list = [Van(van_id='1', max_dist=self.max_dist), Van(van_id='2', max_dist=self.max_dist), Van(van_id='3', max_dist=self.max_dist), Van(van_id='4', max_dist=self.max_dist)]
        self.velocity = 50
        self.dist_factor = self.calculate_dist_factor()
        self.output_df = self.init_output_df()
        self.feasible_solution = self.check_feasible_solution(save_output=save_output)
        self.groups_packages = None

    def save_solution(self, outputpath):
        self.output_df.to_csv(outputpath, index=False)
    def calculate_dist_factor(self):
        coord_c1 = self.df_centers[self.df_centers['id_centro'] == 'C1'][['coord_x', 'coord_y']].reset_index(drop=True)
        coord_c2 = self.df_centers[self.df_centers['id_centro'] == 'C2'][['coord_x', 'coord_y']].reset_index(drop=True)
        dist_c1c2 = np.sqrt((coord_c1.coord_x - coord_c2.coord_x)**2 + (coord_c1.coord_y - coord_c2.coord_y)**2)
        dist_km = 7.54
        factor_km = dist_km/dist_c1c2.iloc[0]
        return factor_km

    def init_output_df(self):
        df = pd.DataFrame(columns=['id_furgo', 'id_pos', 'hora_llegada', 'hora_salida', 'ids_paquetes'])
        return df
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
            return 210

    def calculate_distance_matrix(self, df):
        dist_array = pdist(df)
        dist_matrix = squareform(dist_array)*self.dist_factor
        return dist_matrix

    def _add_pos_to_pckgs(self, packages):
        new_packages = packages.merge(self.df_points, how='left', on='id_pos')
        return new_packages

    def _calculate_dist_packages(self, df_packages, distances):
        # Funcion que ordena los paquetes de menor distancia a mayor.
        # El primer paquete es el más cercano al CL, el segundo paquete es el que tenga el punto de entrega mas cercano
        # a la primera entrega, etc

        list_order = [0]
        list_distances = []
        dist_tmp = distances
        min_dist = np.min([n for n in dist_tmp[0, :] if np.where(dist_tmp[0, :] == n)[0][0] not in list_order])
        min_index = np.where(dist_tmp[0, :] == min_dist)[0]
        min_index = min_index.tolist()
        list_order = list_order + min_index
        if len(min_index) > 1:
            list_distances = [min_dist] + [0]*(len(min_index)-1)
        else:
            list_distances.append(min_dist)

        while len(list_order) < np.shape(distances)[0]:
            # Fila con las distancias del ultimo paquete guardado en la lista al resto de paquetes
            row = dist_tmp[list_order[-1], :]
            # Distancia minima desde el ultimo paquete entregado excluyendo paquetes que se entregan en el mismo punto y
            # paquetes ya guardados
            min_dist = np.min([n for n in row if n > 0 and np.where(row == n)[0][0] not in list_order])
            min_index = np.where(row == min_dist)[0]
            min_index = min_index.tolist()
            # Se concatenan los indices de los paquetes que se entreguen en la posicion mas cercana
            list_order = list_order + min_index
            min_dist = [min_dist]
            # Si mas de un paquete se entregan en la posicion mas cercana se concatena la distancia seguida de tantos
            # ceros como paquetes adicionales se entreguen en ese punto
            if len(min_index) > 1:
                list_distances = list_distances + min_dist + [0] * (len(min_index) - 1)
            else:
                list_distances = list_distances + min_dist

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
        # Funcion que genera una lista de DataFrames. La distancia recorrida para entregar los paquetes de cada df es
        # igual o menor que la distancia maxima que puede recorrer una furgoneta que comienza con la bateria al maximo.
        list_groups = []
        i = 0
        # Bucle en el que se guarda en listas los indices de los paquetes que se repartiran en un mismo trayecto,
        # comenzando en el centro logistico y acabando en el centro logistico
        # (para recargar la bateria antes de la siguiente ruta)
        while i < df_packages_dist.shape[0]:
            group_tmp = []
            group_tmp.append(i)
            if len(group_tmp) == 1:
                dist_tmp = df_packages_dist.distancia_cl[i]
            else:
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
                    # Se añade el ultimo grupo de paquetes a la lista
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

    def get_iddle_van(self):
        iddle_id = None
        for i in range(len(self.vans_list)):
            if len(self.vans_list[i].packages) == 0:
                iddle_id = self.vans_list[i].van_id
                return i, iddle_id
        return iddle_id

    def map_weight2time_deliver(self, weight):
        if weight <= 5:
            time = 5/60
            return time
        if weight <= 20:
            time = 7/60
            return time
        else:
            time = 15/60
            return time

    def individual_pkg_time(self, df):
        df['t_entrega'] = df.peso.apply(self.map_weight2time_deliver)
        return df['t_entrega']

    def get_deliver_time(self, df_packages):
        df_packages['t_entrega'] = self.individual_pkg_time(df_packages)
        total_distance = df_packages['distancia_p'].iloc[1:-1].sum() + df_packages['distancia_cl'].iloc[0] + df_packages['distancia_cl'].iloc[-1]
        # total_distance = total_distance + df_packages['distancia_cl'].iloc[0] + df_packages['distancia_cl'].iloc[-1]
        deliver_time = total_distance/self.velocity
        deliver_time = deliver_time + df_packages['t_entrega'].sum()
        return deliver_time

    def add_group2output(self, van_id, pkgs_groups):
        # Funcion que guarda la ruta de una furgoneta van_id para entregar todos los paquetes de la lista pkgs_groups
        # en el data fram output. Para cada grupo de paquetes se repite:
        #
        #   paso_1: recoger paquetes:
        #      -hora_salida=hora_llegada
        #      -ids_paquetes=lista paquetes que se va a entregar antes de volver al centro logistico.
        #      -id_pos=id centro logistico
        #
        #   paso_2 a paso_n-1: entregar paquetes en order.
        #
        #   paso_n: volver al centro logistico:
        #      -si ya se ha entregado el ultimo grupo de paquetes que le quedaba a la furgoneta por entregar, no se
        #      recarga la bateria.
        #      -si no es el ultimo grupo de paquetes que le queda a la furgoneta por entregar, se recarga la bateria.

        for n_g, group in enumerate(pkgs_groups):
            i = 0
            while True:
                if i == 0:
                    if len(self.output_df.loc[self.output_df.id_furgo==van_id].index) == 0:
                        # El primer paquete repartido por la furgoneta empieza a las 8am
                        hora_llegada = time(8, 0, 0)
                        hora_salida = hora_llegada
                        id_pos = group.iloc[0].id_centro
                        ids_paquetes = group.id_paquete.tolist()
                    else:
                        id_pos = group.iloc[i].id_pos
                        # Ultima hora salida guardada en output_df
                        prev_time = self.output_df.hora_salida.iloc[-1]
                        # Calculo hora_llegada del centro logistico al punto de entrega
                        delta_t_llegada = group.distancia_cl.iloc[i]/self.velocity
                        dt_hora_llegada = datetime.combine(date.today(), prev_time) + timedelta(hours=delta_t_llegada)
                        hora_llegada = dt_hora_llegada.time()
                        # Calculo hora_salida
                        t_entrega = self.individual_pkg_time(group.loc[group.id_pos == group.iloc[i].id_pos])
                        delta_t_salida = t_entrega.sum()
                        dt_hora_salida = dt_hora_llegada + timedelta(hours=delta_t_salida)
                        hora_salida = dt_hora_salida.time()
                        ids_paquetes = group.loc[group.id_pos == group.iloc[i].id_pos]['id_paquete'].tolist()
                        i = i + len(ids_paquetes)

                    self.output_df.loc[len(self.output_df.index)] = [van_id, id_pos, hora_llegada, hora_salida, ids_paquetes]
                else:
                    id_pos = group.iloc[i].id_pos
                    # Ultima hora salida guardada en output_df
                    prev_time = self.output_df.hora_salida.iloc[-1]
                    # Calculo hora_llegada del punto de entrega anterior al nuevo punto de entrega
                    delta_t_llegada = group.distancia_p.iloc[i] / self.velocity
                    dt_hora_llegada = datetime.combine(date.today(), prev_time) + timedelta(hours=delta_t_llegada)
                    hora_llegada = dt_hora_llegada.time()
                    # Calculo hora_salida
                    t_entrega = self.individual_pkg_time(group.loc[group.id_pos == group.iloc[i].id_pos])
                    delta_t_salida = t_entrega.sum()
                    dt_hora_salida = dt_hora_llegada + timedelta(hours=delta_t_salida)
                    hora_salida = dt_hora_salida.time()
                    ids_paquetes = group.loc[group.id_pos == group.iloc[i].id_pos]['id_paquete'].tolist()
                    self.output_df.loc[len(self.output_df.index)] = [van_id, id_pos, hora_llegada, hora_salida,
                                                                     ids_paquetes]
                    i = i + len(ids_paquetes)

                if i == len(group):
                    id_pos = group.iloc[0].id_centro
                    prev_time = self.output_df.hora_salida.iloc[-1]
                    # Calculo hora_llegada del punto de entrega al centro logistico
                    delta_t_llegada = group.distancia_cl.iloc[-1]/self.velocity
                    dt_hora_llegada = datetime.combine(date.today(), prev_time) + timedelta(hours=delta_t_llegada)
                    hora_llegada = dt_hora_llegada.time()
                    if n_g == len(pkgs_groups)-1:
                        # Para despues del ultimo grupo de paquetes no se carga la bateria
                        hora_salida = hora_llegada
                        ids_paquetes = []
                    else:
                        charge_time = 3  # TODO: coger el tiempo de carga rapido o lento segun disponibilidad
                        delta_t_salida = charge_time
                        dt_hora_salida = dt_hora_llegada + timedelta(hours=delta_t_salida)
                        hora_salida = dt_hora_salida.time()
                        ids_paquetes = pkgs_groups[n_g+1]['id_paquete'].tolist()
                    self.output_df.loc[len(self.output_df.index)] = [van_id, id_pos, hora_llegada, hora_salida,
                                                                     ids_paquetes]
                    break




    def assign_pckgs2vans(self, center, save_output=False):
        #
        van_n, van_id = self.get_iddle_van()
        self.vans_list[van_n].packages = center.groups_pckgs
        total_time = 0
        for group in center.groups_pckgs:
            deliver_time = self.get_deliver_time(group)
            total_time = total_time + deliver_time
        if save_output:
            self.add_group2output(van_id, center.groups_pckgs)
        charge_time = 0
        if len(center.groups_pckgs) > 1:
            if center.chargers_occupied[0]:
                charge_time = center.charge_time[1]
                center.chargers_occupied[1] = True
            else:
                charge_time = center.charge_time[0]*(len(center.groups_pckgs)-1)
                center.chargers_occupied[0] = True
        total_time = total_time + charge_time
        return total_time

    def check_feasible_solution(self, save_output=False):
        max_len = 0
        max_len_center = ''

        for center in self.logistic_centers:
            center.groups_pckgs = self.group_packages(center)
            if len(center.groups_pckgs) > max_len:
                max_len = len(center.groups_pckgs)
                max_len_center =center.center_id
        vans_time = []
        for center in self.logistic_centers:
            if center.center_id == max_len_center and max_len > 1:
                n_mid_pckg = np.floor(max_len/2)
                time = self.assign_pckgs2vans(center.groups_pckgs[:n_mid_pckg], save_output=save_output)
                vans_time.append(time)
                time = self.assign_pckgs2vans(center.groups_pckgs[n_mid_pckg:], save_output=save_output)
                vans_time.append(time)
            else:
                time = self.assign_pckgs2vans(center, save_output=save_output)
                vans_time.append(time)

        if max(vans_time) > 8:
            feasible_solution = False
        else:
            feasible_solution = True

        return feasible_solution


if __name__ == "__main__":

    gi = GetInput(100, 50)  # Limite 115 paquetes, 100 puntos
    gi.save()
    path = gi.path_base
    output_path = os.path.join(path, 'output.csv')
    test_solver = Solver(path_input=path, save_output=True)
    test_solver.save_solution(output_path)
    if test_solver.feasible_solution:
        print('There is a feasible solution for the input dataset')
    else:
        print('There is NOT a feasible solution for the input dataset')

