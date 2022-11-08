from utils import pack_prop
from copy import deepcopy
import pandas as pd
import numpy as np

class Package:

    def __init__(self, num_packg, num_points):

        self.num_packg = num_packg
        self.num_points = num_points
        self.pos_list = ['A', 'B', 'C']
        self.types_dict = self.create_types_dict()
        self.df_packg = self.create_random_packages()

    @staticmethod
    def create_types_dict():
        types_dict = {}
        for t in pack_prop['types']:
            _type, prob, min_weight, max_weight = eval(t)
            types_dict[_type] = {}
            types_dict[_type]['prob'] = prob
            types_dict[_type]['min_weight'] = min_weight
            types_dict[_type]['max_weight'] = max_weight

        return types_dict

    def random_package_types(self, k: int) -> [int]:
        '''
        Función para obtener k tipos de paquete aleatorios

        :param k: Número de tipos a generar
        :return: Tipo de paquete aleatorio (1, 2 o 3)
        '''

        types = []
        probs = []
        for type, props in self.types_dict.items():
            types.append(type)
            probs.append(props['prob']/100)
        
        sel_types = np.random.choice(types, p=probs, size=k)
        return sel_types

    def random_package_weights(self, types: [int]) -> [int]:
        '''
        Función para obtener k pesos de paquete aleatorios

        :param types: Lista de tipos
        :return: Tipo de pesos aleatorios (1-50)
        '''

        sel_weights = []
        for type in types:
            min_weight = self.types_dict[type]['min_weight'] + 1
            max_weight = self.types_dict[type]['max_weight']
            sel_weights.append(np.random.randint(min_weight, max_weight))
        return sel_weights

    @staticmethod
    def random_start_pos(pos_list: [str], k: int) -> [str]:
        '''
        Función para obtener k posiciones iniciales de paquete aleatorias

        :param pos_list: Lista de posibles posiciones
        :param k: Número de posiciones
        :return: Lista de k posiciones iniciales
        '''

        return np.random.choice(pos_list, size=k)

    def create_del_post_list(self):
        del_list = [f"P{i}" for i in range(self.num_points)]

        if self.num_packg >= self.num_points:
            del_post_list = del_list[:self.num_points] + list(np.random.choice(del_list, size=self.num_packg - self.num_points))
            del_post_list = list(np.random.permutation(del_post_list))

        else:
            print("Hay más paquetes que puntos")
            del_post_list = []
        
        return del_post_list
        

    def create_random_packages(self) -> pd.DataFrame:

        types = self.random_package_types(self.num_packg)
        weights = self.random_package_weights(types)
        start_pos_list = self.random_start_pos(self.pos_list, self.num_packg)
        del_pos_list = self.create_del_post_list()

        df = pd.DataFrame(
                list(zip(weights, start_pos_list, del_pos_list)),
                columns=["peso","id_centro","id_pos"]
                )

        df["id_paquete"] = pd.Series(df.index).apply(lambda x: f"PK{x}")

        return df