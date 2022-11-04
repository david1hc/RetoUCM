from properties.properties import properties_package as pack_prop
from random import randint, choices
from copy import deepcopy
import pandas as pd

#TODO Añadir id_paquete = index
#TODO Leer los id_pos de la clase Points

class Package:

    def __init__(self, num_packg):

        self.num_packg = num_packg
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
            probs.append(props['prob'])
        sel_types = choices(types, probs, k=k)
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
            sel_weights.append(randint(min_weight, max_weight))
        return sel_weights

    @staticmethod
    def random_start_pos(pos_list: [str], k: int) -> [str]:
        '''
        Función para obtener k posiciones iniciales de paquete aleatorias

        :param pos_list: Lista de posibles posiciones
        :param k: Número de posiciones
        :return: Lista de k posiciones iniciales
        '''

        return choices(pos_list, k=k)


    def create_random_packages(self,pos_list: [str] = ['a', 'b', 'c'], del_list: [str] = []) -> pd.DataFrame:
        '''
        Función para crear una lista con k paquetes aleatorios

        :param pos_list: Lista de posibles posiciones iniciales
        :param del_list: Lista de posibles posiciones de entrega
        :return: Lista de k Paquetes
        '''

        types = self.random_package_types(self.num_packg)
        weights = self.random_package_weights(types)
        start_pos_list = self.random_start_pos(pos_list, self.num_packg)
        if not del_list:
            del_pos_list =  []
            for st_pos in start_pos_list:
                pos_list2 = deepcopy(pos_list)
                pos_list2.remove(st_pos)
                del_pos_list.append(choices(pos_list2, k=1)[0])
        else:
            del_pos_list = choices(del_list, k=self.num_packg)
        df = pd.DataFrame(
                list(zip(weights, start_pos_list, del_pos_list)),
                columns=["peso","id_centro","id_pos"]
                )
        df["id_paquete"] = pd.Series(df.index).apply(lambda x: f"PK{x}")

        return df


if __name__ == '__main__':
    print(Package(4).df_packg)

