from properties.properties import properties_package as pack_prop
from random import randint, choices
from copy import deepcopy

types_dict = {}
for t in pack_prop['types']:
    type, prob, min_weight, max_weight = eval(t)
    types_dict[type] = {}
    types_dict[type]['prob'] = prob
    types_dict[type]['min_weight'] = min_weight
    types_dict[type]['max_weight'] = max_weight


class Package:

    def __init__(self, weight: int, st_pos: str, del_pos: str):
        '''

        :param weight: Peso del paquete
        :param st_pos: Posición inicial del paquete
        :param del_pos: Posición de reparto del paquete
        '''

        self.weight = weight
        self.pos = st_pos
        self.start_pos = st_pos
        self.deliver_pos = del_pos


def random_package_types(k: int) -> [int]:
    '''
    Función para obtener k tipos de paquete aleatorios

    :param k: Número de tipos a generar
    :return: Tipo de paquete aleatorio (1, 2 o 3)
    '''

    types = []
    probs = []
    for type, props in types_dict.items():
        types.append(type)
        probs.append(props['prob'])
    sel_types = choices(types, probs, k=k)
    return sel_types


def random_package_weights(types: [int]) -> [int]:
    '''
    Función para obtener k pesos de paquete aleatorios

    :param types: Lista de tipos
    :return: Tipo de pesos aleatorios (1-50)
    '''

    sel_weights = []
    for type in types:
        min_weight = types_dict[type]['min_weight'] + 1
        max_weight = types_dict[type]['max_weight']
        sel_weights.append(randint(min_weight, max_weight))
    return sel_weights


def random_start_pos(pos_list: [str], k: int) -> [str]:
    '''
    Función para obtener k posiciones iniciales de paquete aleatorias

    :param pos_list: Lista de posibles posiciones
    :param k: Número de posiciones
    :return: Lista de k posiciones iniciales
    '''

    return choices(pos_list, k=k)


def create_random_packages(k: int, pos_list: [str] = ['a', 'b', 'c'], del_list: [str] = []) -> [Package]:
    '''
    Función para crear una lista con k paquetes aleatorios

    :param k: Número de paquetes a crear
    :param pos_list: Lista de posibles posiciones iniciales
    :param del_list: Lista de posibles posiciones de entrega
    :return: Lista de k Paquetes
    '''

    types = random_package_types(k)
    weights = random_package_weights(types)
    start_pos_list = random_start_pos(pos_list, k)
    if not del_list:
        del_pos_list =  []
        for st_pos in start_pos_list:
            pos_list2 = deepcopy(pos_list)
            pos_list2.remove(st_pos)
            del_pos_list.append(choices(pos_list2, k=1)[0])
    else:
        del_pos_list = choices(del_list, k=k)
    data = [Package(*x) for x in list(zip(weights, start_pos_list, del_pos_list))]
    return data


if __name__ == '__main__':
    g = create_random_packages(4)
    p = Package(4, 'a', 'b')
