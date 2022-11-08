import yaml

properties: dict = None

with open("utils/data/package_properties.yml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    dict_properties = yaml.safe_load(file)
    properties_package = dict_properties["types"]
    properties_chargers = dict_properties["charger_type"]