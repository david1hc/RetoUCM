import yaml

properties: dict = None

with open("utils/data/package_properties.yml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    properties_package = yaml.safe_load(file)
