import yaml


properties: dict = None

# Load properties common all project
filename = "package_properties.yml"

with open(fr"properties/{filename}") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    properties_package = yaml.safe_load(file)
