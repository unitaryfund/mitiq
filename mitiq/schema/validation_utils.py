import json
import jsonschema
from jsonschema import validate


def load(schema_path):
    with open(schema_path, 'r') as file:
        return json.load(file)
    

def validate_experiment(experiment, schema):
    try:
        validate(instance=experiment, schema=schema)
        print("expertiment validation passed")
    except jsonschema.exceptions.ValidationError as e:
        print("experiment validation failed")
    return None

