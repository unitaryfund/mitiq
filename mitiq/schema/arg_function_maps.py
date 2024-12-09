from mitiq import zne, ddd, pec, rem
from mitiq.zne.scaling.folding import fold_global, fold_gates_at_random, fold_all
from mitiq.zne.scaling.layer_scaling import get_layer_folding
from mitiq.zne.scaling.identity_insertion import insert_id_layers

def ddd_schema_to_params(experiment):
    ddd_rule_map = {
    "xx": ddd.rules.xx,
    "yy": ddd.rules.yy,
    "xyxy": ddd.rules.xyxy, 
    "general": ddd.rules.general_rule, # need to adjust ddd_schema here to better allow for this 
    "repeated": ddd.rules.repeated_rule, # .. and this
    "custom": None, # ... and this
    }

    rule = ddd_rule_map[experiment['rule']]

    params = {
        "rule": rule
    }

    return params



def zne_schema_to_params(experiment):

    zne_noise_scaling_method_map = {
    "global": fold_global,
    "local_random": fold_gates_at_random,
    "local_all": fold_all,
    "layer": get_layer_folding,
    "identity_scaling": insert_id_layers
    }

    zne_extrapolation_map = {
        "linear": zne.inference.LinearFactory(scale_factors=experiment['noise_scaling_factors']),
        "richardson": zne.inference.RichardsonFactory(scale_factors=experiment['noise_scaling_factors']),
        "polynomial": zne.inference.PolyFactory(scale_factors=experiment['noise_scaling_factors'], order=0), # need to add order as an option in the metashcema here ...
        "exponential": zne.inference.ExpFactory(scale_factors=experiment['noise_scaling_factors']),
        "poly-exp": zne.inference.PolyExpFactory(scale_factors=experiment['noise_scaling_factors'], order=0), # .. and here
        "adaptive-exp": zne.inference.AdaExpFactory(scale_factor=experiment['noise_scaling_factors'][0], steps=4, asymptote=None), # need to adjust metaschema here for steps
    }

    extrapolation = zne_extrapolation_map[experiment['extrapolation']]
    noise_scaling = zne_noise_scaling_method_map[experiment['noise_scaling_method']]

    params = {
        "factory": extrapolation,
        "scale_noise": noise_scaling
    }
    return params



def pec_schema_to_params(experiment):

    params = {
        "representations": experiment['operation_representations'],
        "observable": experiment.get('observable', None),
        "precision": experiment.get('precision', 0.03),
        "num_samples" : experiment.get('num_samples', None),
        "force_run_all": experiment.get('force_to_run_all', False),
        "random_state": experiment.get('random_state', None),
        "full_output": experiment.get('full_ouput', False)

    }
    return params


def schema_to_params(experiment):
    map_dict = {
    "zne": zne_schema_to_params,
    "ddd": ddd_schema_to_params,
    "pec": pec_schema_to_params
    }

    params = map_dict[experiment['technique']](experiment)
    return params


def run_composed_experiment(composed_experiment, circuit, executor):

    experiments = composed_experiment['experiments']

    mitigate_exectuor_map = {
    "zne": zne.mitigate_executor,
    "ddd": ddd.mitigate_executor,
    "pec": pec.mitigate_executor,
    "rem": rem.mitigate_executor
    }

    execute_with_map = {
        "zne": zne.execute_with_zne,
        "ddd": ddd.execute_with_ddd,
        "pec": pec.execute_with_pec,
        "rem": rem.execute_with_rem
    }

    for i, experiment in enumerate(experiments):
        technique = experiment['technique']
        params = schema_to_params(experiment)
        if i < len(experiments) - 1:
            new_executor = mitigate_exectuor_map[technique](executor=executor, **params)
            executor = new_executor
        else:
            result = execute_with_map[technique](executor=executor, circuit=circuit, **params)
        
    return result
