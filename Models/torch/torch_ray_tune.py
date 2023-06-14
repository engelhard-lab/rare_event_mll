import ray
from ray import tune
from ray.air import Checkpoint, session
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from Models.torch.torch_param_search import torch_param_search


def ray_tune(config, fixed_var, data, final_layer_size, combine_labels):
    if not ray.is_initialized():
        ray.init()

    scheduler = ASHAScheduler(
        metric="valid_R2",
        mode="max"
    )

    result = tune.run(
        partial(torch_param_search, fixed_config=fixed_var, data=data,
                final_layer_size=final_layer_size,
                combine_labels=combine_labels),
        resources_per_trial={"cpu": 2},
        config=config,
        scheduler=scheduler,
    )

    best_config = result.get_best_config(metric="valid_R2", mode="max")

    if ray.is_initialized():
        ray.shutdown()

    return best_config
