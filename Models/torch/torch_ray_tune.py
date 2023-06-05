import ray
from ray import tune
from ray.air import Checkpoint, session
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from Models.torch.torch_classifier import torch_classifier

def ray_tune(config, fixed_var, data):

    if not ray.is_initialized():
        ray.init()

    scheduler = ASHAScheduler(
        metric="valid_auc",
        mode="max"
    )


    result = tune.run(
        partial(torch_classifier, fixed_config=fixed_var, data=data, performance=False),
        resources_per_trial={"cpu": 2},
        config=config,
        scheduler=scheduler,
    )

    best_config = result.get_best_config(metric="valid_auc", mode="max")

    if ray.is_initialized():
        ray.shutdown()

    return best_config