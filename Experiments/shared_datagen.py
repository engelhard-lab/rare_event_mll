from Experiments.base_auc_ap import base_auc_ap
import numpy as np
from ray import tune

"""NOTE: the CWD needs to be set to the base directory. And you need to add
the the path to rare_event_mll/ to your PYTHONPATH."""

"""Any arguments that are inside a list allow you to specify multiple values
and they will all be run. be aware when combining multiple lists that this
can lead to very long run time to complete all combinations."""

save_file = 'torch/raytune_test.csv'  # saved inside Results/ folder
n_patients = 50000  # n of samples to generate
n_features = 100  # n of features to generate
event_rate = [0.001]  # event rate for sample
model_types = ['torch']  # options are 'sklearn' and 'torch'
activations = ['relu']  # activation function. currently only support relu
similarity_measures = {
    'n_distinct': [0, 25],  # n of distinct features for each label
    'n_random_features': [25],  # n of hidden features for each label
    'shared_second_layer_weights': [True]  # whether the labels share the same weights of their features
}
param_config = {
    "learning_rate": tune.choice([1e-2, 1e-3, 1e-4, 1e-5]), # find a best lr -2, -3, -4, -5
    "batch_size": tune.choice([200]), # fix at 200
    "regularization": tune.choice([1e-6]), #without regularization
    "hidden_layers": tune.choice([[10], [25], [50], [200]]), # 10, 25, 50, 200
}
n_iters = 5  # n of iterations to run each combination
test_perc = 0.2  # percent of samples to use for test set
print_time = True  # whether to print updates after each combination is complete
print_output = False  # whether to print details about each generated dataset
plot = False  # whether to plot details of each generated dataset
run_refined = False
loss_plot = False  # whether to plot learning loss
early_stop = True  # whether to do early stopping in training

results = base_auc_ap(n=n_patients, p=n_features, event_rate=event_rate,
                      model_types=model_types,
                      activations=activations, param_config=param_config, n_iters=n_iters,
                      datagen='shared_features',
                      similarity_measures=similarity_measures,
                      test_perc=test_perc, print_time=print_time,
                      print_output=print_output, plot=plot,
                      run_refined=run_refined, loss_plot=loss_plot,
                      early_stop=early_stop)

results.to_csv(f'Results/{save_file}', index=False)
