from Experiments.base_auc_ap import base_auc_ap
from ray import tune

"""NOTE: the CWD needs to be set to the base directory. And you need to add
the the path to rare_event_mll/ to your PYTHONPATH."""

"""Any arguments that are inside a list allow you to specify multiple values
and they will all be run. be aware when combining multiple lists that this
can lead to very long run time to complete all combinations."""

save_file = 'torch/raytune_test.csv'  # saved inside Results/ folder
n_patients = 50000  # n of samples to generate
n_features = 10  # n of features to generate
event_rate = [0.01]  # event rate for sample
model_types = ['torch']  # options are 'sklearn' and 'torch'
activations = ['relu']  # activation function. currently only support relu
similarity_measures = {
    'n_distinct': [0],  # n of distinct features for each label
    'n_random_features': [5],  # n of hidden features for each label
    'shared_second_layer_weights': [True]  # whether the labels share the same weights of their features
}
param_config = {
    "learning_rate": tune.grid_search([1e-3, 1e-4]),
    "regularization": tune.grid_search([1e-6]),
    "hidden_layers": tune.grid_search([[200, 2], [10]]),
}

n_iters = 10  # n of iterations to run each combination
test_perc = 0.2  # percent of samples to use for test set
print_time = True  # whether to print updates after each combination is completes
print_output = True  # whether to print details about each generated dataset
plot = False  # whether to plot details of each generated dataset
run_combined = True
loss_plot = False  # whether to plot learning loss
early_stop = True  # whether to do early stopping in training

results = base_auc_ap(n=n_patients, p=n_features, event_rate=event_rate,
                      model_types=model_types,
                      activations=activations, param_config=param_config, n_iters=n_iters,
                      datagen='shared_features',
                      similarity_measures=similarity_measures,
                      test_perc=test_perc, print_time=print_time,
                      print_output=print_output, plot=plot,
                      run_combined=run_combined, loss_plot=loss_plot,
                      early_stop=early_stop)

results.to_csv(f'Results/{save_file}', index=False)
