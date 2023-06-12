from Experiments.base_auc_ap import base_auc_ap
from ray import tune

save_file = 'torch/similarity_results.csv'
n_patients = 50000
n_features = 100
event_rate = [0.01]
model_types = ['torch']
activations = ['relu']
similarity_measures = {'similarity': [1.0]}
param_config = {
    "learning_rate": tune.grid_search([1e-4]),  # find a best lr -2, -3, -4, -5
    "regularization": tune.grid_search([1e-5]),  #without regularization
    "hidden_layers": tune.grid_search([[10]]),  # 10, 25, 50, 200
}
n_iters = 1  # n of iterations to run each combination
test_perc = 0.2  # percent of samples to use for test set
print_time = True  # whether to print updates after each combination is completes
print_output = True  # whether to print details about each generated dataset
plot = False  # whether to plot details of each generated dataset
run_combined = True
loss_plot = True  # whether to plot learning loss
early_stop = True  # whether to do early stopping in training

results = base_auc_ap(n=n_patients, p=n_features, event_rate=event_rate,
                      model_types=model_types,
                      activations=activations, param_config=param_config,
                      n_iters=n_iters,
                      datagen='linear',
                      similarity_measures=similarity_measures,
                      test_perc=test_perc, print_time=print_time,
                      print_output=print_output, plot=plot,
                      run_combined=run_combined, loss_plot=loss_plot,
                      early_stop=early_stop)

results.to_csv(f'Results/{save_file}', index=False)
