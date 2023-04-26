from Experiments.base_auc_ap import base_auc_ap

"""NOTE: the CWD needs to be set to the base directory. And you need to add
the the path to rare_event_mll/ to your PYTHONPATH."""

"""Any arguments that are inside a list allow you to specify multiple values
and they will all be run. be aware when combining multiple lists that this
can lead to very long run time to complete all combinations."""

save_file = 'sklearn/shared_features_results_overlap.csv'  # saved inside Results/ folder
n_patients = 50000  # n of samples to generate
n_features = 100  # n of features to generate
event_rate = 0.01  # event rate for sample
model_types = ['torch']  # options are 'sklearn' and 'torch'
hidden_layers = [[25]]  # list of hidden layer sizes
activations = ['relu']  # activation function. currently only support relu
similarity_measures = {
    'n_distinct': [20, 25],  # n of distinct features for each label
    'n_random_features': [25],  # n of hidden features for each label
    'shared_second_layer_weights': [True]  # whether the labels share the same weights of their features
}
n_iters = 1  # n of iterations to run each combination
test_perc = 0.25  # percent of samples to use for test set
print_time = True  # whether to print updates after each combination is complete
print_output = False  # whether to print details about each generated dataset
plot = False  # whether to plot details of each generated dataset

results = base_auc_ap(n=n_patients, p=n_features, er=event_rate,
                      model_types=model_types, hidden_layers=hidden_layers,
                      activations=activations, n_iters=n_iters,
                      datagen='shared_features',
                      similarity_measures=similarity_measures,
                      test_perc=test_perc, print_time=print_time,
                      print_output=print_output, plot=plot)

results.to_csv(f'Results/{save_file}', index=False)
