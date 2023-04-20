from Experiments.base_sklearn_auc_ap import base_sklearn_auc_ap

n_patients = 1000
n_features = 20
event_rate = 0.01
hidden_layers = [[1]]
activations = ['identity', 'relu']
similarity_measures = {
    'n_distinct': [5, 10],
    'n_overlapping': [5],
    'shared_second_layer_weights': [True]
}
n_iters = 2
test_perc = 0.25
print_time = True
print_output = False
plot = False

results = base_sklearn_auc_ap(n=n_patients, p=n_features, er=event_rate,
                              hidden_layers=hidden_layers,
                              activations=activations, n_iters=n_iters,
                              datagen='shared_features',
                              similarity_measures=similarity_measures,
                              test_perc=test_perc, print_time=print_time,
                              print_output=print_output, plot=plot)

results.to_csv(f'Results/sklearn/shared_features_results.csv', index=False)
