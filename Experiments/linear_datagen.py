from Experiments.base_sklearn_auc_ap import base_sklearn_auc_ap

n_patients = 5000
n_features = 10
event_rate = 0.01
hidden_layers = [[1]]
activations = ['identity', 'relu']
similarity_measures = {'similarity': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
n_iters = 2
test_perc = 0.25
print_time = True
print_output = False
plot = False

results = base_sklearn_auc_ap(n=n_patients, p=n_features, er=event_rate,
                              hidden_layers=hidden_layers,
                              activations=activations, n_iters=n_iters,
                              datagen='linear',
                              similarity_measures=similarity_measures,
                              test_perc=test_perc, print_time=print_time,
                              print_output=print_output, plot=plot)

results.to_csv(f'Results/sklearn/similarity_results.csv', index=False)
