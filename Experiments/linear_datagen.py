from Experiments.base_auc_ap import base_auc_ap

save_file = 'torch/similarity_results.csv'
n_patients = 50000
n_features = 100
event_rate = 0.01
model_types = ['torch']
hidden_layers = [[1]]
activations = ['relu']
similarity_measures = {'similarity': [0.8, 1.0]}
n_iters = 2
test_perc = 0.25
print_time = True
print_output = False
plot = False

results = base_auc_ap(n=n_patients, p=n_features, er=event_rate,
                      model_types=model_types, hidden_layers=hidden_layers,
                      activations=activations, n_iters=n_iters,
                      datagen='linear',
                      similarity_measures=similarity_measures,
                      test_perc=test_perc, print_time=print_time,
                      print_output=print_output, plot=plot)

results.to_csv(f'Results/{save_file}', index=False)
