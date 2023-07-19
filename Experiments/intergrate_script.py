from generate_data import generate_data_linear, generate_data_shared_features
from Models.sklearn.mlp_classifier import sklearn_mlp
from Models.torch.torch_classifier import torch_classifier
from Models.torch.torch_training import torch_classifier, grid_search

import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import time

"""NOTE: the CWD needs to be set to the base directory. And you need to add
the the path to rare_event_mll/ to your PYTHONPATH."""

"""Any arguments that are inside a list allow you to specify multiple values
and they will all be run. be aware when combining multiple lists that this
can lead to very long run time to complete all combinations."""

save_file = 'torch/final_test.csv'  # saved inside Results/ folder
n_patients = 250000  # n of samples to generate
n_features = 50  # n of features to generate
event_rate = 0.001  # event rate for sample
er2_ratio = [5]
model_types = ['new_torch']  # options are 'sklearn' and 'torch'
activations = ['relu']  # activation function. currently only support relu
similarity_measures = {
    'n_distinct': [0, 1, 2, 3, 4, 6, 8, 10],  # n of distinct features for each label
    'n_random_features': [10],  # n of hidden features for each label
    'shared_second_layer_weights': [True]  # whether the labels share the same weights of their features
}
sim_keys, sim_values = zip(*similarity_measures.items())
similarity_combos = [dict(zip(sim_keys, v)) for v in
                        itertools.product(*sim_values)]

param_config = {
    "batch_size": [64, 256],
    "learning_rate": [1e-3, 1e-4],
    "regularization": [1e-5],
    "hidden_layers": [[200],[50],[10]],
}
# param_config = {
#     "learning_rate": 1e-4,
#     "regularization": 1e-6,
#     "hidden_layers": [200]
# }

n_iters = 10  # n of iterations to run each combination
test_perc = 0.8  # percent of samples to use for test set
print_time = True  # whether to print updates after each combination is completes
print_output = True  # whether to print details about each generated dataset
plot = True  # whether to plot details of each generated dataset
run_combined = False
loss_plot = False  # whether to plot learning loss
early_stop = True  # whether to do early stopping in training


columns = ['patients', 'features',
           'event_rate', 'er2_ratio',
           'iter', 
           'auc_single', 'ap_single',
           'r2_single', 'cov_single', 
           'auc_multi', 'ap_multi',
           'r2_multi', 'cov_multi',
           'config_single', 'config_multi']

results = pd.DataFrame(columns=columns + list(similarity_measures.keys()))

start = time.time()
for s in similarity_combos:
    for er2 in er2_ratio:
            for r in range(n_iters):
                datagen_args = {
                    'n_patients': n_patients,
                    'n_features': n_features,
                    'event_rate1': event_rate,
                    'event_rate2': er2*event_rate,
                    'print_output': print_output,
                    'plot': plot
                    }
                datagen_args['random_seed'] = r
                datagen_args.update(s)
                x, p1, e1, e2 = generate_data_shared_features(**datagen_args)
                print(sum(e1))
                e_combine = [str(label1)+str(label2) for label1, label2 in zip (e1, e2)]
                x_train, x_test, \
                p1_train, p1_test, \
                e1_train, e1_test, \
                e2_train, e2_test, \
                _, _ = train_test_split(x, p1, e1, e2, e_combine,
                                                    random_state=r,
                                                    test_size=test_perc,
                                                    stratify=e1)

                input_data = {
                    "x_train": x_train,
                    "x_test": x_test,
                    "y_train": np.concatenate([e1_train.reshape(-1, 1),
                                                e2_train.reshape(-1, 1),
                                                p1_train.reshape(-1, 1)], axis=1),
                    "y_test": np.concatenate([e1_test.reshape(-1, 1),
                                                e2_test.reshape(-1, 1),
                                                p1_test.reshape(-1, 1)], axis=1),
                }

                best_perform_sin, best_perform_mul = 0, 0
                best_config_sin, best_config_mul = {}, {}

                for lr in param_config["learning_rate"]:
                    for batch in param_config["batch_size"]:
                        for h in param_config["hidden_layers"]:
                            for lam in param_config["regularization"]:
                                config = {
                                    "learning_rate": lr,
                                    "batch_size": batch,
                                    "hidden_layers": h,
                                    "regularization": lam,
                                }
                                performance = torch_classifier(config=config,
                                                            input_data=input_data,
                                                            valid=True,
                                                            random_seed=r)
                                print(performance)
                                if performance[0]>best_perform_sin:
                                    best_config_sin = config
                                    best_perform_sin = performance[0]
                                if performance[1]>best_perform_mul:
                                    best_config_mul = config
                                    best_perform_mul = performance[1]
                
                perform_sin = torch_classifier(config=best_config_sin,
                                            input_data=input_data,
                                            valid=False,
                                            random_seed=r)
                perform_mul = torch_classifier(config=best_config_mul,
                                            input_data=input_data,
                                            valid=False,
                                            random_seed=r)
                                
                results.loc[len(results.index)] = [n_patients, n_features,
                                                   event_rate, er2_ratio, r,] + \
                                                perform_sin[:4] + perform_mul[4:] + \
                                                [str(best_config_sin), str(best_config_mul)] + \
                                                list(s.values())
                
                results.to_csv("final_er.csv")


results.to_csv(f'Results/{save_file}', index=False)
