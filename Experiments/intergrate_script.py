import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import time

import sys
sys.path.append('E:\\duke\\MLL\\git\\rare_event_mll')

from generate_data import generate_data_shared_features
from Models.torch.torch_training import torch_classifier, torch_performance, split_set

"""NOTE: the CWD needs to be set to the base directory. And you need to add
the the path to rare_event_mll/ to your PYTHONPATH."""

"""Any arguments that are inside a list allow you to specify multiple values
and they will all be run. be aware when combining multiple lists that this
can lead to very long run time to complete all combinations."""

# change similarity
save_file = 'torch/sim.csv'  # saved inside Results/ folder
n_patients = 250000  # n of samples to generate
n_features = 50  # n of features to generate
event_rate1 = [0.01]  # event rate for sample
er2_ratio = [5]
model_types = ['new_torch']  # options are 'sklearn' and 'torch'
activations = ['relu']  # activation function. currently only support relu
similarity_measures = {
    'n_distinct': [0, 2, 4, 6, 8, 10],  # n of distinct features for each label
    'n_random_features': [10],  # n of hidden features for each label
    'shared_second_layer_weights': [False]  # whether the labels share the same weights of their features
}
sim_keys, sim_values = zip(*similarity_measures.items())
similarity_combos = [dict(zip(sim_keys, v)) for v in
                        itertools.product(*sim_values)]
                        
# # change event rate2 (ratio)
# save_file = 'torch/final_ratio_test_scale.csv'  # saved inside Results/ folder
# n_patients = 250000  # n of samples to generate
# n_features = 50  # n of features to generate
# event_rate1 = [0.01]  # event rate for sample
# er2_ratio = [1,2,3,5,10,30]
# model_types = ['new_torch']  # options are 'sklearn' and 'torch'
# activations = ['relu']  # activation function. currently only support relu
# similarity_measures = {
#     'n_distinct': [1],  # n of distinct features for each label
#     'n_random_features': [10],  # n of hidden features for each label
#     'shared_second_layer_weights': [True]  # whether the labels share the same weights of their features
# }
# sim_keys, sim_values = zip(*similarity_measures.items())
# similarity_combos = [dict(zip(sim_keys, v)) for v in
#                         itertools.product(*sim_values)]

# # change event rate1
# save_file = 'torch/final_er1_test_scale.csv'  # saved inside Results/ folder
# n_patients = 250000  # n of samples to generate
# n_features = 50  # n of features to generate
# event_rate1 = [0.001, 0.002, 0.005, 0.01, 0.02]  # event rate for sample
# er2_ratio = [5]
# model_types = ['new_torch']  # options are 'sklearn' and 'torch'
# activations = ['relu']  # activation function. currently only support relu
# similarity_measures = {
#     'n_distinct': [1],  # n of distinct features for each label
#     'n_random_features': [10],  # n of hidden features for each label
#     'shared_second_layer_weights': [True]  # whether the labels share the same weights of their features
# }
# sim_keys, sim_values = zip(*similarity_measures.items())
# similarity_combos = [dict(zip(sim_keys, v)) for v in
#                         itertools.product(*sim_values)]


param_config = {
    "batch_size": [64, 128, 256],
    "learning_rate": [1e-3, 1e-4],
    "regularization": [1e-5],
    "hidden_layers": [[200],[50],[10]],
}


n_iters = 20  # n of iterations to run each combination
test_perc = 0.8  # percent of samples to use for test set
print_time = True  # whether to print updates after each combination is completes
print_output = True  # whether to print details about each generated dataset
plot = False  # whether to plot details of each generated dataset
run_combined = False
loss_plot = False  # whether to plot learning loss
early_stop = True  # whether to do early stopping in training


columns = ['patients', 'features',
           'event_rate', 'er2_ratio',
           'iter','config',
           'auc', 'ap', 'r2', 'cov',
           ]

results = pd.DataFrame(columns=columns + list(similarity_measures.keys()))

start = time.time()
for event_rate in event_rate1:
    single_first_time = True
    for s in similarity_combos:
        for ratio in er2_ratio:
                for r in range(n_iters):
                    datagen_args = {
                        'n_patients': n_patients,
                        'n_features': n_features,
                        'event_rate1': event_rate,
                        'event_rate2': ratio*event_rate,
                        'print_output': print_output,
                        'plot': plot
                        }
                    datagen_args['random_seed'] = r
                    datagen_args.update(s)
                    x, p1, e1, e2 = generate_data_shared_features(**datagen_args)
                    x_train, x_test, \
                    p1_train, p1_test, \
                    e1_train, e1_test, \
                    e2_train, e2_test = train_test_split(x, p1, e1, e2,
                                                        random_state=r,
                                                        test_size=test_perc,
                                                        stratify=e1)
    
                    y_train = np.concatenate([e1_train.reshape(-1, 1),
                                              e2_train.reshape(-1, 1),
                                              p1_train.reshape(-1, 1)], axis=1)
                    y_test = np.concatenate([e1_test.reshape(-1, 1),
                                             e2_test.reshape(-1, 1),
                                             p1_test.reshape(-1, 1)], axis=1)
                    
                    x_train = x_train.astype('float32')
                    y_train = y_train.astype('float32')
                    x_test = x_test.astype('float32')
                    y_test = y_test.astype('float32')

                    x_sub_train, x_sub_val = split_set(x_train, 0.75)
                    y_sub_train, y_sub_val = split_set(y_train, 0.75)

                    if single_first_time:
                        best_perform_sin = 0
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
                                        sin_model = torch_classifier(x_train=x_train, y_train=y_train[:,:-1],
                                                                     config=config,
                                                                     random_seed=r,
                                                                     method="single")

                                        performance = torch_performance(model=sin_model,
                                                                        x_test=x_sub_val, y_test=y_sub_val[:,0])
                                        if performance["auc"] > best_perform_sin:
                                            best_perform_sin = performance["auc"]
                                            best_config_sin = config
                                            best_model_sin = sin_model
                        
                        perform_sin = {**torch_performance(model=best_model_sin,
                                                           x_test=x_test, y_test=y_test[:,0]),
                                       **torch_performance(model=best_model_sin,
                                                           x_test=x_test, y_test=y_test[:,-1],
                                                           y_is_prob=True)}

                        results.loc[len(results.index)] = [n_patients, n_features, event_rate, 0, r] + \
                                                          [str(best_config_sin)] + \
                                                          list(perform_sin.values()) + \
                                                          list(s.values())

                    best_perform_mul = 0
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
                                    mul_model = torch_classifier(x_train=x_train, y_train=y_train[:,:-1],
                                                                 config=config,
                                                                 random_seed=r,
                                                                 method="multi")

                                    performance = torch_performance(model=mul_model,
                                                                    x_test=x_sub_val, y_test=y_sub_val[:,0])
                                    if performance["auc"] > best_perform_mul:
                                        best_perform_mul = performance["auc"]
                                        best_config_mul = config
                                        best_model_mul = mul_model
                    
                    perform_mul = {**torch_performance(model=best_model_mul,
                                                        x_test=x_test,y_test=y_test[:,0]),
                                    **torch_performance(model=best_model_mul,
                                                        x_test=x_test, y_test=y_test[:,-1],
                                                        y_is_prob=True)}

                    results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                        [str(best_config_mul)] + \
                                                        list(perform_mul.values()) + \
                                                        list(s.values())
                    results.to_csv("test2.csv")
                
                single_first_time=False
                   



results.to_csv(f'Results/{save_file}', index=False)
