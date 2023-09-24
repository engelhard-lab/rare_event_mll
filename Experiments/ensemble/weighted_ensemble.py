import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time
import torch
import copy
from scipy import stats

import sys
sys.path.append('E:\\duke\\MLL\\git\\rare_event_mll')

from generate_data import generate_data_shared_features
from Models.torch.torch_training import split_set, torch_classifier, torch_performance
from Models.torch.torch_training import set_torch_seed, make_ingredients, make_uniform_soup, make_greedy_soup
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
    'n_distinct': [1],  # n of distinct features for each label
    'n_random_features': [10],  # n of hidden features for each label
    'shared_second_layer_weights': [True]  # whether the labels share the same weights of their features
}
sim_keys, sim_values = zip(*similarity_measures.items())
similarity_combos = [dict(zip(sim_keys, v)) for v in
                        itertools.product(*sim_values)]

param_config = {
    "learning_rate": [1e-3],
    "batch_size": [64, 256],
    "hidden_layers": [[10],[50],[200]],
    "regularization": [1e-5],
}

n_iters = 10  # n of iterations to run each combination
test_perc = 0.6  # percent of samples to use for test set
print_time = True  # whether to print updates after each combination is completes
print_output = True  # whether to print details about each generated dataset
plot = False  # whether to plot details of each generated dataset
run_combined = False
loss_plot = False  # whether to plot learning loss
early_stop = True  # whether to do early stopping in training
n_models = 20   # n of soup ingredients
columns = ['patients', 'features',
           'event_rate', 'er2_ratio',
           'iter',
           'learning_method', 'ensemble_method',
           'auc', 'ap', 'r2', 'corr',
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

                # create and randomly split train set and test set by test_perc
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
                
                x_sub_train, x_sub_val,\
                y_sub_train, y_sub_val = train_test_split(x_train, y_train,
                                                          random_state=r, test_size=0.25)

                # single learning
                if single_first_time:

                    # create ingredients
                    fine_tuned_models = {}
                    rs, i, best_perf = 0, 1, 0
                    configs = list(itertools.product(*param_config.values()))
                    while len(fine_tuned_models) < n_models:
                        for config in configs:
                            set_torch_seed(r)
                            model = torch_classifier(x_train=x_sub_train, y_train=y_sub_train,
                                                    config=config, random_seed=r,
                                                    event_idx=[0], val_idx=[0],
                                                    method='single', epoch=25)
                            fine_tuned_models[f"model{i}"]=copy.deepcopy(model)

                            # find best ingredient as baseline model
                            valid_perf = torch_performance(model=model,
                                                        x_test=x_sub_val,
                                                        y_test=y_sub_val,
                                                        event_idx=[0])['auc']
                            if valid_perf > best_perf:
                                best_perf = valid_perf
                                baseline_model = copy.deepcopy(model)
                            
                            i += 1
                        rs += 1

                    # save baseline performance
                    perform_base = {**torch_performance(model=baseline_model,
                                                        x_test=x_test,
                                                        y_test=y_test,
                                                        event_idx=[0]),
                                    **torch_performance(model=baseline_model,
                                                        x_test=x_test,
                                                        y_test=y_test[:,-1],
                                                        event_idx=[0],
                                                        y_is_prob=True)}
                    results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                      ["single", "baseline"] + \
                                                      list(perform_base.values()) + \
                                                      list(s.values())
                    results.to_csv(save_file)
                    
                    # weighted ensemble
                    valid_pred = []
                    for model in fine_tuned_models.values():
                        with torch.no_grad():
                            valid_pred.append(model(torch.from_numpy(x_sub_val)).reshape(-1,1))
                    valid_pred = np.hstack(valid_pred)

                    ensembling_model = LogisticRegression()
                    ensembling_model.fit(valid_pred, y_sub_val[:,0])

                    # test ensemble performance on test set
                    test_pred = []
                    for model in fine_tuned_models.values():
                        with torch.no_grad():
                            test_pred.append(model(torch.from_numpy(x_test)).reshape(-1,1))
                    test_pred = np.hstack(test_pred)
                    final_test_pred = ensembling_model.predict_proba(test_pred)[:,1]
                    
                    auc = roc_auc_score(y_test[:,0], final_test_pred)
                    ap = average_precision_score(y_test[:,0], final_test_pred)
                    r2 = r2_score(y_test[:,-1], final_test_pred)
                    corr = stats.spearmanr(y_test[:,-1], final_test_pred).correlation
                    
                    results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                      ["single", "weighted_ensemble"] + \
                                                      [auc, ap, r2, corr] + \
                                                      list(s.values())
                    results.to_csv(save_file)
            
                # multi learning

                # create ingredients
                fine_tuned_models = {}
                rs, i, best_perf = 0, 1, 0
                configs = list(itertools.product(*param_config.values()))
                while len(fine_tuned_models) < n_models:
                    for config in configs:
                        set_torch_seed(r)
                        model = torch_classifier(x_train=x_sub_train, y_train=y_sub_train,
                                                config=config, random_seed=r,
                                                event_idx=[0,1], val_idx=[0],
                                                method='multi', epoch=40)
                        fine_tuned_models[f"model{i}"]=copy.deepcopy(model)

                        # find best ingredient as baseline model
                        valid_perf = torch_performance(model=model,
                                                    x_test=x_sub_val,
                                                    y_test=y_sub_val,
                                                    event_idx=[0])['auc']
                        if valid_perf > best_perf:
                            best_perf = valid_perf
                            baseline_model = copy.deepcopy(model)
                        
                        i += 1
                    rs += 1

                # save baseline performance
                perform_base = {**torch_performance(model=baseline_model,
                                                    x_test=x_test,
                                                    y_test=y_test,
                                                    event_idx=[0]),
                                **torch_performance(model=baseline_model,
                                                    x_test=x_test,
                                                    y_test=y_test[:,-1],
                                                    event_idx=[0],
                                                    y_is_prob=True)}
                results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                    ["multi", "baseline"] + \
                                                    list(perform_base.values()) + \
                                                    list(s.values())
                results.to_csv(save_file)

                # weighted ensemble
                valid_pred = []
                for model in fine_tuned_models.values():
                    with torch.no_grad():
                        valid_pred.append(model(torch.from_numpy(x_sub_val))[:,0].reshape(-1,1))
                valid_pred = np.hstack(valid_pred)

                ensembling_model = LogisticRegression()
                ensembling_model.fit(valid_pred, y_sub_val[:,0])

                # test ensemble performance on test set
                test_pred = []
                for model in fine_tuned_models.values():
                    with torch.no_grad():
                        test_pred.append(model(torch.from_numpy(x_test))[:,0].reshape(-1,1))
                test_pred = np.hstack(test_pred)
                final_test_pred = ensembling_model.predict_proba(test_pred)[:,1]
                
                auc = roc_auc_score(y_test[:,0], final_test_pred)
                ap = average_precision_score(y_test[:,0], final_test_pred)
                r2 = r2_score(y_test[:,-1], final_test_pred)
                corr = stats.spearmanr(y_test[:,-1], final_test_pred).correlation
                    
                results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                    ["multi", "weighted_ensemble"] + \
                                                    [auc, ap, r2, corr] + \
                                                    list(s.values())
                results.to_csv(save_file)
