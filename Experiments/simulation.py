import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import torch
import copy
from scipy import stats

import sys
# sys.path.append('')

from generate_data import generate_data_shared_features, generate_data_linear
from Classifier.linear import split_set, linear_classifier, torch_performance, set_torch_seed
from Classifier.mlp import mlp_classifier
"""NOTE: the CWD needs to be set to the base directory. And you need to add
the the path to rare_event_mll/ to your PYTHONPATH."""

"""Any arguments that are inside a list allow you to specify multiple values
and they will all be run. be aware when combining multiple lists that this
can lead to very long run time to complete all combinations."""

# change similarity
n_patients = 500000  # n of samples to generate
n_features = 25  # n of features to generate
n_relevant = 20  # n of important features that is relevant to outcome
event_rate1 = [0.01]  # event rate for sample
er2_ratio = [1, 2, 5, 10, 30] # event rate 2 / event rate 1

DGP = 'linear' # linear/shared
similarity = [0, 0.2, 0.4, 0.6, 0.8, 1]
if DGP == 'shared':
    similarity_measures = {
        'similarity': similarity,  # n of distinct features for each label
        'n_random_features': [5],  # n of hidden features for each label
        'shared_second_layer_weights': [True]  # whether the labels share the same weights of their features
    }
if DGP == 'linear':
    similarity_measures = {
        'similarity': similarity, # parameter for generate_data_linear
        }
sim_keys, sim_values = zip(*similarity_measures.items())
similarity_combos = [dict(zip(sim_keys, v)) for v in
                        itertools.product(*sim_values)]

classifier = ['linear_classifier', 'mlp_classifier']
weighted = [False]

param_config = {
    "learning_rate": [5e-3],
    "batch_size": [256],
    "reg_metric": ['L2'],
    "reg_lam": [1e-2, 1e-3, 1e-4]
}
mul_sim_lam = [0] + np.logspace(-3, 2, 6).tolist() # lambda options for multi learning

n_iters = 10  # n of iterations to run each combination
test_perc = 0.7  # percent of samples to use for test set
print_time = True  # whether to print updates after each combination is completes
print_output = False  # whether to print details about each generated dataset
plot = False  # whether to plot details of each generated dataset
loss_plot = False  # whether to plot learning loss
early_stop = True  # whether to do early stopping in training
columns = ['patients', 'features',
           'event_rate', 'er2_ratio', 'similarity',
           'iter', # iteration number
           'lam', # lam for weights similarity regularization
           'clf', # linear classifier / mlp classifier'
           'learning_method', # single_baseline / multi_baseline / mutli_cos / multi_L1
           'event', # index for interested event
           'auc', 'ap', # performance for interested event
           'r2', 'mse', 'corr', # performance for interested event
           'config' # configurations chosed by validation, [learning_rate, batch_size, L1_regularization, similar_regularization, hidden_layer_size]
           ]
results = pd.DataFrame(columns=columns)

save_file = f'simulation_{DGP}.csv'  # saved inside Results/ folder

start = time.time()
for event_rate in event_rate1:
    single_first_time = True
    for s in similarity_combos:
        for ratio in er2_ratio:
            for r in range(n_iters):
                set_torch_seed(r)
                datagen_args = {
                    'n_patients': n_patients,
                    'n_features': n_features,
                    'n_relevant': n_relevant,
                    'event_rate1': event_rate,
                    'event_rate2': ratio*event_rate,
                    'print_output': print_output,
                    'plot': plot
                    }
                datagen_args['random_seed'] = r
                datagen_args.update(s)
                simi = datagen_args['similarity']
                if DGP == 'shared':
                    x, p0, p1, e0, e1 = generate_data_shared_features(**datagen_args)
                if DGP == 'linear':
                    x, p0, p1, e0, e1 = generate_data_linear(**datagen_args)

                # create and randomly split train set and test set by test_perc
                y = np.concatenate([e0.reshape(-1,1),
                                    e1.reshape(-1,1)], axis=1).astype('float32')
                p = np.concatenate([p0.reshape(-1,1),
                                    p1.reshape(-1,1)], axis=1).astype('float32')
                x = x.astype('float32')
                
                x_train, x_test, \
                y_train, y_test, \
                p_train, p_test,  = train_test_split(x, y, p,
                                                    random_state=r,
                                                    test_size=test_perc,
                                                    stratify=y[:,0])
                                                    
                x_sub_train, x_sub_val,\
                y_sub_train, y_sub_val,\
                p_sub_train, p_sub_val  = train_test_split(x_train, y_train, p_train,
                                                          random_state=r, test_size=0.25)

                set_torch_seed(r)
                
                for clf in classifier:
                    if clf == 'mlp_classifier':
                        param_config['hidden_layer']=[[100]] # hidden sizes
                    if clf == 'linear_classifier':
                        param_config['hidden_layer']=[[0]]
                    
                    # single learning
                    for event in range(y_train.shape[1]):
                        fine_tuned_models = {}
                        best_perf = 0
                        param_config['sim_lam'] = [0]
                        configs = list(itertools.product(*param_config.values()))
                        for config in configs:
                            model = eval(clf)(x_train=x_train, y_train=y_train,
                                                    config=config, random_seed=r,
                                                    event_idx=[event], val_idx=[event],
                                                    method='single')
                            fine_tuned_models[f"model{len(fine_tuned_models)}"]=copy.deepcopy(model)
                            valid_perf = torch_performance(model=model,
                                                        x_test=x_sub_val,
                                                        y_test=y_sub_val,
                                                        event_idx=[event])['auc']
                            if valid_perf > best_perf:
                                best_perf = valid_perf
                                best_config = config
                                sin_model = copy.deepcopy(model)
                        perform = {**torch_performance(model=sin_model,
                                                            x_test=x_test,
                                                            y_test=y_test,
                                                            event_idx=[event])}
                        perform2 = {**torch_performance(model=sin_model,
                                                            x_test=x_test,
                                                            y_test=p_test,
                                                            event_idx=[event],
                                                            y_is_prob=True)}
                        print('single_best', best_config, perform2)
                        results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, simi] +\
                                                        [r, 0, clf, 'single', event] + \
                                                        list(perform.values()) + list(perform2.values()) + \
                                                        [str(best_config)]
                        results.to_csv(save_file)
                            
                    # multi learning baseline
                    fine_tuned_models = {}
                    best_perf = 0
                    param_config['sim_lam'] = [0]
                    configs = list(itertools.product(*param_config.values()))
                    for config in configs:
                        set_torch_seed(r)
                        model = eval(clf)(x_train=x_train, y_train=y_train,
                                                config=config, random_seed=r,
                                                event_idx=list(range(y_train.shape[1])),
                                                val_idx=list(range(y_train.shape[1])), # train one model for all events
                                                method='multi')
                        fine_tuned_models[f"model{len(fine_tuned_models)}"]=copy.deepcopy(model)
                        valid_perf = torch_performance(model=model,
                                                    x_test=x_sub_val,
                                                    y_test=y_sub_val,
                                                    event_idx=list(range(y_train.shape[1])))['auc'] # valid on all events
                        if valid_perf > best_perf:
                            best_perf = valid_perf
                            best_config = config
                            mul_model = copy.deepcopy(model)
                    for event in range(y_train.shape[1]):
                        perform = {**torch_performance(model=mul_model,
                                                        x_test=x_test,
                                                        y_test=y_test,
                                                        event_idx=[event])}
                        perform2 = {**torch_performance(model=mul_model,
                                                            x_test=x_test,
                                                            y_test=p_test,
                                                            event_idx=[event],
                                                            y_is_prob=True)}
                        
                        print('multi_base', config, perform2)
                        results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, simi] +\
                                                        [r, 0, clf, 'multi_base', event] + \
                                                        list(perform.values()) + list(perform2.values()) + \
                                                        [str(best_config)]
                    results.to_csv(save_file)

                    # new multi learning (L2)
                    sim_loss= 'L2'
                    param_config['sim_lam'] = mul_sim_lam
                    fine_tuned_models = {}
                    best_perf = 0
                    configs = list(itertools.product(*param_config.values()))
                    for config in configs:
                        set_torch_seed(r)
                        model = eval(clf)(x_train=x_train, y_train=y_train,
                                                config=config, random_seed=r,
                                                event_idx=list(range(y.shape[1])),
                                                val_idx=list(range(y.shape[1])), # train one model for all events
                                                method='multi',
                                                sim_penalty=sim_loss,
                                                loss_plot=loss_plot)
                        fine_tuned_models[f"model{len(fine_tuned_models)}"]=copy.deepcopy(model)
                        valid_perf = torch_performance(model=model,
                                                    x_test=x_sub_val,
                                                    y_test=y_sub_val,
                                                    event_idx=list(range(y.shape[1])))['auc'] # valid on all events
                        if valid_perf > best_perf:
                            best_perf = valid_perf
                            best_config = config
                            baseline_model = copy.deepcopy(model)
                                
                    for event in range(y_train.shape[1]):
                        perform = {**torch_performance(model=baseline_model,
                                                        x_test=x_test,
                                                        y_test=y_test,
                                                        event_idx=[event])}
                        perform2 = {**torch_performance(model=baseline_model,
                                                            x_test=x_test,
                                                            y_test=p_test,
                                                            event_idx=[event],
                                                            y_is_prob=True)}
                        print('L2', best_config, perform2)
                        results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, simi] +\
                                                        [r, 'tuned', clf, 'multi_cos', event] + \
                                                        list(perform.values()) + list(perform2.values()) + \
                                                        [str(best_config)]
                    results.to_csv(save_file)        

                    # new multi learning (COS)
                    sim_loss= 'cos'
                    param_config['sim_lam'] = mul_sim_lam
                    fine_tuned_models = {}
                    best_perf = 0
                    configs = list(itertools.product(*param_config.values()))
                    for config in configs:
                        set_torch_seed(r)
                        model = eval(clf)(x_train=x_train, y_train=y_train,
                                                config=config, random_seed=r,
                                                event_idx=list(range(y.shape[1])),
                                                val_idx=list(range(y.shape[1])), # train one model for all events
                                                method='multi',
                                                sim_penalty=sim_loss,
                                                loss_plot=True)
                        fine_tuned_models[f"model{len(fine_tuned_models)}"]=copy.deepcopy(model)
                        valid_perf = torch_performance(model=model,
                                                    x_test=x_sub_val,
                                                    y_test=y_sub_val,
                                                    event_idx=list(range(y.shape[1])))['auc'] # valid on all events
                        if valid_perf > best_perf:
                            best_perf = valid_perf
                            best_config = config
                            baseline_model = copy.deepcopy(model)
                                
                    for event in range(y_train.shape[1]):
                        perform = {**torch_performance(model=baseline_model,
                                                        x_test=x_test,
                                                        y_test=y_test,
                                                        event_idx=[event])}
                        perform2 = {**torch_performance(model=baseline_model,
                                                            x_test=x_test,
                                                            y_test=p_test,
                                                            event_idx=[event],
                                                            y_is_prob=True)}
                        results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, simi] +\
                                                        [r, 'tuned', clf, 'multi_cos', event] + \
                                                        list(perform.values()) + list(perform2.values()) + \
                                                        [str(best_config)]
                    results.to_csv(save_file)        
                    
                    # new multi learning (L1)
                    sim_loss= 'L1'
                    param_config['sim_lam'] = mul_sim_lam
                    fine_tuned_models = {}
                    best_perf = 0
                    configs = list(itertools.product(*param_config.values()))
                    for config in configs:
                        set_torch_seed(r)
                        model = eval(clf)(x_train=x_train, y_train=y_train,
                                                config=config, random_seed=r,
                                                event_idx=list(range(y.shape[1])),
                                                val_idx=list(range(y.shape[1])), # train one model for all events
                                                method='multi',
                                                sim_penalty=sim_loss,
                                                loss_plot=loss_plot)
                        fine_tuned_models[f"model{len(fine_tuned_models)}"]=copy.deepcopy(model)
                        valid_perf = torch_performance(model=model,
                                                    x_test=x_sub_val,
                                                    y_test=y_sub_val,
                                                    event_idx=list(range(y.shape[1])))['auc'] # valid on all events
                        if valid_perf > best_perf:
                            best_perf = valid_perf
                            best_config = config
                            baseline_model = copy.deepcopy(model)
                                
                    for event in range(y_train.shape[1]):
                        perform = {**torch_performance(model=baseline_model,
                                                        x_test=x_test,
                                                        y_test=y_test,
                                                        event_idx=[event])}
                        perform2 = {**torch_performance(model=baseline_model,
                                                            x_test=x_test,
                                                            y_test=p_test,
                                                            event_idx=[event],
                                                            y_is_prob=True)}
                        print('L1', best_config, perform2)
                        results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, simi] +\
                                                        [r, 'tuned', clf, 'multi_L1', event] + \
                                                        list(perform.values()) + list(perform2.values()) + \
                                                        [str(best_config)]
                    results.to_csv(save_file)
