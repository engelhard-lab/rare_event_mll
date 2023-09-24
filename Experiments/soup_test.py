import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.model_selection import train_test_split
import time
import torch
import copy
from scipy import stats

import sys
sys.path.append('E:\\duke\\MLL\\git\\rare_event_mll')

from generate_data import generate_data_shared_features
from Models.torch.torch_training import split_set, torch_classifier, torch_performance
from Models.torch.torch_training import make_ingredients, make_uniform_soup, make_greedy_soup
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
    'n_distinct': [0],  # n of distinct features for each label
    'n_random_features': [10],  # n of hidden features for each label
    'shared_second_layer_weights': [True]  # whether the labels share the same weights of their features
}
sim_keys, sim_values = zip(*similarity_measures.items())
similarity_combos = [dict(zip(sim_keys, v)) for v in
                        itertools.product(*sim_values)]

param_config = {
    "learning_rate": [1e-3],
    "batch_size": [128, 256],
    "hidden_layers": [[10]],
    "regularization": [1e-5],
}
param_config_soup = {
    "learning_rate": [1e-3],
    "batch_size": [128, 256],
    "hidden_layers": [[10]],
    "regularization": [1e-5],
}
n_iters = 10  # n of iterations to run each combination
test_perc = 0.75  # percent of samples to use for test set
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
           'learning_method', 'soup_method',
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

                # held out 20% from train set for greedy soup
                x_train, greedy_val_x = split_set(x_train, 0.8)
                y_train, greedy_val_y = split_set(y_train, 0.8)
                
                if single_first_time:
                    # single
                    # baseline model with tuned hidden size
                    n_configs = len(list(itertools.product(*param_config.values())))
                    base_models, base_rank = make_ingredients(data_x=x_train,
                                                         data_y=y_train[:,:-1], # drop p1 in y
                                                         event_idx=[0], val_idx=[0],# all valid on event0
                                                         param_config=param_config, n_models=n_configs,
                                                         learning_method="single")
                    base_model = copy.deepcopy(base_models[base_rank[0]])
                                       
                    # fix hidden size
                    # get n_models of fine_tuned models as soup ingredents
                    ingredients, rank = make_ingredients(data_x=x_train,
                                                   data_y=y_train[:,:-1], # drop p1 in y
                                                   event_idx=[0], val_idx=[0],
                                                   param_config=param_config_soup, n_models=n_models,
                                                   learning_method="single")
                    # 4 methods of model soup
                    best_ingredient = copy.deepcopy(ingredients[rank[0]])
                    ensemble_set = ingredients.copy()
                    unif_soup = make_uniform_soup(ingredients, rank)
                    greedy_soup = make_greedy_soup(ingredients=ingredients, rank_list=rank,
                                                   greedy_val_x=greedy_val_x,
                                                   greedy_val_y=greedy_val_y,
                                                   val_idx=[0])
                
                    # record performance
                    # ensemble
                    pred = torch.zeros(x_test.shape[0]) 
                    for model_indiv in ensemble_set.values():
                        with torch.no_grad():
                            pred += model_indiv(torch.from_numpy((x_test)))[:,0]
                    pred = (pred/len(ensemble_set)).reshape(-1,1)
                    auc = roc_auc_score(y_test[:,0], pred)
                    ap = average_precision_score(y_test[:,0], pred)
                    r2 = r2_score(y_test[:,-1], pred)
                    cor = stats.spearmanr(y_test[:,-1], pred).correlation
                    results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                      ["single", "ensemble"] + \
                                                      [auc,ap,r2,cor] + \
                                                      list(s.values())
                    # baseline
                    perform_base = {**torch_performance(model=base_model,
                                                        x_test=x_test,
                                                        y_test=y_test,
                                                        event_idx=[0]),
                                    **torch_performance(model=base_model,
                                                        x_test=x_test,
                                                        y_test=y_test[:,-1],
                                                        event_idx=[0],
                                                        y_is_prob=True)}
                    results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                      ["single", "baseline"] + \
                                                      list(perform_base.values()) + \
                                                      list(s.values())
                    # best ingredient
                    perform_ingredient = {**torch_performance(model=best_ingredient,
                                                        x_test=x_test,
                                                        y_test=y_test,
                                                        event_idx=[0]),
                                        **torch_performance(model=best_ingredient,
                                                            x_test=x_test,
                                                            y_test=y_test[:,-1],
                                                            event_idx=[0],
                                                            y_is_prob=True)}
                    results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                      ["single", "ingredient"] + \
                                                      list(perform_ingredient.values()) + \
                                                      list(s.values())
                    # uniform soup
                    perform_unif = {**torch_performance(model=unif_soup,
                                                        x_test=x_test, y_test=y_test,
                                                        event_idx=[0]),
                                    **torch_performance(model=unif_soup,
                                                        x_test=x_test, y_test=y_test[:,-1],
                                                        event_idx=[0],
                                                        y_is_prob=True)}
                    results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                      ["single", "uniform"] + \
                                                      list(perform_unif.values()) + \
                                                      list(s.values())
                    # greedy soups
                    perform_greedy = {**torch_performance(model=greedy_soup,
                                                        x_test=x_test,y_test=y_test,
                                                        event_idx=[0]),
                                    **torch_performance(model=greedy_soup,
                                                        x_test=x_test, y_test=y_test[:,-1],
                                                        event_idx=[0],
                                                        y_is_prob=True)}                                                                        
                    results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                      ["single", "greedy"] + \
                                                      list(perform_greedy.values()) + \
                                                      list(s.values())
                    results.to_csv("soup_test.csv")
            
                # multi
                # baseline model with tuned hidden size
                n_configs = len(list(itertools.product(*param_config.values())))
                base_models, base_rank = make_ingredients(data_x=x_train,
                                                     data_y=y_train[:,:-1], # drop p1 in y
                                                     event_idx=[0], val_idx=[0],
                                                     param_config=param_config, n_models=n_configs,
                                                     learning_method="multi")
                base_model = copy.deepcopy(base_models[base_rank[0]])
                                    
                # fix hidden size
                # get n_models of fine_tuned models as soup ingredents
                ingredients, rank = make_ingredients(data_x=x_train,
                                                data_y=y_train[:,:-1], # drop p1 in y
                                                event_idx=[0], val_idx=[0],
                                                param_config=param_config_soup, n_models=n_models,
                                                learning_method="multi")
                # 4 methods of model soup
                best_ingredient = copy.deepcopy(ingredients[rank[0]])
                ensemble_set = ingredients.copy()
                unif_soup = make_uniform_soup(ingredients, rank)
                greedy_soup = make_greedy_soup(ingredients=ingredients, rank_list=rank,
                                                greedy_val_x=greedy_val_x,
                                                greedy_val_y=greedy_val_y,
                                                val_idx=[0])
            
                # record performance
                # ensemble
                pred = torch.zeros(x_test.shape[0]) 
                for model_indiv in ensemble_set.values():
                    with torch.no_grad():
                        pred += model_indiv(torch.from_numpy((x_test)))[:,0]
                pred = (pred/len(ensemble_set)).reshape(-1,1)
                auc = roc_auc_score(y_test[:,0], pred)
                ap = average_precision_score(y_test[:,0], pred)
                r2 = r2_score(y_test[:,-1], pred)
                cor = stats.spearmanr(y_test[:,-1], pred).correlation
                results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                    ["multi", "ensemble"] + \
                                                    [auc,ap,r2,cor] + \
                                                    list(s.values())
                # baseline
                perform_base = {**torch_performance(model=base_model,
                                                    x_test=x_test,
                                                    y_test=y_test,
                                                    event_idx=[0]),
                                **torch_performance(model=base_model,
                                                    x_test=x_test,
                                                    y_test=y_test[:,-1],
                                                    event_idx=[0],
                                                    y_is_prob=True)}
                results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                    ["multi", "baseline"] + \
                                                    list(perform_base.values()) + \
                                                    list(s.values())
                # best ingredient
                perform_ingredient = {**torch_performance(model=best_ingredient,
                                                    x_test=x_test,
                                                    y_test=y_test,
                                                    event_idx=[0]),
                                    **torch_performance(model=best_ingredient,
                                                        x_test=x_test,
                                                        y_test=y_test[:,-1],
                                                        event_idx=[0],
                                                        y_is_prob=True)}
                results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                    ["multi", "ingredient"] + \
                                                    list(perform_ingredient.values()) + \
                                                    list(s.values())
                # uniform soup
                perform_unif = {**torch_performance(model=unif_soup,
                                                    x_test=x_test, y_test=y_test,
                                                    event_idx=[0]),
                                **torch_performance(model=unif_soup,
                                                    x_test=x_test, y_test=y_test[:,-1],
                                                    event_idx=[0],
                                                    y_is_prob=True)}
                results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                    ["multi", "uniform"] + \
                                                    list(perform_unif.values()) + \
                                                    list(s.values())
                # greedy soup
                perform_greedy = {**torch_performance(model=greedy_soup,
                                                    x_test=x_test, y_test=y_test,
                                                    event_idx=[0]),
                                **torch_performance(model=greedy_soup,
                                                    x_test=x_test, y_test=y_test[:,-1],
                                                    event_idx=[0],
                                                    y_is_prob=True)}                                                                        
                results.loc[len(results.index)] = [n_patients, n_features, event_rate, ratio, r] + \
                                                    ["multi", "greedy"] + \
                                                    list(perform_greedy.values()) + \
                                                    list(s.values())
                results.to_csv("soup_test.csv")

            single_first_time = False


