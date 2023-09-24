#%%
import numpy as np
import pandas as pd
import random
import copy
import itertools
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy import stats

random_seed=2023

import sys
sys.path.append('E:\duke\MLL\git\\rare_event_mll')

from Models.torch.torch_training import split_set, torch_classifier, torch_performance
from Models.torch.torch_training import make_ingredients, make_uniform_soup, make_greedy_soup

#%%
dataset = 'premier'
event = 'model2'
n_models = 50
event_idx = [1]
val_idx = [1]

param_config = {
    "learning_rate": [1e-3, 1e-4],
    "batch_size": [64, 256],
    "hidden_layers": [[50], [200], [1000]],
    "regularization": [1e-5],
}
param_config_soup = {
    "learning_rate": [1e-3, 1e-4],
    "batch_size": [64, 256],
    "hidden_layers": [[2]],
    "regularization": [1e-5],
}

#%% import data
if (dataset=='premier') & (event=='model1'):
    data = pd.read_csv('E:\duke\FAER\working repo\data\premier\Model 1 data.csv')

    ident_var = ["PAT_KEY",
                 "MEDREC_KEY",
                 "PROV_ID",
                 ]
    model2_var = ["HF_edema_1y_postDelivery", "HF_edema_1y",
                  "puerperal_1y_postDelivery", "puerperal_1y",
                  "acuteRF_1y_postDelivery", "acuteRF_1y"
                  ]
    outcome_var = [
                   'HF_edema',
                   'puerperal',
                   'acuteRF',
                   ]
    rm_var = ['SMM_delivery',
              'furosemideIV_beforeDelivery',
              'RACE',
              'HISPANIC_IND',
              ]
    numeric_var = ['AGE', 'GAnum']
    data = data.drop(columns=ident_var+model2_var+rm_var)

if  (dataset=='premier') & (event=='model2'):
    data = pd.read_csv('E:\duke\FAER\working repo\data\premier\Model 2 data.csv')
    ident_var = ["PAT_KEY",
                "MEDREC_KEY",
                "PROV_ID",
                ]
    model1_var = ["HF_edema", "HF_edema_1y",
                "puerperal", "puerperal_1y",
                "acuteRF", "acuteRF_1y"
                ]
    outcome_var = [
                'HF_edema_1y_postDelivery',
                'puerperal_1y_postDelivery',
                'acuteRF_1y_postDelivery',
                ]
    rm_var = ['SMM_1y_postDelivery',
              'furosemideIV_beforeDelivery',
              'RACE',
              'HISPANIC_IND',
              ]
    numeric_var = ['AGE', 'GAnum']
    data = data.drop(columns=ident_var+model1_var+rm_var)

#%%
data = pd.get_dummies(data, drop_first=True)
data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# normalize the numeric predictors
mean = data[numeric_var].mean()
std = data[numeric_var].std()
data.loc[:, numeric_var] = (data[numeric_var] - mean) / std

# outcome
outcome = data[outcome_var]
features = data.drop(columns=outcome_var)

x = features.values.astype('float32')
y = outcome.values.astype('float32')

#%%
def set_torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_torch_seed(random_seed)

#%%
columns = ['iter',
           'learning_method', 'soup_method',
           'auc', 'ap',
           ]

results = pd.DataFrame(columns=columns)

i = 0
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
for test_index, train_index in kf.split(x, y[:,event_idx]):
    set_torch_seed(i)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # held out 20% from train set for greedy soup
    x_train, greedy_val_x,\
    y_train, greedy_val_y = train_test_split(x_train, y_train,
                                    random_state=i,
                                    test_size=0.2,
                                    stratify=y_train[:,event_idx])
    # single
    # baseline model with tuned hidden size
    n_configs = len(list(itertools.product(*param_config.values())))
    base_models, base_rank = make_ingredients(data_x=x_train,
                                         data_y=y_train,
                                         event_idx=event_idx, val_idx=val_idx, # all valid on cere
                                         param_config=param_config, n_models=n_configs,
                                         learning_method="single")
    base_model = copy.deepcopy(base_models[base_rank[0]])
                        
    # fix hidden size
    # get n_models of fine_tuned models as soup ingredents
    ingredients, rank = make_ingredients(data_x=x_train,
                                    data_y=y_train, # drop p1 in y
                                    event_idx=event_idx, val_idx=val_idx,
                                    param_config=param_config_soup, n_models=n_models,
                                    learning_method="single")
    # 4 methods of model soup
    best_ingredient = copy.deepcopy(ingredients[rank[0]])
    ensemble_set = ingredients.copy()
    # unif_soup = make_uniform_soup(ingredients, rank)
    # greedy_soup = make_greedy_soup(ingredients=ingredients, rank_list=rank,
    #                                 greedy_val_x=greedy_val_x,
    #                                 greedy_val_y=greedy_val_y,
    #                                 val_idx=event_idx)

    # record performance
    # ensemble
    pred = torch.zeros(x_test.shape[0]) 
    for model_indiv in ensemble_set.values():
        with torch.no_grad():
            pred += model_indiv(torch.from_numpy(x_test))[:,0]
    pred = (pred/len(ensemble_set)).reshape(-1,1)
    auc = roc_auc_score(y_test[:,event_idx], pred)
    ap = average_precision_score(y_test[:,event_idx], pred)
    results.loc[len(results.index)] = [i, "single", "ensemble"] + [auc,ap]
    # baseline
    perform_base = {**torch_performance(model=base_model,
                                        x_test=x_test,
                                        y_test=y_test,
                                        event_idx=event_idx)}
    results.loc[len(results.index)] = [i, "single", "baseline"] + list(perform_base.values())
    # best ingredient
    perform_ingredient = {**torch_performance(model=best_ingredient,
                                        x_test=x_test,
                                        y_test=y_test,
                                        event_idx=event_idx)}
    results.loc[len(results.index)] = [i, "single", "ingredient"] + list(perform_ingredient.values())
    # # uniform soup
    # perform_unif = {**torch_performance(model=unif_soup,
    #                                     x_test=x_test, y_test=y_test,
    #                                     event_idx=event_idx)}
    # results.loc[len(results.index)] = [i, "single", "uniform"] + list(perform_unif.values())
    # # greedy soups
    # perform_greedy = {**torch_performance(model=greedy_soup,
    #                                     x_test=x_test,y_test=y_test,
    #                                     event_idx=event_idx)}                                                                        
    # results.loc[len(results.index)] = [i, "single", "greedy"] + list(perform_greedy.values())
    results.to_csv("50soup_test_FAER2.csv")


    # multi
    # baseline model with tuned hidden size
    n_configs = len(list(itertools.product(*param_config.values())))
    base_models, base_rank = make_ingredients(data_x=x_train,
                                            data_y=y_train,
                                            event_idx=event_idx, val_idx=val_idx,
                                            param_config=param_config, n_models=n_configs,
                                            learning_method="multi")
    # base_model = base_models[base_rank[0]]
                        
    # fix hidden size
    # get n_models of fine_tuned models as soup ingredents
    ingredients, rank = make_ingredients(data_x=x_train,
                                    data_y=y_train, # drop p1 in y
                                    event_idx=event_idx, val_idx=val_idx,
                                    param_config=param_config_soup, n_models=n_models,
                                    learning_method="multi")
    # 4 methods of model soup
    best_ingredient = copy.deepcopy(ingredients[rank[0]])
    ensemble_set = ingredients.copy()
    unif_soup = make_uniform_soup(ingredients, rank)
    greedy_soup = make_greedy_soup(ingredients=ingredients, rank_list=rank,
                                    greedy_val_x=greedy_val_x,
                                    greedy_val_y=greedy_val_y,
                                    val_idx=event_idx)


    # record performance
    # ensemble
    pred = torch.zeros(x_test.shape[0]).reshape(-1,1)
    for model_indiv in ensemble_set.values():
        with torch.no_grad():
            pred += model_indiv(torch.from_numpy(x_test))[:,event_idx].reshape(-1,1)
    pred = (pred/len(ensemble_set)).reshape(-1,1)
    auc = roc_auc_score(y_test[:,event_idx], pred)
    ap = average_precision_score(y_test[:,event_idx], pred)
    results.loc[len(results.index)] = [i, "multi", "ensemble"] + [auc,ap]
    # baseline
    perform_base = {**torch_performance(model=base_model,
                                        x_test=x_test,
                                        y_test=y_test,
                                        event_idx=event_idx)}
    results.loc[len(results.index)] = [i, "multi", "baseline"] + list(perform_base.values())
    # best ingredient
    perform_ingredient = {**torch_performance(model=best_ingredient,
                                        x_test=x_test,
                                        y_test=y_test,
                                        event_idx=event_idx)}
    results.loc[len(results.index)] = [i, "multi", "ingredient"] + list(perform_ingredient.values())
    # # uniform soup
    # perform_unif = {**torch_performance(model=unif_soup,
    #                                     x_test=x_test, y_test=y_test,
    #                                     event_idx=event_idx)}
    # results.loc[len(results.index)] = [i, "multi", "uniform"] + list(perform_unif.values())
    # # greedy soup
    # perform_greedy = {**torch_performance(model=greedy_soup,
    #                                     x_test=x_test, y_test=y_test,
    #                                     event_idx=event_idx)}                                                                        
    # results.loc[len(results.index)] = [i, "multi", "greedy"] + list(perform_greedy.values())
    i+=1
    results.to_csv("50soup_test_FAER2.csv")
# %%