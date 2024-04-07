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
# sys.path.append('')

from Models.torch.similar_weights.linear import linear_classifier
from Models.torch.similar_weights.mlp import mlp_classifier, torch_performance, set_torch_seed
"""NOTE: the CWD needs to be set to the base directory. And you need to add
the the path to rare_event_mll/ to your PYTHONPATH."""

"""Any arguments that are inside a list allow you to specify multiple values
and they will all be run. be aware when combining multiple lists that this
can lead to very long run time to complete all combinations."""

# choose classifier
classifier = ['mlp_classifier', 'linear_classifier']

# edit output file route
save_file = 'MLLtest_results.csv'

# customize hyper parameter configurations
param_config = {
    "learning_rate": [1e-3],
    "batch_size": [256],
    "reg_metric": ['L2'],
    "reg_lam": [1e-6, 1e-5, 1e-4, 1e-3]
}

mul_sim_lam = [0] + np.logspace(-4, 1, 6).tolist()


CV_times = 10
random_seed = 2023
test_perc = 0.2  # percent of samples to use for test set
loss_plot = False  # whether to plot learning loss
early_stop = True  # whether to do early stopping in training

columns = ['classifier', # linear/mlp
           'iter', # iteration number
           'lam', # lam for weights similarity regularization
           'learning_method', # single_baseline / multi_baseline / mutli_new
           'event', # index for interested event
           'auc', 'ap', # performance for interested event
           'config' # configurations chosed by validation, [learning_rate, batch_size, L1_regularization, similar_regularization, hidden_layer_size]
           ]
results = pd.DataFrame(columns=columns)


#%% data input
# data = pd.read_csv("")

# ident_var = ["PAT_KEY",
#                 "MEDREC_KEY",
#                 "PROV_ID",
#                 ]
# model2_var = ["HF_edema_1y_postDelivery", "HF_edema_1y",
#                 "puerperal_1y_postDelivery", "puerperal_1y",
#                 "acuteRF_1y_postDelivery", "acuteRF_1y"
#                 ]
# outcome_var = ['SMM_delivery',
#                 'HF_edema',
#                 'puerperal',
#                 'acuteRF',
#                 ]
# rm_var = ['furosemideIV_beforeDelivery',
#             'RACE',
#             'HISPANIC_IND',
#             ]
# numeric_var = ['AGE', 'GAnum']
# data = data.drop(columns=ident_var+model2_var+rm_var)

# rm_adapt_var = [
# 'URBAN_RURAL', 'TEACHING',
# 'BEDS_GRP',
# 'preeclampsia_POA', 'inclusionGroup',
# 'labetalolPO_day1', 'metoprololPO_day1',
# 'nifedipinePO_day1', 'methyldopaPO_day1',
# 'amlodipinePO_day1', 'labetalolPO_beforeDelivery',
# 'metoprololPO_beforeDelivery', 'nifedipinePO_beforeDelivery',
# 'methyldopaPO_beforeDelivery', 'amlodipinePO_beforeDelivery',
# 'hydralazineIV_beforeDelivery', 'nicardipineIV_beforeDelivery',
# 'labetalolIV_beforeDelivery', 'nifedipineIV_beforeDelivery',
# 'furosemidePO_day1', 'hydrochlorothiazidePO_day1',
# 'furosemidePO_beforeDelivery', 
# 'hydrochlorotzPO_beforeDelivery', 'aspirin_day1',
# 'aspirin_beforeDelivery', 'atorvastatin_day1',
# 'rosuvastatin_day1', 'simvastatin_day1',
# 'pravastatin_day1', 'lovastatin_day1',
# 'pitavastatin_day1', 'ezetimibe_day1',
# 'fenofibrate_day1', 'gemfibrozil_day1',
# 'niacin_day1', 'metformin_day1', 'insulin_day1']
# data = data.drop(columns=rm_adapt_var)

        
# data = pd.get_dummies(data, drop_first=True)
# data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# # normalize the numeric predictors
# mean = data[numeric_var].mean()
# std = data[numeric_var].std()
# data.loc[:, numeric_var] = (data[numeric_var] - mean) / std

# # outcome
# outcome = data[outcome_var]
# features = data.drop(columns=outcome_var)


#%% training
x = features.values.astype('float32')
y = outcome.values.astype('float32')

for i in range(CV_times):
    x_train, x_test, \
    y_train, y_test = train_test_split(x, y,
                                       random_state=i,
                                       test_size=test_perc)
    
    x_sub_train, x_sub_val,\
    y_sub_train, y_sub_val = train_test_split(x_train, y_train,
                                              random_state=i,
                                              test_size=0.25)
    r = i
    set_torch_seed(r)
    
    for clf in classifier:
        if clf == 'mlp_classifier':
            param_config['hidden_layer']=[[100]] # hidden size
        if clf == 'linear_classifier':
            param_config['hidden_layer']=[[0]] # hidden size

        # single learning
        for event in range(y_train.shape[1]): # define interested event
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
            results.loc[len(results.index)] = [clf, i, 'tuned', 'single', event] + \
                                                list(perform.values()) + \
                                                [str(best_config)]
            results.to_csv(save_file)
        
        # multi baseline
        fine_tuned_models = {}
        best_perf = 0
        param_config['sim_lam'] = [0]
        configs = list(itertools.product(*param_config.values()))
        for config in configs:
            model = eval(clf)(x_train=x_train, y_train=y_train,
                                    config=config, random_seed=r,
                                    event_idx=list(range(y_train.shape[1])),
                                    val_idx=[0], # validate on event of interest
                                    method='multi')
            fine_tuned_models[f"model{len(fine_tuned_models)}"]=copy.deepcopy(model)
            valid_perf = torch_performance(model=model,
                                        x_test=x_sub_val,
                                        y_test=y_sub_val,
                                        event_idx=[0])['auc']
            if valid_perf > best_perf:
                best_perf = valid_perf
                best_config = config
                baseline_model = copy.deepcopy(model)
        for event in range(y_train.shape[1]):
            perform = {**torch_performance(model=baseline_model,
                                            x_test=x_test,
                                            y_test=y_test,
                                            event_idx=[0])}
            results.loc[len(results.index)] = [clf, i, 'tuned', 'multi_base', event] + \
                                            list(perform.values()) + \
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
                                    val_idx=[0],
                                    method='multi',
                                    sim_penalty=sim_loss,
                                    loss_plot=True)
            fine_tuned_models[f"model{len(fine_tuned_models)}"]=copy.deepcopy(model)
            valid_perf = torch_performance(model=model,
                                        x_test=x_sub_val,
                                        y_test=y_sub_val,
                                        event_idx=[0])['auc']
            if valid_perf > best_perf:
                best_perf = valid_perf
                best_config = config
                baseline_model = copy.deepcopy(model)
                    
        for event in range(y_train.shape[1]):
            perform = {**torch_performance(model=baseline_model,
                                            x_test=x_test,
                                            y_test=y_test,
                                            event_idx=[event])}
            results.loc[len(results.index)] = [clf, i, 'tuned', 'multi_cos', event] + \
                                            list(perform.values()) + \
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
                                    val_idx=[0],
                                    method='multi',
                                    sim_penalty=sim_loss,
                                    loss_plot=True)
            fine_tuned_models[f"model{len(fine_tuned_models)}"]=copy.deepcopy(model)
            valid_perf = torch_performance(model=model,
                                        x_test=x_sub_val,
                                        y_test=y_sub_val,
                                        event_idx=[0])['auc']
            if valid_perf > best_perf:
                best_perf = valid_perf
                best_config = config
                baseline_model = copy.deepcopy(model)
                    
        for event in range(y_train.shape[1]):
            perform = {**torch_performance(model=baseline_model,
                                            x_test=x_test,
                                            y_test=y_test,
                                            event_idx=[event])}
            results.loc[len(results.index)] = [clf, i, 'tuned', 'multi_L1', event] + \
                                            list(perform.values()) + \
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
                                    val_idx=[0],
                                    method='multi',
                                    sim_penalty=sim_loss,
                                    loss_plot=True)
            fine_tuned_models[f"model{len(fine_tuned_models)}"]=copy.deepcopy(model)
            valid_perf = torch_performance(model=model,
                                        x_test=x_sub_val,
                                        y_test=y_sub_val,
                                        event_idx=[0])['auc']
            if valid_perf > best_perf:
                best_perf = valid_perf
                best_config = config
                baseline_model = copy.deepcopy(model)
                    
        for event in range(y_train.shape[1]):
            perform = {**torch_performance(model=baseline_model,
                                            x_test=x_test,
                                            y_test=y_test,
                                            event_idx=[event])}
            results.loc[len(results.index)] = [clf, i, 'tuned', 'multi_L2', event] + \
                                            list(perform.values()) + \
                                            [str(best_config)]
        results.to_csv(save_file)
