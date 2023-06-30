#%%
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch_training import split_set, grid_search

random_seed=2023
same_stop = False
config_options = {
    "learning_rate": [1e-3],
    "batch_size": [32, 64, 256],
    "hidden_layers": [[20], [50], [200], [1000]],
    "regularization": [1e-5],
}


#%%
def set_torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#%%
set_torch_seed(random_seed)

data = pd.read_csv("Model 2 data.csv")
ident_var = ["PAT_KEY",
             "MEDREC_KEY",
             "PROV_ID",
             ]
model1_var = ["HF_edema", "HF_edema_1y",
              "puerperal", "puerperal_1y",
              "acuteRF", "acuteRF_1y"
              ]
outcome_var = ['HF_edema_1y_postDelivery',
               'puerperal_1y_postDelivery',
               'acuteRF_1y_postDelivery',
               ]
rm_var = ['SMM_1y_postDelivery',
          'furosemideIV_beforeDelivery',
          'RACE',
          'HISPANIC_IND',
          ]

data = data.drop(columns=ident_var+model1_var+rm_var)
data = pd.get_dummies(data, drop_first=True)
data = data.sample(frac=1).reset_index(drop=True)

# normalize the numeric predictors
numeric_var = ['AGE', 'GAnum']
mean = data[numeric_var].mean()
std = data[numeric_var].std()
data.loc[:, numeric_var] = (data[numeric_var] - mean) / std

outcome = data[outcome_var]
outcome["no_event"] = (outcome.sum(axis=1)==0).astype('float32')
outcome = np.array(outcome)
x = np.array(data.drop(columns=outcome_var))

#%%
x_train, x_test = split_set(x, 0.8)
y_train, y_test = split_set(outcome, 0.8)

input_data = {
    "x_train":x_train,
    "x_test":x_test,
    "y_train":y_train,
    "y_test":y_test,
}
    
columns = ['sin_auc0', 'sig_auc0', 'smx_auc0',
           'sin_auc1', 'sig_auc1', 'smx_auc1',
           'sin_auc2', 'sig_auc2', 'smx_auc2',
           'sin_ap0', 'sig_ap0', 'smx_ap0',
           'sin_ap1', 'sig_ap1', 'smx_ap1',
           'sin_ap2', 'sig_ap2', 'smx_ap2']
results = pd.DataFrame(columns=columns)

best_performance, best_config = grid_search(
    configs = config_options,
    input_data = input_data,
    random_seed = random_seed,
    same_stop = same_stop,
    )
results.loc[0,:] = best_performance
results.to_csv("model2_performance.csv")
best_config.to_csv("model2_config.csv")


# %%

