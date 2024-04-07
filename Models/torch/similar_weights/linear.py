#%%
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import copy
from itertools import product
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt

#%%
# Create a linear regression model
class MLLR(nn.Module):
    def __init__(self, dim_in, dim_out, sigmoid):
        super().__init__()
        if sigmoid == True:
            self.forward_pass = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.Sigmoid()
            )
        else: 
            self.forward_pass = nn.Sequential(
                nn.Linear(dim_in, dim_out))
    def forward(self, x):
        logits = self.forward_pass(x)
        return logits

#%%
def set_torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def split_set(data: np.ndarray, test_ratio):
    """Split train set and test set by test ratio."""
    train = data[:int(test_ratio*len(data))]
    test = data[int(test_ratio*len(data)):]
    return train, test

def torch_load(x_train: np.ndarray, y_train: np.ndarray, batch_size, random_seed):
    """Load and reformat data for torch."""
    set_torch_seed(random_seed)
    class FormatData():
        def __init__(self, xx, yy):
            self.X = xx
            self.y = yy

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx, :], np.array(self.y[idx])

    train_set = FormatData(xx=x_train, yy=y_train)
    train_set = DataLoader(train_set, batch_size, shuffle=True)

    return train_set

def linear_classifier(x_train, y_train,
                     config, # [lr, batch, reg, reg_lam, sim_lam]
                     random_seed,
                     method,
                     event_idx, val_idx,
                     epoch='dynamic',
                     sim_penalty='L2',
                     loss_plot=False,
                     pos_weighted=False,
                     ):

    set_torch_seed(random_seed)
    if epoch == 'dynamic':
        max_epochs = 200
        x_sub_train, x_sub_val,\
        y_sub_train, y_sub_val = train_test_split(x_train, y_train,
                                                  random_state=random_seed,
                                                  test_size=0.25)
    else:
        x_sub_train = x_train
        y_sub_train = y_train
        max_epochs = epoch
    
    learning_rate, batch_size, reg, reg_lam, hidden_layers, sim_reg = config

    train_batch = torch_load(x_train=x_sub_train,
                            y_train=y_sub_train,
                            batch_size=batch_size,
                            random_seed=random_seed)
    
    valid_loss_list = []
    
    if method == "single":
        # single learning
        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        sin_model = MLLR(dim_in=x_train.shape[1], dim_out=1, sigmoid=True)
        sin_optimizer = torch.optim.Adam(list(sin_model.parameters()), lr=learning_rate)
        for iter in range (max_epochs):
            train_loss = 0
            for X, Y in train_batch:
                batch_loss = nn.functional.binary_cross_entropy(
                    sin_model(X), Y[:,event_idx].reshape(-1,1),
                    reduction='sum',
                )
                regularization_loss = 0
                if reg == 'L1':
                    for param in sin_model.parameters():
                        regularization_loss += torch.sum(torch.abs(param))
                if reg == 'L2':
                    for param in sin_model.parameters():
                        regularization_loss += torch.sum(torch.sum(param ** 2))
                batch_loss += reg_lam * regularization_loss
    
                sin_optimizer.zero_grad()
                batch_loss.backward()
                sin_optimizer.step()
    
            #     train_loss += batch_loss.item()
            # train_loss /= len(train_batch)
            # train_loss_list.append(train_loss)

            if epoch == 'dynamic':
                with torch.no_grad():
                    valid_loss = nn.functional.binary_cross_entropy(
                        sin_model(torch.from_numpy(x_sub_val)),
                        torch.from_numpy(y_sub_val[:,val_idx].reshape(-1,1)))
                    valid_loss = valid_loss.item()
                    valid_loss_list.append(valid_loss)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
        
                if patience_counter >= patience:
                    break
                
        if loss_plot:
            plt.plot(range(1, len(valid_loss_list)+1), valid_loss_list)
            plt.show()
        print(iter)
        return sin_model

    if method == "multi":
        # multi sigmoid learning
        best_loss = float('inf')
        patience_counter = 0
        patience = 5
    
        mul_model = MLLR(dim_in=x_sub_train.shape[1], dim_out=len(event_idx), sigmoid=True)
        mul_optimizer = torch.optim.Adam(list(mul_model.parameters()), lr=learning_rate)
        
        for iter in range (max_epochs):
            train_loss = 0
            for X, Y in train_batch:
                batch_loss = nn.functional.binary_cross_entropy(
                    mul_model(X), Y[:,event_idx],
                    reduction='sum'
                    )
                regularization_loss = 0
                if reg == 'L1':
                    for param in mul_model.parameters():
                        regularization_loss += torch.sum(torch.abs(param))
                if reg == 'L2':
                    for param in mul_model.parameters():
                        regularization_loss += torch.sum(torch.sum(param ** 2))
                
                param_sim_loss=0
                coef, bias = list(mul_model.parameters())
                weights = torch.cat((coef, bias.unsqueeze(1)), dim=1)
                for i in range(1,weights.shape[0]):
                    for j in range(i):
                        if sim_penalty == 'cos':
                            param_sim_loss += (1-nn.functional.cosine_similarity(weights[i], weights[j],dim=0))
                        if sim_penalty == 'L1':
                            param_sim_loss += torch.sum(torch.abs(weights[i]-weights[j]))
                        if sim_penalty == 'L2':
                            param_sim_loss += torch.sum(torch.sum((weights[i]-weights[j]) ** 2))

                batch_loss += reg_lam * regularization_loss
                batch_loss += sim_reg * param_sim_loss
    
                mul_optimizer.zero_grad()
                batch_loss.backward()
                mul_optimizer.step()
    
            #     train_loss += batch_loss.item()
            # train_loss /= len(train_batch)
            # train_loss_list.append(train_loss)

            if epoch == 'dynamic':
                with torch.no_grad():
                    valid_loss =nn.functional.binary_cross_entropy(
                        mul_model(torch.from_numpy(x_sub_val))[:,val_idx],
                        torch.from_numpy(y_sub_val)[:,val_idx],
                    )
                    valid_loss_list.append(valid_loss)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break
        if loss_plot:
            plt.plot(range(1, len(valid_loss_list)+1), valid_loss_list)
            plt.show()
        
        print(iter)
        return mul_model
    
    
def torch_performance(model,
                      x_test, y_test,
                      event_idx,
                      y_is_prob=False):
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test))
        if pred.shape[1]>1:
            pred = pred[:,event_idx].reshape(-1,1)
    
    y_test = y_test[:,event_idx].reshape(-1,1)
    if y_is_prob:
        r2 = r2_score(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        corr = stats.spearmanr(y_test, pred).correlation
        return {"r2": r2, "mse": mse, "corr": corr}
    else:
        auc = roc_auc_score(y_test, pred)
        ap = average_precision_score(y_test, pred)
        return {"auc": auc, "ap": ap}

