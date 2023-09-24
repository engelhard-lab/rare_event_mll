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
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from scipy import stats

#%%
class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.flatten = nn.Flatten()
        sequence = []
        for i in range(len(layers)-1):
            sequence.append(nn.Linear(layers[i], layers[i+1]))
            sequence.append(nn.ReLU())
        self.forward_pass = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.flatten(x)
        features = self.forward_pass(x)
        return features

class Decoder(nn.Module):
    def __init__(self, layers, sigmoid):
        super().__init__()
        if sigmoid == True:
            self.forward_pass = nn.Sequential(
                nn.Linear(layers[0], layers[1]),
                nn.Sigmoid()
            )
        else: 
            self.forward_pass = nn.Sequential(
                nn.Linear(layers[0], layers[1]))

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

def torch_classifier(x_train, y_train,
                     config, # [lr, batch, h, lam]
                     random_seed,
                     method,
                     event_idx, val_idx,
                     epoch='dynamic',
                     ):

    set_torch_seed(random_seed)
    if epoch == 'dynamic':
        max_epochs = 200
        x_sub_train, x_sub_val,\
        y_sub_train, y_sub_val = train_test_split(x_train, y_train,
                                                  random_state=random_seed,
                                                  test_size=0.2)
    else:
        x_sub_train = x_train
        y_sub_train = y_train
        max_epochs = epoch
    
    learning_rate, batch_size, hidden_layers, regularization = config

    train_batch = torch_load(x_train=x_sub_train,
                            y_train=y_sub_train,
                            batch_size=batch_size,
                            random_seed=random_seed)
    
    if method == "single":
        # single learning
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # train_loss_list = []
        # valid_loss_list = []
    
        sin_encoder = Encoder([x_train.shape[1]]+ hidden_layers)
        sin_decoder = Decoder(hidden_layers+[1], True)
        sin_optimizer = torch.optim.Adam(list(sin_encoder.parameters())+list(sin_decoder.parameters()),
                                    lr=learning_rate)
        for _ in range (max_epochs):
            # train_loss = 0
            for X, Y in train_batch:
                batch_loss = nn.functional.binary_cross_entropy(
                    sin_decoder(sin_encoder(X)), Y[:,event_idx].reshape(-1,1),
                )
                regularization_loss = 0
                for param in sin_encoder.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
                for param in sin_decoder.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
    
                batch_loss += regularization * regularization_loss
    
                sin_optimizer.zero_grad()
                batch_loss.backward()
                sin_optimizer.step()
    
                # train_loss += batch_loss.item()
    
            # train_loss /= len(train_batch)
            # train_loss_list.append(train_loss)

            if epoch == 'dynamic':
                with torch.no_grad():
                    valid_loss = nn.functional.binary_cross_entropy(
                        torch.from_numpy(y_sub_val[:,event_idx].reshape(-1,1)),
                        sin_decoder(sin_encoder(torch.from_numpy(x_sub_val))))
                    valid_loss = valid_loss.item()
                    # valid_loss_list.append(valid_loss)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
        
                if patience_counter >= patience:
                    break

        sin_model = nn.Sequential(sin_encoder, sin_decoder)
        return sin_model

    if method == "multi":
        # multi sigmoid learning
        best_loss = float('inf')
        patience_counter = 0
        patience = 5
    
        mul_encoder = Encoder([x_sub_train.shape[1]]+ hidden_layers)
        mul_decoder = Decoder(hidden_layers+[y_sub_train.shape[1]], True)
        mul_optimizer = torch.optim.Adam(list(mul_encoder.parameters())+list(mul_decoder.parameters()),
                                        lr=learning_rate)
        
        for e in range (max_epochs):
            # train_loss = 0
            for X, Y in train_batch:
                batch_loss = nn.functional.binary_cross_entropy(
                    mul_decoder(mul_encoder(X)), Y,
                )
                regularization_loss = 0
                for param in mul_encoder.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
                for param in mul_decoder.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
    
                batch_loss += regularization * regularization_loss
    
                mul_optimizer.zero_grad()
                batch_loss.backward()
                mul_optimizer.step()
    
                # train_loss += batch_loss.item()
            # train_loss /= len(train_batch)
            # train_loss_list.append(train_loss)

            if epoch == 'dynamic':
                with torch.no_grad():
                    valid_loss =nn.functional.binary_cross_entropy(
                        mul_decoder(mul_encoder(torch.from_numpy(x_sub_val)))[:,val_idx],
                        torch.from_numpy(y_sub_val)[:,val_idx],
                    )
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break
    
        mul_model = nn.Sequential(mul_encoder, mul_decoder)
        return mul_model
    
def torch_performance(model,
                      x_test, y_test,
                      event_idx,
                      y_is_prob=False):
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test))
        if pred.shape[1]>1:
            pred = pred[:,event_idx].reshape(-1,1)
    if y_is_prob:
        r2 = r2_score(y_test, pred)
        cor = stats.spearmanr(y_test, pred).correlation
        return {"r2": r2, "cov": cor}
    else:
        y_test = y_test[:,event_idx].reshape(-1,1)
        auc = roc_auc_score(y_test, pred)
        ap = average_precision_score(y_test, pred)
        return {"auc": auc, "ap": ap}

def make_ingredients(data_x, data_y,
                     learning_method,
                     event_idx, val_idx,
                     param_config, n_models,
                     epoch='dynamic'):
    fine_tuned_models = {}
    model_val_perform = {}
    rs, i = 0, 0
    configs = list(product(*param_config.values()))
    while i < n_models:
        for config in configs:
            i += 1
            set_torch_seed(rs)
            _, x_sub_val,\
            _, y_sub_val = train_test_split(data_x, data_y,
                                            random_state=rs,
                                            test_size=0.25)
            model = torch_classifier(x_train=data_x, y_train=data_y,
                                     config=config,
                                     random_seed=rs,
                                     event_idx=event_idx,
                                     val_idx=val_idx,
                                     method=learning_method,
                                     epoch=epoch)
            performance = torch_performance(model=model,
                                            x_test=x_sub_val,
                                            y_test=y_sub_val,
                                            event_idx=event_idx)
            fine_tuned_models[f"model{i}"]=copy.deepcopy(model)
            model_val_perform[f"model{i}"]=performance["auc"]
        rs += 1
    # ordered fine_tuned model by auc in valid set
    model_rank = sorted(model_val_perform.keys(),
                        key=lambda x: model_val_perform[x],
                        reverse=True)
    return fine_tuned_models, model_rank
    
def make_uniform_soup(ingredients, rank_list):
    n_models = len(rank_list)
    for i in range(n_models):
        model_ingredient = copy.deepcopy(ingredients[rank_list[i]])
        if i == 0:
            unif_model = copy.deepcopy(model_ingredient)
            unif_param = model_ingredient.state_dict()
        else:
            ingredient_param = model_ingredient.state_dict()
            for key in unif_param.keys():
                unif_param[key] += ingredient_param[key]
    for key in unif_param.keys():
        unif_param[key] /= n_models
    unif_model.load_state_dict(unif_param)
    return unif_model

def make_greedy_soup(ingredients, rank_list,
                     greedy_val_x, greedy_val_y,
                     val_idx):
    n_models = len(rank_list)
    for i in range(n_models):
        model_ingredient = copy.deepcopy(ingredients[rank_list[i]])
        if i == 0:
            n_ingredient=1
            greedy_model = copy.deepcopy(model_ingredient)
            greedy_param = greedy_model.state_dict()
            best_val_auc = torch_performance(model=greedy_model,
                                             x_test=greedy_val_x,
                                             y_test=greedy_val_y,
                                             event_idx=val_idx)["auc"]
        else:
            greedy_model_tempo = copy.deepcopy(greedy_model)
            ingredient_param = model_ingredient.state_dict()
            greedy_param = greedy_model.state_dict()
            greedy_param_tempo = greedy_model_tempo.state_dict()
            for key in greedy_param_tempo.keys():
                greedy_param_tempo[key] = (greedy_param[key]*n_ingredient+ingredient_param[key])/(n_ingredient+1)
            greedy_model_tempo.load_state_dict(greedy_param_tempo)
            valid_auc_tempo = torch_performance(model=greedy_model_tempo,
                                                x_test=greedy_val_x,
                                                y_test=greedy_val_y,
                                                event_idx=val_idx)["auc"]
            if valid_auc_tempo>=best_val_auc:
                best_val_auc = valid_auc_tempo
                greedy_model = copy.deepcopy(greedy_model_tempo)
                n_ingredient+=1
                print(i)
    return greedy_model
    
def make_greedy_ensemble(ingredients, rank_list,
                     greedy_val_x, greedy_val_y,
                     val_idx, learning_method):
    if learning_method == "single":
        val_idx = 0
    n_models = len(rank_list)
    ensemble_set = {}
    for i in range(n_models):
        model_ingredient = copy.deepcopy(ingredients[rank_list[i]])
        if i == 0:
            n_ingredient=1
            ensemble_set[rank_list[i]] = copy.deepcopy(model_ingredient)
            with torch.no_grad():
                pred = model_ingredient(torch.from_numpy((greedy_val_x)))[:,val_idx].reshape(-1,1)
            best_val_auc = roc_auc_score(greedy_val_y[:,val_idx], pred)
        else:
            set_tempo = copy.deepcopy(ensemble_set)
            set_tempo[rank_list[i]] = copy.deepcopy(model_ingredient)
            for model_indiv in set_tempo.values():
                with torch.no_grad():
                    pred += model_indiv(torch.from_numpy((greedy_val_x)))[:,val_idx].reshape(-1,1)
            pred = (pred/(n_ingredient+1)).reshape(-1,1)
            valid_auc_tempo = roc_auc_score(greedy_val_y[:,0], pred)
            if valid_auc_tempo >= best_val_auc:
                best_val_auc = valid_auc_tempo
                ensemble_set = copy.deepcopy(set_tempo)
                n_ingredient+=1
                print(i)
    return ensemble_set