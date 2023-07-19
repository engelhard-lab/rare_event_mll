#%%
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from scipy import stats
from imblearn.over_sampling import SMOTE

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

def torch_classifier(config,
                     input_data,
                     valid,
                     random_seed,
                     same_stop=True,
                     max_epochs=200,
                     ):

    set_torch_seed(random_seed)
    
    x_train = input_data["x_train"].astype('float32')
    x_test = input_data["x_test"].astype('float32')
    y_train = input_data["y_train"].astype('float32')
    y_test = input_data["y_test"].astype('float32')

    x_sub_train, x_sub_val = split_set(x_train, 0.75)
    y_sub_train, y_sub_val = split_set(y_train, 0.75)
    
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    hidden_layers = config["hidden_layers"]
    regularization = config["regularization"]

    train_batch = torch_load(x_train=x_sub_train,
                            y_train=y_sub_train,
                            batch_size=batch_size,
                            random_seed=random_seed)

    # single learning
    best_loss = 1
    patience = 5
    patience_counter = 0

    sin_encoder = Encoder([x_sub_train.shape[1]]+ hidden_layers)
    sin_decoder = Decoder(hidden_layers+[1], True)
    sin_optimizer = torch.optim.Adam(list(sin_encoder.parameters())+list(sin_decoder.parameters()),
                                lr=learning_rate)
    for _ in range (max_epochs):
        # train_loss = 0
        for X, Y in train_batch:
            batch_loss = nn.functional.binary_cross_entropy(
                sin_decoder(sin_encoder(X)), Y[:, 0].reshape(-1,1)
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

        with torch.no_grad():
            valid_loss = nn.functional.binary_cross_entropy(
                torch.from_numpy(y_sub_val[:,0].reshape(-1,1)),
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

    if valid:
        with torch.no_grad():
            pred = sin_decoder(sin_encoder(torch.from_numpy(x_sub_val)))
        sin_auc = roc_auc_score(y_sub_val[:,0], pred)

    else:
        with torch.no_grad():
            pred = sin_decoder(sin_encoder(torch.from_numpy(x_test)))
        sin_auc = roc_auc_score(y_test[:,0], pred)
        sin_ap = average_precision_score(y_test[:,0], pred)
        sin_r2 = r2_score(y_test[:,-1], pred)
        sin_cov = stats.spearmanr(y_test[:,-1], pred).correlation
    
    # multi sigmoid learning
    best_loss = 1
    patience_counter = 0
    patience = 5

    sig_encoder = Encoder([x_sub_train.shape[1]]+ hidden_layers)
    sig_decoder = Decoder(hidden_layers+[y_sub_train.shape[1]-1], True)
    sig_optimizer = torch.optim.Adam(list(sig_encoder.parameters())+list(sig_decoder.parameters()),
                                    lr=learning_rate)
    
    for e in range (max_epochs):
        # train_loss = 0
        for X, Y in train_batch:
            batch_loss = nn.functional.binary_cross_entropy(
                sig_decoder(sig_encoder(X)), Y[:,:-1]
            )
            regularization_loss = 0
            for param in sig_encoder.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            for param in sig_decoder.parameters():
                regularization_loss += torch.sum(torch.abs(param))

            batch_loss += regularization * regularization_loss

            sig_optimizer.zero_grad()
            batch_loss.backward()
            sig_optimizer.step()

            # train_loss += batch_loss.item()
        # train_loss /= len(train_batch)
        # train_loss_list.append(train_loss)

        if same_stop==True:
            valid_loss =nn.functional.binary_cross_entropy(
                sig_decoder(sig_encoder(torch.from_numpy(x_sub_val)))[:,0],
                torch.from_numpy(y_sub_val[:,0]),
            )
            if valid_loss < best_loss:
                best_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

    if valid:
        with torch.no_grad():
            pred = sig_decoder(sig_encoder(torch.from_numpy(x_sub_val)))
        mul_auc = roc_auc_score(y_sub_val[:,0], pred[:,0])

    else:
        with torch.no_grad():
            pred = sig_decoder(sig_encoder(torch.from_numpy(x_test)))[:,0]
        mul_auc = roc_auc_score(y_test[:,0], pred)
        mul_ap = average_precision_score(y_test[:,0], pred)
        mul_r2 = r2_score(y_test[:,-1], pred)
        mul_cov = stats.spearmanr(y_test[:,-1], pred).correlation           
                
    # with torch.no_grad():
    #     pred = sig_decoder(sig_encoder(torch.from_numpy(x_test)))
    # sig_auc0 = roc_auc_score(y_test[:,0], pred[:,0])
    # sig_ap0 = average_precision_score(y_test[:,0], pred[:,0])
    # sig_auc1 = roc_auc_score(y_test[:,1], pred[:,1])
    # sig_ap1 = average_precision_score(y_test[:,1], pred[:,1])
    # sig_auc2 = roc_auc_score(y_test[:,2], pred[:,2])
    # sig_ap2 = average_precision_score(y_test[:,2], pred[:,2])

    # # multi softmax learning
    # best_loss0, best_loss1, best_loss2 = 1,1,1
    # patience_counter0, patience_counter1, patience_counter2 = 0,0,0
    # patience = 5
    # stop = [0, 0, 0]

    # smx_encoder = Encoder([x_sub_train.shape[1]]+ hidden_layers)
    # smx_decoder = Decoder(hidden_layers+[y_sub_train.shape[1]], False)
    # smx_optimizer = torch.optim.Adam(list(smx_encoder.parameters())+list(smx_decoder.parameters()),
    #                                 lr=learning_rate)
    
    # for e in range (max_epochs):
    #     # train_loss = 0
    #     for X, Y in train_batch:
    #         batch_loss = nn.functional.cross_entropy(
    #             smx_decoder(smx_encoder(X)), Y
    #         )
    #         regularization_loss = 0
    #         for param in smx_encoder.parameters():
    #             regularization_loss += torch.sum(torch.abs(param))
    #         for param in smx_decoder.parameters():
    #             regularization_loss += torch.sum(torch.abs(param))

    #         batch_loss += regularization * regularization_loss

    #         smx_optimizer.zero_grad()
    #         batch_loss.backward()
    #         smx_optimizer.step()

    #     if same_stop==True:
    #         valid_loss0 =nn.functional.cross_entropy(
    #             smx_decoder(smx_encoder(torch.from_numpy(x_sub_val))),
    #             torch.from_numpy(y_sub_val),
    #         )
    #         if valid_loss0 < best_loss0:
    #             best_loss0 = valid_loss0
    #             patience_counter0 = 0
    #         else:
    #             patience_counter0 += 1
    #         if patience_counter0 >= patience:             
    #             break

    # with torch.no_grad():
    #     pred1 = (torch.nn.functional.softmax(
    #         smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,1] /
    #     torch.nn.functional.softmax(
    #         smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,[1,-1]].sum(axis=1)).numpy()
    # smx_auc1 = roc_auc_score(y_test[:,1], pred1)
    # smx_ap1 = average_precision_score(y_test[:,1], pred1)


    # # combined learning
    # best_loss = 0
    # patience = 5
    # patience_counter = 0

    # com_encoder = Encoder([x_sub_train.shape[1]]+ hidden_layers)
    # com_decoder = Decoder(hidden_layers+[1], True)
    # com_optimizer = torch.optim.Adam(list(com_encoder.parameters())+list(com_decoder.parameters()),
    #                             lr=learning_rate)

    # for _ in range (10):
    #     for X, Y in train_batch:
    #         batch_loss = nn.functional.binary_cross_entropy(
    #             com_decoder(com_encoder(X)), Y[:, 2].reshape(-1,1)
    #         )
    #         regularization_loss = 0
    #         for param in com_encoder.parameters():
    #             regularization_loss += torch.sum(torch.abs(param))
    #         for param in com_decoder.parameters():
    #             regularization_loss += torch.sum(torch.abs(param))

    #         batch_loss += regularization * regularization_loss

    #         com_optimizer.zero_grad()
    #         batch_loss.backward()
    #         com_optimizer.step()
    
    # for _ in range (max_epochs):
    #     for X, Y in train_batch:
    #         batch_loss = nn.functional.binary_cross_entropy(
    #             com_decoder(com_encoder(X)), Y[:, 1].reshape(-1,1)
    #         )
    #         regularization_loss = 0
    #         for param in com_encoder.parameters():
    #             regularization_loss += torch.sum(torch.abs(param))
    #         for param in com_decoder.parameters():
    #             regularization_loss += torch.sum(torch.abs(param))

    #         batch_loss += regularization * regularization_loss

    #         com_optimizer.zero_grad()
    #         batch_loss.backward()
    #         com_optimizer.step()

    #     with torch.no_grad():
    #         valid_loss = nn.functional.binary_cross_entropy(
    #             torch.from_numpy(y_sub_val[:,1].reshape(-1,1)),
    #             com_decoder(com_encoder(torch.from_numpy(x_sub_val))))
    #         valid_loss = valid_loss.item()
        
    #     if valid_loss > best_loss:
    #         best_loss = valid_loss
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1

    #     if patience_counter >= patience:
    #         break

    # with torch.no_grad():
    #     pred = com_decoder(com_encoder(torch.from_numpy(x_test)))
    # com_auc1 = roc_auc_score(y_test[:,1], pred)
    # com_ap1 = average_precision_score(y_test[:,1], pred)

    if valid:
        performance = [sin_auc, mul_auc]
    else:
        performance = [sin_auc, sin_ap, sin_r2, sin_cov,
                       mul_auc, mul_ap, mul_r2, mul_cov,
        ]
    # performance = [
    #     sin_auc1, sig_auc1, smx_auc1, com_auc1, 
    #     sin_ap1, sig_ap1, smx_ap1, com_ap1,
    # ]
        

    return performance
    
#%%
def grid_search(configs, input_data, random_seed, same_stop=True):
    learning_rate = configs["learning_rate"]
    batch_size = configs["batch_size"]
    hidden_layers = configs["hidden_layers"]
    regularization = configs["regularization"]
    
    records = pd.DataFrame(columns = [
        'lr', 'batch', 'hidden',
        'sin_auc1', 'sig_auc1', 'smx_auc1', 'com_auc1'])
           
    best_performance = [0]*8
    best_config = pd.DataFrame([[0]*3]*4, columns = ["lr", "batch", "hidden"])
    for lr in learning_rate:
        for batch in batch_size:
            for h in hidden_layers:
                for lam in regularization:
                    param = {
                        "learning_rate": lr,
                        "batch_size": batch,
                        "hidden_layers": h,
                        "regularization": lam,
                    }
                    
                    performance = torch_classifier(
                        config = param,
                        input_data = input_data,
                        random_seed = random_seed,
                        same_stop = same_stop,
                        )
                        
                    records.loc[len(records.index)] = [lr, batch, str(h)] + performance[:4]
                        
                    for i in range(4):
                        if performance[i] > best_performance[i]:
                            best_performance[i] = performance[i]
                            best_performance[i+4] = performance[i+4]
                            best_config.iloc[i,:] = [lr, batch, str(h)]
                        
    return best_performance, best_config, records
# %%
