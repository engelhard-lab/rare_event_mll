#%%
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

def torch_classifier(config,
                     input_data,
                     valid,
                     random_seed,
                     method,
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
    if method == "single":
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
            
            return sin_auc
    
        else:
            with torch.no_grad():
                pred = sin_decoder(sin_encoder(torch.from_numpy(x_test)))
            sin_auc = roc_auc_score(y_test[:,0], pred)
            sin_ap = average_precision_score(y_test[:,0], pred)
            sin_r2 = r2_score(y_test[:,-1], pred)
            sin_cov = stats.spearmanr(y_test[:,-1], pred).correlation
            
            return [sin_auc, sin_ap, sin_r2, sin_cov]
    
    if method == "multi":
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
            
            return mul_auc
    
        else:
            with torch.no_grad():
                pred = sig_decoder(sig_encoder(torch.from_numpy(x_test)))[:,0]
            mul_auc = roc_auc_score(y_test[:,0], pred)
            mul_ap = average_precision_score(y_test[:,0], pred)
            mul_r2 = r2_score(y_test[:,-1], pred)
            mul_cov = stats.spearmanr(y_test[:,-1], pred).correlation        
            
            return [mul_auc, mul_ap, mul_r2, mul_cov]

    # if valid:
    #     performance = [sin_auc, mul_auc]
    # else:
    #     performance = [sin_auc, sin_ap, sin_r2, sin_cov,
    #                   mul_auc, mul_ap, mul_r2, mul_cov,
    #     ]
    # performance = [
    #     sin_auc1, sig_auc1, smx_auc1, com_auc1, 
    #     sin_ap1, sig_ap1, smx_ap1, com_ap1,
    # ]
        

    # return performance
