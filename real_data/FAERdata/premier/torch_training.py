#%%
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

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
                     same_stop,
                     random_seed,
                     max_epochs=20,
                     ):

    set_torch_seed(random_seed)
    
    x_train = input_data["x_train"].astype('float32')
    x_test = input_data["x_test"].astype('float32')
    y_train = input_data["y_train"].astype('float32')
    y_test = input_data["y_test"].astype('float32')

    x_sub_train, x_sub_val = split_set(x_train, 0.8)
    y_sub_train, y_sub_val = split_set(y_train, 0.8)
    
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    hidden_layers = config["hidden_layers"]
    regularization = config["regularization"]

    train_batch = torch_load(x_train=x_sub_train,
                            y_train=y_sub_train,
                            batch_size=batch_size,
                            random_seed=random_seed)

    # single learning
    ## event 0
    best_loss = 0
    patience = 10
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
            pred = sin_decoder(sin_encoder(torch.from_numpy(x_sub_val)))[:,0]
            valid_loss = roc_auc_score(y_sub_val[:,0], pred)
            valid_loss = valid_loss.item()
            # valid_loss_list.append(valid_loss)
        
        if valid_loss > best_loss:
            best_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    with torch.no_grad():
        pred = sin_decoder(sin_encoder(torch.from_numpy(x_test)))
    sin_auc0 = roc_auc_score(y_test[:,0], pred)
    sin_ap0 = average_precision_score(y_test[:,0], pred)

    ## event 1
    best_loss = 0
    patience = 10
    patience_counter = 0

    sin_encoder = Encoder([x_sub_train.shape[1]]+ hidden_layers)
    sin_decoder = Decoder(hidden_layers+[1], True)
    sin_optimizer = torch.optim.Adam(list(sin_encoder.parameters())+list(sin_decoder.parameters()),
                                lr=learning_rate)
    for _ in range (max_epochs):
        # train_loss = 0
        for X, Y in train_batch:
            batch_loss = nn.functional.binary_cross_entropy(
                sin_decoder(sin_encoder(X)), Y[:, 1].reshape(-1,1)
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
            pred = sin_decoder(sin_encoder(torch.from_numpy(x_sub_val)))[:,0]
            valid_loss = roc_auc_score(y_sub_val[:,1], pred)
            valid_loss = valid_loss.item()
            # valid_loss_list.append(valid_loss)
        
        if valid_loss > best_loss:
            best_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    with torch.no_grad():
        pred = sin_decoder(sin_encoder(torch.from_numpy(x_test)))
    sin_auc1 = roc_auc_score(y_test[:,1], pred)
    sin_ap1 = average_precision_score(y_test[:,1], pred)
    
    ## event 2
    best_loss = 0
    patience = 10
    patience_counter = 0

    sin_encoder = Encoder([x_sub_train.shape[1]]+ hidden_layers)
    sin_decoder = Decoder(hidden_layers+[1], True)
    sin_optimizer = torch.optim.Adam(list(sin_encoder.parameters())+list(sin_decoder.parameters()),
                                lr=learning_rate)
    for _ in range (max_epochs):
        # train_loss = 0
        for X, Y in train_batch:
            batch_loss = nn.functional.binary_cross_entropy(
                sin_decoder(sin_encoder(X)), Y[:, 2].reshape(-1,1)
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
            pred = sin_decoder(sin_encoder(torch.from_numpy(x_sub_val)))[:,0]
            valid_loss = roc_auc_score(y_sub_val[:,2], pred)
            valid_loss = valid_loss.item()
            # valid_loss_list.append(valid_loss)
        
        if valid_loss > best_loss:
            best_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    with torch.no_grad():
        pred = sin_decoder(sin_encoder(torch.from_numpy(x_test)))
    sin_auc2 = roc_auc_score(y_test[:,2], pred)
    sin_ap2 = average_precision_score(y_test[:,2], pred)

    # multi sigmoid learning
    # train_loss_list = []
    # valid_loss_list = []
    best_loss0, best_loss1, best_loss2 = 0,0,0
    patience_counter0, patience_counter1, patience_counter2 = 0,0,0
    patience = 10
    stop = [0,0,0]

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
            valid_loss0 =nn.functional.binary_cross_entropy(
                sig_decoder(sig_encoder(torch.from_numpy(x_sub_val))), y_sub_val[:,:-1]
            )
            if valid_loss0 > best_loss0:
                best_loss0 = valid_loss0
                patience_counter0 = 0
            else:
                patience_counter0 += 1
            if patience_counter0 >= patience:
                with torch.no_grad():
                    pred = sig_decoder(sig_encoder(torch.from_numpy(x_test)))
                sig_auc0 = roc_auc_score(y_test[:,0], pred[:,0])
                sig_ap0 = average_precision_score(y_test[:,0], pred[:,0])
                sig_auc1 = roc_auc_score(y_test[:,1], pred[:,1])
                sig_ap1 = average_precision_score(y_test[:,1], pred[:,1])
                sig_auc2 = roc_auc_score(y_test[:,2], pred[:,2])
                sig_ap2 = average_precision_score(y_test[:,2], pred[:,2])
                break

        else:
            with torch.no_grad():
                pred = sig_decoder(sig_encoder(torch.from_numpy(x_sub_val)))
            valid_loss0 = roc_auc_score(y_sub_val[:,0], pred[:,0])
            valid_loss1 = roc_auc_score(y_sub_val[:,1], pred[:,1])
            valid_loss2 = roc_auc_score(y_sub_val[:,2], pred[:,2])
            if valid_loss0 > best_loss0:
                best_loss0 = valid_loss0
                patience_counter0 = 0
            else:
                patience_counter0 += 1
            if valid_loss1 > best_loss1:
                best_loss1 = valid_loss1
                patience_counter1 = 0
            else:
                patience_counter1 += 1
            if valid_loss2 > best_loss2:
                best_loss2 = valid_loss2
                patience_counter2 = 0
            else:
                patience_counter2 += 1

            if patience_counter0 >= patience and stop[0]==0:
                with torch.no_grad():
                    pred = sig_decoder(sig_encoder(torch.from_numpy(x_test)))
                sig_auc0 = roc_auc_score(y_test[:,0], pred[:,0])
                sig_ap0 = average_precision_score(y_test[:,0], pred[:,0])
                stop[0] = 1
            if patience_counter1 >= patience and stop[1]==0:
                with torch.no_grad():
                    pred = sig_decoder(sig_encoder(torch.from_numpy(x_test)))
                sig_auc1 = roc_auc_score(y_test[:,1], pred[:,1])
                sig_ap1 = average_precision_score(y_test[:,1], pred[:,1])
                stop[1] = 1
            if patience_counter2 >= patience and stop[2]==0:
                with torch.no_grad():
                    pred = sig_decoder(sig_encoder(torch.from_numpy(x_test)))
                sig_auc2 = roc_auc_score(y_test[:,2], pred[:,2])
                sig_ap2 = average_precision_score(y_test[:,2], pred[:,2])
                stop[2] = 1
            if sum(stop)==3:
                break


    # multi softmax learning
    # train_loss_list = []
    # valid_loss_list = []
    best_loss0, best_loss1, best_loss2 = 0,0,0
    patience_counter0, patience_counter1, patience_counter2 = 0,0,0
    patience = 10
    stop = [0, 0, 0]

    smx_encoder = Encoder([x_sub_train.shape[1]]+ hidden_layers)
    smx_decoder = Decoder(hidden_layers+[y_sub_train.shape[1]], False)
    smx_optimizer = torch.optim.Adam(list(smx_encoder.parameters())+list(smx_decoder.parameters()),
                                    lr=learning_rate)
    
    for e in range (max_epochs):
        # train_loss = 0
        for X, Y in train_batch:
            batch_loss = nn.functional.cross_entropy(
                smx_decoder(smx_encoder(X)), Y
            )
            regularization_loss = 0
            for param in smx_encoder.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            for param in smx_decoder.parameters():
                regularization_loss += torch.sum(torch.abs(param))

            batch_loss += regularization * regularization_loss

            smx_optimizer.zero_grad()
            batch_loss.backward()
            smx_optimizer.step()

        if same_stop==True:
            valid_loss0 =nn.functional.cross_entropy(
                smx_decoder(smx_encoder(torch.from_numpy(x_sub_val))), y_sub_val
            )
            if valid_loss0 > best_loss0:
                best_loss0 = valid_loss0
                patience_counter0 = 0
            else:
                patience_counter0 += 1
            if patience_counter0 >= patience:
                with torch.no_grad():
                    pred0 = (torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,0] /
                    torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,[0,-1]].sum(axis=1)).numpy()
                    pred1 = (torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,1] /
                    torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,[1,-1]].sum(axis=1)).numpy()
                    pred2 = (torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,2] /
                    torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,[2,-1]].sum(axis=1)).numpy()
                smx_auc0 = roc_auc_score(y_test[:,0], pred0)
                smx_ap0 = average_precision_score(y_test[:,0], pred0)
                smx_auc1 = roc_auc_score(y_test[:,1], pred1)
                smx_ap1 = average_precision_score(y_test[:,1], pred1)
                smx_auc2 = roc_auc_score(y_test[:,2], pred2)
                smx_ap2 = average_precision_score(y_test[:,2], pred2)                
                break

        else:
            with torch.no_grad():
                pred0 = (torch.nn.functional.softmax(
                    smx_decoder(smx_encoder(torch.from_numpy(x_sub_val))))[:,0] /
                torch.nn.functional.softmax(
                    smx_decoder(smx_encoder(torch.from_numpy(x_sub_val))))[:,[0,-1]].sum(axis=1)).numpy()
                pred1 = (torch.nn.functional.softmax(
                    smx_decoder(smx_encoder(torch.from_numpy(x_sub_val))))[:,1] /
                torch.nn.functional.softmax(
                    smx_decoder(smx_encoder(torch.from_numpy(x_sub_val))))[:,[1,-1]].sum(axis=1)).numpy()
                pred2 = (torch.nn.functional.softmax(
                    smx_decoder(smx_encoder(torch.from_numpy(x_sub_val))))[:,2] /
                torch.nn.functional.softmax(
                    smx_decoder(smx_encoder(torch.from_numpy(x_sub_val))))[:,[2,-1]].sum(axis=1)).numpy()
                        
            valid_loss0 = roc_auc_score(y_sub_val[:,0], pred0)
            valid_loss1 = roc_auc_score(y_sub_val[:,1], pred1)
            valid_loss2 = roc_auc_score(y_sub_val[:,2], pred2)

            if valid_loss0 > best_loss0:
                best_loss0 = valid_loss0
                patience_counter0 = 0
            else:
                patience_counter0 += 1
            if valid_loss1 > best_loss1:
                best_loss1 = valid_loss1
                patience_counter1 = 0
            else:
                patience_counter1 += 1
            if valid_loss2 > best_loss2:
                best_loss2 = valid_loss2
                patience_counter2 = 0
            else:
                patience_counter2 += 1

            if patience_counter0 >= patience and stop[0]==0:
                with torch.no_grad():
                    pred0 = (torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,0] /
                    torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,[0,-1]].sum(axis=1)).numpy()
                smx_auc0 = roc_auc_score(y_test[:,0], pred0)
                smx_ap0 = average_precision_score(y_test[:,0], pred0)
                stop[0] = 1
            if patience_counter1 >= patience and stop[1]==0:
                with torch.no_grad():
                    pred1 = (torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,1] /
                    torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,[1,-1]].sum(axis=1)).numpy()
                smx_auc1 = roc_auc_score(y_test[:,1], pred1)
                smx_ap1 = average_precision_score(y_test[:,1], pred1)
                stop[1] = 1
            if patience_counter2 >= patience and stop[2]==0:
                with torch.no_grad():
                    pred2 = (torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,2] /
                    torch.nn.functional.softmax(
                        smx_decoder(smx_encoder(torch.from_numpy(x_test))))[:,[2,-1]].sum(axis=1)).numpy()
                smx_auc2 = roc_auc_score(y_test[:,2], pred2)
                smx_ap2 = average_precision_score(y_test[:,2], pred2)
                stop[2] = 1
            if sum(stop)==3:
                break

    performance = [sin_auc0, sig_auc0, smx_auc0,
                   sin_auc1, sig_auc1, smx_auc1,
                   sin_auc2, sig_auc2, smx_auc2,
                   sin_ap0, sig_ap0, smx_ap0,
                   sin_ap1, sig_ap1, smx_ap1,
                   sin_ap2, sig_ap2, smx_ap2,
                   ]

    return performance
    
#%%
def grid_search(configs, input_data, random_seed, same_stop=True):
    learning_rate = configs["learning_rate"]
    batch_size = configs["batch_size"]
    hidden_layers = configs["hidden_layers"]
    regularization = configs["regularization"]
    
    best_performance = [0]*18
    best_config = pd.DataFrame([[0]*3]*9, columns = ["lr", "batch", "hidden"])
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
                        
                    for i in range(9):
                        if performance[i] > best_performance[i]:
                            best_performance[i] = performance[i]
                            best_performance[i+9] = performance[i+9]
                            best_config.iloc[i,:] = [lr, batch, h]
                        
    return best_performance, best_config
# %%
