import copy
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import random_split
from Models.torch.torch_base_nn import NeuralNet
from Models.dataloader import create_batches
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import ray
from ray import tune


def set_torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch_classifier(config,
                     fixed_config,
                     data,
                     performance,
                     run_refined=False,
                     loss_plot=False,
                     epochs=200,
                     ):
    
    x_train = data["x_train"]
    e1_train = data["e1_train"]
    e2_train = data["e2_train"]
    x_test = data["x_test"]
    e1_test = data["e1_test"]

    method = fixed_config["method"]
    activation = fixed_config["activation"]
    random_seed = fixed_config["random_seed"]

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    regularization = config["regularization"]
    hidden_layers = config["hidden_layers"]


    # convert dtypes to play nicely with torch
    x_train = x_train.astype('float32')
    e1_train = e1_train.astype('float32')
    x_test = x_test.astype('float32')

    # split 20% of train set to valid set
    train_abs = int(len(x_train)* 0.8)
    x_sub_train = x_train[:train_abs]
    x_sub_val = x_train[train_abs:]
    e1_sub_train = e1_train[:train_abs]
    e1_sub_val = e1_train[train_abs:]

    set_torch_seed(random_seed)

    train_epochs = create_batches(num_samples=x_sub_train.shape[0], 
                                  batch_size=batch_size,
                                  num_epochs=epochs, 
                                  random_seed=random_seed)

    layers = [x_sub_train.shape[1]] + hidden_layers

            
    train_loss_list = []
    valid_loss_list = []
    best_loss = float("inf")
    patience = 5
    
    print(method)


    # single learning
    if method == "single":
        model = NeuralNet(layers + [1], activation=activation)

        optimizer_single = torch.optim.Adam(model.parameters(),
                                            lr=learning_rate)

        for e in train_epochs:
            train_loss = 0
            for batch in e:
                X = torch.from_numpy(x_sub_train[batch, :])
                Y = torch.from_numpy(e1_sub_train[batch].reshape(-1, 1))
                
                batch_loss = nn.functional.binary_cross_entropy(
                    model(X), Y
                )
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
                batch_loss += regularization * regularization_loss

                optimizer_single.zero_grad()
                batch_loss.backward()
                optimizer_single.step()

                train_loss += batch_loss.item()

            train_loss /= len(e)
            train_loss_list.append(train_loss)
        
            with torch.no_grad():
                valid_loss = nn.functional.binary_cross_entropy(
                    model(torch.from_numpy(x_sub_val)),
                    torch.from_numpy(e1_sub_val.reshape(-1, 1))
                    )
                valid_loss = valid_loss.item()
                valid_loss_list.append(valid_loss)
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                patientce_counter = 0
            else:
                patientce_counter += 1
            
            if patientce_counter >= patience:
                break
        
        accuracy = 1-valid_loss
    

    # multi learning
    elif method=="multi":
        e2_train = e2_train.astype('float32')
        e2_sub_train = e2_train[:train_abs]
        e2_sub_val = e2_train[train_abs:]
        
        model = NeuralNet(layers + [2], activation=activation)

        optimizer_multi = torch.optim.Adam(model.parameters(),
                                        lr=learning_rate)

        multi_epochs = epochs // 2 if run_refined else epochs
        for e in train_epochs[:multi_epochs]:
            train_loss = 0
            for batch in e:
                X = torch.from_numpy(x_sub_train[batch, :])
                Y = torch.from_numpy(
                    np.concatenate([e1_sub_train[batch].reshape(-1, 1),
                                    e2_sub_train[batch].reshape(-1, 1)], axis=1)
                )
                batch_loss = nn.functional.binary_cross_entropy(
                    model(X), Y
                )
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
                batch_loss += regularization * regularization_loss

                optimizer_multi.zero_grad()
                batch_loss.backward()
                optimizer_multi.step()

                train_loss += batch_loss.item()

            train_loss /= len(e)
            train_loss_list.append(train_loss)

            with torch.no_grad():
                valid_loss = nn.functional.binary_cross_entropy(
                    model(torch.from_numpy(x_sub_val))[:,0].reshape(-1,1),
                    torch.from_numpy(e1_sub_val).reshape(-1, 1)
                    )
                valid_loss = valid_loss.item()
                valid_loss_list.append(valid_loss)
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                patientce_counter = 0
            else:
                patientce_counter += 1

            if patientce_counter >= patience:
                break

        accuracy = 1-valid_loss

        
    if loss_plot:
        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, color="red", label="single train loss")
        plt.plot(range(len(valid_loss_list)), valid_loss_list, color="blue", label="singel test loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()     

    if performance:  
        test = torch.from_numpy(x_test.astype('float32'))
        with torch.no_grad():
            pred = model(test).numpy()[:, 0]
        auc = roc_auc_score(e1_test, pred)
        ap = average_precision_score(e1_test, pred)

        return auc, ap 
    
    else:
        return {"valid_loss": valid_loss, "accuracy": accuracy}

