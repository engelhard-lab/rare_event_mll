import numpy as np
import random

import torch
import torch.nn as nn
from Models.torch.torch_base_nn import NeuralNet
from Models.dataloader import create_batches
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


def set_torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch_param_search(config, fixed_config, data, final_layer_size=1,
                       combine_labels=False, epochs=800, patience=5,
                       start_up=20):
    
    x_train = data["x_train"]
    e1_train = data["e1_train"]
    e2_train = data["e2_train"]
    p1_train = data["p1_train"]

    activation = fixed_config["activation"]
    random_seed = fixed_config["random_seed"]
    batch_size = fixed_config["batch_size"]

    learning_rate = config["learning_rate"]
    regularization = config["regularization"]
    hidden_layers = config["hidden_layers"]

    # convert dtypes to play nicely with torch
    x_train = x_train.astype('float32')
    e1_train = e1_train.astype('float32')
    e2_train = e2_train.astype('float32')
    p1_train = p1_train.astype('float32')

    x_sub_train, x_sub_val, \
    e1_sub_train, e1_sub_val,\
    e2_sub_train, e2_sub_val, \
    p1_sub_train, p1_sub_val = \
        train_test_split(x_train, e1_train, e2_train, p1_train, 
                         random_state=random_seed,
                         test_size=0.2, stratify=e1_train)

    train_epochs = create_batches(num_samples=x_sub_train.shape[0], 
                                  batch_size=batch_size,
                                  num_epochs=epochs,
                                  random_seed=random_seed)

    best_val_loss = 0
    start_up_counter = 0
    patience_counter = 0
    set_torch_seed(random_seed)

    model = NeuralNet([x_sub_train.shape[1]] +
                      hidden_layers + [final_layer_size],
                      activation=activation)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if combine_labels:
        e1_sub_train = np.array(
            [1 if (e1_sub_train[i] == 1) or (e2_sub_train[i] == 1) else 0 for i
             in range(e1_sub_train.shape[0])]).astype('float32')

    for e in train_epochs:
        for batch in e:
            X = torch.from_numpy(x_sub_train[batch, :])
            if final_layer_size == 1:
                Y = torch.from_numpy(e1_sub_train[batch].reshape(-1, 1))
                batch_loss = nn.functional.binary_cross_entropy(
                    model(X), Y
                )
            else:
                Y = torch.from_numpy(
                    np.concatenate([(1 - np.logical_or(e1_sub_train[batch], e2_sub_train[batch], where=True).astype('float32')).reshape(-1, 1),
                                    e1_sub_train[batch].reshape(-1, 1),
                                    e2_sub_train[batch].reshape(-1, 1)],
                                   axis=1)
                )
                batch_loss = nn.functional.cross_entropy(
                    model(X), Y
                )


            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            batch_loss += regularization * regularization_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        with torch.no_grad():
            if final_layer_size == 1:
                pred = model(torch.from_numpy(x_sub_val)).numpy()[:, 0]
            else:
                pred = (torch.nn.functional.softmax(model(torch.from_numpy(x_sub_val)))[:, 1] /
                        torch.nn.functional.softmax(model(torch.from_numpy(x_sub_val)))[:, :2].sum(axis=1)).numpy()
        valid_loss = roc_auc_score(e1_sub_val, pred)

        start_up_counter += 1
        if start_up_counter >= start_up:
            if valid_loss > best_val_loss:
                best_val_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    with torch.no_grad():
        if final_layer_size == 1:
            pred = model(torch.from_numpy(x_sub_val)).numpy()[:, 0]
        else:
            pred = (torch.nn.functional.softmax(model(torch.from_numpy(x_sub_val)))[:, 1] /
                    torch.nn.functional.softmax(model(torch.from_numpy(x_sub_val)))[:, :2].sum(axis=1)).numpy()
    valid_R2 = 1-np.sum((p1_sub_val-pred)**2)/np.sum((p1_sub_val-np.mean(p1_sub_val))**2)
    return {"valid_R2": valid_R2}
