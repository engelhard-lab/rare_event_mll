import copy
import numpy as np
import random

import torch
import torch.nn as nn
from Models.torch.torch_base_nn import NeuralNet
from Models.dataloader import create_batches
import matplotlib.pyplot as plt


def set_torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch_classifier(x_train, e1_train, e2_train, x_test, e1_test,
                     hidden_layers, activation, random_seed, run_refined=False,
                     loss_plot=False,
                     early_stop=True,
                     learning_rate=0.001,
                     batch_size=200,
                     epochs=200, regularization=0.0001,
                     ):

    # convert dtypes to play nicely with torch
    x_train = x_train.astype('float32')
    e1_train = e1_train.astype('float32')
    e2_train = e2_train.astype('float32')
    x_test = x_test.astype('float32')
    e1_test = e1_test.astype('float32')

    set_torch_seed(random_seed)

    train_epochs = create_batches(x_train.shape[0], batch_size, epochs,
                                  random_seed)

    layers = [x_train.shape[1]] + hidden_layers

    single_model = NeuralNet(layers + [1], activation=activation)
    multi_model = NeuralNet(layers + [2], activation=activation)

    optimizer_single = torch.optim.Adam(single_model.parameters(),
                                        lr=learning_rate)
    optimizer_multi = torch.optim.Adam(multi_model.parameters(),
                                       lr=learning_rate)
    train_loss_single = []
    test_loss_single = []
    best_loss = float("inf")
    patience = 10

    for e in train_epochs:
        train_loss = 0
        for batch in e:
            X = torch.from_numpy(x_train[batch, :])
            Y = torch.from_numpy(e1_train[batch].reshape(-1, 1))
            # single
            batch_loss_single = nn.functional.binary_cross_entropy(
                single_model(X), Y
            )
            single_regularization_loss = 0
            for param in single_model.parameters():
                single_regularization_loss += torch.sum(torch.abs(param))
            batch_loss_single += regularization * single_regularization_loss

            optimizer_single.zero_grad()
            batch_loss_single.backward()
            optimizer_single.step()

            train_loss += batch_loss_single.item()

        if early_stop:
            train_loss /= len(e)
            train_loss_single.append(train_loss)
        
            with torch.no_grad():
                test_loss = nn.functional.binary_cross_entropy(
                    single_model(torch.from_numpy(x_test)),
                    torch.from_numpy(e1_test.reshape(-1, 1))
                    )
                test_loss_single.append(test_loss)
            
            if test_loss < best_loss:
                best_loss = test_loss
                patientce_counter = 0
            else:
                patientce_counter += 1
            
            if patientce_counter >= patience:
                break


    train_loss_multi = []
    test_loss_multi = []
    best_loss = float("inf")
    patience = 10

    multi_epochs = epochs // 2 if run_refined else epochs
    for e in train_epochs[:multi_epochs]:
        train_loss = 0
        for batch in e:
            X = torch.from_numpy(x_train[batch, :])
            Y = torch.from_numpy(
                np.concatenate([e1_train[batch].reshape(-1, 1),
                                e2_train[batch].reshape(-1, 1)], axis=1)
            )
            # multi
            batch_loss_multi = nn.functional.binary_cross_entropy(
                multi_model(X), Y
            )
            multi_regularization_loss = 0
            for param in multi_model.parameters():
                multi_regularization_loss += torch.sum(torch.abs(param))
            batch_loss_multi += regularization * multi_regularization_loss

            optimizer_multi.zero_grad()
            batch_loss_multi.backward()
            optimizer_multi.step()

            train_loss += batch_loss_multi.item()

        if early_stop:
            train_loss /= len(e)
            train_loss_multi.append(train_loss)

            with torch.no_grad():
                test_loss = nn.functional.binary_cross_entropy(
                    multi_model(torch.from_numpy(x_test))[:,0].reshape(-1,1),
                    torch.from_numpy(e1_test).reshape(-1, 1)
                    )
                test_loss_multi.append(test_loss)
            
            if test_loss < best_loss:
                best_loss = test_loss
                patientce_counter = 0
            else:
                patientce_counter += 1

            if patientce_counter >= patience:
                break

    if loss_plot:
        plt.figure()
        plt.plot(range(len(train_loss_single)), train_loss_single, color="red", label="single train loss")
        plt.plot(range(len(test_loss_single)), test_loss_single, color="blue", label="singel test loss")
        plt.plot(range(len(train_loss_multi)), train_loss_multi, color="purple", label="multi train loss")
        plt.plot(range(len(test_loss_multi)), test_loss_multi, color="navy", label="multi train loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()



    if run_refined:
        multi_params = []
        for _, param in multi_model.named_parameters():
            if param.requires_grad:
                multi_params.append(param.data)
        multi_params = copy.deepcopy(multi_params)
        multi_refined = NeuralNet(layers + [1], activation=activation,
                                  preset_weights=multi_params)
        optimizer_refined = torch.optim.Adam(multi_refined.parameters(),
                                             lr=learning_rate)
        for e in train_epochs[multi_epochs:]:
            for batch in e:
                X = torch.from_numpy(x_train[batch, :])
                Y = torch.from_numpy(
                    np.concatenate([e1_train[batch].reshape(-1, 1),
                                    e2_train[batch].reshape(-1, 1)], axis=1)
                )
                # multi
                batch_loss_multi = nn.functional.binary_cross_entropy(
                    multi_model(X), Y
                )
                multi_regularization_loss = 0
                for param in multi_model.parameters():
                    multi_regularization_loss += torch.sum(torch.abs(param))
                batch_loss_multi += regularization * multi_regularization_loss

                optimizer_multi.zero_grad()
                batch_loss_multi.backward()
                optimizer_multi.step()

        for e in train_epochs[multi_epochs:]:
            for batch in e:
                X = torch.from_numpy(x_train[batch, :])
                Y = torch.from_numpy(e1_train[batch].reshape(-1, 1))
                # multi refined
                batch_loss_refined = nn.functional.binary_cross_entropy(
                    multi_refined(X), Y
                )
                refined_regularization_loss = 0
                for param in multi_refined.parameters():
                    refined_regularization_loss += torch.sum(torch.abs(param))
                batch_loss_refined += regularization * refined_regularization_loss

                optimizer_refined.zero_grad()
                batch_loss_refined.backward()
                optimizer_refined.step()

    test = torch.from_numpy(x_test.astype('float32'))
    with torch.no_grad():
        single_preds = single_model(test).numpy()[:, 0]
        multi_preds = multi_model(test).numpy()[:, 0]
        if run_refined:
            refined_preds = multi_refined(test).numpy()[:, 0]
        else:
            refined_preds = None

    return single_preds, multi_preds, refined_preds
