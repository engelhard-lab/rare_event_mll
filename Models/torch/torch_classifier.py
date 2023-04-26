import copy

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.torch.torch_dataset import Dataset
from Models.torch.torch_base_nn import NeuralNet


def torch_classifier(x_train, e1_train, e2_train, x_test, random_seed,
                     hidden_layers, activation, learning_rate=0.001,
                     batch_size=200,
                     epochs=200, regularization=0.0001):
    
    
    torch.manual_seed(random_seed)

    train = DataLoader(Dataset(x_train, e1_train, e2_train),
                       batch_size=batch_size, shuffle=True)

    layers = [x_train.shape[1]] + hidden_layers

    torch.manual_seed(random_seed)
    single_model = NeuralNet(layers + [1], activation=activation)
    multi_model = NeuralNet(layers + [2], activation=activation)

    torch.manual_seed(random_seed)
    optimizer_single = torch.optim.Adam(single_model.parameters(),
                                        lr=learning_rate)
    optimizer_multi = torch.optim.Adam(multi_model.parameters(),
                                       lr=learning_rate)

    for t in range(epochs // 2):
        for X, Y in train:
            # single
            batch_loss_single = nn.functional.binary_cross_entropy(
                single_model(X), Y[:, 0].unsqueeze(1)
            )
            regularization_loss = 0
            for param in single_model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            batch_loss_single += regularization * regularization_loss

            optimizer_single.zero_grad()
            batch_loss_single.backward()
            optimizer_single.step()

            # multi
            batch_loss_multi = nn.functional.binary_cross_entropy(
                multi_model(X), Y
            )
            regularization_loss = 0
            for param in multi_model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            batch_loss_multi += regularization * regularization_loss

            optimizer_multi.zero_grad()
            batch_loss_multi.backward()
            optimizer_multi.step()

    multi_params = []
    for _, param in multi_model.named_parameters():
        if param.requires_grad:
            multi_params.append(param.data)
    multi_params = copy.deepcopy(multi_params)
    multi_refined = NeuralNet(layers + [1], activation=activation,
                              preset_weights=multi_params)
    optimizer_refined = torch.optim.Adam(multi_refined.parameters(),
                                         lr=learning_rate)
    for t in range(epochs // 2):
        for X, Y in train:
            # single
            batch_loss_single = nn.functional.binary_cross_entropy(
                single_model(X), Y[:, 0].unsqueeze(1)
            )
            regularization_loss = 0
            for param in single_model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            batch_loss_single += regularization * regularization_loss

            optimizer_single.zero_grad()
            batch_loss_single.backward()
            optimizer_single.step()

            # multi
            batch_loss_multi = nn.functional.binary_cross_entropy(
                multi_model(X), Y
            )
            regularization_loss = 0
            for param in multi_model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            batch_loss_multi += regularization * regularization_loss

            optimizer_multi.zero_grad()
            batch_loss_multi.backward()
            optimizer_multi.step()

            # multi refined
            batch_loss_refined = nn.functional.binary_cross_entropy(
                multi_refined(X), Y[:, 0].unsqueeze(1)
            )
            regularization_loss = 0
            for param in multi_refined.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            batch_loss_refined += regularization * regularization_loss

            optimizer_refined.zero_grad()
            batch_loss_refined.backward()
            optimizer_refined.step()

    test = torch.from_numpy(x_test.astype('float32'))
    with torch.no_grad():
        single_preds = single_model(test).numpy()[:, 0]
        multi_preds = multi_model(test).numpy()[:, 0]
        refined_preds = multi_refined(test).numpy()[:, 0]

    return single_preds, multi_preds, refined_preds
