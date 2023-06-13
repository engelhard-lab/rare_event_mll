import numpy as np
import random

import torch
import torch.nn as nn
from Models.torch.torch_base_nn import NeuralNet
from Models.dataloader import create_batches
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def set_torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch_classifier(single_config, multi_config, fixed_config, data,
                     loss_plot=False, epochs=800, patience=5, start_up=20,
                     combined_config=None):
    
    x_train = data["x_train"]
    e1_train = data["e1_train"]
    e2_train = data["e2_train"]
    x_test = data["x_test"]
    e1_test = data["e1_test"]

    activation = fixed_config["activation"]
    random_seed = fixed_config["random_seed"]
    batch_size = fixed_config["batch_size"]

    if single_config is not None:
        single_learning_rate = single_config["learning_rate"]
        single_regularization = single_config["regularization"]
        single_hidden_layers = single_config["hidden_layers"]

    if multi_config is not None:
        multi_learning_rate = multi_config["learning_rate"]
        multi_regularization = multi_config["regularization"]
        multi_hidden_layers = multi_config["hidden_layers"]


    # convert dtypes to play nicely with torch
    x_train = x_train.astype('float32')
    e1_train = e1_train.astype('float32')
    x_test = x_test.astype('float32')
    e1_test = e1_test.astype('float32')
    e2_train = e2_train.astype('float32')

    x_sub_train, x_sub_val, e1_sub_train, \
    e1_sub_val, e2_sub_train, e2_sub_val = \
        train_test_split(x_train, e1_train.astype('float32'),
                         e2_train.astype('float32'), random_state=random_seed,
                         test_size=0.2, stratify=e1_train)

    train_epochs = create_batches(num_samples=x_sub_train.shape[0], 
                                  batch_size=batch_size,
                                  num_epochs=epochs,
                                  random_seed=random_seed)

            
    single_train_loss_list = []
    single_valid_loss_list = []
    best_single_val_loss = 0  # float("inf")
    start_up_counter = 0
    patience_counter = 0
    set_torch_seed(random_seed)

    # single learning
    if single_config is not None:
        single_model = NeuralNet([x_sub_train.shape[1]] +
                                 single_hidden_layers + [1],
                                 activation=activation)

        optimizer_single = torch.optim.Adam(single_model.parameters(),
                                            lr=single_learning_rate)

        for e in train_epochs:
            single_train_loss = 0
            for batch in e:
                X = torch.from_numpy(x_sub_train[batch, :])
                Y = torch.from_numpy(e1_sub_train[batch].reshape(-1, 1))

                batch_loss = nn.functional.binary_cross_entropy(
                    single_model(X), Y
                )

                single_train_loss += batch_loss.item()

                single_regularization_loss = 0
                for param in single_model.parameters():
                    single_regularization_loss += torch.sum(torch.abs(param))
                batch_loss += single_regularization * \
                              single_regularization_loss

                optimizer_single.zero_grad()
                batch_loss.backward()
                optimizer_single.step()

            single_train_loss /= len(e)
            single_train_loss_list.append(single_train_loss)

            with torch.no_grad():
                # single_valid_loss = nn.functional.binary_cross_entropy(
                #     single_model(torch.from_numpy(x_sub_val)),
                #     torch.from_numpy(e1_sub_val.reshape(-1, 1))
                #     )
                # single_valid_loss = single_valid_loss.item()
                # single_valid_loss_list.append(single_valid_loss)
                single_pred = single_model(torch.from_numpy(x_sub_val)).numpy()[
                             :, 0]
            single_valid_loss = roc_auc_score(e1_sub_val, single_pred)
            single_valid_loss_list.append(single_valid_loss)

            start_up_counter += 1
            if start_up_counter >= start_up:
                if single_valid_loss > best_single_val_loss:
                    best_single_val_loss = single_valid_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        with torch.no_grad():
            single_pred = single_model(torch.from_numpy(x_sub_val)).numpy()[:, 0]
        single_valid_auc = roc_auc_score(e1_sub_val, single_pred)
        single_valid_ap = average_precision_score(e1_sub_val, single_pred)
        
    

    multi_train_loss_list = []
    multi_valid_loss_list = []
    best_multi_val_loss = 0  # float("inf")
    patience_counter = 0
    start_up_counter = 0
    set_torch_seed(random_seed)

    # multi learning
    if multi_config is not None:
        multi_model = NeuralNet([x_sub_train.shape[1]] +
                                multi_hidden_layers + [3],
                                activation=activation)

        # with torch.no_grad():
        #     multi_model.forward_pass[4].weight.data[1] = torch.zeros(
        #         (multi_hidden_layers[-1],), requires_grad=True)
        #     multi_model.forward_pass[4].bias.data[1] = torch.zeros(
        #         (1,), requires_grad=True)

        optimizer_multi = torch.optim.Adam(multi_model.parameters(),
                                           lr=multi_learning_rate)

        for e in train_epochs:
            multi_train_loss = 0
            for batch in e:
                X = torch.from_numpy(x_sub_train[batch, :])
                Y = torch.from_numpy(
                    np.concatenate([(1 - np.logical_or(e1_sub_train[batch], e2_sub_train[batch], where=True).astype('float32')).reshape(-1, 1),
                                    e1_sub_train[batch].reshape(-1, 1),
                                    e2_sub_train[batch].reshape(-1, 1)], axis=1)
                )
                # Y = torch.from_numpy(e1_sub_train[batch].reshape(-1, 1))
                batch_loss = nn.functional.cross_entropy(multi_model(X), Y)
                multi_regularization_loss = 0
                for param in multi_model.parameters():
                    multi_regularization_loss += torch.sum(torch.abs(param))
                batch_loss += multi_regularization * multi_regularization_loss

                optimizer_multi.zero_grad()
                batch_loss.backward()
                optimizer_multi.step()

                multi_train_loss += batch_loss.item()

            multi_train_loss /= len(e)
            multi_train_loss_list.append(multi_train_loss)

            with torch.no_grad():
                # multi_valid_loss = nn.functional.binary_cross_entropy(
                #     # multi_model(torch.from_numpy(x_sub_val)),
                #     multi_model(torch.from_numpy(x_sub_val))[:, 0].reshape(-1,
                #                                                            1),
                #     torch.from_numpy(e1_sub_val).reshape(-1, 1)
                #     )
                # multi_valid_loss = multi_valid_loss.item()
                # multi_valid_loss_list.append(multi_valid_loss)
                # multi_pred = multi_model(torch.from_numpy(x_sub_val)).numpy()[
                #              :, 0]
                multi_pred = (torch.nn.functional.softmax(multi_model(torch.from_numpy(x_sub_val)))[:, 1] /
                              torch.nn.functional.softmax(multi_model(torch.from_numpy(x_sub_val)))[:, :2].sum(axis=1)).numpy()
            multi_valid_loss = roc_auc_score(e1_sub_val, multi_pred)
            multi_valid_loss_list.append(multi_valid_loss)

            start_up_counter += 1
            if start_up_counter >= start_up:
                if multi_valid_loss > best_multi_val_loss:
                    best_multi_val_loss = multi_valid_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        with torch.no_grad():
            multi_pred = multi_model(torch.from_numpy(x_sub_val)).numpy()[:, 0]
        multi_valid_auc = roc_auc_score(e1_sub_val, multi_pred)
        multi_valid_ap = average_precision_score(e1_sub_val, multi_pred)

    if combined_config is not None:
        combined_learning_rate = combined_config["learning_rate"]
        combined_regularization = combined_config["regularization"]
        combined_hidden_layers = combined_config["hidden_layers"]

        combined_train_loss_list = []
        combined_valid_loss_list = []
        best_combined_val_loss = 0  # float("inf")
        patience_counter = 0
        start_up_counter = 0
        set_torch_seed(random_seed)

        # multi_params = []
        # for _, param in multi_model.named_parameters():
        #     if param.requires_grad:
        #         multi_params.append(param.data)
        # multi_params = copy.deepcopy(multi_params)
        # refined_model = NeuralNet([x_sub_train.shape[1]] +
        #                           multi_hidden_layers + [1],
        #                           activation=activation,
        #                           preset_weights=multi_params)
        # optimizer_refined = torch.optim.Adam(refined_model.parameters(),
        #                                      lr=multi_learning_rate)
        #
        # for e in train_epochs:
        #     refined_train_loss = 0
        #     for batch in e:
        #         X = torch.from_numpy(x_sub_train[batch, :])
        #         Y = torch.from_numpy(e1_train[batch].reshape(-1, 1))
        #         batch_loss = nn.functional.binary_cross_entropy(
        #             refined_model(X), Y
        #         )
        #         refined_regularization_loss = 0
        #         for param in refined_model.parameters():
        #             refined_regularization_loss += torch.sum(torch.abs(param))
        #         batch_loss += multi_regularization * refined_regularization_loss
        #
        #         optimizer_refined.zero_grad()
        #         batch_loss.backward()
        #         optimizer_refined.step()
        #
        #         refined_train_loss += batch_loss.item()
        #
        #     refined_train_loss /= len(e)
        #     refined_train_loss_list.append(refined_train_loss)
        #
        #     with torch.no_grad():
        #         # multi_valid_loss = nn.functional.binary_cross_entropy(
        #         #     # multi_model(torch.from_numpy(x_sub_val)),
        #         #     multi_model(torch.from_numpy(x_sub_val))[:, 0].reshape(-1,
        #         #                                                            1),
        #         #     torch.from_numpy(e1_sub_val).reshape(-1, 1)
        #         #     )
        #         # multi_valid_loss = multi_valid_loss.item()
        #         # multi_valid_loss_list.append(multi_valid_loss)
        #         refined_pred = refined_model(torch.from_numpy(x_sub_val)).numpy()[
        #                      :, 0]
        #     refined_valid_loss = roc_auc_score(e1_sub_val, refined_pred)
        #     refined_valid_loss_list.append(refined_valid_loss)
        #
        #     start_up_counter += 1
        #     if start_up_counter >= start_up:
        #         if refined_valid_loss > best_refined_val_loss:
        #             best_refined_val_loss = refined_valid_loss
        #             patience_counter = 0
        #         else:
        #             patience_counter += 1
        #
        #         if patience_counter >= patience:
        #             break
        combined_model = NeuralNet([x_sub_train.shape[1]] +
                                   combined_hidden_layers + [1],
                                   activation=activation)

        # with torch.no_grad():
        #     multi_model.forward_pass[4].weight.data[1] = torch.zeros(
        #         (multi_hidden_layers[-1],), requires_grad=True)
        #     multi_model.forward_pass[4].bias.data[1] = torch.zeros(
        #         (1,), requires_grad=True)

        optimizer_combined = torch.optim.Adam(combined_model.parameters(),
                                              lr=combined_learning_rate)

        e_combined = np.array(
            [1 if (e1_sub_train[i] == 1) or (e2_sub_train[i] == 1) else 0 for i
             in range(e1_sub_train.shape[0])]).astype('float32')

        for e in train_epochs:
            combined_train_loss = 0
            for batch in e:
                X = torch.from_numpy(x_sub_train[batch, :])
                Y = torch.from_numpy(e_combined[batch].reshape(-1, 1))
                batch_loss = nn.functional.binary_cross_entropy(
                    combined_model(X), Y
                )
                combined_regularization_loss = 0
                for param in combined_model.parameters():
                    combined_regularization_loss += torch.sum(torch.abs(param))
                batch_loss += combined_regularization * \
                              combined_regularization_loss

                optimizer_combined.zero_grad()
                batch_loss.backward()
                optimizer_combined.step()

                combined_train_loss += batch_loss.item()

            combined_train_loss /= len(e)
            combined_train_loss_list.append(combined_train_loss)

            with torch.no_grad():
                combined_pred = combined_model(
                    torch.from_numpy(x_sub_val)).numpy()[:, 0]
            combined_valid_loss = roc_auc_score(e1_sub_val, combined_pred)
            combined_valid_loss_list.append(combined_valid_loss)

            start_up_counter += 1
            if start_up_counter >= start_up:
                if combined_valid_loss > best_combined_val_loss:
                    best_combined_val_loss = combined_valid_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

    if loss_plot:
        plt.figure()
        plt.plot(range(len(single_train_loss_list)), single_train_loss_list,
                 color="red", label="single train loss", linewidth=1.5)
        plt.plot(range(len(single_valid_loss_list)), single_valid_loss_list,
                 color="blue", label="single val loss", linewidth=2)
        plt.plot(range(len(multi_train_loss_list)), multi_train_loss_list,
                 color="black", label="multi train loss", linestyle='dashed',
                 linewidth=1.5)
        plt.plot(range(len(multi_valid_loss_list)), multi_valid_loss_list,
                 color="gray", label="multi val loss", linestyle='dashed',
                 linewidth=2)
        if combined_config is not None:
            plt.plot(range(len(combined_valid_loss_list)), combined_valid_loss_list,
                     color="orange", label="combined val loss", linestyle="dotted",
                     linewidth=2)

        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    test = torch.from_numpy(x_test.astype('float32'))
    with torch.no_grad():
        single_pred = single_model(test).numpy()[:, 0]
        multi_pred = (torch.nn.functional.softmax(multi_model(test))[:, 1] /
                              torch.nn.functional.softmax(multi_model(test))[:, :2].sum(axis=1)).numpy()
        if combined_config is not None:
            combined_pred = combined_model(test).numpy()[:, 0]
    single_auc = roc_auc_score(e1_test, single_pred)
    multi_auc = roc_auc_score(e1_test, multi_pred)
    single_ap = average_precision_score(e1_test, single_pred)
    multi_ap = average_precision_score(e1_test, multi_pred)
    if combined_config is not None:
        combined_auc = roc_auc_score(e1_test, combined_pred)
        combined_ap = average_precision_score(e1_test, combined_pred)
        return single_auc, multi_auc, combined_auc, single_ap, multi_ap, combined_ap

    return single_auc, multi_auc, single_ap, multi_ap
