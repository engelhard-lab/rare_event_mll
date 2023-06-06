import itertools
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import time

from generate_data import generate_data_linear, generate_data_shared_features
from Models.sklearn.mlp_classifier import sklearn_mlp
from Models.torch.torch_classifier import torch_classifier
from Models.torch.torch_ray_tune import ray_tune


def base_auc_ap(n, p, event_rate, model_types, activations, param_config,
                n_iters, datagen, similarity_measures, test_perc,
                run_refined=False, print_time=True, print_output=False,
                plot=False, loss_plot=False, early_stop=True, batch_size=200):

    sim_keys, sim_values = zip(*similarity_measures.items())
    similarity_combos = [dict(zip(sim_keys, v)) for v in
                         itertools.product(*sim_values)]
    if datagen == 'linear':
        datagen = generate_data_linear
    elif datagen == 'shared_features':
        datagen =generate_data_shared_features
    else:
        print(f'datagen argument not a valid datagen process')
        raise ValueError

    if run_refined:
        columns = ['n', 'p', 'er', 'model',
                   'activation', 'iter', 'auc_single', 'auc_multi', 'auc_refined',
                   'ap_single', 'ap_multi', 'ap_refined', 'config_single', 'config_multi']
    else:
        columns = ['n', 'p', 'er', 'model',
                   'activation', 'iter', 'auc_single', 'auc_multi',
                   'ap_single', 'ap_multi', 'config_single', 'config_multi']

    results = pd.DataFrame(columns=columns + list(similarity_measures.keys()))
    start = time.time()
    for act in activations:
        for s in similarity_combos:
            for er in event_rate:
                for r in range(n_iters):    
                    r = r+5            
                    datagen_args = {
                        'n_patients': n,
                        'n_features': p,
                        'event_rate': er,
                        'print_output': print_output,
                        'plot': plot
                        }
                    datagen_args['random_seed'] = r
                    datagen_args.update(s)
                    x, e1, e2 = datagen(**datagen_args)
                    x_train, x_test, \
                    e1_train, e1_test, \
                    e2_train, e2_test = train_test_split(x, e1, e2,
                                                        random_state=r,
                                                        test_size=test_perc,
                                                         stratify=e1)
                    for m in model_types:
                        if m == 'sklearn':
                            single_proba, multi_proba, multi_refined_proba = sklearn_mlp(
                                x_train=x_train, e1_train=e1_train, e2_train=e2_train,
                                x_test=x_test, random_state=r,
                                activation=act, run_refined=run_refined,
                                loss_plot = loss_plot,
                                early_stop=early_stop)
                        elif m == 'torch':
                            data = {
                                "x_train": x_train,
                                "e1_train": e1_train,
                                "e2_train": e2_train, 
                                "x_test": x_test,
                                "e1_test": e1_test,
                            }
                            other_var = {
                                "activation": act, 
                                "random_seed": r,
                                "batch_size": batch_size
                            }
                            # best_config_single = ray_tune(config=param_config,
                            #                               fixed_var=other_var,
                            #                               data=data,
                            #                               )
                            #
                            # print(best_config_single)
                            # best_config_multi = ray_tune(config=param_config,
                            #                              fixed_var=other_var,
                            #                              data=data,
                            #                              )
                            # print(best_config_multi)
                            best_config_single = {
                                'learning_rate': 1e-5,
                                'batch_size': 200,
                                'regularization': 1e-5,
                                'hidden_layers': [200]
                            }
                            best_config_multi = {
                                'learning_rate': 1e-5,
                                'batch_size': 200,
                                'regularization': 1e-5,
                                'hidden_layers': [200, 2]
                            }

                        if run_refined:
                            single_auc, multi_auc, refined_auc, single_ap, \
                            multi_ap, refined_ap = torch_classifier(
                                single_config=best_config_single,
                                multi_config=best_config_multi,
                                fixed_config=other_var,
                                data=data,
                                performance=True, loss_plot=loss_plot,
                                run_refined=True)
                            results.loc[len(results.index)] = [n, p, er, m,
                                                            act, r, single_auc,
                                                            multi_auc, refined_auc,
                                                            single_ap,
                                                            multi_ap, refined_ap,
                                                            best_config_single,
                                                            best_config_multi] + \
                                                            list(s.values())
                        else:
                            single_auc, multi_auc, single_ap, multi_ap = torch_classifier(
                                single_config=best_config_single,
                                multi_config=best_config_multi,
                                fixed_config=other_var,
                                data=data,
                                performance=True, loss_plot=loss_plot)
                            results.loc[len(results.index)] = [n, p, er, m,
                                                            act, r, single_auc,
                                                            multi_auc,
                                                            single_ap,
                                                            multi_ap,
                                                            best_config_single,
                                                            best_config_multi] + \
                                                            list(s.values())
                        results.to_csv("tempo2.csv")
                if print_time:
                    print(f'Activation: {act}, '
                        f'{" ".join([str(k) + ": "+str(v) for k,v in s.items()])}. '
                        f'Total Time: {time.time() - start}')
    return results
