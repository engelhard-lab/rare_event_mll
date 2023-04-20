import itertools
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import time

from generate_data import generate_data_linear, generate_data_shared_features
from Models.sklearn.mlp_classifier import sklearn_mlp


def base_sklearn_auc_ap(n, p, er, hidden_layers, activations,
                        n_iters, datagen, similarity_measures, test_perc,
                        print_time=True, print_output=False, plot=False):
    datagen_args = {
        'n_patients': n,
        'n_features': p,
        'event_rate': er,
        'print_output': print_output,
        'plot': plot
    }
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

    results = pd.DataFrame(columns=['n', 'p', 'er', 'hidden_layers',
                                    'activation',
                                    'iter', 'auc_single', 'auc_multi',
                                    'auc_multi_refined', 'ap_single',
                                    'ap_multi',
                                    'ap_multi_refined'] +
                                   list(similarity_measures.keys()))
    start = time.time()
    for h in hidden_layers:
        hidden_label = '|'.join([str(layer) for layer in h])
        for act in activations:
            for s in similarity_combos:
                for r in range(n_iters):
                    datagen_args['random_seed'] = r
                    datagen_args.update(s)
                    x, e1, e2 = datagen(**datagen_args)
                    x_train, x_test, \
                    e1_train, e1_test, \
                    e2_train, e2_test = train_test_split(x, e1, e2,
                                                         random_state=r,
                                                         test_size=test_perc)
                    single_proba, multi_proba, multi_refined_proba = sklearn_mlp(
                        x_train=x_train, e1_train=e1_train, e2_train=e2_train,
                        x_test=x_test, random_state=r, hidden_layers=h,
                        activation=act)

                    single_auc = roc_auc_score(e1_test, single_proba)
                    multi_auc = roc_auc_score(e1_test, multi_proba)
                    multi_refined_auc = roc_auc_score(e1_test,
                                                      multi_refined_proba)

                    single_ap = average_precision_score(e1_test, single_proba)
                    multi_ap = average_precision_score(e1_test, multi_proba)
                    multi_refined_ap = average_precision_score(
                        e1_test, multi_refined_proba
                    )

                    results.loc[len(results.index)] = [n, p, er, hidden_label,
                                                       act, r, single_auc,
                                                       multi_auc,
                                                       multi_refined_auc,
                                                       single_ap, multi_ap,
                                                       multi_refined_ap] + \
                                                      list(s.values())
                if print_time:
                    print(f'Hidden Layers: {hidden_label}, Activation: {act}, '
                          f'{" ".join([str(k) + ": "+str(v) for k,v in s.items()])}. '
                          f'Total Time: {time.time() - start}')
    return results