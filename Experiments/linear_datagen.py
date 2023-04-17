import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import time

from generate_data import generate_data_linear
from Models.sklearn.mlp_classifier import sklearn_mlp

n_patients = 5000
n_features = 10
event_rate = 0.01
activations = ['identity', 'relu']
hidden_layers = [[1]]
similarities = [0.8, 1.0]  # used only for linear datagen
n_iters = 2
test_perc = 0.25

results = pd.DataFrame(columns=['n', 'p', 'er', 'hidden_layers', 'activation',
                                'sim', 'iter', 'auc_single', 'auc_multi',
                                'ap_single', 'ap_multi'])

start = time.time()
for h in hidden_layers:
    hidden_label = '|'.join([str(layer) for layer in h])
    for act in activations:
        for s in similarities:
            for r in range(n_iters):
                x, e1, e2, u1, u2 = generate_data_linear(n_patients=n_patients,
                                                         n_features=n_features,
                                                         event_rate=event_rate,
                                                         similarity=s,
                                                         print_output=False,
                                                         plot=False,
                                                         random_seed=r)
                x_train, x_test, \
                e1_train, e1_test, \
                e2_train, e2_test = train_test_split(x, e1, e2,
                                                     random_state=r,
                                                     test_size=test_perc)
                single_proba, multi_proba = sklearn_mlp(x_train=x_train,
                                                        e1_train=e1_train,
                                                        e2_train=e2_train,
                                                        x_test=x_test,
                                                        random_state=r,
                                                        hidden_layers=h,
                                                        activation=act)

                single_auc = roc_auc_score(e1_test, single_proba)
                multi_auc = roc_auc_score(e1_test, multi_proba)

                single_ap = average_precision_score(e1_test, single_proba)
                multi_ap = average_precision_score(e1_test, multi_proba)

                results.loc[len(results.index)] = [n_patients, n_features,
                                                   event_rate, hidden_label,
                                                   act, s, r,
                                                   single_auc, multi_auc,
                                                   single_ap, multi_ap]
            print(f'Hidden Layers: {hidden_label}, Activation: {act}, '
                  f'Similarity: {s}. Total Time: {time.time() - start}')

results.to_csv(f'Results/sklearn/results2.csv', index=False)
