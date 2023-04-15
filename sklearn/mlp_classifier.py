import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import time

from generate_data import generate_data

n_patients = 50000
n_features = 100
event_rate = 0.01
activations = ['identity', 'relu']
similarities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
n_iters = 25
test_perc = 0.25

results = pd.DataFrame(columns=['n', 'p', 'er', 'activation', 'sim',
                                'iter', 'auc_single', 'auc_multi',
                                'ap_single', 'ap_multi'])

start = time.time()
for act in activations:
    for s in similarities:
        for r in range(n_iters):
            x, e1, e2, u1, u2 = generate_data(n_patients=n_patients,
                                              n_features=n_features,
                                              event_rate=event_rate,
                                              similarity=s,
                                              print_output=False, plot=False,
                                              random_seed=r)

            x_train, x_test, e1_train, \
            e1_test, e2_train, e2_test = train_test_split(x, e1, e2,
                                                          random_state=r,
                                                          test_size=test_perc)

            single_model = MLPClassifier(hidden_layer_sizes=[1],
                                         activation=act,
                                         random_state=r).fit(x_train, e1_train)

            double_model = MLPClassifier(hidden_layer_sizes=[1],
                                         activation=act,
                                         random_state=r)\
                .fit(x_train,
                     np.concatenate([e1_train.reshape(-1, 1),
                                     e2_train.reshape(-1, 1)], axis=1)
                     )

            single_auc = roc_auc_score(e1_test,
                                       single_model.predict_proba(x_test)[:, 1])
            double_auc = roc_auc_score(e1_test,
                                       double_model.predict_proba(x_test)[:, 0])

            single_ap = average_precision_score(
                e1_test, single_model.predict_proba(x_test)[:, 1])
            double_ap = average_precision_score(
                e1_test, double_model.predict_proba(x_test)[:, 0])

            results.loc[len(results.index)] = [n_patients, n_features,
                                               event_rate, act, s, r,
                                               single_auc, double_auc,
                                               single_ap, double_ap]
        print(f'Activation {act} with similarity '
              f'{s} complete: {time.time() - start}')

results.to_csv(f'results/results.csv', index=False)
