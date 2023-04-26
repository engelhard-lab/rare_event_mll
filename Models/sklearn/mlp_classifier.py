import copy
import numpy as np
from sklearn.neural_network import MLPClassifier
from Models.sklearn.mlp_preset_weights import MLPClassifierOverride


def sklearn_mlp(
        x_train, e1_train, e2_train, x_test, random_state, hidden_layers,
        activation, run_refined=False
):
    single_model = MLPClassifier(hidden_layer_sizes=hidden_layers,
                                 activation=activation,
                                 random_state=random_state).fit(x_train,
                                                                e1_train)

    multi_model = MLPClassifier(hidden_layer_sizes=hidden_layers,
                                 activation=activation,
                                 random_state=random_state)\
        .fit(x_train,
             np.concatenate([e1_train.reshape(-1, 1),
                             e2_train.reshape(-1, 1)], axis=1)
             )

    if run_refined:
        multi_coefs = copy.deepcopy(multi_model.coefs_)
        multi_intercepts = copy.deepcopy(multi_model.intercepts_)
        multi_coefs[-1] = multi_coefs[-1][:, [0]]
        multi_intercepts[-1] = multi_intercepts[-1][0]

        multi_refined_model = MLPClassifierOverride(
            hidden_layer_sizes=hidden_layers, activation=activation,
            random_state=random_state, init_coefs_=multi_coefs,
            init_intercepts_=multi_intercepts)
        multi_refined_model.fit(x_train, e1_train)
        multi_refined_results = multi_refined_model.predict_proba(x_test)[:, 0]
    else:
        multi_refined_results = None

    return single_model.predict_proba(x_test)[:, 1], \
           multi_model.predict_proba(x_test)[:, 1], \
           multi_refined_results
