import numpy as np
from sklearn.neural_network import MLPClassifier


def sklearn_mlp(
        x_train, e1_train, e2_train, x_test, random_state, hidden_layers,
        activation
):
    single_model = MLPClassifier(hidden_layer_sizes=hidden_layers,
                                 activation=activation,
                                 random_state=random_state).fit(x_train,
                                                                e1_train)

    double_model = MLPClassifier(hidden_layer_sizes=hidden_layers,
                                 activation=activation,
                                 random_state=random_state)\
        .fit(x_train,
             np.concatenate([e1_train.reshape(-1, 1),
                             e2_train.reshape(-1, 1)], axis=1)
             )

    return single_model.predict_proba(x_test)[:, 1], \
           double_model.predict_proba(x_test)[:, 0]
