"""
  For tuning models using skopt we specify the space to search via metadata.
  We list below a set of pre-configured search spaces that have been found to perform well for the specific models.
  For each one we define the bounds, the corresponding scikit-learn parameter name, as well as how to sample values
  from that dimension (`'log-uniform'` for the learning rate)
"""
from skopt import forest_minimize, dummy_minimize, gbrt_minimize, gp_minimize, expected_minimum
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


def tune_model(X_train, y_train, X_test, y_test, model,
               space, metric, n_calls=25, minimize=True, min_func=gp_minimize):
    """
    :param X_train: array-like, shape = [n_samples, n_features] The input samples.
    :param y_train: array of shape = [n_samples] The training values.
    :param X_train: array-like, shape = [n_samples, n_features] The validation samples.
    :param y_train: array of shape = [n_samples] The validation values.
    :param model: An Sklearn api conformant model with named arguments
    :param space: The hyper-parameter search space for the model
    :param metric: The metric function to tune the model against e.g sklearn.metrics.
    :param minimize: Are we maximizing or minimizing the cost function.
    :param min_func: the minimization strategy from `skopt` to employ for finding minimum of the function.
    :return: model object with optimial hyperparameters.
    """

    # Using skopt decorator allows the objective function to receive the parameters as keyword arguments.
    @use_named_args(space)
    def objective(**params):
        model.set_params(**params)
        model.fit(X_train, y_train)
        if minimize:
            out_sample_score = metric(y_test, model.predict(X_test))
        else:
            out_sample_score = -metric(y_test, model.predict(X_test))
        return out_sample_score

    res_gp = min_func(objective, space, n_calls=n_calls, random_state=42, verbose=True)
    # Get the names of the parameter
    params = {dim.name: value for dim, value in zip(space, res_gp.x)}
    model.set_params(**params)
    print(model)
    return model


def space_lightgbm():
    space = [Categorical(['gbdt', 'gbrt', 'dart', 'goss'], name='boosting_type'),
             Integer(150, 1500, name="num_leaves"),
             Integer(5, 25, name='max_depth'),
             Real(0.005, 0.05, "log-uniform", name='learning_rate'),
             Integer(100, 1000, name="max_bin"),
             Integer(250, 2500, name="n_estimators"),
             Real(0.6, 0.9, name="bagging_fraction"),
             Real(0.6, 0.9, name="colsample_bytree"),
             Real(2., 20., name="min_child_weight"),
             Integer(25, 50, name="min_child_samples"),
             Real(0.1, 0.5, name="reg_alpha"),
             ]
    return space


def space_xlearn():
    space = [Integer(300, 1000, name='block_size'),
             Real(0.005, 0.1, name='lr'),
             Integer(4, 20, name='k'),
             Real(0.01, 0.5, name='reg_lambda'),
             Real(0.01, 0.5, name='init'),
             Integer(4, 50, name='epoch'),
             Categorical(['adagrad', 'ftrl'], name='opt'),
             Real(0.1, 2.5, name='alpha'),
             Real(0.1, 2.5, name='beta'),
             Real(0.1, 2.5, name='lambda_1'),
             Real(0.1, 2.5, name='lambda_2'),
             ]
    return space


def space_sklearn_gbm():
    space = [Integer(50, 250, name='n_estimators'),
             Integer(2, 12, name='max_depth'),
             Real(10 ** -5, 10 ** 0, "log-uniform", name='learning_rate'),
             Integer(2, 100, name='min_samples_split'),
             Integer(1, 100, name='min_samples_leaf')]
    return space


def space_sklearn_rf():
    space = [Integer(50, 250, name='n_estimators'),
             Integer(2, 100, name='min_samples_split'),
             Integer(1, 100, name='min_samples_leaf')]
    return space
