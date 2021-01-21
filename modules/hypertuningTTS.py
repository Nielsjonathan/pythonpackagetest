import pandas as pd
import numpy as np
import itertools
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

class HypertunerTTS(object):
    def __init__(self, estimator, tuning_params, validation_mapping,
    target = "sellingtime"
    ):
        self.estimator = estimator
        self.tuning_params = tuning_params
        self.validation_mapping = validation_mapping
        self.target = target

    def calculate_mean_cv_error(self, train_set, estimator_cv):
        splits = train_set['cv_split'].unique().tolist()
        splits.sort()

        cv_errors = []

        for i in splits:
            train_split = train_set.query(f"cv_split != {i}")
            X_train = train_split.drop(['globalId', 'sellingtime', 'cv_split'],axis=1)
            y_train = train_split['sellingtime']
            estimator_cv.fit(X=X_train, y = y_train)
            test_obs = train_set.query(f"cv_split == {i}")
            X_test = test_obs.drop(['globalId', 'sellingtime', 'cv_split'],axis=1)
            y_test = test_obs['sellingtime']
            y_pred = estimator_cv.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            cv_errors.append(rmse)
        mean_rmse = np.mean(cv_errors)
        return mean_rmse

    def tune_model(self, train_set):
        parameter_combos = []
        parameter_combos_dicts = []

        for a in itertools.product(*self.tuning_params.values()):
            parameter_combos.append(a)

        for i in parameter_combos:
            d = {}
            for j in range(len(i)):
                d[list(self.tuning_params.keys())[j]] = i[j]
            parameter_combos_dicts.append(d)

        validation_mapping_train = self.validation_mapping.query("test == False")
        train_set = train_set.merge(validation_mapping_train[['globalId', 'cv_split']])
        
        # the cv errors for each parameter combo inside paramter_combos_dict
        cv_errors = []
        best_rmse = 1000000000
        best_parameters = ''


        for d in parameter_combos_dicts:
            estimator_cv = deepcopy(self.estimator)
            estimator_cv = estimator_cv.set_params(**d)
            mean_cv_error = self.calculate_mean_cv_error(train_set, estimator_cv)
            cv_errors.append(mean_cv_error)
            print(d, 'RMSE: ' , mean_cv_error)
            if mean_cv_error < best_rmse:
                best_parameters = d
                best_rmse = mean_cv_error

        # creating train set
        train_set_model = train_set.drop(columns=['sellingtime', 'globalId', 'cv_split'])

        # train the best model on all train set
        final_estimator = deepcopy(self.estimator)
        final_estimator = self.estimator.set_params(**best_parameters)
        final_estimator.fit(X=train_set_model, y=train_set[self.target])
        return final_estimator, best_parameters
