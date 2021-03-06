import pandas as pd
import numpy as np
import itertools
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

class Hypertuner(object):
    def __init__(self, estimator, tuning_params, validation_mapping,
    target = "sellingPrice"
    ):
        self.estimator = estimator
        self.tuning_params = tuning_params
        self.validation_mapping = validation_mapping
        self.target = target

    def calculate_mean_cv_error(self, train_set, estimator_cv):
        # now perform cross validation fitting for each split
        splits = train_set['cv_split'].unique().tolist()
        splits.sort()

        cv_errors = []

        for i in splits:
            train_split = train_set.query(f"cv_split != {i}")
            X_train = train_split.drop(['globalId', 'sellingPrice', 'cv_split'],axis=1)
            print(X_train.columns)
            y_train = train_split['sellingPrice']
            estimator_cv.fit(X=X_train, y = y_train)
            # evaluate the model on split 1
            test_obs = train_set.query(f"cv_split == {i}")
            X_test = test_obs.drop(['globalId', 'sellingPrice', 'cv_split'],axis=1)
            y_test = test_obs['sellingPrice']
            y_pred = estimator_cv.predict(X_test)
            # calculate error measure on this fold for the estimator with the
            # given parameters
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            #rmse = np.sqrt(np.sum(np.square(X_test['sellingPrice'] - y_pred))/X_test.shape[0])
            #np.sqrt(mean_squared_error(X_test['sellingPrice'], y_pred))
            cv_errors.append(rmse)
        mean_rmse = np.mean(cv_errors)
        return mean_rmse

    def tune_model(self, train_set):
        '''Perform the hypertuning of the estimator on the train set
        for all the combinations of the hyperparameters
        '''
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

        #return best_parameters, best_rmse



        
        # get the best parameter combo out of the lot
        #best_parameters = {'n_estimators': 40, 'max_depth': 12} # needs to be replaced with the best parameter combo i.e. with smallers cv error value

        # train the best model on all train set
        final_estimator = deepcopy(self.estimator)
        final_estimator = self.estimator.set_params(**best_parameters)
        final_estimator.fit(X=train_set, y=train_set[self.target]) # check that the columns are righjt
        return final_estimator, best_parameters


# estimator = RandomForestRegressor(random_state=1234)
# tuning_params = conf["training_params"]["hypertuning"]["RF_params"]
# validation_mapping = validation_mapping

# self = Hypertuner(estimator = RandomForestRegressor(random_state=1234),
# tuning_params = conf["training_params"]["hypertuning"]["RF_params"],
# validation_mapping = validation_mapping)

