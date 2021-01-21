import pandas as pd
import numpy as np

class DataPartitioner(object):

    def __init__(self, test_perc, train_cv_splits = 5, seed = 1235):
        self.test_perc = test_perc
        self.train_cv_splits = train_cv_splits
        self.seed = seed

    def partition_data(self, houses):
        obs = houses[['globalId']].copy()
        obs = obs.sort_values(['globalId'])
        obs_test = obs.sample(frac =self.test_perc, random_state = self.seed).copy()
        obs_test = obs_test.assign(test = True).assign(cv_split=np.nan)

        obs_train = (
        obs.merge(obs_test[['globalId']], how = 'left', indicator = True)
        .query("_merge == 'left_only'")
        .drop('_merge', axis=1)
        .assign(test=False))
        cv_splits = np.random.randint(1, self.train_cv_splits + 1, obs_train.shape[0])
        obs_train = obs_train.assign(cv_split = cv_splits)
        final_obs = pd.concat([obs_test, obs_train]).reset_index(drop=True)
        return final_obs