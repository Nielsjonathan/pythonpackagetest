#importing existing modules
import json
import os
import datetime
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pickle

#importing our modules
from modules.load import DataLoader
from modules.clean import DataCleaner
from modules.validation import DataPartitioner
from modules.featurizing import Featurizer
from modules.hypertuning import Hypertuner

def main():
    #saving the run time as id
    run_id_start_time = datetime.datetime.now()
    print(f'Starting with run at time: {run_id_start_time}')

    #reading the config file
    with open('./conf.json', 'r') as f:
        conf = json.load(f)

    #setting the run folder
    run_folder = os.path.join(conf['base_folder'], 'run_' + run_id_start_time.strftime("%Y%m%d_%H%M"))

    #checking the folders where the output will be stored
    for i in ['clean', 'logs', 'prepared', 'models', 'predictions']:
        Path(run_folder, i).mkdir(parents=True, exist_ok=True)

    #checking if the raw folder exists and if not throws an error
    assert os.path.exists(os.path.join(conf['base_folder'], 'raw')), "Raw folder not found!"
    
    #save the config information
    with open(os.path.join(run_folder, 'logs', 'run_config.json'), 'w') as f:
        json.dump(conf, f)

    #load the data
    data_loader = DataLoader(conf['base_folder'] + 'raw')
    #houses = data_loader.load_data()
    #final 
    houses, codes, services, infrastructure, leisure = data_loader.load_data()

    #clean the data
    data_cleaner = DataCleaner()
    houses, codes, services, infrastructure, leisure = data_cleaner.clean_data(houses, codes, services, infrastructure, leisure)

    #storing the clean data on disk
    houses.reset_index(drop=True).to_feather(os.path.join(run_folder, 'clean', 'houses.feather'))
    codes.reset_index(drop=True).to_feather(os.path.join(run_folder, 'clean', 'codes.feather'))
    services.reset_index(drop=True).to_feather(os.path.join(run_folder, 'clean', 'services.feather'))
    infrastructure.reset_index(drop=True).to_feather(os.path.join(run_folder, 'clean', 'infrastructure.feather'))
    leisure.reset_index(drop=True).to_feather(os.path.join(run_folder, 'clean', 'leisure.feather'))
    print("Data loaded and cleaned")

    #create train and test
    validation_mapping = DataPartitioner().partition_data(houses)
    validation_mapping.to_feather(os.path.join(run_folder, "prepared", "validation_mapping.feather"))

    #featurize the data
    featurized_data = Featurizer(houses).transform(houses, codes, services, infrastructure, leisure)
    # featurized_data = transform(houses, codes, services, infrastructure, leisure)
    featurized_data.reset_index(drop=True).to_feather(os.path.join(run_folder, "prepared", "features.feather"))

    #hypertuning the model 
    train_set = featurized_data.merge(validation_mapping.query("test == False")[['globalId', 'cv_split']])
    
    hypertuner_rf = Hypertuner(estimator = RandomForestRegressor(random_state=1234),
    tuning_params = conf["training_params"]["hypertuning"]["RF_params"],
    validation_mapping = validation_mapping)

    hypertuner_NN = Hypertuner(estimator = MLPRegressor(random_state=1234),
    tuning_params = conf["training_params"]["hypertuning"]["NN_params"],
    validation_mapping = validation_mapping)

    #model RF
    #train_set = featurized_data.merge(validation_mapping.query("test == False")[['globalId', 'cv_split']])
    best_model_RF, best_model_params_RF = hypertuner_rf.tune_model(train_set)

    #model NN
    best_model_NN, best_model_params_NN = hypertuner_NN.tune_model(train_set)

    # log best parameters RF
    with open(os.path.join(run_folder, 'logs', 'best_model_params_RF_sellingprice.txt'), 'w') as f:
        f.write(json.dumps(best_model_params_RF))

    # log best parameters NN
    with open(os.path.join(run_folder, 'logs', 'best_model_params_NN_sellingprice.txt'), 'w') as f:
        f.write(json.dumps(best_model_params_NN))

    # predict on test set with RF
    test_set = featurized_data.merge(validation_mapping.query("test == True")[['globalId', 'cv_split']])
    truth = test_set.sellingPrice
    test_set.drop(columns=['sellingPrice', 'globalId', 'cv_split'], inplace = True) 
    predictions_RF = best_model_RF.predict(test_set)

    # predict on test set with NN
    test_set = featurized_data.merge(validation_mapping.query("test == True")[['globalId', 'cv_split']])
    truth = test_set.sellingPrice
    test_set.drop(columns=['sellingPrice', 'globalId', 'cv_split'], inplace = True) 
    predictions_NN = best_model_NN.predict(test_set)

    # save RF to disk
    pickle.dump(best_model_RF, open(os.path.join(run_folder, 'models', 'best_model_RF_sellingprice{}.sav'.format(str(best_model_params_RF))), 'wb'))

    # save NN to disk
    pickle.dump(best_model_NN, open(os.path.join(run_folder, 'models', 'best_model_NN_sellingprice{}.sav'.format(str(best_model_NN))), 'wb'))

    # accuracy metrics RF
    mean_dif_RF = predictions_RF / truth * 100 - 100
    np.mean(mean_dif_RF)

    # accuracy metrics NN
    mean_dif_NN = predictions_NN / truth * 100 - 100
    np.mean(mean_dif_NN)


    # # calculate accuracy and build plots
    # # calculate rmse on test set using predictions_test and actuals
    # rmse_test = None # filled
    # # other accuracy measures like MAE enz


    # # build plots actuals vs predicted

    # # save to disk best_model as pickle in model folder, accuracy metrics

    # # in log folder, plots in log folder

    print('Pipeline completed')

if __name__ == "__main__":
    main()