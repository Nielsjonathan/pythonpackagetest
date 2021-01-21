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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from matplotlib import pyplot as plt
import warnings 
warnings.filterwarnings('ignore')


#importing our modules
from modules.load import DataLoader
from modules.clean import DataCleaner
from modules.validation import DataPartitioner
from modules.featurizing import Featurizer
from modules.hypertuning import Hypertuner
from modules.hypertuningTTS import HypertunerTTS

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
    validation_mapping = DataPartitioner(test_perc= conf['training_params']['test_perc']).partition_data(houses)
    validation_mapping.to_feather(os.path.join(run_folder, "prepared", "validation_mapping.feather"))

    #featurize the data
    featurized_data = Featurizer(houses).transform(houses, codes, services, infrastructure, leisure)
    featurized_data.reset_index(drop=True).to_feather(os.path.join(run_folder, "prepared", "features.feather"))

    #hypertuning the model 
    train_set = featurized_data.merge(validation_mapping.query("test == False")[['globalId', 'cv_split']])
    
    hypertuner_rf = Hypertuner(estimator = RandomForestRegressor(random_state=1234),
    tuning_params = conf["training_params"]["hypertuning"]["RF_params"],
    validation_mapping = validation_mapping)

    hypertuner_NN = Hypertuner(estimator = MLPRegressor(random_state=1234),
    tuning_params = conf["training_params"]["hypertuning"]["NN_params"],
    validation_mapping = validation_mapping)


    ########## test tts #########################################################################
    hypertuner_rf_tts = HypertunerTTS(estimator = RandomForestRegressor(random_state=1234),
    tuning_params = conf["training_params"]["hypertuning"]["RF_params"],
    validation_mapping = validation_mapping)

    ### model rf sellingtime ###
    best_model_rf_tts, best_model_params_tts = hypertuner_rf_tts.tune_model(train_set)

    # predict on test set with RF
    test_set = featurized_data.merge(validation_mapping.query("test == True")[['globalId', 'cv_split']])
    truth = test_set.sellingtime
    test_set.drop(columns=['sellingtime', 'globalId', 'cv_split'], inplace = True) 
    predictions_RF_tts = best_model_rf_tts.predict(test_set)

    #rmse for RF tts
    rmse_RF_tts = np.sqrt(mean_squared_error(truth, predictions_RF_tts))
    rmse_RF_tts_result = str(rmse_RF_tts)
    with open(os.path.join(run_folder, 'predictions', 'rmse_RF.txt'), 'w') as f:
        f.write(rmse_RF_tts_result)

    #mae for RF tts
    mae_RF_tts = mean_absolute_error(truth, predictions_RF_tts)
    mae_RF_tts_result = str(mae_RF_tts)
    with open(os.path.join(run_folder, 'predictions', 'mae_RF.txt'), 'w') as f:
        f.write(mae_RF_tts_result)
    ########## end test tts #########################################################################

    print('Training models')

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
    #mean percentage difference between truth and prediction
    dif_RF = predictions_RF / truth * 100 - 100
    mean_dif_RF = np.mean(dif_RF)
    mean_dif_RF_result = str(mean_dif_RF)
    with open(os.path.join(run_folder, 'predictions', 'mean_%dif_RF.txt'), 'w') as f:
        f.write(mean_dif_RF_result)

    #rmse for RF
    rmse_RF = np.sqrt(mean_squared_error(truth, predictions_RF))
    rmse_RF_result = str(rmse_RF)
    with open(os.path.join(run_folder, 'predictions', 'rmse_RF.txt'), 'w') as f:
        f.write(rmse_RF_result)

    #mae for RF
    mae_RF = mean_absolute_error(truth, predictions_RF)
    mae_RF_result = str(mae_RF)
    with open(os.path.join(run_folder, 'predictions', 'mae_RF.txt'), 'w') as f:
        f.write(mae_RF_result)

    # accuracy metrics NN
    mean_dif_NN = predictions_NN / truth * 100 - 100
    np.mean(mean_dif_NN)


    # # calculate accuracy and build plots
    # # calculate rmse on test set using predictions_test and actuals
    # rmse_test = None # filled
    # # other accuracy measures like MAE enz


    # # build plots actuals vs predicted 
    #plot RF scatter
    fig = plt.figure()
    plt.scatter(predictions_RF, truth, alpha=0.2)
    plt.xlabel('Random Forest Predictions')
    plt.ylabel('True Values')
    lims = [0, 800000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'RF scatter.png'
    fig.savefig(os.path.join(graph_path, graph_file))

    #plots error RF
    error = predictions_RF - truth
    fig = plt.figure()
    plt.hist(error, bins = 30)
    plt.xlabel("Absolute Prediction Error for Random Forest")
    plt.ylabel("Count")
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'RF error.png'
    fig.savefig(os.path.join(graph_path, graph_file))


    #plot NN scatter
    fig = plt.figure()
    plt.scatter(predictions_NN, truth, alpha=0.2)
    plt.xlabel('NN predictions')
    plt.ylabel('True Values')
    lims = [0, 800000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'NN scatter.png'
    fig.savefig(os.path.join(graph_path, graph_file))


    #plot NN error
    error = predictions_NN - truth
    fig = plt.figure()
    plt.hist(error, bins = 30)
    plt.xlabel("Absolute Prediction Error for Random Forest")
    plt.ylabel("Count")
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'NN error.png'
    fig.savefig(os.path.join(graph_path, graph_file))


    run_id_end_time = datetime.datetime.now()
    print(f'Ended run at time: {run_id_end_time}')
    total_runtime = run_id_end_time - run_id_start_time
    total_runtime = str(total_runtime)
    with open(os.path.join(run_folder, 'logs', 'total runtime.txt'), 'w') as f:
        f.write(total_runtime)
    print(f'Total run time: {total_runtime}')
    print('Pipeline completed')


if __name__ == "__main__":
    main()