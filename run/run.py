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
    print('Loading data')
    data_loader = DataLoader(conf['base_folder'] + 'raw')
    houses, codes, services, infrastructure, leisure = data_loader.load_data()

    #clean the data
    print('Cleaning data')
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

    ########## RF sellingprice ######################################################

    #hypertuning RF sellingprice 
    print('Training models')

    print('Training RF model for sellingprice')
    
    train_set = featurized_data.merge(validation_mapping.query("test == False")[['globalId', 'cv_split']])
    
    hypertuner_rf = Hypertuner(estimator = RandomForestRegressor(random_state=1234),
    tuning_params = conf["training_params"]["hypertuning"]["RF_params"]["RF_sellingprice"],
    validation_mapping = validation_mapping)

    #model RF sellingprice
    best_model_RF_sellingprice, best_model_params_RF_sellingprice = hypertuner_rf.tune_model(train_set)
    print(f'Best model parameters: {best_model_params_RF_sellingprice}')

    # log best parameters RF sellingprice
    with open(os.path.join(run_folder, 'logs', 'best_model_params_RF_sellingprice.txt'), 'w') as f:
        f.write(json.dumps(best_model_params_RF_sellingprice))

    # predict sellingprice on test set with RF
    test_set = featurized_data.merge(validation_mapping.query("test == True")[['globalId', 'cv_split']])
    truth = test_set.sellingPrice
    test_set.drop(columns=['sellingPrice', 'globalId', 'cv_split'], inplace = True) 
    predictions_RF_sellingprice = best_model_RF_sellingprice.predict(test_set)

    # save RF sellingprice to disk
    pickle.dump(best_model_RF_sellingprice, open(os.path.join(run_folder, 'models', 'best_model_RF_sellingprice{}.sav'.format(str(best_model_params_RF_sellingprice))), 'wb'))
    
    # accuracy metrics RF sellingprice

    #mean percentage difference between truth and prediction RF sellingprice
    dif_RF_sellingprice = predictions_RF_sellingprice / truth * 100 - 100
    mean_dif_RF_sellingprice = np.mean(dif_RF_sellingprice)
    mean_dif_RF_sellingprice_result = str(mean_dif_RF_sellingprice)
    with open(os.path.join(run_folder, 'predictions', 'mean_%dif_RF_sellingprice.txt'), 'w') as f:
        f.write(mean_dif_RF_sellingprice_result)

    #rmse for RF sellingprice
    rmse_RF_sellingprice = np.sqrt(mean_squared_error(truth, predictions_RF_sellingprice))
    rmse_RF_sellingprice_result = str(rmse_RF_sellingprice)
    with open(os.path.join(run_folder, 'predictions', 'rmse_RF_sellingprice.txt'), 'w') as f:
        f.write(rmse_RF_sellingprice_result)

    #mae for RF sellingprice
    mae_RF_sellingprice = mean_absolute_error(truth, predictions_RF_sellingprice)
    mae_RF_sellingprice_result = str(mae_RF_sellingprice)
    with open(os.path.join(run_folder, 'predictions', 'mae_RF_sellingprice.txt'), 'w') as f:
        f.write(mae_RF_sellingprice_result)   

    #plot RF sellingprice scatter
    fig = plt.figure()
    plt.scatter(predictions_RF_sellingprice, truth, alpha=0.2)
    plt.xlabel('Random Forest Predictions Sellingprice')
    plt.ylabel('True Values')
    lims = [0, 800000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'RF sellingprice scatter.png'
    fig.savefig(os.path.join(graph_path, graph_file))

    #plots error RF sellingprice
    error = predictions_RF_sellingprice - truth
    fig = plt.figure()
    plt.hist(error, bins = 30)
    plt.xlabel("Absolute Prediction Error for Random Forest Sellingprice")
    plt.ylabel("Count")
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'RF sellingprice error.png'
    fig.savefig(os.path.join(graph_path, graph_file))
    
    print('Finished RF model for sellingprice')
    
    ########## RF sellingtime ####################################################

    print('Training RF model for sellingtime')

    hypertuner_RF_tts = HypertunerTTS(estimator = RandomForestRegressor(random_state=1234),
    tuning_params = conf["training_params"]["hypertuning"]["RF_params"]["RF_sellingtime"],
    validation_mapping = validation_mapping)

    #model RF sellingtime
    best_model_RF_sellingtime, best_model_params_RF_sellingtime = hypertuner_RF_tts.tune_model(train_set)
    print(f'Best model parameters: {best_model_params_RF_sellingtime}')

    # log best parameters RF sellingtime
    with open(os.path.join(run_folder, 'logs', 'best_model_params_RF_sellingtime.txt'), 'w') as f:
        f.write(json.dumps(best_model_params_RF_sellingtime))

    # predict on test set with RF sellingtime
    test_set = featurized_data.merge(validation_mapping.query("test == True")[['globalId', 'cv_split']])
    truth = test_set.sellingtime
    test_set.drop(columns=['sellingtime', 'globalId', 'cv_split'], inplace = True) 
    predictions_RF_sellingtime = best_model_RF_sellingtime.predict(test_set)

    # save RF sellingtime to disk
    pickle.dump(best_model_RF_sellingtime, open(os.path.join(run_folder, 'models', 'best_model_RF_sellingtime{}.sav'.format(str(best_model_params_RF_sellingtime))), 'wb'))

    #rmse for RF tts
    rmse_RF_sellingtime = np.sqrt(mean_squared_error(truth, predictions_RF_sellingtime))
    rmse_RF_sellingtime_result = str(rmse_RF_sellingtime)
    with open(os.path.join(run_folder, 'predictions', 'rmse_RF_sellingtime.txt'), 'w') as f:
        f.write(rmse_RF_sellingtime_result)

    #mae for RF tts
    mae_RF_sellingtime = mean_absolute_error(truth, predictions_RF_sellingtime)
    mae_RF_sellingtime_result = str(mae_RF_sellingtime)
    with open(os.path.join(run_folder, 'predictions', 'mae_RF_sellingtime.txt'), 'w') as f:
        f.write(mae_RF_sellingtime_result)

    #plot RF sellingtime scatter
    fig = plt.figure()
    plt.scatter(predictions_RF_sellingtime, truth, alpha=0.2)
    plt.xlabel('Random Forest Predictions sellingtime')
    plt.ylabel('True Values')
    lims_x = [0, 200]
    lims_y = [0, 500]
    plt.xlim(lims_x)
    plt.ylim(lims_y)
    _ = plt.plot(lims, lims)
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'RF sellingtime scatter.png'
    fig.savefig(os.path.join(graph_path, graph_file))

    #plots error RF sellingtime
    error = predictions_RF_sellingtime - truth
    fig = plt.figure()
    plt.hist(error, bins = 30)
    plt.xlabel("Absolute Prediction Error for Random Forest sellingtime")
    plt.ylabel("Count")
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'RF sellingtime error.png'
    fig.savefig(os.path.join(graph_path, graph_file))

    print('Finished RF model for sellingtime')

    ########### NN sellingprice ##############################################################

    print('Training NN model for sellingprice')

    hypertuner_NN = Hypertuner(estimator = MLPRegressor(random_state=1234),
    tuning_params = conf["training_params"]["hypertuning"]["NN_params"]["NN_sellingprice"],
    validation_mapping = validation_mapping)

    #model NN
    best_model_NN_sellingprice, best_model_params_NN_sellingprice = hypertuner_NN.tune_model(train_set)
    print(f'Best model parameters: {best_model_params_NN_sellingprice}')

    # log best parameters NN
    with open(os.path.join(run_folder, 'logs', 'best_model_params_NN_sellingprice.txt'), 'w') as f:
        f.write(json.dumps(best_model_params_NN_sellingprice))

    # predict on test set with NN sellingprice
    test_set = featurized_data.merge(validation_mapping.query("test == True")[['globalId', 'cv_split']])
    truth = test_set.sellingPrice
    test_set.drop(columns=['sellingPrice', 'globalId', 'cv_split'], inplace = True) 
    predictions_NN_sellingprice = best_model_NN_sellingprice.predict(test_set)

    # save NN sellingpriceto disk
    pickle.dump(best_model_NN_sellingprice, open(os.path.join(run_folder, 'models', 'best_model_NN_sellingtime{}.sav'.format(str(best_model_NN_sellingprice))), 'wb'))

    # accuracy metrics NN
    #mean percentage difference between truth and prediction RF sellingprice
    dif_NN_sellingprice = predictions_NN_sellingprice / truth * 100 - 100
    mean_dif_NN_sellingprice = np.mean(dif_NN_sellingprice)
    mean_dif_NN_sellingprice_result = str(mean_dif_NN_sellingprice)
    with open(os.path.join(run_folder, 'predictions', 'mean_%dif_NN_sellingprice.txt'), 'w') as f:
        f.write(mean_dif_NN_sellingprice_result)

    #rmse for RF sellingprice
    rmse_NN_sellingprice = np.sqrt(mean_squared_error(truth, predictions_NN_sellingprice))
    rmse_NN_sellingprice_result = str(rmse_NN_sellingprice)
    with open(os.path.join(run_folder, 'predictions', 'rmse_NN_sellingprice.txt'), 'w') as f:
        f.write(rmse_NN_sellingprice_result)

    #mae for RF sellingprice
    mae_NN_sellingprice = mean_absolute_error(truth, predictions_NN_sellingprice)
    mae_NN_sellingprice_result = str(mae_NN_sellingprice)
    with open(os.path.join(run_folder, 'predictions', 'mae_NN_sellingprice.txt'), 'w') as f:
        f.write(mae_NN_sellingprice_result)

    #plot NN scatter
    fig = plt.figure()
    plt.scatter(predictions_NN_sellingprice, truth, alpha=0.2)
    plt.xlabel('NN sellingprice predictions')
    plt.ylabel('True Values')
    lims = [0, 800000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'NN sellingprice scatter.png'
    fig.savefig(os.path.join(graph_path, graph_file))

    #plot NN error
    error = predictions_NN_sellingprice - truth
    fig = plt.figure()
    plt.hist(error, bins = 30)
    plt.xlabel("Absolute Prediction Error for NN sellingprice")
    plt.ylabel("Count")
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'NN sellingprice error.png'
    fig.savefig(os.path.join(graph_path, graph_file))

    print('Finished NN model for sellingtime')

    ############# NN sellingtime ##########################################################

    print('Training NN model for sellingtime')

    hypertuner_NN_tts = HypertunerTTS(estimator = MLPRegressor(random_state=1234),
    tuning_params = conf["training_params"]["hypertuning"]["NN_params"]["NN_sellingtime"],
    validation_mapping = validation_mapping)

    #model NN sellingtime
    best_model_NN_sellingtime, best_model_params_NN_sellingtime = hypertuner_NN_tts.tune_model(train_set)
    print(f'Best model parameters: {best_model_params_NN_sellingtime}')

    # log best parameters NN sellingtime
    with open(os.path.join(run_folder, 'logs', 'best_model_params_NN_sellingtime.txt'), 'w') as f:
        f.write(json.dumps(best_model_params_NN_sellingtime))

    # predict on test set with NN sellingtime
    test_set = featurized_data.merge(validation_mapping.query("test == True")[['globalId', 'cv_split']])
    truth = test_set.sellingtime
    test_set.drop(columns=['sellingtime', 'globalId', 'cv_split'], inplace = True) 
    predictions_NN_sellingtime = best_model_NN_sellingtime.predict(test_set)

    # save NN sellingtime to disk
    pickle.dump(best_model_NN_sellingtime, open(os.path.join(run_folder, 'models', 'best_model_NN_sellingtime{}.sav'.format(str(best_model_params_NN_sellingtime))), 'wb'))

    #rmse for NN sellingtime
    rmse_NN_sellingtime = np.sqrt(mean_squared_error(truth, predictions_NN_sellingtime))
    rmse_NN_sellingtime_result = str(rmse_NN_sellingtime)
    with open(os.path.join(run_folder, 'predictions', 'rmse_NN_sellingtime.txt'), 'w') as f:
        f.write(rmse_NN_sellingtime_result)

    #mae for NN sellingtime
    mae_NN_sellingtime = mean_absolute_error(truth, predictions_NN_sellingtime)
    mae_NN_sellingtime_result = str(mae_NN_sellingtime)
    with open(os.path.join(run_folder, 'predictions', 'mae_NN_sellingtime.txt'), 'w') as f:
        f.write(mae_NN_sellingtime_result)

    #plot NN sellingtime scatter
    fig = plt.figure()
    plt.scatter(predictions_NN_sellingtime, truth, alpha=0.2)
    plt.xlabel('NN Predictions sellingtime')
    plt.ylabel('True Values')
    lims_x = [0, 200]
    lims_y = [0, 500]
    plt.xlim(lims_x)
    plt.ylim(lims_y)
    _ = plt.plot(lims, lims)
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'NN sellingtime scatter.png'
    fig.savefig(os.path.join(graph_path, graph_file))

    #plots error NN sellingtime
    error = predictions_RF_sellingtime - truth
    fig = plt.figure()
    plt.hist(error, bins = 30)
    plt.xlabel("Absolute Prediction Error for NN sellingtime")
    plt.ylabel("Count")
    plt.show()
    graph_path = (os.path.join(run_folder, 'logs'))
    graph_file = 'NN sellingtime error.png'
    fig.savefig(os.path.join(graph_path, graph_file))

    print('Finished NN model for sellingtime')

    ###### all models finished

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