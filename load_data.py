import os
import shutil
import pandas as pd
import yaml
import subprocess

data_models_path = '/mnt/home/jean.alves/Sept_data_models'

if not os.path.isfile('./DeCCaF/data/BAF.parquet'):

    Input_Data = pd.read_csv(data_models_path + '/data/Base.csv')

    Input_Data.sort_values(by = 'month', inplace = True)
    Input_Data.reset_index(inplace=True)
    Input_Data.drop(columns = 'index', inplace = True)
    Input_Data.index.rename('case_id', inplace=True)

    data_cfg_path = './DeCCaF/data/dataset_cfg.yaml'
    with open(data_cfg_path, 'r') as infile:
        data_cfg = yaml.safe_load(infile)

    Input_Data.loc[:,data_cfg['data_cols']['categorical']] = Input_Data.loc[:,data_cfg['data_cols']['categorical']].astype('category')

    Input_Data.to_parquet('./DeCCaF/data/BAF.parquet')

if not os.path.isdir('./DeCCaF/ml_model/model'):  
    shutil.copytree(data_models_path + '/ml_model', './DeCCaF/ml_model/model')

if not os.path.isdir('./DeCCaF/experts/transformed_data'):  
    shutil.copytree(data_models_path + '/experts/transformed_data', './DeCCaF/experts/transformed_data')

if not os.path.isdir('./DeCCaF/expertise_models/models'):  
    shutil.copytree(data_models_path + '/expert_models/HEM', './DeCCaF/expertise_models/models')

if not os.path.isdir('./DeCCaF/ova_models/models'):  
    shutil.copytree(data_models_path + '/expert_models/OvA', './DeCCaF/ova_models/models')

if not os.path.isdir('./DeCCaF/deferral_results'):  
    shutil.copytree(data_models_path + '/deferral_results', './DeCCaF/deferral_results')

if not os.path.isfile('./DeCCaF/test_results_random.parquet'):  
    shutil.copy(data_models_path + '/test_results_random.parquet', './DeCCaF/test_results_random.parquet')

if not os.path.isfile('./DeCCaF/ova_models/ova_predictions.pkl'):  
    shutil.copy(data_models_path + '/expert_models/OvA/ova_predictions.pkl', './DeCCaF/ova_models/ova_predictions.pkl')

if not os.path.isfile('./DeCCaF/expertise_models/calibrated_predictions.pkl'):  
    shutil.copy(data_models_path + '/expert_models/HEM/calibrated_predictions.pkl', './DeCCaF/expertise_models/calibrated_predictions.pkl')

if not os.path.isfile('./DeCCaF/expertise_models/calibrators.pkl'):  
    shutil.copy(data_models_path + '/expert_models/HEM/calibrators.pkl', './DeCCaF/expertise_models/calibrators.pkl')

if not os.path.isfile('./DeCCaF/test_results.parquet'):  
    shutil.copy(data_models_path + '/test_results.parquet', './DeCCaF/test_results.parquet')

if not os.path.isdir('./DeCCaF/testbed/train_alert'):  
    shutil.copytree(data_models_path + '/training_datasets', './DeCCaF/testbed/train_alert')

subprocess.run(["python", "./DeCCaF/ml_model/training_and_predicting.py"])
subprocess.run(["python", "./DeCCaF/experts/expert_gen.py"])
subprocess.run(["python", "./DeCCaF/testbed/testbed_test_generation.py"])
