# %%
import pandas as pd
import numpy as np
import os
import yaml
import hpo
import pickle


with open('../data/dataset_cfg.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

with open('cfg.yaml', 'r') as infile:
    cfg = yaml.safe_load(infile)

CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

cat_dict = data_cfg['categorical_dict']

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype != 'category':
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
        elif new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data


scens = os.listdir('../../Data_and_models/data/alerts/')
costs = cfg['costs']

for scen in scens:
    if len(scen.split('-')) == 3:
        sub = True
    else:
        sub = False
    for l in costs:
        if sub and (l not in cfg['run_sub']):
            continue
        scen = scen.split('.parquet')[0]
        expert_ids_path = f'../../Data_and_models/experts/{scen}-l_{l}/expert_ids.yaml'

        with open(expert_ids_path, 'r') as infile:
            EXPERT_IDS = yaml.safe_load(infile)

        cat_dict['assignment'] = EXPERT_IDS['human_ids'] + EXPERT_IDS['model_ids']

        for train in os.listdir(f'../../Data_and_models/testbed/{scen}-l_{l}/train_alert'):
            train_set = pd.read_parquet(f'../../Data_and_models/testbed/{scen}-l_{l}/train_alert/{train}/train.parquet')
            train_set = train_set.loc[train_set["assignment"] != 'model#0']
            train_set = cat_checker(train_set, data_cfg['data_cols']['categorical'] + ['assignment'], cat_dict)
            print(f'Fitting start for {train}, conjoined')

            val_set = train_set.loc[train_set['month'] == 6]
            train_set = train_set.loc[train_set['month'] != 6]
            
            train_w = train_set['fraud_bool'].replace([0,1],[l,1])
            val_w = val_set['fraud_bool'].replace([0,1],[l,1])

            train_x = train_set.drop(columns = ['fraud_bool', 'batch','month', 'decision'])
            val_x = val_set.drop(columns = ['fraud_bool', 'batch','month', 'decision'])

            train_y = (train_set['decision'] == train_set['fraud_bool']).astype(int)
            val_y = (val_set['decision'] == val_set['fraud_bool']).astype(int)

            if not (os.path.exists(f'../../Data_and_models/expert_models/deccaf/{scen}-l_{l}/{train}/')):
                os.makedirs(f'../../Data_and_models/expert_models/deccaf/{scen}-l_{l}/{train}/')
            
            if not (os.path.exists(f'../../Data_and_models/expert_models/deccaf/{scen}-l_{l}/{train}/best_model.pickle')):
                opt = hpo.HPO(train_x,val_x,train_y,val_y,train_w, val_w, method = 'TPE', path = f'../../Data_and_models/expert_models/deccaf/{scen}-l_{l}/{train}/')
                opt.initialize_optimizer(CATEGORICAL_COLS, 10)

            
for scen in os.listdir('../../Data_and_models/data/alerts'):
    if len(scen.split('-')) == 3:
        sub = True
    else:
        sub = False
    for l in costs:
        if sub and (l not in cfg['run_sub']):
            continue
        scen = scen.split('.parquet')[0]
        test = pd.read_parquet(f'../../Data_and_models/testbed/{scen}-l_{l}/test/test.parquet')
        test_expert_pred = pd.read_parquet(f'../../Data_and_models/testbed/{scen}-l_{l}/test/test_expert_pred.parquet')

        expert_ids_path = f'../../Data_and_models/experts/{scen}-l_{l}/expert_ids.yaml'
        
        with open(expert_ids_path, 'r') as infile:
            EXPERT_IDS = yaml.safe_load(infile)

        cat_dict['assignment'] = EXPERT_IDS['human_ids'] + EXPERT_IDS['model_ids']

        test = cat_checker(test, data_cfg['data_cols']['categorical'], cat_dict)

        test = test.drop(columns = ['month','fraud_label'])

        preds_conj = dict()

        for env_id in os.listdir(f'../../Data_and_models/expert_models/deccaf/{scen}-l_{l}'):
            table = pd.DataFrame(index = test.index, columns = EXPERT_IDS['human_ids'])
            with open(f"../../Data_and_models/expert_models/deccaf/{scen}-l_{l}/{env_id}/best_model.pickle", 'rb') as fp:
                model = pickle.load(fp)
            
            for expert in EXPERT_IDS['human_ids']:
                test['assignment'] = expert

                test = cat_checker(test, data_cfg['data_cols']['categorical'] + ['assignment'], cat_dict)

                table.loc[:,expert] = model.predict_proba(test)[:,1]
            
            for expert in EXPERT_IDS['model_ids']:
                table.loc[:,expert] = np.maximum(test_expert_pred[expert],  1-test_expert_pred[expert])

            preds_conj[env_id.split('#')[0] + '#' + env_id.split('#')[1]] = table
        
        os.makedirs(f"../../Data_and_models/deferral/test_preds/{scen}-l_{l}/", exist_ok = True)
        with open(f"../../Data_and_models/deferral/test_preds/{scen}-l_{l}/deccaf.pkl", "wb") as out_file:
            pickle.dump(preds_conj, out_file)

