# %%
import pandas as pd
import yaml
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler
from sklearn.metrics import confusion_matrix
import pickle
import expert_src as experts
import numpy as np
import os
from pathlib import Path
import shutil

def sample(mu,sigma,n,prev_at_5, l, max_cost):
    slope = -(prev_at_5)/(l*(1-prev_at_5))
    costs = np.random.normal(loc = mu, scale = sigma, size = n)
    costs = np.clip(costs, 0, max_cost)
    experts = []
    for cost in costs:
        print(cost)
        line = pd.DataFrame()
        line['x'] = np.random.uniform(0.01,0.99,size = 10000)
        line['y'] = line['x']*slope + cost/(l*(1-prev_at_5))
        print(line)
        print(mu)
        print(sigma)
        line = line.loc[(line['y']>=0.01) & (line['y']<=0.99)]
        selec = np.random.choice(line.index)
        experts.append([line['x'].loc[selec],line['y'].loc[selec]])
    
    experts = np.array(experts)
    experts = np.clip(experts, 0,1)
    return experts


data_cfg_path = Path(__file__).parent/'../data/dataset_cfg.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

cat_dict = data_cfg['categorical_dict']

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            print(f'{feature} has been reencoded')
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data



LABEL_COL = data_cfg['data_cols']['label']
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
PROTECTED_COL = data_cfg['data_cols']['protected']
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

scens = os.listdir('../data/alerts/')
scens = sorted(scens, key=len)
seeds_set = 0
#Loading ML Model and its properties
with open(Path(__file__).parent/'../alert_model/model/best_model.pickle', 'rb') as infile:
    ml_model = pickle.load(infile)

with open(Path(__file__).parent/'../alert_model/model/model_properties.pickle', 'rb') as infile:
    ml_model_properties = pickle.load(infile)

cfg_path = Path(__file__).parent/'./cfg.yaml'
with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

costs = cfg['costs']
for scen in scens:
    
    scen = scen.split('.parquet')[0]
    if len(scen.split('-')) == 3:
        sub = True
    else:
        sub = False
    for l in costs:
        np.random.seed(42)
        if sub and (l not in cfg['run_sub']):
            continue
        data = pd.read_parquet(f'../data/alerts/{scen}.parquet')
        data = cat_checker(data, CATEGORICAL_COLS, cat_dict)

        if sub:
            os.makedirs(f'./teams/{scen}-l_{l}/expert_info', exist_ok=True)
            orig_scen = scen.split('-')[0] + '-' + scen.split('-')[1]
            for direc in os.listdir(f'./teams/{orig_scen}-l_{l}/expert_info/'):
                if direc in ['expert_ids.yaml','full_w_table.parquet','expert_properties.parquet']:
                    shutil.copy(f'./teams/{orig_scen}-l_{l}/expert_info/{direc}', f'./teams/{scen}-l_{l}/expert_info/{direc}')
                else:
                    a = pd.read_parquet(f'./teams/{orig_scen}-l_{l}/expert_info/{direc}')
                    a.loc[a.index.intersection(data.index)].to_parquet(f'./teams/{scen}-l_{l}/expert_info/{direc}')
                    if direc == 'train_predictions.parquet':
                        assert(len( a.loc[a.index.intersection(data.index)]) == len(data.loc[data['month'] == 6]))
                    if direc == 'deployment_predictions.parquet':
                        assert(len( a.loc[a.index.intersection(data.index)]) == len(data.loc[data['month'] != 6])) 
            
            continue

        train_test = data.loc[data["month"] != 6]
        val = data.loc[data["month"] == 6]
        y_val = val['fraud_bool']

        with open(cfg_path, 'r') as infile:
            cfg = yaml.safe_load(infile)

        tn, fp, fn, tp = confusion_matrix(y_val, np.ones(len(y_val))).ravel()
        cost_rejec_all_val = (fn+fp*l)/(len(y_val))

        # Creating ExpertTeam object. 
        expert_team = experts.ExpertTeam()
        EXPERT_IDS = dict(model_ids=list(), human_ids=list())
        THRESHOLDS = dict()


        ml_model_threshold = ml_model_properties['threshold']
        ml_model_recall = 1 - ml_model_properties['fnr']
        ml_model_fpr_diff = ml_model_properties['disparity']
        ml_model_fpr = ml_model_properties['fpr']

        with open(Path(__file__).parent/f'../classifier_h/selected_models/{scen}-l_{l}/best_model.pickle', 'rb') as infile:
            aux_model = pickle.load(infile)

        with open(Path(__file__).parent/f'../classifier_h/selected_models/{scen}-l_{l}/model_properties.yaml', 'r') as infile:
            aux_model_properties = yaml.safe_load(infile)

        #Inserting ML Model to the team.
        expert_team['model#0'] = experts.MLModelExpert(fitted_model=aux_model, threshold=None, init_score = aux_model_properties['init_score'])
        EXPERT_IDS['model_ids'].append('model#0')
        THRESHOLDS['model#0'] = ml_model_threshold

        #Loading or creating the transformed data for expert generation
        if( os.path.isfile(Path(__file__).parent/f'./teams/{scen}-l_{l}/transformed_data/X_deployment_experts.parquet') and os.path.isfile(Path(__file__).parent/f'./teams/{scen}-l_{l}/transformed_data/X_deployment_experts.parquet')):
            experts_deployment_X = pd.read_parquet(Path(__file__).parent/f'./teams/{scen}-l_{l}/transformed_data/X_deployment_experts.parquet')
            experts_train_X = pd.read_parquet(Path(__file__).parent/f'./teams/{scen}-l_{l}/transformed_data/X_train_experts.parquet')
            experts_train_X = experts_train_X.drop(columns = 'month')
            experts_deployment_X = experts_deployment_X.drop(columns = 'month')
        else:
            #We use the ML Model training split to fit our experts.
            #The expert fitting process involves determining the ideal Beta_0 and Beta_1 to obtain the user's desired target FPR and FNR
            experts_train_X = val.copy().drop(columns=LABEL_COL)
            #Change customer_age variable to a binary
            experts_train_X[PROTECTED_COL] = (experts_train_X[PROTECTED_COL] >= 50).astype(int)

            #Apply same process to the deployment split
            experts_deployment_X = train_test.copy().drop(columns=LABEL_COL)
            experts_deployment_X[PROTECTED_COL] = (experts_deployment_X[PROTECTED_COL] >= 50).astype(int)
        
            #Transform the numerical columns into quantiles and subtract 0.5 so they exist in the [-0.5, 0.5] interval
            cols_to_quantile = experts_train_X.drop(columns=CATEGORICAL_COLS).columns.tolist()
            qt = QuantileTransformer(random_state=42)
            experts_train_X[cols_to_quantile] = (
                qt.fit_transform(experts_train_X[cols_to_quantile])
                - 0.5  # centered on 0
            )

            #Target encode and transform the categorical columns
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            experts_train_X[CATEGORICAL_COLS] = oe.fit_transform(experts_train_X[CATEGORICAL_COLS])

            ss = StandardScaler(with_std=False)
            experts_train_X[:] = ss.fit_transform(experts_train_X)

            cols_to_scale = [c for c in experts_train_X.columns if c not in cols_to_quantile]
            desired_range = 1
            scaling_factors = (
                desired_range /
                (experts_train_X[cols_to_scale].max() - experts_train_X[cols_to_scale].min())
            )
            experts_train_X[cols_to_scale] *= scaling_factors

            # Preprocess the deployment splits and save the transformed data
            def preprocess(df):
                processed_X = df.copy()
                processed_X[cols_to_quantile] = qt.transform(processed_X[cols_to_quantile]) - 0.5  # centered on 0
                processed_X[CATEGORICAL_COLS] = oe.transform(processed_X[CATEGORICAL_COLS])
                processed_X[:] = ss.transform(processed_X)
                processed_X[cols_to_scale] *= scaling_factors

                return processed_X

            experts_train_X['month'] = val['month']
            
            experts_deployment_X = preprocess(experts_deployment_X)
            experts_deployment_X['month'] = train_test['month']
            os.makedirs(Path(__file__).parent/f'./teams/{scen}-l_{l}/transformed_data/', exist_ok= True)
            experts_deployment_X.to_parquet(Path(__file__).parent/f'./teams/{scen}-l_{l}/transformed_data/X_deployment_experts.parquet')
            experts_train_X.to_parquet(Path(__file__).parent/f'./teams/{scen}-l_{l}/transformed_data/X_train_experts.parquet')
            experts_train_X = experts_train_X.drop(columns = 'month')
            experts_deployment_X = experts_deployment_X.drop(columns = 'month')

        # Synthetic Expert Generation -----------------------------------------------------------------------------------
        if  os.path.isdir(Path(__file__).parent/f'./teams/{scen}-l_{l}/expert_info/'):
            print('Experts already Generated')
        else:
            #This function allows a user to create other groups by only defining the parameters that differ from the regular experts
            def process_groups_cfg(groups_cfg, baseline_name='standard'):
                full_groups_cfg = dict()
                for g_name in groups_cfg:
                    if g_name == baseline_name:
                        full_groups_cfg[g_name] = groups_cfg[g_name]
                    else:
                        full_groups_cfg[g_name] = dict()
                        for k in groups_cfg[baseline_name]:
                            if k not in list(groups_cfg[g_name].keys()):
                                full_groups_cfg[g_name][k] = full_groups_cfg[baseline_name][k]
                            elif isinstance(groups_cfg[g_name][k], dict):
                                full_groups_cfg[g_name][k] = {  # update baseline cfg
                                    **groups_cfg[baseline_name][k],
                                    **groups_cfg[g_name][k]
                                }
                            else:
                                full_groups_cfg[g_name][k] = groups_cfg[g_name][k]

                return full_groups_cfg

            
            ensemble_cfg = process_groups_cfg(cfg['experts']['groups'])
            expert_properties_list = list()

            #For each expert group generate the number of experts
            for group_name, group_cfg in ensemble_cfg.items():
                #Setting group random seed
                
                if group_cfg['cost']['setting'] == 'proportion':
                    group_cfg['cost']['target_mean'] = aux_model_properties['cost_val']*group_cfg['cost']['target_mean']
                    group_cfg['cost']['target_stdev'] = aux_model_properties['cost_val']*group_cfg['cost']['target_stdev']

                print(group_cfg['cost']['target_mean'])
                coefs_gen = dict()
                #Generate the set of w_M, w_p, and alpha for the group
                for coef in ['score', 'protected', 'alpha']:
                    coefs_gen[coef] = np.random.normal(
                            loc=group_cfg[f'{coef}_mean'],
                            scale=group_cfg[f'{coef}_stdev'],
                            size=group_cfg['n']
                    )
                
                coefs_spe = dict()
                coefs_spe['fnr'] = dict()
                coefs_spe['fpr'] = dict()
                #Generate the set of T_FPR, T_FNR for the group
                generated = sample(group_cfg['cost']['target_mean'],group_cfg['cost']['target_stdev'],group_cfg['n'], aux_model_properties['prev_val'], l, cost_rejec_all_val - group_cfg['cost']['rejec_all_margin']*cost_rejec_all_val)
                coefs_spe['fnr']['target'] = generated.T[0]
                coefs_spe['fpr']['target'] = generated.T[1]
                
                #Setting each expert's seed (for sampling of individual feature weights)
                if seeds_set == 0: 
                    expert_seeds = np.random.randint(low = 2**32-1, size = group_cfg['n'])
                    seeds_set = 1

                for i in range(group_cfg['n']):
                    expert_name = f'{group_name}#{i}'
                    expert_args = dict(
                        fnr_target=coefs_spe['fnr']['target'][i],
                        fpr_target=coefs_spe['fpr']['target'][i],
                        features_w_std = group_cfg['w_std'],
                        alpha = coefs_gen['alpha'][i],
                        fpr_noise = 0.0,
                        fnr_noise = 0.0,
                        protected_w = coefs_gen['protected'][i],
                        score_w = coefs_gen['score'][i],
                        seed = expert_seeds[i],
                        theta = group_cfg['theta']
                    )
                    #Creating the expert objects
                    expert_team[expert_name] = experts.SigmoidExpert(**expert_args)
                    expert_properties_list.append({**{'expert': expert_name}, **expert_args})
                    EXPERT_IDS['human_ids'].append(expert_name)

            #Fitting the experts
            expert_team.fit(
                X=experts_train_X,
                y=val[LABEL_COL],
                score_col='model_score',
                protected_col=PROTECTED_COL,
            )

            #Saving expert's properties and parameters
            full_w_table = pd.DataFrame(columns = experts_train_X.columns)
            for expert in expert_team:
                if(expert) != "model#0":
                    full_w_table.loc[expert] = expert_team[expert].w

            for expert in expert_team:
                if(expert) != "model#0":
                    full_w_table.loc[expert, 'fp_beta'] = expert_team[expert].fpr_beta
                    full_w_table.loc[expert, 'fn_beta'] = expert_team[expert].fnr_beta
                    full_w_table.loc[expert, 'alpha'] = expert_team[expert].alpha

            os.makedirs(Path(__file__).parent/f'./teams/{scen}-l_{l}/expert_info', exist_ok = True)
            full_w_table.to_parquet(Path(__file__).parent/f'./teams/{scen}-l_{l}/expert_info/full_w_table.parquet')

            expert_properties = pd.DataFrame(expert_properties_list)

            expert_properties.to_parquet(Path(__file__).parent/f'./teams/{scen}-l_{l}/expert_info/expert_properties.parquet')

            #Obtaining the predictions ----------------------------------------------------------------------------------

            ml_train = val.copy()
            ml_train[CATEGORICAL_COLS] = ml_train[CATEGORICAL_COLS].astype('category')

            train_expert_pred = expert_team.predict(
                index=val.index,
                predict_kwargs={
                    experts.SigmoidExpert: {
                        'X': experts_train_X,
                        'y': val[LABEL_COL]
                    },
                    experts.MLModelExpert: {
                        'X': ml_train.drop(columns=[LABEL_COL,'model_score','month'])
                    }}
            )

            train_expert_pred.to_parquet(Path(__file__).parent/f'./teams/{scen}-l_{l}/expert_info/train_predictions.parquet')

            deployment_expert_pred = expert_team.predict(
                index=train_test.index,
                predict_kwargs={
                    experts.SigmoidExpert: {
                        'X': experts_deployment_X,
                        'y': train_test[LABEL_COL]
                    },
                    experts.MLModelExpert: {
                        'X': train_test.drop(columns=[LABEL_COL,'model_score','month'])
                    }
                },
            )
            deployment_expert_pred.to_parquet(Path(__file__).parent/f'./teams/{scen}-l_{l}/expert_info/deployment_predictions.parquet')

            #saving the probability of error associated with each instance
            perror = pd.DataFrame()

            for expert in expert_team:
                if(expert) != "model#0":
                    column1 = f'p_fn_{expert}'
                    column2 = f'p_fp_{expert}'
                    perror[column1] = expert_team[expert].error_prob['p_of_fn']
                    perror[column2] = expert_team[expert].error_prob['p_of_fp']


            perror.to_parquet(Path(__file__).parent/f'./teams/{scen}-l_{l}/expert_info/p_of_error.parquet')


            #Saving the generated experts' ids
            # %%
            with open(Path(__file__).parent/f'./teams/{scen}-l_{l}/expert_info/expert_ids.yaml', 'w') as outfile:
                yaml.dump(EXPERT_IDS, outfile)


            print('Experts generated.')