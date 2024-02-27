import numpy as np
import pandas as pd
import pickle
import yaml
import os
from ortools.sat.python import cp_model
import random
from joblib import Parallel, delayed

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

def full_auto_func(capacities, batches, testset, env, model, scen, l):
    print(f'solving {env}/{scen}-l_{l}: fullreject')
    if os.path.isdir(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/'  + env):
        return

    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for i in np.arange(1,batches['batch'].max()+1):
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        for ix in cases.index:
            assignments.loc[ix] = 'auto-reject'
            results.loc[ix] = 1
        
            
    if not os.path.isdir(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/'  + env):
        os.makedirs(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/' + env +'/assignments.parquet')
    results.to_parquet(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/' + env +'/results.parquet')

    return assignments, results

def full_model_func(capacities, batches, testset, expert_preds, env, model, scen, l):
    print(f'solving {env}/{scen}-l_{l}: fullauto')
    if os.path.isdir(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/'  + env):
        return

    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for i in np.arange(1,batches['batch'].max()+1):
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        for ix in cases.index:
            assignments.loc[ix] = 'model#0'
            results.loc[ix] = expert_preds.loc[ix,'model#0']
        
            
    if not os.path.isdir(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/'  + env):
        os.makedirs(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/' + env +'/assignments.parquet')
    results.to_parquet(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/' + env +'/results.parquet')

    return assignments, results

def rand_deferral_func(capacities, batches, testset, expert_preds, env, model, scen, l):
    print(f'solving {env}/{scen}-l_{l}: rand')
    if os.path.isdir(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/{env}/'):
        return

    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for i in np.arange(1,batches['batch'].max()+1):
        c = capacities.loc[capacities['batch_id'] == i,:].iloc[0]
        c.loc['model#0'] = c['batch_size'] - c[2:].sum()
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        experts = expert_preds.columns.to_list()
        for ix in cases.index:
            done = 0
            while (done != 1):
                choice  = random.choice(experts)
                if c[choice]>0:
                    c[choice] -= 1
                    assignments.loc[ix] = choice
                    results.loc[ix] = expert_preds.loc[ix, choice]
                    done = 1
                else:
                    experts.remove(choice)
                if len(choice) == 0:
                    done = 1
            
    if not os.path.isdir(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/'  + env):
        os.makedirs(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/' + env +'/assignments.parquet')
    results.to_parquet(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{model}/' + env +'/results.parquet')

    return assignments, results

def ova_deferral_func(capacities, batches, testset, expert_preds, model_preds, env, model, seed, scen, l):
    print(f'solving {env}/{scen}-l_{l}: ova')
    if os.path.isdir(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{seed}/{model}/'  + env):
        return

    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for i in np.arange(1,batches['batch'].max()+1):
        c = capacities.loc[capacities['batch_id'] == i,:].iloc[0]
        c.loc['model#0'] = c['batch_size'] - c[2:].sum()
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        preds = model_preds.loc[cases.index]
        for ix, row in preds.iterrows():
            sorted = row.sort_values(ascending = False)
            for choice in sorted.index:
                if c[choice]>0:
                    c[choice] -= 1
                    assignments.loc[ix] = choice
                    results.loc[ix] = expert_preds.loc[ix, choice]
                    break

    if not os.path.isdir(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{seed}/{model}/' + env):
        os.makedirs(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{seed}/{model}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{seed}/{model}/' + env +'/assignments.parquet')
    results.to_parquet(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{seed}/{model}/' + env +'/results.parquet')

    return assignments, results


def deccaf_cp_deferral_func(capacities, batches, testset, expert_preds, model_preds, env, model, seed, scen, l):
    print(f'solving {env}/{scen}-l_{l}: deccaf')
    if os.path.isdir(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{seed}/{model}/{env}/'):
        return

    model_name = model
    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for b in np.arange(1,batches['batch'].max()+1):
        c = capacities.loc[capacities['batch_id'] == b,:].iloc[0]
        c.loc['model#0'] = c['batch_size'] - c[2:].sum()
        cases = testset.loc[batches.loc[batches['batch'] == b]['case_id'],:] 
        preds = model_preds.loc[cases.index]
        cost_matrix_df = preds.T
        for d in c.index:
            if c.loc[d] == 0:
                cost_matrix_df = cost_matrix_df.drop(index=d)
                

        cost_matrix = cost_matrix_df.values
        num_workers, num_tasks = cost_matrix.shape
        workers = list(cost_matrix_df.index)

        model = cp_model.CpModel()
        x = []
        for i in range(num_workers):
            t = []
            for j in range(num_tasks):
                t.append(model.NewBoolVar(f'x[{i},{j}]'))
            x.append(t)

        # capacity constraints
        for i in range(num_workers):
            model.Add(sum([x[i][j] for j in range(num_tasks)]) == c[workers[i]])

        # Each task is assigned to exactly one worker.
        for j in range(num_tasks):
            model.AddExactlyOne(x[i][j] for i in range(num_workers))

        objective_terms = []
        for i in range(num_workers):
            for j in range(num_tasks):
                objective_terms.append(cost_matrix[i, j] * x[i][j])
        model.Minimize(sum(objective_terms))
        #This sum(objective_terms) is the loss of the batch.
        solver = cp_model.CpSolver()
        # solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = 60.0
        status = solver.Solve(model)

        if not status == cp_model.OPTIMAL and not status == cp_model.FEASIBLE:
            print('Solution not found!')
            stop
            return None

        print('Batch solved')
        
        for j in range(num_tasks):
            ix = cost_matrix_df.columns.to_list()[j]
            for i in range(num_workers):
                if solver.BooleanValue(x[i][j]):
                    assignments.loc[ix] = workers[i]
                    results.loc[ix] = expert_preds.loc[ix, workers[i]]

    
    
    if not os.path.isdir(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{seed}/{model_name}/'  + env):
        os.makedirs(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{seed}/{model_name}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{seed}/{model_name}/' + env +'/assignments.parquet')
    results.to_parquet(f'../../Data_and_models/deferral/def_results_alert/{scen}-l_{l}/{seed}/{model_name}/' + env +'/results.parquet')

    return assignments, results


with open('../data/dataset_cfg.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

cat_dict = data_cfg['categorical_dict']

with open('cfg.yaml', 'r') as infile:
    cfg = yaml.safe_load(infile)


costs_l = cfg['costs']

for scen in os.listdir('../../Data_and_models/data/alerts'):
    if len(scen.split('-')) == 3:
        sub = True
    else:
        sub = False
    for l in costs_l:
        if sub and (l not in cfg['run_sub']):
            continue
        scen = scen.split('.parquet')[0]
        alerts = pd.read_parquet(f'../../Data_and_models/data/alerts/{scen}.parquet')
        exp_pred = (pd.read_parquet(f'../../Data_and_models/experts/{scen}-l_{l}/deployment_predictions.parquet')>=0.5).astype(int)
        test = alerts.loc[alerts['month'] == 7]
        Classes = np.array(['fn', 'fp', 'tn', 'tp'])
        CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

        with open(f'../../Data_and_models/deferral/test_preds/{scen}-l_{l}/deccaf.pkl', 'rb') as fp:
                deccaf_model_preds = pickle.load(fp)

        with open(f'../../Data_and_models/deferral/test_preds/{scen}-l_{l}/ova.pkl', 'rb') as fp:
                ova_model_preds = pickle.load(fp)

        a = dict()
        for direc in os.listdir(f'../../Data_and_models/testbed/{scen}-l_{l}/test'):
            if os.path.isfile(f'../../Data_and_models/testbed/{scen}-l_{l}/test/' + direc):
                continue
            a[direc] = dict()
            a[direc]['bat'] = pd.read_csv(f'../../Data_and_models/testbed/{scen}-l_{l}/test/' + direc + '/batches.csv')
            a[direc]['cap'] = pd.read_csv(f'../../Data_and_models/testbed/{scen}-l_{l}/test/' + direc + '/capacity.csv')

        for seed in deccaf_model_preds:
            Parallel(n_jobs=5)(
                    delayed(deccaf_cp_deferral_func)(
                        a[env]['cap'],
                        a[env]['bat'],
                        test,
                        exp_pred,
                        1-deccaf_model_preds[seed],
                        env,
                        f'DeCCaF',
                        seed,
                        scen,
                        l
                    )
                    for env in a 
                )
        for seed in ova_model_preds:
            Parallel(n_jobs=5)(
                    delayed(ova_deferral_func)(
                        a[env]['cap'],
                        a[env]['bat'],
                        test,
                        exp_pred,
                        ova_model_preds[seed],
                        env,
                        f'OvA',
                        seed,
                        scen,
                        l
                    )
                    for env in a 
                )

        Parallel(n_jobs=5)(
                    delayed(full_auto_func)(
                        a[env]['cap'],
                        a[env]['bat'],
                        test,
                        env,
                        f'Full_Rej',
                        scen,
                        l
                    )
                    for env in a 
                )

        Parallel(n_jobs=5)(
                    delayed(full_model_func)(
                        a[env]['cap'],
                        a[env]['bat'],
                        test,
                        exp_pred,
                        env,
                        f'Only_Classifier',
                        scen,
                        l
                    )
                    for env in a 
                )

        Parallel(n_jobs=5)(
                    delayed(rand_deferral_func)(
                        a[env]['cap'],
                        a[env]['bat'],
                        test,
                        exp_pred,
                        env,
                        f'Random',
                        scen,
                        l
                    )
                    for env in a 
                )

