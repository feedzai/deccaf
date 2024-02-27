import yaml
import pandas as pd
import pickle
from sklearn.metrics import recall_score
import numpy as np
import os

def tpr_at_fpr(labels, preds,fpr):
    is_higher_better = True
    results = pd.DataFrame()
    results["true"] = labels
    results["score"] = preds

    temp = results.sort_values(by="score", ascending=False)

    FPR = fpr
    N = (temp["true"] == 0).sum()
    FP = round(FPR * N)
    aux = temp[temp["true"] == 0]

    threshold = aux.iloc[FP - 1, 1]

    y_pred = np.where(results["score"] >= threshold, 1, 0)
    tpr = recall_score(labels, y_pred)
    return tpr, threshold

BAF = pd.read_csv('../../Data_and_models/data/Base.csv')
BAF.sort_values(by = 'month', inplace = True)
BAF.reset_index(inplace=True)
BAF.drop(columns = 'index', inplace = True)
BAF.index.rename('case_id', inplace=True)

data_cfg_path = 'dataset_cfg.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

BAF.loc[:,data_cfg['data_cols']['categorical']] = BAF.loc[:,data_cfg['data_cols']['categorical']].astype('category')

if not os.path.isfile('../../Data_and_models/alert_model/best_model.pickle'):
    print('The Alert Model is not Trained! - Please run ./alert_model/training_and_predicting.py')
else:
    BAF_dep = pd.read_parquet('../../Data_and_models/data/BAF_deployment_score.parquet')
    BAF_dep["month"] = BAF.loc[BAF_dep.index,"month"]

    BAF_val = BAF_dep.loc[BAF_dep['month'] == 3]
    tpr, t = tpr_at_fpr(BAF_val['fraud_bool'], BAF_val['model_score'], 0.05)
    alerts_5 = BAF_dep.loc[BAF_dep['model_score'] > t]

    os.makedirs('../../Data_and_models/data/alerts/', exist_ok=True)
    if not os.path.isfile('../../Data_and_models/data/alerts/alert_0.05-data_0.05.parquet'):
        alerts_5.to_parquet('../../Data_and_models/data/alerts/alert_0.05-data_0.05.parquet')
    else:
        alerts_5 = pd.read_parquet('../../Data_and_models/data/alerts/alert_0.05-data_0.05.parquet')
    alerts = dict()
    alerts[0.05] = alerts_5

    tpr, t = tpr_at_fpr(BAF_val['fraud_bool'], BAF_val['model_score'], 0.15)
    alerts_temp = BAF_dep.loc[BAF_dep['model_score'] > t]
    temp = []
    for month in alerts_5['month'].unique():
        size = int(len(alerts_5.loc[alerts_5['month'] == month]))
        alerts_temp_month = alerts_temp.loc[alerts_temp['month'] == month]
        temp.append(alerts_temp_month.sample(n = size, random_state = 42))

    alerts_15 = pd.concat(temp)
    alerts_15.to_parquet('../../Data_and_models/data/alerts/alert_0.15-data_0.05.parquet')

    alerts[0.15] = alerts_15

    desired_alerts = [0.05,0.15]
    desired_subsample = [0.50,0.25]

    for fpr_alert_rate in desired_alerts:
        alerts[fpr_alert_rate] = dict()
        for sub in desired_subsample:
            tpr, t = tpr_at_fpr(BAF_val['fraud_bool'], BAF_val['model_score'], fpr_alert_rate)
            alerts_temp = BAF_dep.loc[BAF_dep['model_score'] > t]
            alerts_temp_subsample = []
            for month in alerts_5['month'].unique():
                size = int(len(alerts_5.loc[alerts_5['month'] == month]))
                alerts_temp_month = alerts_temp.loc[alerts_temp['month'] == month]
                alerts_temp_month_sample = alerts_temp_month.sample(n = size, random_state = 42)
                if month !=7:
                    alerts_temp_month_sample = alerts_temp_month_sample.sample(n = int(size*sub), random_state = 42)
                alerts_temp_subsample.append(alerts_temp_month_sample)

            alerts_temp_subsample = pd.concat(alerts_temp_subsample)
            alerts[fpr_alert_rate][sub] = alerts_temp_subsample
            alerts_temp_subsample.to_parquet(f'../../Data_and_models/data/alerts/alert_{fpr_alert_rate:.2f}-data_0.05-sub_{sub:.2f}.parquet')


