import pandas as pd 
import numpy as np

import rpy2.robjects as ro
from rpy2.robjects  import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

from crisp_nam.metrics.calibration import brier_score, integrated_brier_score
from crisp_nam.metrics.discrimination import auc_td

risk_regression = importr('riskRegression')
prodlim = importr('prodlim')
data_table = importr('data.table')
dpylr = importr('dplyr')
geepack = importr('geepack')
ggplot2 = importr('ggplot2')
rms = importr('rms')
tableone = importr('tableone')
Hmisc = importr('Hmisc')
base = importr('base')
grdevices = importr('grDevices')

#Load model from RDS file and convert to pickle
readRDS = ro.r['readRDS']
model = readRDS('model/final_FGR_clean.rds')

####### PREDICT ON 1 DATA POINT ########

#Predictions on new data
single_data_pt = ro.DataFrame({'age': ro.IntVector([67]),
    'sex_f': ro.IntVector([1]),
    'elective_adm': ro.IntVector([1]),
    'homelessness': ro.IntVector([0]),
    'peripheral_AD': ro.IntVector([0]),
    'coronary_AD': ro.IntVector([1]),
    'stroke': ro.IntVector([0]),
    'CHF': ro.IntVector([0]),
    'hypertension': ro.IntVector([1]),
    'COPD': ro.IntVector([0]),
    'CKD': ro.IntVector([0]),
    'malignancy': ro.IntVector([0]),
    'mental_illness': ro.IntVector([0]),
    'creatinine': ro.FloatVector([140]),
    'Hb_A1C': ro.FloatVector([8.5]),
    'albumin': ro.FloatVector([32.1]),
    'Hb_A1C_missing': ro.IntVector([0]),
    'creatinine_missing': ro.IntVector([0]),
    'albumin_missing': ro.IntVector([0])})


age_splines = Hmisc.rcspline_eval(single_data_pt.rx2('age'), knots=model.rx2('splines').rx2('age_knots'), inclx=True)
creatinine_splines = Hmisc.rcspline_eval(single_data_pt.rx2('creatinine'), knots=model.rx2('splines').rx2('creatinine_knots'), inclx=True)
Hb_A1C_splines = Hmisc.rcspline_eval(single_data_pt.rx2('Hb_A1C'), knots=model.rx2('splines').rx2('hba1c_knots'), inclx=True)
albumin_splines = Hmisc.rcspline_eval(single_data_pt.rx2('albumin'), knots=model.rx2('splines').rx2('albumin_knots'), inclx=True)

single_data_pt = ro.r.cbind(single_data_pt, age1= age_splines.rx2(2))
single_data_pt = ro.r.cbind(single_data_pt, age2= age_splines.rx2(3))
single_data_pt = ro.r.cbind(single_data_pt, creatinine1= creatinine_splines.rx2(2))
single_data_pt = ro.r.cbind(single_data_pt, creatinine2= creatinine_splines.rx2(3))
single_data_pt = ro.r.cbind(single_data_pt, Hb_A1C1= Hb_A1C_splines.rx2(2))
single_data_pt = ro.r.cbind(single_data_pt, Hb_A1C2= Hb_A1C_splines.rx2(3))
single_data_pt = ro.r.cbind(single_data_pt, albumin1= albumin_splines.rx2(2))

risk_1_year = ro.r('predict')(model, newdata=single_data_pt, times=ro.FloatVector([365.25]))
print(f'Predictions for 1 year: {risk_1_year}')

########## PREDICT ON SIMULATED DATA ##########

#Generate simulated data: Rscript data/simulate_data.R

#Read data from data file
data = readRDS('data/dummy_data_no_na.rds')
print(f'data shape: {data.nrow} rows, {data.ncol} columns')
r_data_df = ro.DataFrame(data)

#Performance metrics
try:    
    score_result = ro.r('Score')(
                    ro.ListVector({'model1': model}),
                    data=r_data_df,
                    formula=ro.Formula('Hist(time, status) ~ 1'),
                    cause=ro.IntVector([1]),
                    times=ro.FloatVector([365.25]),
                    metrics=ro.StrVector(["auc", "brier"]),
                    summary=ro.StrVector(["risks", "ipa"]),
                    plots=ro.StrVector(["ROC", "calibration"])
    )

except Exception as e:
    print(f'Error during prediction: {e}')

#Plot AUC ROC
grdevices.png(file="plots/roc.png", width=500, height=600)
roc = risk_regression.plotROC(score_result,
                                times = ro.FloatVector([365.25]),
                                ylab = ro.r.paste0("Sensitivity at 1 year"),
                                xlab = ro.r.paste0("1-Specificity at 1 year")
                            )

grdevices.dev_off()

grdevices.png(file="plots/caliberation.png", width=500, height=600)
params  = { 'method' : "nne",
            'xlim': ro.FloatVector([0, 0.05]),
            'round': ro.BoolVector([False]),
            'ylim' : ro.FloatVector([0, 0.05]),
            'rug' : ro.BoolVector([True])
        }
  

caliberation_plot = risk_regression.plotCalibration(score_result,
                                                        **params)
grdevices.dev_off()


####### CRISP-NAM Metrics #########
data_pd, score = pd.DataFrame(), pd.DataFrame()
r_score_result_df = ro.DataFrame(score_result)

with localconverter(ro.default_converter + pandas2ri.converter):
    data_pd = ro.conversion.rpy2py(data)

data_pd = data_pd.reset_index()
data_pd.drop(columns='index', inplace=True)


e_val, t_val = data_pd['status'], data_pd['time']
quantiles, safe_max = [0.25, 0.5, 0.75], 0.99 * np.max(data_pd['time'])
eval_times = np.quantile(data_pd['time'][data_pd['time'] <= safe_max], quantiles)

risk_predicted = np.zeros(shape=(data_pd.shape[0], len(eval_times)))
for t, time in enumerate(eval_times):
    curr_risk = ro.r('predictRisk')(model, newdata=r_data_df, times=ro.FloatVector([time]))
    risk_predicted[:, t] = np.array(curr_risk).flatten()

    brier_score_val = brier_score(
                                e_val, 
                                t_val,
                                risk_predicted,
                                times=eval_times,
                                t=time,
                                #km=(e_train, t_train),
                                #primary_risk=k+1
                        )
    print(f'brier_score_val for time {time}: {brier_score_val[0]}')


integrated_brier_score_val = integrated_brier_score(
                                e_val, 
                                t_val,
                                risk_predicted, 
                                times=eval_times,
                                t_eval=None,
                                km=None,
                                primary_risk=1)[0]

print(f'integrated_brier_score_val from CRISP NAM : {integrated_brier_score_val}')

"""
auc_td_val = auc_td(e_val,
                    t_val,
                    dummy_pred_np,
                    times,
                    366, km=None, primary_risk=1)
print(f'auc_td_val for sample : {auc_td_val}')
"""