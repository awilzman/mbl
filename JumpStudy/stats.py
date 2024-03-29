# -*- coding: utf-8 -*-
'''
Created on Wed Jan  3 10:14:22 2024

@author: arwilzman
'''

import os
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt

import itertools
import statsmodels.api as sm
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

save_fig = True
show_fig = True
save_results = True

if save_fig:
    show_fig = True

date = datetime.datetime.now()
month = date.strftime('%m')
day = date.strftime('%d')
year = date.strftime('%y')
date = ('_'+month + '_' + day + '_' + year)
directory = 'Z:/_Current IRB Approved Studies/Jumping_Study/'
data = pd.read_csv(directory+f'Data/Jump_Study_Data{date}.csv')


predictor_col = ['Height','Landing Limbs',
                 'Hip Flexion ROM', 'Knee Flexion ROM',
                 'Ankle Flexion ROM', 'Hip Flexion at Contact',
                 'Knee Flexion at Contact', 'Ankle Flexion at Contact']
val_col = ['DIS','IMU_FFT',
           'RXF','RXF_R','RXF_FFT',
           'JCF', 'JCF_R','JCF_FFT',
           'RXF_FE','RXF_FE_FFT','RXF_SMR',
           'JCF_FE','JCF_FE_FFT','JCF_SMR']

#upscale strain to microstrain
data['RXF_FE'] = data['RXF_FE']*10e6
data['JCF_FE'] = data['JCF_FE']*10e6
data['FFT_RXF_FE'] = data['RXF_FE_FFT']*10e6
data['FFT_JCF_FE'] = data['JCF_FE_FFT']*10e6

def fit_n_print_LM(X, y, alpha=0.05, show_fig=True, plt_file=''):
    
    outcome = y.columns[0]
    y = y.values.ravel()
    if X.shape[1] > 1:
        X = sm.add_constant(X)
        reg = sm.OLS(y, X).fit()
        coefficients = reg.params[1:]
        intercept = reg.params[0]
        total_r2 = reg.rsquared
        f_statistic = reg.fvalue
        p_value_total = reg.f_pvalue
        dual_total = f'{total_r2:.3e}' #total r2
        # p value testing
        if p_value_total < alpha:
            dual_total += '*'
            if p_value_total < alpha/10:
                dual_total += '*'
                if p_value_total < alpha/100:
                    dual_total += '*'
        print('R2 total:\t', total_r2)

        string = []
        for i, x in enumerate(X.columns[1:]):  # Exclude the constant term
            # get individual constant statistics for flavor
            uni_reg = sm.OLS(y, sm.add_constant(X[x])).fit()
            r2 = uni_reg.rsquared
            p_value = uni_reg.pvalues[1]
            b_value = uni_reg.params[1]
            print(f'R2 {x}:\t', r2)
            print(f'P-value {x}:\t', p_value)
            # p value testing has to be done like this because we're appending
            # to a string and I don't feel like changing it
            if p_value < alpha:
                if p_value < alpha / 10:
                    if p_value < alpha / 100:
                        string.append(f'{r2:.3e}, β = {b_value:.3e}***')
                    else:
                        string.append(f'{r2:.3e}, β = {b_value:.3e}**')
                else:
                    string.append(f'{r2:.3e}, β = {b_value:.3e}*')
            else:
                string.append(f'{r2:.3e}, β = {b_value:.3e}')
        if show_fig:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[X.columns[1]], X[X.columns[2]], abs(y))
            ax.scatter(X[X.columns[1]], X[X.columns[2]], abs(reg.predict(X)), color='red', s=50)
            ax.set_title(f'{X.columns[1]} r2 = {string[0]}'
                         f'\nand {X.columns[2]} r2 = {string[1]}'
                         f'\nvs. {outcome}: r2 = {dual_total}',fontsize=20)
            if plt_file != '':
                plt.savefig(plt_file)
                
            plt.show()
    else:
        reg = LinearRegression().fit(X, y)
        pred = reg.predict(X)
        coefficients = reg.coef_
        intercept = reg.intercept_
        b_value = coefficients[0]
        f_statistic, p_value_total = f_regression(X, y)

        total_r2 = reg.score(X, y)
        print('Coefficients:\t', coefficients)
        print('Intercept:\t', intercept)
        print('F-statistic:\t', f_statistic[0])
        print('P-value:\t', p_value_total[0])
        print(f'R2 {X.columns[0]}:\t', total_r2)

        # p value testing
        if p_value_total[0] < alpha:
            total_string = f'{total_r2:.3e}, β = {b_value:.3e}*'
            if p_value_total[0] < alpha / 10:
                total_string += '*'
                if p_value_total[0] < alpha / 100:
                    total_string += '*'
        else:
            total_string = f'{total_r2:.3e}'
        if show_fig:
            plt.figure(figsize=(8, 6)) 
            plt.scatter(X[X.columns[0]], abs(y))
            plt.plot(X[X.columns[0]], abs(pred))
            plt.title(f'{X.columns[0]} vs. {outcome}\nr2 = {total_string}')
            if plt_file != '':
                plt.savefig(plt_file)
                
            plt.show()
    
    return coefficients, intercept, f_statistic, total_r2, p_value_total

def bonf(df,alpha=0.05):
    df = df.sort_values(by='P-value').reset_index()
    df.columns = ['old index'] + list(df.columns[1:])
    df['corrected alpha'] = np.linspace(alpha/len(df),alpha, num=len(df))
    df['sig'] = (df['P-value'] < df['corrected alpha']).astype(int)
    df = df.sort_values(by='old index')
    df = df.drop(columns='old index')
    return df

def Lasso_split_n_train(X,y,test_size,show_fig,plt_file='',title=''):
    y = y.dropna()
    X = X.loc[y.index]
    X = X.dropna()
    y = y.loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=551)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lasso_model = LassoCV()
    lasso_model.fit(X_train_scaled, y_train)
    y_pred = lasso_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred
    
    # Significance testing
    n = len(y_train)
    p = X_train_scaled.shape[1]
    dof = max(0, n - p)  # Degrees of freedom
    t_scores = lasso_model.coef_ / np.sqrt((mse / dof) * np.linalg.inv(np.dot(X_train_scaled.T, X_train_scaled)).diagonal())
    p_values = [2 * (1 - stats.t.cdf(np.abs(score), dof)) for score in t_scores]
    
    # Plotting
    if show_fig or plt_file != '':
        plt.figure(figsize=(8, 8)) 
        plt.scatter(y_test, y_pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Actual',fontsize=16)
        plt.ylabel('Predicted',fontsize=16)
        plt.title(title,fontsize=20)
        if plt_file != '':
            plt.savefig(plt_file)
        if show_fig:
            plt.show()
        
        plt.figure(figsize=(8, 8)) 
        stats.probplot(residuals, plot=plt)
        plt.title(f'{title}\nQ-Q Plot',fontsize=20)
        plt.xlabel('Fitted values',fontsize=16)
        plt.ylabel('Residuals',fontsize=16)
        if plt_file != '':
            plt.savefig(f"{plt_file[:-4]}_residuals.png")
        if show_fig:
            plt.show()
    
    return mse, mae, r2, lasso_model, p_values
#%%
for val in val_col:
    formula = f"{val} ~ Height + Q('Landing Limbs')"
    ram_data = data.dropna(subset=[val])
    group_id = ram_data['ID']
    model = sm.MixedLM.from_formula(formula, data=ram_data, groups=group_id)
    result = model.fit()
    print(result.summary())
    
    residuals = result.resid
    
    # Create a residuals plot
    sns.residplot(x=result.fittedvalues, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.title(f'{val} Residuals Plot')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.show()
    
    sns.lmplot(x='Height', y=val, hue='Landing Limbs', data=data, ci=None)
    plt.title(f'Interaction Plot: Height vs. {val} by Landing Limbs')
    plt.show()
    
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{val} Q-Q Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Standardized Residuals')
    plt.show()
    
#%% t-tests
t_stats = []
p_vals = []
group_names = []

for i,val in enumerate(val_col):
    group_names.append(f"{val}: all data; unilateral vs bilateral")
    G1 = data[data['Landing Limbs']==1][val].dropna()
    G2 = data[data['Landing Limbs']==2][val].dropna()
    t_stat, p_val = stats.ttest_ind(G1,G2,equal_var=False)
    t_stats.append(t_stat)
    p_vals.append(p_val)
    
    
corrected_LL_test = bonf(pd.DataFrame({'Group Comparison': group_names,
                                       'T-statistic': t_stats,
                                       'P-value': p_vals}))

t_stats = []
p_vals = []
group_names = []

for val in val_col:
    for pair in itertools.combinations(pd.unique(data['Height']),2):
        group_names.append(f"{val}: all data; {pair[0]} vs {pair[1]}")
        G1 = data[data['Height'] == pair[0]][val].dropna()
        G2 = data[data['Height'] == pair[1]][val].dropna()
        t_stat, p_val = stats.ttest_ind(G1, G2, equal_var=False)
        t_stats.append(t_stat)
        p_vals.append(p_val)
height_test = pd.DataFrame({
    'Group Comparison': group_names,
    'T-statistic': t_stats,
    'P-value': p_vals
})

corrected_height_test = bonf(height_test)

t_stats = []
p_vals = []
group_names = []
conditions = [(data['Landing Limbs']==1),(data['Landing Limbs']==2)]
for val in val_col:
    for i,cond in enumerate(conditions):
        if i == 0:
            string = 'unilateral'
        else:
            string = 'bilateral'
        for pair in itertools.combinations(pd.unique(data[cond]['Height']),2):
            group_names.append(f"{val}: {string}; {pair[0]} vs {pair[1]}")
            G1 = data[cond]
            G2 = data[cond]
            G1 = G1[G1['Height'] == pair[0]][val].dropna()
            G2 = G2[G2['Height'] == pair[1]][val].dropna()
            t_stat, p_val = stats.ttest_ind(G1, G2, equal_var=False)
            t_stats.append(t_stat)
            p_vals.append(p_val)
height_LL_test = pd.DataFrame({
    'Group Comparison': group_names,
    'T-statistic': t_stats,
    'P-value': p_vals
})

corrected_height_LL_test = bonf(height_LL_test)

t_tests = pd.concat(
    [corrected_LL_test,corrected_height_test,corrected_height_LL_test],
    axis=0,ignore_index=True).reset_index(drop=True)

if save_results:
    t_tests.to_csv(f'{directory}Statistics/main_t-tests{date}.csv', index=False)

#%% Linear modeling
results = []
file_path = ''

for val in val_col:
    for predictor in predictor_col:
        # Extract relevant columns and drop NaN values
        clean_data = data[[val, predictor]].dropna()
        
        if save_fig:
            file_path = f'{directory}Statistics/graphs/{val}_{predictor}{date}_LM.png'
        
        c,i,f,r,p = fit_n_print_LM(clean_data[[predictor]], clean_data[[val]],0.05,show_fig,file_path)
        
        results.append({'Val': val, 'Predictor 1': predictor, 'Predictor 2': None,
                        'Coefficient 1': c[0], 'Coefficient 2': None,'Intercept': i,
                        'F-statistic': f[0],'Total R2': r,'P-value': p[0]})
        
#%%
for pair in itertools.combinations(predictor_col, 2):
    for val in val_col:
        clean_data = data[[pair[0],pair[1],val]].dropna()
        
        if save_fig:
            file_path = f'{directory}Statistics/graphs/{val}_{pair[0]}_{pair[1]}{date}_LM.png'
            
        c,i,f,r,p = fit_n_print_LM(clean_data[[pair[0],pair[1]]],clean_data[[val]],0.05,show_fig,file_path)
        results.append({'Val': val, 'Predictor 1': pair[0], 'Predictor 2': pair[1],
                        'Coefficient 1': c[0], 'Coefficient 2': c[1],'Intercept': i,
                        'F-statistic': f,'Total R2': r,'P-value': p})
df_results = pd.DataFrame(results)

if save_results:
    df_results.to_csv(f'{directory}Statistics/LM_results{date}.csv', index=False)
    
#%%

singles = df_results[pd.isnull(df_results['Predictor 2'])][['Val','Predictor 1','Predictor 2','Total R2','P-value']]
singles_resort = bonf(singles)
singles_resort = singles_resort.sort_values(by=['Val','Predictor 1'])
    
doubles = df_results[~pd.isnull(df_results['Predictor 2'])][['Val','Predictor 1','Predictor 2','Total R2','P-value']]
doubles_resort = bonf(doubles)
doubles_resort = doubles_resort.sort_values(by=['Val','Predictor 1','Predictor 2'])

bonf_data = pd.concat([singles_resort,doubles_resort],axis=0)
if save_results:
    bonf_data.to_csv(f'{directory}Statistics/Bonferroni.csv')
    
#%%
test_size = 0.2
metrics = pd.DataFrame()
complex0 = predictor_col
complex1 = ['DIS', 'IMU_FFT']
complex2 = ['RXF', 'RXF_R','RXF_FFT']
complex3 = ['JCF', 'JCF_R','JCF_FFT']
complex4 = ['RXF_FE','RXF_FE_FFT','RXF_SMR']
complex5 = ['JCF_FE','JCF_FE_FFT','JCF_SMR']
X = data[complex0]

if save_fig:
    file_path = f'{directory}Statistics/graphs/'
else:
    file_path = ''
    
title_ = ''

for i in val_col:
    file_path_ = file_path
    if file_path_ != '':
        file_path_ += f'{i}_LASSO_kinematics_only.png'
    title_ = f'{i} LASSO Regression Test Results,\nkinematic predictors only'
    mse,mae,r2,lasso_model,p_vals = Lasso_split_n_train(X,data[i],test_size,show_fig,file_path_,title_)
    coefficients = lasso_model.coef_
    result_dict = {'outcome': [f'{i}_kinematics only'], 'mean': data[i].mean(),
                   'std': data[i].std(), 'mse': [mse], 'mae': [mae], 'r2': [r2]}
    for j, coef in enumerate(coefficients):
        result_dict[f'coef_{X.columns[j]}'] = coef
        result_dict[f'p_value_{X.columns[j]}'] = p_vals[j]
    outcome_df = pd.DataFrame(result_dict, index=[0])
    
    metrics = pd.concat([metrics, outcome_df], axis=0, ignore_index=True)
    
X = pd.concat([X,data[complex1]],axis=1) # add complex1 to predict complex2
for i in complex2:
    file_path_ = file_path
    if file_path_ != '':
        file_path_ += f'{i}_LASSO.png'
    title_ = f'{i} LASSO Regression Test Results'
    mse,mae,r2,lasso_model,p_vals = Lasso_split_n_train(X,data[i],test_size,show_fig,file_path_,title_)
    coefficients = lasso_model.coef_
    result_dict = {'outcome': [f'{i}'], 'mean': data[i].mean(),
                   'std': data[i].std(), 'mse': [mse], 'mae': [mae], 'r2': [r2]}
    for j, coef in enumerate(coefficients):
        result_dict[f'coef_{X.columns[j]}'] = coef
        result_dict[f'p_value_{X.columns[j]}'] = p_vals[j]
    outcome_df = pd.DataFrame(result_dict, index=[0])
    metrics = pd.concat([metrics, outcome_df], axis=0, ignore_index=True)
    
X = pd.concat([X,data[complex2]],axis=1) # add complex2 to predict complex3
for i in complex3:
    file_path_ = file_path
    if file_path_ != '':
        file_path_ += f'{i}_LASSO.png'
    title_ = f'{i} LASSO Regression Test Results'
    mse,mae,r2,lasso_model,p_vals = Lasso_split_n_train(X,data[i],test_size,show_fig,file_path_,title_)
    coefficients = lasso_model.coef_
    result_dict = {'outcome': [f'{i}'], 'mean': data[i].mean(),
                   'std': data[i].std(), 'mse': [mse], 'mae': [mae], 'r2': [r2]}
    for j, coef in enumerate(coefficients):
        result_dict[f'coef_{X.columns[j]}'] = coef
        result_dict[f'p_value_{X.columns[j]}'] = p_vals[j]
    outcome_df = pd.DataFrame(result_dict, index=[0])
    metrics = pd.concat([metrics, outcome_df], axis=0, ignore_index=True)
    
X = pd.concat([X,data[complex3]],axis=1) # add complex3 to predict complex4
for i in complex4:
    file_path_ = file_path
    if file_path_ != '':
        file_path_ += f'{i}_LASSO.png'
    title_ = f'{i} LASSO Regression Test Results'
    mse,mae,r2,lasso_model,p_vals = Lasso_split_n_train(X,data[i],test_size,show_fig,file_path_,title_)
    coefficients = lasso_model.coef_
    result_dict = {'outcome': [f'{i}'], 'mean': data[i].mean(),
                   'std': data[i].std(), 'mse': [mse], 'mae': [mae], 'r2': [r2]}
    for j, coef in enumerate(coefficients):
        result_dict[f'coef_{X.columns[j]}'] = coef
        result_dict[f'p_value_{X.columns[j]}'] = p_vals[j]
    outcome_df = pd.DataFrame(result_dict, index=[0])
    metrics = pd.concat([metrics, outcome_df], axis=0, ignore_index=True)
    
# do NOT add complex4 to predict complex5
for i in complex5:
    file_path_ = file_path
    if file_path_ != '':
        file_path_ += f'{i}_LASSO.png'
    title_ = f'{i} LASSO Regression Test Results'
    mse,mae,r2,lasso_model,p_vals = Lasso_split_n_train(X,data[i],test_size,show_fig,file_path_,title_)
    coefficients = lasso_model.coef_
    result_dict = {'outcome': [f'{i}'], 'mean': data[i].mean(),
                   'std': data[i].std(), 'mse': [mse], 'mae': [mae], 'r2': [r2]}
    for j, coef in enumerate(coefficients):
        result_dict[f'coef_{X.columns[j]}'] = coef
        result_dict[f'p_value_{X.columns[j]}'] = p_vals[j]
    outcome_df = pd.DataFrame(result_dict, index=[0])
    metrics = pd.concat([metrics, outcome_df], axis=0, ignore_index=True)
    
if save_results:
    metrics.to_csv(f'{directory}Statistics/LASSO.csv')
    
