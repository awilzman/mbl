# -*- coding: utf-8 -*-
"""
Created on Fri May 31 08:45:44 2024

@author: arwilzman
"""

from instron_read import TestProcessor

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import kendalltau
    

class StatsDoer:
    def __init__(self,
        directory: str,
        bone_map: Dict[str, int],
        items: Optional[List[str]] = None,
        save: bool = True,
        plot: bool = False) -> None:
        
        self.directory = directory
        self.items = items if items else ['']
        self.save = save
        self.plot = plot
        self.bone_map = bone_map
        self.data = None
        
    def grab_fe_data(self,
        sides: Tuple[str, str] = ("L", "R")) -> pd.DataFrame:
        data = []
        for side in sides:
            for item in self.items:
                file = os.path.join(self.directory, item, side, 'abaqus_files', f'{item}_fe_data.csv')
                if os.path.isfile(file):
                    ram = pd.read_csv(file)
                    for _, r in ram.iterrows():
                        
                        segments = r['Step Key'].split('_')
                        mtno = self.bone_map.get(segments[0], 0)
                        load = segments[2]
                        angle = int(segments[-1][:-3]) if 'deg' in segments[-1] else 0
                        
                        data.append({
                            'ID': item,
                            'MtNo': mtno,
                            'Side': side,
                            'Load N': load,
                            'Angle deg': angle,
                            'Max Principal Strain': r['Max Principal Strain'],
                            'Min Principal Strain': r['Min Principal Strain'],
                            'Max von Mises Stress': r['Max von Mises Stress'],
                            'Tsai-Wu Strained Volume': r['Tsai-Wu Strained Volume'],
                            'von Mises Stressed Volume': r['von Mises Stressed Volume'],
                            'von Mises Strained Volume': r['von Mises Strained Volume'],
                            'Max Displacement': r['Max Displacement'],
                            'Total Volume': r['Total Volume']
                        })
        df = pd.DataFrame(data)        
        for angle in [0, 30]:
            subset = df[df['Angle deg'] == angle]
            if not subset.empty:
                print(f"Angle = {angle}°")
                for col in ['Max Principal Strain', 'Min Principal Strain', 'Tsai-Wu Strained Volume', 
                            'von Mises Stressed Volume', 'von Mises Strained Volume']:
                    mean_val = subset[col].mean()
                    std_val = subset[col].std()
                    print(f"  {col}: Mean = {mean_val:.6f}, Std = {std_val:.6f}")
                
        return df

    def load_computer(self,
        ram: int,
        target: str) -> int:
        X = np.array(ram['Load N']).reshape(-1, 1)
        y = abs(np.array(ram['Tsai-Wu Strained Volume']).reshape(-1, 1))

        reg = LinearRegression().fit(y, X)

        max_tw = float(reg.predict([[target]]).squeeze())

        return round(max_tw)

    def define_loadcases(self,
        load_case: "pd.DataFrame",
        mtnos: Tuple[int, int, int] = (2, 3, 4),
        sides: Tuple[str, str] = ("L", "R")) -> None:
        
        data_df = self.grab_fe_data(sides)
        load_case_final = pd.DataFrame()
        
        demo = {
            "2108482": {"Age": 93, "Height_cm": 65 * 2.54, "Weight_kg": 178 / 2.2, "Sex": 0},
            "2202457M": {"Age": 97, "Height_cm": 63 * 2.54, "Weight_kg": 82 / 2.2, "Sex": 0},
            "2202474M": {"Age": 83, "Height_cm": 63 * 2.54, "Weight_kg": 117 / 2.2, "Sex": 0},
            "2202556M": {"Age": 65, "Height_cm": 63 * 2.54, "Weight_kg": 169 / 2.2, "Sex": 0},
            "2203581M": {"Age": 24, "Height_cm": 64 * 2.54, "Weight_kg": 136 / 2.2, "Sex": 0},
            "2204751M": {"Age": 40, "Height_cm": 71 * 2.54, "Weight_kg": 189 / 2.2, "Sex": 1},
            "2204828M": {"Age": 60, "Height_cm": 69 * 2.54, "Weight_kg": 140 / 2.2, "Sex": 1},
            "2204869M": {"Age": 90, "Height_cm": 70 * 2.54, "Weight_kg": 195 / 2.2, "Sex": 1},
            "2205030M": {"Age": 74, "Height_cm": 61 * 2.54, "Weight_kg": 95 / 2.2, "Sex": 0},
            "2205033M": {"Age": 82, "Height_cm": 72 * 2.54, "Weight_kg": 157 / 2.2, "Sex": 1},
            "2205041M": {"Age": 69, "Height_cm": 66 * 2.54, "Weight_kg": 153 / 2.2, "Sex": 0},
            "2205048M": {"Age": 81, "Height_cm": 71 * 2.54, "Weight_kg": 163 / 2.2, "Sex": 1},
            "2205976M": {"Age": 62, "Height_cm": 66 * 2.54, "Weight_kg": 191 / 2.2, "Sex": 1},
            "2206149M": {"Age": 65, "Height_cm": 64 * 2.54, "Weight_kg": 151 / 2.2, "Sex": 0},
        }
        
        demo_df = pd.DataFrame.from_dict(demo, orient="index").reset_index()
        demo_df.rename(columns={"index": "Foot"}, inplace=True)
        
        if isinstance(mtnos, int):
            mtnos = (mtnos,)
            
        for item in self.items:
            for side in sides:
                for mtno in mtnos:
                    angle = load_case[(load_case['Foot'] == item) & (load_case['Side'] == side)]['Angle'].iloc[0]
                    condition = ((data_df['ID'] == item) & (data_df['MtNo'] == mtno) & 
                                 (data_df['Side'] == side) & (data_df['Angle deg'] == angle))
                    l_condition = ((load_case['Foot'] == item) & (load_case['Side'] == side))

                    ram = data_df[condition]
                    if ram.dropna().empty:
                        print(f"Skipping {item} {side} MT{mtno}: contains no data.")
                        continue

                    ram2 = load_case[l_condition]
                    var_ = f'MT{mtno} Load'
                    if not pd.isna(ram2[var_].values[0]):
                        load_target = ram2[var_].values[0]
                    else:
                        #load_target = self.load_computer(ram, 1500 if ram2.iloc[0]['Mag'] == 'High' else 1000)
                        load_target = 0
                        
                    X = np.array(ram['Load N'],dtype=float)

                    idx = np.searchsorted(X, load_target)
                    
                    #Load target must be included in the FE load list
                    
                    vm_str_o_vol = ram.iloc[idx]['von Mises Stressed Volume']
                    vm_str_e_vol = ram.iloc[idx]['von Mises Strained Volume']
                    tw_strvol = ram.iloc[idx]['Tsai-Wu Strained Volume']
                    pred_displ = ram.iloc[idx]['Max Displacement']
                    min_strain = ram.iloc[idx]['Min Principal Strain']
                    max_strain = ram.iloc[idx]['Max Principal Strain']
                    tot_vol = ram.iloc[idx]['Total Volume']
                    
                    slope = load_target/pred_displ
                    
                    FL = ram2.iloc[0][f'MT{mtno} Fatigue Life']
                    C1 = ram2.iloc[0][f'MT{mtno} Initial Stiffness']
                    C2 = ram2.iloc[0][f'MT{mtno} Maximum Stiffness']
                    ratio = C2/C1 if C1 != 0 else 0
                    
                    ram3 = {
                        'Foot': item,
                        'Age': demo[item]['Age'],
                        'Sex': demo[item]['Sex'],
                        'Side': side,
                        'Mtno': mtno,
                        'Mag': ram2.iloc[0]['Mag'],
                        'Angle': angle,
                        'Load': load_target,
                        'Tsai-Wu Strained Volume': tw_strvol,
                        'von Mises Stressed Volume': vm_str_o_vol,
                        'von Mises Strained Volume': vm_str_e_vol,
                        'Total Volume': tot_vol,
                        'Max Principal Strain': max_strain,
                        'Min Principal Strain': min_strain,
                        'Predicted Displacement': pred_displ,
                        'Predicted Stiffness': slope,
                        'Fatigue Life': FL, 
                        'Initial Stiffness': C1,
                        'Maximum Stiffness': C2,
                        'Max Hardening Ratio': ratio
                    }
                    
                    load_case_final = pd.concat((load_case_final,pd.DataFrame([ram3])))

                    if self.plot:
                        for col in ram.columns[5:]:
                            if 'Volume' in col:
                                continue# for now...
                            elif col == 'Max Displacement':
                                plt.figure()
                            else:
                                continue
                            plt.plot(X, ram[col], marker='o', linestyle='-', label=col)
                            plt.xlabel('Load N')
                            plt.ylabel(col)
                            plt.title(f'{col}, {item} {side} {mtno} {angle} deg')
                            if self.save:
                                filename = f"{self.directory}/graphs/{item}{col}_{side}_{mtno}.png"
                                plt.savefig(filename)
                            #plt.show()

        if self.save:
            file_path = os.path.join(self.directory, 'Cadaver_Loadcases.xlsx')
            load_case_final.to_excel(file_path, index=False)
            
    def load_the_cases(self,
        mtnos: Tuple[int, int, int] = (2, 3, 4),
        sides: Tuple[str, str] = ("L", "R")) -> None:
        
        main_file = os.path.join(self.directory, 'Cadaver_Loadcases.xlsx')
        if not os.path.exists(main_file):
            print("Error: 'Cadaver_Loadcases.xlsx' not found.")
        
        self.data = pd.read_excel(main_file)
        
    def tune_parameters(self,
        fatigue_lives: "np.ndarray",
        tw_str_vols: List["np.ndarray"],
        vm_str_o_vols: List["np.ndarray"],
        vm_str_e_vols: List["np.ndarray"],
        tw_exp: float,
        vm_exp: float,
        lhs_data: "pd.DataFrame") -> None:
        
        self.load_the_cases((2,3,4))
        
        model = LinearRegression()
        tw_mse_1 = []
        vm_o_mse_1 = []
        vm_e_mse_1 = []
        for i, strains in enumerate(lhs_data.T):
            tw_mse_2 = []
            vm_o_mse_2 = []
            vm_e_mse_2 = []
            
            tw_str_vol = [np.array(t,dtype='float') for t in tw_str_vols]
            tw_str_vol = np.array([t[i] for t in tw_str_vol])
            
            if i < vm_exp:
                vm_str_o_vol = [np.array(v,dtype='float') for v in vm_str_o_vols]
                vm_str_o_vol = np.array([v[i] for v in vm_str_o_vol])
                
                vm_str_e_vol = [np.array(v,dtype='float') for v in vm_str_e_vols]
                vm_str_e_vol = np.array([v[i] for v in vm_str_e_vol])
            
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            kf_split = kf.split(fatigue_lives)
            
            for train_idx, test_idx in kf_split:
                tw_train, tw_test = tw_str_vol[train_idx], tw_str_vol[test_idx]
                
                y_train, y_test = fatigue_lives[train_idx], fatigue_lives[test_idx]
                y_train = np.log10(y_train)
                y_test = np.log10(y_test)
                
                model.fit(tw_train.reshape(-1,1), y_train.reshape(-1,1))
                y_pred = model.predict(tw_test.reshape(-1,1))
                tw_mse_2.append(mean_squared_error(10**y_test, 10**y_pred)**0.5)
                
                if i < vm_exp:
                    vm_train, vm_test = vm_str_o_vol[train_idx], vm_str_o_vol[test_idx]
                    model.fit(vm_train.reshape(-1,1), y_train.reshape(-1,1))
                    y_pred = model.predict(vm_test.reshape(-1,1))
                    vm_o_mse_2.append(mean_squared_error(10**y_test, 10**y_pred)**0.5)
                    
                    vm_train, vm_test = vm_str_e_vol[train_idx], vm_str_e_vol[test_idx]
                    model.fit(vm_train.reshape(-1,1), y_train.reshape(-1,1))
                    y_pred = model.predict(vm_test.reshape(-1,1))
                    vm_e_mse_2.append(mean_squared_error(10**y_test, 10**y_pred)**0.5)
            
            tw_mse_1.append(np.mean(tw_mse_2))
            if i < vm_exp:
                vm_o_mse_1.append(np.mean(vm_o_mse_2))
                vm_e_mse_1.append(np.mean(vm_e_mse_2))
        
        if self.plot:
            for i in range(5):    
                fig, ax = plt.subplots(figsize=(22, 16))
                ax.scatter(lhs_data.iloc[:,i],tw_mse_1,s=420, edgecolor='black', marker='o')
                ax.tick_params(axis='both', which='major', labelsize=48)
                ax.set_ylabel('Average Fatigue Life RMSE', fontsize=48)
                ax.set_xlabel(lhs_data.columns[i], fontsize=48)
                if self.save:
                    filename = f"{self.directory}/graphs/tuning {lhs_data.columns[i]}.png"
                    plt.savefig(filename)
                #plt.show()
        
        tw_choice = min(range(len(tw_mse_1)), key=tw_mse_1.__getitem__)
        
        #this needs to match everywhere
        vm_yield_stress = np.arange(20e6, 30e6, (30e6 - 20e6) / vm_exp)
        vm_yield_strain = np.arange(1e-4, 2e-3, (2e-3 - 1e-4) / vm_exp)
        
        if self.plot:
            X = vm_yield_stress
            y = vm_o_mse_1
            fig,ax = plt.subplots(figsize=(22,16))
            plt.title('Threshold calibration')
            ax.scatter(X,y,s=420,edgecolor='black',marker='o')
            ax.tick_params(axis='both',which='major',labelsize=48)
            ax.set_ylabel('Average RMSE',fontsize=48)
            ax.set_xlabel('von Mises stress threshold',fontsize=48)
            if self.save:
                filename = f"{self.directory}/graphs/tuning von Mises stress.png"
                plt.savefig(filename)
            #plt.show()
            
            X = vm_yield_strain
            y = vm_e_mse_1
            fig,ax = plt.subplots(figsize=(22,16))
            plt.title('Threshold calibration')
            ax.scatter(X,y,s=420,edgecolor='black',marker='o')
            ax.tick_params(axis='both',which='major',labelsize=48)
            ax.set_ylabel('Average RMSE',fontsize=48)
            ax.set_xlabel('von Mises strain threshold',fontsize=48)
            if self.save:
                filename = f"{self.directory}/graphs/tuning von Mises strain.png"
                plt.savefig(filename)
            #plt.show()
        
        vm_o_choice = min(range(len(vm_o_mse_1)), key=vm_o_mse_1.__getitem__)
        vm_e_choice = min(range(len(vm_e_mse_1)), key=vm_e_mse_1.__getitem__)
        if self.save:
            # this removes all data except for minimum
            file = self.directory + 'LHS.csv'
            df = lhs_data.iloc[[tw_choice]].copy()
            df['vm_o'] = vm_yield_stress[vm_o_choice]
            df['vm_e'] = vm_yield_strain[vm_e_choice]
            df.to_csv(file, index=False)
            
            self.data['Tsai-Wu Strained Volume'] = self.data['Tsai-Wu Strained Volume'].apply(
                lambda s: float(s.split(',')[tw_choice]))
            
            self.data['von Mises Stressed Volume'] = self.data['von Mises Stressed Volume'].apply(
                lambda s: float(s.split(',')[vm_o_choice]))
            
            self.data['von Mises Strained Volume'] = self.data['von Mises Strained Volume'].apply(
                lambda s: float(s.split(',')[vm_e_choice]))
            
            main_file = os.path.join(self.directory, 'Cadaver_Loadcases.xlsx')
            self.data.to_excel(main_file, index=False, engine='openpyxl')
            
    def define_sn_curves(self,
        mtnos: Tuple[int, int, int] = (2, 3, 4),
        cycles: List[str] = [
            'Fatigue Life', 'Initial Stiffness', 'Maximum Stiffness', 'Max Hardening Ratio'
        ],
        mags: List[str] = [
            'von Mises Stressed Volume', 'von Mises Strained Volume', 'Tsai-Wu Strained Volume',
            'Max Principal Strain', 'Min Principal Strain', 'Predicted Stiffness']) -> pd.DataFrame: 
    
        if isinstance(mtnos, int):
            mtnos = (mtnos,)
        self.load_the_cases(mtnos)
    
        # Plot colors
        color_map = {'Tsai-Wu Strained Volume': 'fuchsia', 'von Mises Stressed Volume': 'blue',
                     'von Mises Stressed Volume': 'green',
                     'Max Principal Strain': 'red', 'Min Principal Strain': 'blue', 'Predicted Stiffness': 'black'}
    
        res = []
    
        for angle in [0, 30]:
            data_group = self.data[self.data['Angle'] == angle]
        
            for cyc in cycles:
                for mag in mags:
                    required_cols = [cyc, mag, 'Foot', 'Side', 'Mtno']
                    df = data_group.dropna(subset=required_cols).copy()
        
                    df[cyc] = df[cyc].replace(0, np.nan)
                    df.dropna(subset=[cyc, mag], inplace=True)
        
                    # Skip if insufficient data
                    if len(df) < 3:
                        continue
        
                    mask = df[cyc] > 0
                    X = df.loc[mask, cyc]
                    y = df.loc[mask, mag]
        
                    log_model = False  # Flag to indicate if log transformation is used
        
                    if cyc == 'Fatigue Life':
                        mask = (df[cyc] < 250000) & (df[cyc] > 0)
                        survivors = df[cyc] >= 250000
        
                        # Log transformation (avoid log(0))
                        X = np.log10(df.loc[mask, cyc])
                        y = np.abs(df.loc[mask, mag])
                        y_log = np.log10(y+1)
        
                        # Survival data (handle empty cases safely)
                        if survivors.any():
                            X_surv = np.log10(df.loc[survivors, cyc])
                            y_surv = np.abs(df.loc[survivors, mag])
                        else:
                            X_surv, y_surv = np.array([]), np.array([])
        
                    # Add constant for regression
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    #model_log = sm.OLS(y_log, X).fit() if cyc == 'Fatigue Life' else None
        
                    # Calculate error metrics
                    y_pred = model.predict(X)
                    rmse, mae = np.sqrt(mean_squared_error(y, y_pred)), mean_absolute_error(y, y_pred)
        
                    # if model_log:
                        # At one point I wanted to check to see if logging both axes would
                        # change the curve fits
                    #     y_pred_log = model_log.predict(X)
                    #     rmse_log, mae_log = np.sqrt(mean_squared_error(10**y_log, 10**y_pred_log)), mean_absolute_error(10**y_log, 10**y_pred_log)
        
                    #     if mae_log < mae:
                    #         model, log_model = model_log, True
                    #         y_surv = np.log10(y_surv+1)
                    #         y, y_pred, rmse, mae = y_log, y_pred_log, rmse_log, mae_log
        
                    # Extract Model Parameters
                    slope, intercept = model.params.iloc[1], model.params.iloc[0]
                    p_val, r_squared = model.pvalues.iloc[1], getattr(model, "rsquared", None)
        
                    # Survival Prediction Rate
                    surv_pred_rate = 0
                    if len(X_surv) > 0:
                        if cyc == 'Fatigue Life':
                            X = np.log10(df.loc[mask, cyc])
                        else:
                            X = df.loc[mask, cyc]
                        y_c = sm.add_constant(y)
                        y_surv = sm.add_constant(y_surv)
                        model = sm.OLS(X, y_c).fit()
                        pred_surv = model.predict(y_surv)
                        surv_pred_rate = np.mean(10**pred_surv > 250000)
        
                    # Append results
                    res.append({
                        'Angle': angle,
                        'Cycle': cyc,
                        'Magnitude': mag,
                        'Slope_X': slope,
                        'Intercept': intercept,
                        'R-Squared': r_squared,
                        'P-Value_X': p_val,
                        'RMSE': rmse,
                        'MAE': mae,
                        'Survival Prediction Rate': surv_pred_rate
                    })
    
                        # Plot if conditions met
                    if self.plot and model.pvalues.iloc[1] < 0.25:
                        # Determine the ylabel based on the 'mag' variable
                        if 'volume' in mag.lower():
                            ylabel = 'Failed Volume (mm\u00B3)'
                        elif 'stiff' in mag.lower():
                            ylabel = 'Stiffness (N/mm)'
                        else:
                            ylabel = 'Strain (mm/mm)'
                    
                        if isinstance(X, pd.DataFrame):
                            X = X.to_numpy()
                        if isinstance(y, pd.Series):
                            y = y.to_numpy()
                    
                        fig = plt.figure(figsize=(30, 24))
                        c = color_map.get(mag, 'blue')
                    
                        if (cyc == 'Fatigue Life'):
                            X_plot = 10**X
                            if log_model:
                                y_plot = 10**y
                                y_pred_plot = 10**y_pred
                            else:
                                y_plot = y
                                y_pred_plot = y_pred
                        else:
                            X_plot = X
                            y_plot = y
                            y_pred_plot = y_pred
                    

                        # 2D scatter and plot
                        fig, ax = plt.subplots(figsize=(22, 16))
                        if cyc == 'Fatigue Life':
                            ax.set_xscale('log')
                            if log_model:
                                ax.set_yscale('log')
                            xlabel = cyc
                        elif 'initial' in cyc.lower():
                            xlabel = 'Initial Stiffness (N/mm)'
                        elif 'maximum stiffness' in cyc.lower():
                            xlabel = 'Maximum Stiffness (N/mm)'
                        elif 'max hardening' in cyc.lower():
                            xlabel = 'Maximum Hardening Ratio ([N/mm]/[N/mm])'
                            
                        scatter = ax.scatter(X_plot, y_plot, s=420, edgecolor='black', marker='o')
                        ax.plot(X_plot, y_pred_plot, color=c, linestyle='dashed', alpha=0.5)

                        # Axis Labels and Titles with increased separation
                        # add text here that shows equation, p val, and r2
                        ax.set_xlabel(xlabel, fontsize=48, labelpad=20)
                        ax.set_ylabel(ylabel, fontsize=48, labelpad=20)
                        ax.set_title(f'{mag}: {angle} degrees', fontsize=64, pad=30)
                        ax.grid(True, which="both", ls="--", lw=0.25)
                        eq_text = f"y = {slope:.2f}x + {intercept:.2f}\nP-value: {p_val:.3g}\nR²: {model.rsquared:.3f}"
                        ax.text(1.05, 0.1, eq_text, transform=ax.transAxes, fontsize=36)

                        ax.tick_params(axis='both', which='major', labelsize=48)
                        ax.tick_params(axis='both', which='minor', labelsize=48)
                        # Save the plot
                        if self.save:
                            graph_dir = os.path.join(self.directory, 'graphs')
                            os.makedirs(graph_dir, exist_ok=True)
                            plt.savefig(os.path.join(graph_dir, f'Angle_{angle}_{cyc}_{mag}_2d.png'), bbox_inches='tight', dpi=300)
                        plt.tight_layout(pad=5)
                        #plt.show()
                        plt.close()
        
        # Convert Results to DataFrame
        res_df = pd.DataFrame(res)
    
        # Save to Excel if required
        if self.save:
            file_path = os.path.join(self.directory, 'sn_curve_results.xlsx')
            res_df.to_excel(file_path, index=False)
    
        return res_df
            
    def corr_plots(self,
        mags: Optional[List[str]] = None) -> None:
        
        def significance_marker(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return ''
    
        if mags is None:
            mags = ['von Mises Stressed Volume', 'von Mises Strained Volume', 'Tsai-Wu Strained Volume']
        if self.data is None:
            print("Data not loaded. Run load_the_cases() or define_loadcases() first.")
            return
        df_ax = self.data[(self.data['Angle']==0) & (self.data['Fatigue Life'] > 0)][mags].dropna()
        df_bd = self.data[(self.data['Angle']==30) & (self.data['Fatigue Life'] > 0)][mags].dropna()
        
        cols_ax = df_ax.columns
        cols_bd = df_bd.columns
        n_ax = len(cols_ax)
        n_bd = len(cols_bd)
        
        pvals_ax = pd.DataFrame(np.ones((n_ax, n_ax)), columns=cols_ax, index=cols_ax)
        pvals_bd = pd.DataFrame(np.ones((n_bd, n_bd)), columns=cols_bd, index=cols_bd)
        
        corr_ax = df_ax.corr()
        corr_bd = df_bd.corr()
        
        for i in range(n_ax):
            for j in range(n_ax):
                if i != j:
                    r_k, p_k = kendalltau(df_ax[cols_ax[i]], df_ax[cols_ax[j]])
                    corr_ax.iloc[i, j] = r_k
                    pvals_ax.iloc[i, j] = p_k
                else:
                    pvals_ax.iloc[i, j] = 1
        for i in range(n_bd):
            for j in range(n_bd):
                if i != j:
                    r_k, p_k = kendalltau(df_bd[cols_bd[i]], df_bd[cols_bd[j]])
                    corr_bd.iloc[i, j] = r_k
                    pvals_bd.iloc[i, j] = p_k
                else:
                    pvals_bd.iloc[i, j] = 1
        
        annot_ax = corr_ax.copy()
        for i in range(n_ax):
            for j in range(n_ax):
                r = corr_ax.iloc[i, j]**2
                sig = significance_marker(pvals_ax.iloc[i, j])
                annot_ax.iloc[i, j] = f'{r:.2f}{sig}'
                
        annot_bd = corr_bd.copy()
        for i in range(n_bd):
            for j in range(n_bd):
                r = corr_bd.iloc[i, j]**2
                sig = significance_marker(pvals_bd.iloc[i, j])
                annot_bd.iloc[i, j] = f'{r:.2f}{sig}'
        
        plt.figure(figsize=(16, 16))
        #get r2
        ax = sns.heatmap(corr_ax, annot=annot_ax.values, fmt='', cmap='viridis',
                 annot_kws={'fontsize': 36}, cbar=True)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=28)
        
        ax.tick_params(axis='x', which='major', labelsize=28, rotation=25)
        ax.tick_params(axis='y', which='major', labelsize=28, rotation=0)
        plt.title('Axial Correlation Heatmap', fontsize=36)
        ax.set_xticklabels(ax.get_xticklabels(), ha='right', position=(0.05, 0))
        
        if self.save:
            file_path = os.path.join(self.directory, 'graphs/axial_correlations.png')
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
        #plt.show()
        
        plt.figure(figsize=(16, 16))
        bd = sns.heatmap(corr_bd, annot=annot_bd.values, fmt='', cmap='viridis',
                 annot_kws={'fontsize': 36}, cbar=True)
        cbar = bd.collections[0].colorbar
        cbar.ax.tick_params(labelsize=28)
        
        bd.tick_params(axis='x', which='major', labelsize=28, rotation=25)
        bd.tick_params(axis='y', which='major', labelsize=28, rotation=0)
        plt.title('Bending Correlation Heatmap', fontsize=36)
        bd.set_xticklabels(bd.get_xticklabels(), ha='right', position=(0.05, 0))
        
        if self.save:
            file_path = os.path.join(self.directory, 'graphs/bending_correlations.png')
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
        #plt.show()
        
        # Pairplot
        g = sns.pairplot(df_ax)
        for ax in g.axes.flatten():
            if ax is not None:
                ax.tick_params(axis='x', which='major', labelsize=16)
                ax.tick_params(axis='y', which='major', labelsize=16)
    
        g.fig.suptitle('Axial Pairwise Correlation Plot', fontsize=36, y=1.1)
        for ax in g.axes.flatten():
            ax.set_xlabel(ax.get_xlabel().replace('ed Volume', ''), fontsize=18)
            ax.set_ylabel(ax.get_ylabel().replace('ed Volume', ''), fontsize=18)
        if self.save:
            file_path = os.path.join(self.directory, 'graphs/scatter_bending_corr.png')
            g.fig.savefig(file_path, bbox_inches='tight', dpi=300)
        #plt.show()
        
        g = sns.pairplot(df_bd)
        for ax in g.axes.flatten():
            if ax is not None:
                ax.tick_params(axis='x', which='major', labelsize=16)
                ax.tick_params(axis='y', which='major', labelsize=16)
    
        g.fig.suptitle('Bending Pairwise Correlation Plot', fontsize=36, y=1.1)
        for ax in g.axes.flatten():
            ax.set_xlabel(ax.get_xlabel().replace('ed Volume', ''), fontsize=18)
            ax.set_ylabel(ax.get_ylabel().replace('ed Volume', ''), fontsize=18)
        if self.save:
            file_path = os.path.join(self.directory, 'scatter_axial_corr.png')
            g.fig.savefig(file_path, bbox_inches='tight', dpi=300)
        #plt.show()
        
    def multi_model(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        ax_cond = (((self.data['Angle']==0) & 
                    (self.data['Fatigue Life'] > 0) & 
                    (self.data['Fatigue Life'] < 250000)))
                   
        bd_cond = (((self.data['Angle']==30) & 
                    (self.data['Fatigue Life'] > 0) & 
                    (self.data['Fatigue Life'] < 250000)))
        
        y_ax = self.data[ax_cond]['Fatigue Life']
        y_bd = self.data[bd_cond]['Fatigue Life']
        
        Xax_1 = self.data[ax_cond]['Age']
        Xbd_1 = self.data[bd_cond]['Age']
        
        Xax_2 = self.data[ax_cond]['Sex']
        Xbd_2 = self.data[bd_cond]['Sex']
        
        cols = ['von Mises Stressed Volume', 'von Mises Strained Volume', 'Tsai-Wu Strained Volume',
                                'Max Principal Strain', 'Min Principal Strain', 'Predicted Stiffness']
        ind_ = ['Age coef','Age p-value','Sex coef','Sex p-value','X coef','X p-value','Overall p-value','Mean Absolute Error']
        Xax_3s = self.data[ax_cond][cols]
        Xbd_3s = self.data[bd_cond][cols]
        
        metrics_ax = pd.DataFrame(np.zeros((len(ind_), len(cols))), columns=cols)
        metrics_bd = pd.DataFrame(np.zeros((len(ind_), len(cols))), columns=cols)
        
        metrics_ax.index = ind_
        metrics_bd.index = ind_
    
        y_ax = y_ax.reset_index(drop=True)
        y_bd = y_bd.reset_index(drop=True)
        
        thresh = 0.05  # Significance threshold
        
        for i in range(Xax_3s.shape[1]):    
            # Stack predictor variables
            X_ax = np.column_stack((Xax_1, Xax_2, Xax_3s.iloc[:, i]))
            X_bd = np.column_stack((Xbd_1, Xbd_2, Xbd_3s.iloc[:, i]))
            
            # Convert to DataFrame with meaningful column names
            X_ax = sm.add_constant(pd.DataFrame(X_ax, columns=['Age', 'Sex', Xax_3s.columns[i]]))
            X_bd = sm.add_constant(pd.DataFrame(X_bd, columns=['Age', 'Sex', Xbd_3s.columns[i]]))
        
            # Reset indices to prevent misalignment
            X_ax = X_ax.reset_index(drop=True)
            X_bd = X_bd.reset_index(drop=True)
        
            # Backward Elimination for X_ax
            while X_ax.shape[1] > 1:
                model_ax = sm.OLS(np.log10(y_ax), X_ax).fit()
                pvals_ax = model_ax.pvalues
                
                max_p = pvals_ax[1:].max()
                if max_p < thresh:
                    break  # Stop if all predictors are significant
                
                worst_feature = pvals_ax[1:].idxmax()
                X_ax = X_ax.drop(columns=[worst_feature])  # Drop worst predictor
        
            # Backward Elimination for X_bd
            while X_bd.shape[1] > 1:
                model_bd = sm.OLS(np.log10(y_bd), X_bd).fit()
                pvals_bd = model_bd.pvalues
                
                max_p = pvals_bd[1:].max()
                if max_p < thresh:
                    break  
                
                worst_feature = pvals_bd[1:].idxmax()
                X_bd = X_bd.drop(columns=[worst_feature])
        
            
            model_ax = sm.OLS(np.log10(y_ax), X_ax).fit()
            model_bd = sm.OLS(np.log10(y_bd), X_bd).fit()
        
            # Predictions
            y_pred_ax = 10**model_ax.predict(X_ax)
            y_pred_bd = 10**model_bd.predict(X_bd)
        
            # Get remaining predictor names
            remaining_ax = list(model_ax.pvalues.index[1:])  # Exclude intercept
            remaining_bd = list(model_bd.pvalues.index[1:])
        
            for var in remaining_ax:  
                if 'age' in var.lower():
                    idx_c = 'Age coef'
                    idx_p = 'Age p-value'
                    
                elif 'sex' in var.lower():
                    idx_c = 'Sex coef'
                    idx_p = 'Sex p-value'
                else:
                    idx_c = 'X coef'
                    idx_p = 'X p-value'
                metrics_ax.loc[idx_p, cols[i]] = model_ax.pvalues[var]
                metrics_ax.loc[idx_c, cols[i]] = model_ax.params[var]
        
            for var in remaining_bd:
                if 'age' in var.lower():
                    idx_c = 'Age coef'
                    idx_p = 'Age p-value'
                    
                elif 'sex' in var.lower():
                    idx_c = 'Sex coef'
                    idx_p = 'Sex p-value'
                else:
                    idx_c = 'X coef'
                    idx_p = 'X p-value'
                metrics_bd.loc[idx_p, cols[i]] = model_bd.pvalues[var]
                metrics_bd.loc[idx_c, cols[i]] = model_bd.params[var]
        
            # Assign F-statistic p-value
            metrics_ax.loc[ind_[6], cols[i]] = model_ax.f_pvalue
            metrics_bd.loc[ind_[6], cols[i]] = model_bd.f_pvalue
        
            # Assign mean absolute error
            metrics_ax.loc[ind_[7], cols[i]] = mean_absolute_error(y_ax, y_pred_ax)
            metrics_bd.loc[ind_[7], cols[i]] = mean_absolute_error(y_bd, y_pred_bd)
            
            if self.plot:
                min_val = min(min(y_pred_ax), min(y_ax))
                max_val = max(max(y_pred_ax), max(y_ax))
                
                plt.figure(figsize=(16, 16))
                plt.scatter(y_pred_ax, y_ax,s=420)
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3)
                
                ax = plt.gca()  # Get current axes
                ax.tick_params(axis='both', which='major', labelsize=48)
                ax.set_ylabel('Actual Fatigue Life', fontsize=48)
                ax.set_xlabel('Predicted Fatigue Life', fontsize=48)
                ax.set_xscale('log')
                ax.set_yscale('log')
                plt.title('Axial Fatigue Life vs. f(' + ', '.join(remaining_ax) + ')', fontsize=48)
                
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()
                
                #plt.text(x_lim[1] * 0.5, y_lim[0] * 1.1, f'p = {model_ax.f_pvalue:.6f}', fontsize=32, color='red')
                #plt.show()
                
                min_val = min(min(y_pred_bd), min(y_bd))
                max_val = max(max(y_pred_bd), max(y_bd))
                plt.figure(figsize=(16, 16))
                plt.scatter(y_pred_bd, y_bd,s=420)
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3)
                
                ax = plt.gca()  # Get current axes
                ax.tick_params(axis='both', which='major', labelsize=48)
                ax.set_ylabel('Actual Fatigue Life', fontsize=48)
                ax.set_xlabel('Predicted Fatigue Life', fontsize=48)
                ax.set_xscale('log')
                ax.set_yscale('log')
                plt.title('Bending Fatigue Life vs. f(' + ', '.join(remaining_bd) + ')', fontsize=48)
                
                #plt.text(x_lim[1] * 0.5, y_lim[0] * 1.1, f'p = {model_bd.f_pvalue:.6f}', fontsize=32, color='red')
                #plt.show()
                
        if self.save:
            file = self.directory + 'multivariate_fatigue_life.csv'
            metrics_combined = pd.concat([metrics_ax, metrics_bd], keys=['Axial', 'Bending'])
            metrics_combined.to_csv(file, index=True)
            
        return metrics_ax, metrics_bd
    
def get_subjects(subs: str, base_dir: str) -> List[str]:
    if subs:
        return subs.split(',')
    return [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and ('220' in d or '2108482' in d)
    ]

def process_repro(directory: str, df: pd.DataFrame, args: argparse.Namespace, subs: List[str]) -> None:
    processor = TestProcessor(directory, df, args.save, args.plot)
    mme: List[float] = []
    count: int = 0
    if args.subs:
        for sub in subs:
            sub_dir = os.path.join(directory, sub)
            if os.path.isdir(sub_dir):
                try:
                    mme.append(processor.process_all_tests(sub_dir, sub))
                    count += 1
                except Exception as e:
                    print(f"Error processing {sub}: {e}")
            else:
                print(f"Sub-directory {sub} not found in {directory}. Exiting.")
    else:
        for sub in os.listdir(directory):
            sub_dir = os.path.join(directory, sub)
            if os.path.isdir(sub_dir):
                mme.append(processor.process_all_tests(sub_dir, sub))
    if mme:
        mean_mme = sum(mme) / len(mme)
        print(f'mean minimum error: {mean_mme} mm')

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stats Doer')
    parser.add_argument('--subs', type=str, default='')
    parser.add_argument('-p', '--plot', action='store_true', default=False)
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('--redef', action='store_true', default=False)
    parser.add_argument('--repro', action='store_true', default=False)
    parser.add_argument('--stats', action='store_true', default=False)
    parser.add_argument('--mtnos', type=int, nargs='+', default=[2, 3, 4])
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    directory: str = 'Z:/_Current IRB Approved Studies/Karens_Metatarsal_Stress_Fractures/Cadaver_Data/'
    csv_file: str = os.path.join(directory, 'LHS.csv')
    lhs_data: pd.DataFrame = pd.read_csv(csv_file)
    re_search: bool = len(lhs_data) > 1

    items: List[str] = get_subjects(args.subs, directory)
    mt_map: Dict[str, int] = {'MT2': 2, 'MT3': 3, 'MT4': 4}
    main_file: str = os.path.join(directory, 'Cadaver_Loadcases.xlsx')

    load_case: pd.DataFrame = pd.read_excel(os.path.join(directory, 'Cadaver_Tracking.xlsx'), sheet_name='Load Case')
    load_case['Foot'] = load_case['Foot'].astype(str)

    doer = StatsDoer(directory, mt_map, items, args.save, args.plot)

    if args.redef:
        doer.define_loadcases(load_case, args.mtnos)

    if args.repro:
        df: pd.DataFrame = pd.read_excel(main_file)
        df.to_excel(main_file, index=False)
        process_repro(directory, df, args, items)

    if args.stats:
        if re_search:
            df: pd.DataFrame = pd.read_excel(main_file)
            df = df[(df['Fatigue Life'] > 0) & (df['Fatigue Life'] < 250000)]
            df = df[df['Angle'] == 30]
            tw_exp, vm_exp = 64.0, 10.0
            tw_str_vols: List[np.ndarray] = [np.array(t.split(',')) for t in df['Tsai-Wu Strained Volume']]
            vm_str_o_vols: List[np.ndarray] = [np.array(v.split(',')) for v in df['von Mises Stressed Volume']]
            vm_str_e_vols: List[np.ndarray] = [np.array(v.split(',')) for v in df['von Mises Strained Volume']]
            fatigue_lives: np.ndarray = df['Fatigue Life'].to_numpy()
            doer.tune_parameters(fatigue_lives, tw_str_vols, vm_str_o_vols, vm_str_e_vols, tw_exp, vm_exp, lhs_data)

        results: Dict[str, float] = doer.define_sn_curves(args.mtnos)
        if args.plot:
            doer.corr_plots()
        metrics_ax, metrics_bd = doer.multi_model()
#%%
if __name__ == "__main__":
    main([
        #'-s',
        '-p',
        #'--redef', #takes a medium time, resets to Cadaver_Tracking.xlsx
        '--repro', # takes a long time, recreates parquets in Fatigue study
        #'--stats'
        ])
