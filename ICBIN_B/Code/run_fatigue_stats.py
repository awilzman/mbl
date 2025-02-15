# -*- coding: utf-8 -*-
"""
Created on Fri May 31 08:45:44 2024

@author: arwilzman
"""
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import statsmodels.api as sm
from instron_read import TestProcessor
import seaborn as sns

class StatsDoer:
    def __init__(self, directory, bone_map, items=None, save=True, plot=False):
        self.directory = directory
        self.items = items if items else ['']
        self.save = save
        self.plot = plot
        self.bone_map = bone_map
        self.data = None

    def grab_data(self, sides=('L', 'R')):
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
                            'Tsai Wu Strained Volume': r['Tsai Wu Strained Volume'],
                            'von Mises Stressed Volume': r['von Mises Stressed Volume'],
                            'Max Tsai Wu Ratio': r['Max Tsai Wu Ratio'],
                            'Max von Mises Ratio': r['Max von Mises Ratio'],
                            'Max Displacement': r['Max Displacement']
                        })
        return pd.DataFrame(data)

    def load_computer(self, ram, target):
        X = np.array(ram['Load N']).reshape(-1, 1)
        y = abs(np.array(ram['Tsai Wu Strained Volume']).reshape(-1, 1))

        reg = LinearRegression().fit(y, X)

        max_tw = float(reg.predict([[target]]).squeeze())

        return round(max_tw)

    def define_loadcases(self, load_case, mtnos=(2, 3, 4), sides=('L', 'R')):
        data_df = self.grab_data(sides)
        load_case_final = pd.DataFrame()
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
                        load_target = self.load_computer(ram, 1500 if ram2.iloc[0]['Mag'] == 'High' else 1000)
                    
                    X = np.array(ram['Load N'],dtype=float)

                    idx = np.searchsorted(X, load_target)
                    
                    #Load target must be included in the FE load list
                    
                    vm_strvol = ram.iloc[idx]['von Mises Stressed Volume']
                    tw_strvol = ram.iloc[idx]['Tsai Wu Strained Volume']
                    pred_displ = ram.iloc[idx]['Max Displacement']
                    min_strain = ram.iloc[idx]['Min Principal Strain']
                    max_strain = ram.iloc[idx]['Max Principal Strain']
                    tw_ratio = ram.iloc[idx]['Max Tsai Wu Ratio']
                    vm_ratio = ram.iloc[idx]['Max von Mises Ratio']
                    
                    slope = load_target/pred_displ
                    
                    FL = ram2.iloc[0][f'MT{mtno} Fatigue Life']
                    C1 = ram2.iloc[0][f'MT{mtno} cyc1']
                    C2 = ram2.iloc[0][f'MT{mtno} cyc10']
                    C3 = ram2.iloc[0][f'MT{mtno} cyc100']
                    C4 = ram2.iloc[0][f'MT{mtno} cyc1000']
                    C5 = ram2.iloc[0][f'MT{mtno} cyc10000']
                    ram3 = {
                        'Foot': item,
                        'Side': side,
                        'Mtno': mtno,
                        'Mag': ram2.iloc[0]['Mag'],
                        'Angle': angle,
                        'Load': load_target,
                        'Tsai Wu Strained Volume': int(tw_strvol),
                        'von Mises Stressed Volume': int(vm_strvol),
                        'Max Principal Strain': max_strain,
                        'Min Principal Strain': min_strain,
                        'Max Tsai Wu Ratio': tw_ratio,
                        'Max von Mises Ratio': vm_ratio,
                        'Predicted Displacement': pred_displ,
                        'Predicted Stiffness': slope,
                        'Fatigue Life': FL, 
                        'cyc1': C1,
                        'cyc10': C2,
                        'cyc100': C3,
                        'cyc1000': C4,
                        'cyc10000': C5                         
                    }
                    
                    load_case_final = pd.concat((load_case_final,pd.DataFrame([ram3])))

                    if self.plot:
                        for col in ram.columns[5:]:
                            if 'Tsai Wu Strained Volume' in col:
                                plt.figure()
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
                            plt.show()

        if self.save:
            file_path = os.path.join(self.directory, 'Cadaver_Loadcases.xlsx')
            load_case_final.to_excel(file_path, index=False)
            
    def load_the_cases(self, mtnos=(2, 3, 4), sides=('L', 'R')):
        main_file = os.path.join(self.directory, 'Cadaver_Loadcases.xlsx')
        if not os.path.exists(main_file):
            print("Error: 'Cadaver_Loadcases.xlsx' not found.")
            return pd.DataFrame()
        
        self.data = pd.read_excel(main_file)
    
    def define_sn_curves(self, mtnos=(2, 3, 4), cycles=['Fatigue Life',
                                                        'cyc1','cyc10','cyc100','cyc1000','cyc10000'],
                         mags=['von Mises Stressed Volume', 'Tsai Wu Strained Volume',
                               'Max Tsai Wu Ratio','Max von Mises Ratio',
                               'Max Principal Strain','Min Principal Strain','Predicted Stiffness']):
        if isinstance(mtnos, int):
            mtnos = (mtnos,)
        self.load_the_cases(mtnos)
        
        color_map = {'Tsai Wu Strained Volume': 'fuchsia', 'von Mises Stressed Volume': 'blue', 
                     'Max Tsai Wu Ratio': 'fuchsia', 'Max von Mises Ratio': 'blue',
                     'Max Principal Strain':'red','Min Principal Strain':'blue','Predicted Stiffness':'black'}  
        res=[]
        for angle in [0,30]:
            data_group = self.data[self.data['Angle'] == angle]
            for cyc in cycles:
                
                plotted_mags = [] 
                co = 0
                for ind, mag in enumerate(mags):
                    df = data_group.dropna(subset=[cyc, mag, 'Foot', 'Side', 'Mtno'])
                    df.loc[df[cyc] == 0, cyc] = np.nan
                    df = df.dropna(subset=[cyc, mag])
            
                    if df.empty or len(df) < 3:
                        continue
            
                    if cyc == 'Fatigue Life':
                        mask = (df[cyc] < 250000) & (df[cyc] > 0)
                        X = np.log10(df.loc[mask, cyc])
                        y = np.log10(abs(df.loc[mask, mag]))
                    else:
                        mask = df[cyc] > 0
                        X = df.loc[mask, cyc]
                        y = df.loc[mask, mag]
            
                    model = sm.OLS(y, sm.add_constant(X)).fit()
                    y_pred = model.predict(sm.add_constant(X))
            
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    mae = mean_absolute_error(y, y_pred)
            
                    res.append({
                        'Angle': angle,
                        'Cycle': cyc,
                        'Magnitude': mag,
                        'Slope': model.params.iloc[1],
                        'Intercept': model.params.iloc[0],
                        'R-Squared': model.rsquared,
                        'P-Value': model.pvalues.iloc[1],
                        'RMSE': rmse,
                        'MAE': mae
                    })
                    
                    if self.plot and model.pvalues.iloc[1] < 0.1:
                        plt.figure(figsize=(20, 16))
                        c = color_map.get(mag, 'blue')  # Assign color based on magnitude
                    
                        if cyc == 'Fatigue Life':
                            X_plot = 10**X
                            y_plot = 10**y
                            y_pred_plot = 10**y_pred
                            model_label = f'log(y) = {model.params.iloc[1]:.2f}*log(x) + {model.params.iloc[0]:.2f}\n' \
                                          f'$r^2$ = {model.rsquared:.2f}, $p$ = {model.pvalues.iloc[1]:.3g}'
                        else:
                            X_plot = X
                            y_plot = y
                            y_pred_plot = y_pred
                            model_label = f'y = {model.params.iloc[1]:.2f}*x + {model.params.iloc[0]:.2f}\n' \
                                          f'$r^2$ = {model.rsquared:.2f}, $p$ = {model.pvalues.iloc[1]:.3g}'
                    
                        plt.scatter(X_plot, y_plot, color=c, s=420, edgecolor='black', marker='o')
                        plt.plot(X_plot, y_pred_plot, color=c, linestyle='dashed',
                                 label=model_label)
                        
                        for i in range(len(X_plot)):
                            label = f"{df.iloc[i]['Foot']}-{df.iloc[i]['Side']}-{df.iloc[i]['Mtno']}"
                            plt.text(X_plot.iloc[i], y_plot.iloc[i], label, fontsize=28, ha='left', va='bottom',
                                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))
                            
                        plotted_mags.append(mag)
                        if 'volume' in mag.lower():
                            plt.ylabel('Failed Volume (mm\u00B3)', fontsize=48)
                        elif 'stiff' in mag.lower():
                            plt.ylabel('Stiffness (N/mm)', fontsize=48)
                        elif 'ratio' in mag.lower():
                            plt.ylabel('Max Damaged Volume Ratio', fontsize=48)
                        else:
                            plt.ylabel('Strain (mm/mm)', fontsize=48)
                        plt.xlabel(cyc, fontsize=48)
                        plt.xscale('log' if cyc == 'Fatigue Life' else 'linear')
                        plt.yscale('log' if cyc == 'Fatigue Life' else 'linear')
                        plt.title(f'{mag}: {angle} degrees', fontsize=64)
                        plt.grid(True, which="both", ls="--", lw=0.25)
                        
                        plt.tick_params(axis='both', which='major', labelsize=48)
                        plt.tick_params(axis='both', which='minor', labelsize=48)
                        plt.legend(fontsize=16)
                
                        if self.save:
                            graph_dir = os.path.join(self.directory, 'graphs')
                            os.makedirs(graph_dir, exist_ok=True)
                            plt.savefig(os.path.join(graph_dir, f'Angle_{angle}_{cyc}_{mag}.png'), bbox_inches='tight', dpi=300)
                
                        plt.show()
                        plt.close()
        
        res_df = pd.DataFrame(res)
        
        if self.save:
            file_path = os.path.join(self.directory, 'sn_curve_results.xlsx')
            res_df.to_excel(file_path, index=False)
        
        return res_df
    
    def corr_plots(self, mags=None):
        if mags is None:
            mags = ['von Mises Stressed Volume', 'Tsai Wu Strained Volume',
                    'Max Principal Strain', 'Min Principal Strain', 
                    'Max Tsai Wu Ratio','Max von Mises Ratio',
                    'Predicted Stiffness']
        if self.data is None:
            print("Data not loaded. Run load_the_cases() or define_loadcases() first.")
            return
        df = self.data[mags].dropna()
        corr = df.corr()
    
        plt.figure(figsize=(16, 16))
        ax = sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis', annot_kws={'fontsize': 36}, cbar=True)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=28)
        
        ax.tick_params(axis='x', which='major', labelsize=28, rotation=25)
        ax.tick_params(axis='y', which='major', labelsize=28, rotation=0)
        plt.title('Correlation Heatmap', fontsize=36)
        ax.set_xticklabels(ax.get_xticklabels(), ha='right', position=(0.05, 0))
        
        if self.save:
            file_path = os.path.join(self.directory, 'correlations.png')
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.show()
        
        # Pairplot
        g = sns.pairplot(df)
        for ax in g.axes.flatten():
            if ax is not None:
                ax.tick_params(axis='x', which='major', labelsize=20)
                ax.tick_params(axis='y', which='major', labelsize=20)
    
        g.fig.suptitle('Pairwise Correlation Plot', fontsize=36, y=1.02)
        for ax in g.axes.flatten():
            ax.set_xlabel(ax.get_xlabel(), fontsize=26, rotation=25, ha='right')
            ax.set_ylabel(ax.get_ylabel(), fontsize=26, rotation=25, labelpad=140)
        if self.save:
            file_path = os.path.join(self.directory, 'scatter_correlations.png')
            g.fig.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.show()
        
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stats Doer')
    parser.add_argument('--subs', type=str, default='', help='Comma separated IDs')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='generate plots')
    parser.add_argument('-s', '--save', action='store_true', default=False, help='save data')
    parser.add_argument('--redef', action='store_true', default=False, help='redefine loadcases')
    parser.add_argument('--repro', action='store_true', default=False, help='reprocess instron data')
    parser.add_argument('--stats', action='store_true', default=False, help='perform stats')
    parser.add_argument('--mtnos', type=int, default=(2,3,4), help='MT numbers to analyze')
    args = parser.parse_args([
        '-s',
        '-p',
        #'--redef',
        '--repro',
        '--stats'
        ])

    directory = 'Z:/_Current IRB Approved Studies/Karens_Metatarsal_Stress_Fractures/Cadaver_Data/'
    
    if args.subs:
        items = args.subs.split(',')
    else:
        items = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and ('220' in d or '2108482' in d)]
   
    mt_map = {'MT2': 2, 'MT3': 3, 'MT4': 4}
    
    load_case = pd.read_excel(f'{directory}/Cadaver_Tracking.xlsx', sheet_name='Load Case')
    load_case['Foot'] = load_case['Foot'].astype(str)
    
    doer = StatsDoer(directory, mt_map, items, args.save, args.plot)
    
    if args.redef:
        doer.define_loadcases(load_case,args.mtnos)
    
    if args.repro:
        main_file = os.path.join(directory, 'Cadaver_Loadcases.xlsx')
        df = pd.read_excel(main_file)
        df.loc[:, df.columns.str.startswith("cyc")] = 0
        df.to_excel(main_file, index=False)
        processor = TestProcessor(
            directory=directory,
            load_case=df,
            save=args.save,
            plot=args.plot)
        mme = []
        if args.subs:
            for sub in args.subs.split(','):
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
            # Process all subjects
            for sub in os.listdir(directory):
                sub_dir = os.path.join(directory, sub)
                if os.path.isdir(sub_dir):
                    mme.append(processor.process_all_tests(sub_dir, sub))
        mean_mme = sum(mme)/len(mme)
        print(f'mean minimum error: {mean_mme} mm')
    
    if args.stats:
        
        results = doer.define_sn_curves(args.mtnos)
        
        doer.corr_plots()