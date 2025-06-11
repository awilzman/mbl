import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from typing import List, Dict, Optional, Tuple

class TestProcessor:
    def __init__(self,
                 directory: str,
                 load_case: pd.DataFrame,
                 save: bool = False,
                 plot: bool = False,
                 figsize: tuple[int, int] = (18, 14),
                 fontsize: int = 60,) -> None:
        
        self.directory = directory
        self.load_case = load_case
        self.save = save
        self.plot = plot
        self.figsize = figsize
        self.fontsize = fontsize
        self.cycle_hist = 0
        self.current_mt = None
        self.mean_min_err = 0
        self.count = 0
        self.max_stiff = 0.0
        self.overwrite = True
    
    def process_all_tests(self,
                          sub_directory: str,
                          sub_name: str) -> float:
        
        for side in ['L', 'R']:
            path = os.path.join(sub_directory, side, 'Tests')
            if os.path.isdir(path):
                self.process_tests(path, side, sub_name)
        return self.mean_min_err/self.count
        
    
    def process_tests(self,
                      path: str,
                      side: str,
                      sub_name: str) -> None:
        
        for item in os.listdir(path):
            
            item_path = os.path.join(path, item)
            if not os.path.isdir(item_path):
                continue

            mtno = self.identify_metatarsal(item)
            
            a = self.load_case
            condition = (a['Foot'] == sub_name) & (a['Side'] == side) & (a['Mtno'] == mtno)
            a = a.loc[condition]
            try:
                if a.iloc[0]['Fatigue Life'] == 0:
                    continue
            except:
                continue
            pred_disp = self.get_predicted_displacement(sub_name, side, mtno)
            
            try:
                pred_disp = float(pred_disp.iloc[0])
            except:
                print(f"{sub_name}_{side}_MT{mtno} needs FE")
                continue
            
            test_file = os.path.join(item_path, 'Test1', 'Test1.steps.tracking.csv')
            if not os.path.isfile(test_file):
                continue

            data = self.load_test_data(test_file)
            if data.empty:
                continue

            results = self.process_cycles(data, f'{sub_name}{side}{mtno}', pred_disp)
            if results is not None:
                hi = self.handle_results(results, sub_name, side, mtno, item)
                
                if hi is None:
                    continue
                else:
                    self.mean_min_err += hi
                    self.count += 1
    
    def identify_metatarsal(self, item: str) -> int:
        
        if 'MT2' in item:
            return 2
        if 'MT3' in item:
            return 3
        return 4  # Default to MT4
    
    def get_predicted_displacement(self,
                                   sub_name: str,
                                   side: str,
                                   mtno: int) -> Optional[pd.Series]:
        
        row = self.load_case[(self.load_case['Foot'] == sub_name) & (self.load_case['Side'] == side) & (self.load_case['Mtno'] == mtno)]
        return row.get(f'Predicted Displacement', None)
    
    def load_test_data(self, file_path: str) -> pd.DataFrame:
        
        try:
            data = pd.read_csv(file_path)
            columns = ['Time', 'Cycle Time', 'Cycle', 'eCycle', 'Step', 'Cycle Again', 'Position', 'Force', 'Pos']
            if data.shape[1] == 10:
                columns.append('na')
            data.columns = columns
            return data
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def process_cycles(self,
                       data: pd.DataFrame,
                       mtna: str,
                       pred_disp: float) -> Optional[pd.DataFrame]:
        
        if self.current_mt != mtna:
            self.current_mt = mtna
            self.reset_cycle_state()
        else:
            self.overwrite = False

        results = []
        max_cycle = data['Cycle'].max()
        for j in range(1, max_cycle + 1):
            cycle_data = data[(data['Cycle'] == j) & (data['Step'] > 1)]
            
            if cycle_data.empty:
                continue

            res = self.analyze_cycle(cycle_data, pred_disp, j)
            if res:
                results.append(res)
                
        return pd.DataFrame(results) if results else None

    def analyze_cycle(self,
                      cycle_data: pd.DataFrame,
                      pred_disp: float,
                      cycle_num: int) -> Optional[dict]:
        
        cycle_data = cycle_data.reset_index(drop=True)
        a = len(cycle_data)
        if a <= 10:
            return None
        
        max_load = cycle_data['Force'].min()
        min_load = cycle_data['Force'].max()
        
        start_idx = cycle_data['Pos'].idxmax()
        start_pos = cycle_data.loc[start_idx,'Pos']
        
        inflection_idx = cycle_data['Pos'].idxmin()
        inflx_pos = cycle_data.loc[inflection_idx, 'Pos']
        
        dis_range = start_pos - inflx_pos
        
        start_restricted = start_pos - (dis_range/2)
        end_restricted = inflx_pos + (dis_range/2)
        
        start_res_idx = cycle_data[cycle_data['Pos']<start_restricted].index[0]
        end_res_idx = cycle_data[cycle_data['Pos']<end_restricted].index[-1]
        
        restricted_data = cycle_data.iloc[start_res_idx:end_res_idx]
        
        full_range = cycle_data['Pos'].max() - cycle_data['Pos'].min()
        if full_range == 0:
            return None
        p_err = pred_disp - full_range
        
        # # Plot in debug only
        # #
        # # Uncomment this whole block and place debugs between (before and after).
        # # Then run the plt.figure and then plt.legend lines, and remove the 
        # # debug stop before this block. Now, continue for the amount of cycles 
        # # you want to show and then run plt.show()
        # # 
        # # plt.figure(figsize=(20,20))
        
        # plt.plot(restricted_data['Pos'], restricted_data['Force'] * -1000, color='blue', linewidth=8, label='Eligible Data')
        # plt.plot(cycle_data['Pos'], cycle_data['Force'] * -1000, color='fuchsia', linewidth=2, label='Full Data')
        
        # plt.xlabel('Position (mm)', fontsize=self.fontsize)
        # plt.ylabel('Force (N)', fontsize=self.fontsize)
        # plt.xticks(fontsize=self.fontsize)
        # plt.yticks(fontsize=self.fontsize)
        
        # # plt.legend(fontsize=self.fontsize, loc="lower left")
        # # plt.show()

        slope_loading, slope_unloading, dissipation, load_work = self.calculate_stiffness(
            restricted_data, inflection_idx, max_load, min_load, inflx_pos, cycle_num)
        
        if slope_loading == 0 and slope_unloading == 0:
            return None
        
        if self.initialized:
            h_rat = slope_loading/self.init_stiff
        else:
            h_rat = 0
        
        return {
            'Time': cycle_data['Cycle Time'].values[-1],
            'Cycle': cycle_num+self.cycle_hist,
            'ForceMin': max_load * 1000,
            'ForceMax': min_load * 1000,
            'PosMin': cycle_data['Pos'].min(),
            'PosMax': cycle_data['Pos'].max(),
            'Loading Stiffness': slope_loading,
            'Unloading Stiffness': slope_unloading,
            'Hardening Ratio': h_rat,
            'Stiffness Loss': self.stiff_loss,
            'Energy Dissipation': dissipation,
            'Loading Work': load_work,
            'Predicted Displacement Error': p_err
        }

    def calculate_stiffness(self,
                            data: pd.DataFrame,
                            inflx_idx: int,
                            max_load: float,
                            min_load: float,
                            inflx_pos: float,
                            cycle_num: int) -> tuple[float, float, float, float]:
        def is_valid_for_regression(subset: pd.DataFrame) -> bool:
            return subset['Pos'].nunique() > 1 and subset['Force'].nunique() > 1
        
        # Loading Stiffness
        inflx2 = data['Force'].idxmin()
        
        load_data = data.loc[:inflx2]
        if is_valid_for_regression(load_data):
            slope_load = linregress(load_data['Pos'], load_data['Force'] * 1000)[0]
        else:
            return 0, 0, 0, 0  # Exit if invalid for regression
        
        # Unloading Stiffness
        unload_data = data.loc[inflx_idx:]
        if is_valid_for_regression(unload_data):
            slope_unload = linregress(unload_data['Pos'], unload_data['Force'] * 1000)[0]
        else:
            return 0, 0, 0, 0  # Exit if invalid for regression

        Y_L = load_data['Force']
        X_L = load_data['Pos']
        Y_U = unload_data['Force']
        X_U = unload_data['Pos']

        # Calculate dissipation energies
        # hysteresis energy / total energy
        # Clean and enforce monotonicity
        # Remove non-finite values
        valid_L = np.isfinite(X_L) & np.isfinite(Y_L)
        valid_U = np.isfinite(X_U) & np.isfinite(Y_U)

        X_L, Y_L = X_L[valid_L], Y_L[valid_L]
        X_U, Y_U = X_U[valid_U], Y_U[valid_U]
        no_diss = False
        
        # Ensure enough points
        if len(X_L) < 2 or len(X_U) < 2:
            no_diss = True
            print('not enough points for energy')

        # Ensure unique and aligned
        X_L_unique, idx_L = np.unique(X_L, return_index=True)
        Y_L = Y_L.iloc[idx_L] if isinstance(Y_L, pd.Series) else Y_L[idx_L]

        X_U_unique, idx_U = np.unique(X_U, return_index=True)
        Y_U = Y_U.iloc[idx_U] if isinstance(Y_U, pd.Series) else Y_U[idx_U]

        X_L = X_L_unique
        X_U = X_U_unique
        # Shared domain
        x_min, x_max = max(X_L.min(), X_U.min()), min(X_L.max(), X_U.max())
        if x_min >= x_max:
            no_diss = True
            print('theres no overlap in loading curves')

        common_x = np.linspace(x_min, x_max, num=500)

        # Interpolation
        interp_L = interp1d(X_L, Y_L, kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_U = interp1d(X_U, Y_U, kind='linear', bounds_error=False, fill_value='extrapolate')

        Y_L_r = interp_L(common_x)
        Y_U_r = interp_U(common_x)

        # Integration
        load_diss = abs(integrate.trapezoid(Y_L_r, common_x))
        unload_diss = abs(integrate.trapezoid(Y_U_r, common_x))

        # Final check
        if load_diss == 0 or np.isnan(load_diss) or np.isnan(unload_diss):
            print('something terrible happened')
            no_diss = True

        if no_diss:
            dissipation = 0
        else:
            dissipation = (load_diss - unload_diss) / load_diss
            
        # some stiffnesses start out extrememly low, < 10, this must be the 
        # very beginning of the test or an incorrect measurement.
        if not self.initialized and slope_load > 30:
            try:
                self.init_stiff = slope_load
                self.initialized = True
            except:
                return 0, 0, 0, 0
        else:
            self.stiff_loss = (slope_load / self.init_stiff) - 1
        
        # Clamp the returned values between 0 and 1000
        slope_load = np.clip(slope_load, 0, 1000)
        slope_unload = np.clip(slope_unload, 0, 1000)
        #Loading work is currently in kN*mm which works out to N*m
        return slope_load, slope_unload, dissipation, load_diss

    def reset_cycle_state(self) -> None:
        self.overwrite = True
        self.init_point = 0
        self.stiff_loss = 0
        self.init_stiff = 0.0
        self.cycle_hist = 0
        self.initialized = False
        self.max_stiff = 0.0

    def handle_results(self,
                       results: pd.DataFrame,
                       sub_name: str,
                       side: str,
                       mtno: int,
                       sheet_name: str) -> Optional[float]:
        
        save_parquet = True
        try:
            min_err = min(abs(results[results['Cycle']<200]['Predicted Displacement Error']))
        except:
            min_err = None
        # m_err = results['Predicted Displacement Error'].mean(skipna=True)
        # m_std = results['Predicted Displacement Error'].std(skipna=True)
        try:
            b, a = butter(4, 0.1, btype='low', analog=False)
            for col in ['Loading Stiffness', 'Unloading Stiffness', 'Stiffness Loss',
                        'Energy Dissipation', 'Loading Work', 'Predicted Displacement Error', 'Hardening Ratio']:
                results[col] = filtfilt(b, a, results[col])
        except:
            save_parquet = False
            print(f"Low output in a test from {sub_name}{side}{mtno}")
            
        condition = ((self.load_case['Foot']==sub_name) & 
                     (self.load_case['Side']==side) & 
                     (self.load_case['Mtno']==mtno))
            
        fatigue_life = self.load_case[condition]['Fatigue Life'].values[0]
        exp_cols = ['Cycle','Loading Stiffness','Unloading Stiffness','Energy Dissipation']
        
        if fatigue_life >= 250000:
            fatigue_life = 500000 # 500k = LD50 as a guess for calculation
            end_cycle = 50000 #100k cycles should be very low prob
            save_data = results[results['Cycle']<end_cycle][exp_cols]
        else:
            save_data = results[results['Cycle']<fatigue_life][exp_cols]
            if len(save_data) > 0:
                if max(save_data['Loading Stiffness']) > self.max_stiff:
                    self.max_stiff = max(save_data['Loading Stiffness'])
        
        if len(save_data) == 0:
            print(f'{sub_name}_{side}_{mtno} has an empty test')
            return None
        
        data_dir = 'Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/Fatigue/Mech_Test/'
        data_file = f'{data_dir}{sub_name}_{side}_{mtno}.parquet'
        
        k = 2
        lambda_w = fatigue_life / (np.log(2) ** (1/k))
        save_data['Failure Probability'] = 1 - np.exp(-((save_data['Cycle'] / lambda_w) ** k))
        
        if self.save:
            results = self.save_results(results, sub_name, side, mtno, sheet_name)
            
            if save_parquet:
                if self.overwrite or not os.path.exists(data_file):
                    save_data.to_parquet(data_file, engine='pyarrow', index=False)
                else:
                    # Append to the existing file
                    existing_data = pd.read_parquet(data_file, engine='pyarrow')
                    combined_data = pd.concat([existing_data, save_data], ignore_index=True)
                    combined_data.to_parquet(data_file, engine='pyarrow', index=False)
            
        if self.plot:
            self.plot_results(results, sheet_name, sub_name, side)

        self.cycle_hist = results['Cycle'].values[-1]
            
        return min_err
    
    def save_results(self,
                     results: pd.DataFrame,
                     subj: str,
                     side: str,
                     mt: int,
                     sheet: str) -> pd.DataFrame:
        
        main_file = os.path.join(self.directory, 'Cadaver_Loadcases.xlsx')
        
        if os.path.exists(main_file):
            df = pd.read_excel(main_file)
            df['Initial Stiffness'] = df['Initial Stiffness'].astype(float)
            df['Maximum Stiffness'] = df['Maximum Stiffness'].astype(float)
            df['Max Hardening Ratio'] = df['Max Hardening Ratio'].astype(float)
            row = (df['Foot'] == subj) & (df['Side'] == side) & (df['Mtno'] == mt)
            
            if pd.isna(df.loc[row, 'Initial Stiffness'].values[0]) or df.loc[row, 'Initial Stiffness'].values[0] == 0:
                df.loc[row,'Initial Stiffness'] = self.init_stiff
                
            if df.loc[row, 'Maximum Stiffness'].values[0] < self.max_stiff:
                df.loc[row,'Maximum Stiffness'] = self.max_stiff
            
            if self.init_stiff > 0:
                hrat = (self.max_stiff/self.init_stiff)
            else:
                hrat = 0
            if df.loc[row, 'Max Hardening Ratio'].values[0] < hrat:
                df.loc[row,'Max Hardening Ratio'] = hrat
            df.to_excel(main_file, index=False)
        else:
            print(f"Error: {main_file} not found.")
    
        test_file = os.path.join(self.directory, subj, side, 'Tests', f'{subj}_{side}_MT{mt}.xlsx')
        
        if self.cycle_hist > 0 and os.path.exists(test_file):
            try:
                with pd.ExcelFile(test_file, engine='openpyxl') as xls:
                    existing_df = pd.read_excel(xls)
                combined_df = pd.concat([existing_df, results], ignore_index=True)
                # Write combined data back to the Excel file
                with pd.ExcelWriter(test_file, engine='openpyxl', mode='w') as writer:
                    combined_df.to_excel(writer, sheet_name=sheet, index=False)
            except Exception as e:
                print(f"Failed to save results: {e}")
        else:
            try:
                with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
                    results.to_excel(writer, sheet_name=sheet, index=False)
            except Exception as e:
                print(f"Failed to save results: {e}")
                
        return results
                    
    def plot_results(self,
                     results: pd.DataFrame,
                     sheet: str,
                     subj: str,
                     side: str) -> None:
        
        force_app = int(abs(results['ForceMin'].min()))
        mtno = self.identify_metatarsal(sheet)
        
        angle_value = self.load_case[(self.load_case['Foot'] == subj) & (
            self.load_case['Side'] == side)]['Angle'].values[0]
        
        if angle_value == 0:
            case = 'Axial'
        else:
            case = 'Bending 30 deg'
        
        first_segment = results[results['Cycle'] < 10000]
        last_segment = results[results['Cycle'] > 10000]
    
        fig_size = (40, 40)
        
        if len(last_segment) > 1:
            fig, axes = plt.subplots(2, 2, figsize=fig_size, constrained_layout=True)
        else:
            fig, axes = plt.subplots(2, 1, figsize=fig_size, constrained_layout=True)
    
        # Shared plot settings
        grid_params = dict(axis='x', linestyle='--', alpha=0.6, color='black')
    
        # First subplot (Cycle < 10000) - Stiffness Metrics
        ax = axes[0, 0] if len(last_segment) > 1 else axes[0]
        ax.plot(first_segment['Cycle'], first_segment['Loading Stiffness'], color='red', label='Loading Stiffness', linewidth=8)
        ax.plot(first_segment['Cycle'], first_segment['Unloading Stiffness'], color='maroon', label='Unloading Stiffness', linewidth=8)
        ax.set_xlabel('Cycles', fontsize=self.fontsize)
        ax.set_ylabel('Stiffness (N/mm)', fontsize=self.fontsize)
        ax.legend(fontsize=self.fontsize * 0.8)
        ax.tick_params(axis='both', labelsize=self.fontsize)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', labelsize=self.fontsize * 0.9)
    
        # Second subplot (Cycle < 10000) - Energy Dissipation
        ax = axes[1, 0] if len(last_segment) > 1 else axes[1]
        ax.plot(first_segment['Cycle'], first_segment['Energy Dissipation'], color='red', linewidth=8)
        ax.set_xlabel('Cycles', fontsize=self.fontsize)
        ax.set_ylabel('Energy Dissipation', fontsize=self.fontsize)
        ax.set_title(f'{subj} {side} MT{mtno}; {force_app}N {case}', fontsize=self.fontsize)
        ax.tick_params(axis='both', labelsize=self.fontsize)
        ax.grid(**grid_params)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', labelsize=self.fontsize * 0.9)
    
        if len(last_segment) > 1:
            # Third subplot (Cycle > 10000) - Stiffness Metrics (Log Scale)
            ax = axes[0, 1]
            ax.plot(last_segment['Cycle'], last_segment['Loading Stiffness'], color='red', label='Loading Stiffness', linewidth=8)
            ax.plot(last_segment['Cycle'], last_segment['Unloading Stiffness'], color='maroon', label='Unloading Stiffness', linewidth=8)
            ax.set_xscale('log')
            ax.set_xlabel('Cycles (log)', fontsize=self.fontsize)
            ax.tick_params(axis='both', labelsize=self.fontsize)
            ax.grid(True, which='major', **grid_params)
            ax.grid(True, which='minor', **grid_params)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='minor', labelsize=self.fontsize * 0.9)
    
            # Fourth subplot (Cycle > 10000) - Energy Dissipation (Log Scale)
            ax = axes[1, 1]
            ax.plot(last_segment['Cycle'], last_segment['Energy Dissipation'], color='red', linewidth=8)
            ax.set_xscale('log')
            ax.set_xlabel('Cycles (log)', fontsize=self.fontsize)
            ax.set_title('(Cycle > 10000)', fontsize=self.fontsize)
            ax.tick_params(axis='both', labelsize=self.fontsize)
            ax.grid(True, which='major', **grid_params)
            ax.grid(True, which='minor', **grid_params)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='minor', labelsize=self.fontsize * 0.9)
            
        if self.save:
            file_path = os.path.join(self.directory, subj, side, 'Tests', f'{subj}_{side}_MT{mtno}.png')
            plt.savefig(file_path)
            
        #plt.show()
