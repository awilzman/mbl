# -*- coding: utf-8 -*-
"""
Created on Fri May 31 08:45:44 2024

@author: arwilzman
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
import re
from statsmodels.stats.anova import AnovaRM
import scipy.stats as stats
import pingouin as pg
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import LogLocator, LogFormatter

def run_anova_to_excel(data_df, metric_cols, angle, output_file, min_rows_per_group=3):

    from matplotlib import pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")

    data_df = data_df[data_df['Injured'] != 1]

    all_anova_results = []

    for metric in metric_cols:
        print(f"Running ANOVA for {metric} at {angle}deg...")
        dv = f'{angle}deg_{metric}'
        if metric == "Max Principal Strain":
            new_dv = f'{dv}_Nf'
            data_df[new_dv] = 10 ** (-139.6 * (data_df[dv] - 0.0569))
            dv = new_dv

        problematic = data_df[['ID', 'Perturb', 'sex', dv]].copy()
        problematic['Perturb'] = pd.to_numeric(problematic['Perturb'], errors='coerce')
        problematic = problematic[problematic['Perturb'] != 0]  # exclude baseline
        grouped_data = problematic.groupby(['ID', 'Perturb', 'sex']).mean().reset_index()

        grouped_data = grouped_data.dropna(subset=[dv])
        print(f"Rows after dropping missing values: {grouped_data.shape[0]}")

        valid_groups = grouped_data.groupby('Perturb').size()
        valid_groups = valid_groups[valid_groups >= min_rows_per_group].index
        grouped_data = grouped_data[grouped_data['Perturb'].isin(valid_groups)]
        print(f"Valid groups after min row filter: {len(valid_groups)}")

        valid_ids = grouped_data.groupby('ID')[['Perturb', 'sex']].nunique()
        valid_ids = valid_ids[
            (valid_ids['Perturb'] == grouped_data['Perturb'].nunique()) &
            (valid_ids['sex'] == grouped_data['sex'].nunique())
        ].index
        grouped_data = grouped_data[grouped_data['ID'].isin(valid_ids)]
        print(f"Valid IDs after filtering: {len(valid_ids)}")

        if grouped_data.empty or grouped_data[dv].std() == 0:
            print(f"Skipping ANOVA for {metric}: no variability or empty data.")
            continue

        try:
            mauchly_result = pg.sphericity(data=grouped_data, dv=dv, within=['Perturb', 'sex'])
            pval = mauchly_result.pval
            correction = 'auto' if pval < 0.05 else None

            anova_result = pg.rm_anova(data=grouped_data, dv=dv, within=['Perturb', 'sex'],
                                       subject='ID', correction=correction)

            if isinstance(anova_result, pd.DataFrame):
                anova_result['Response_Variable'] = dv
                all_anova_results.append(anova_result)
                print(f"Results for {metric} added.")
            else:
                print(f"ANOVA result is not a DataFrame for {metric}.")

            # Optional: posthoc Tukey test (univariate)
            try:
                tukey = pg.pairwise_tukey(data=grouped_data, dv=dv, between='Perturb')
                print(tukey[['A', 'B', 'p-tukey']])
            except Exception as e:
                print(f"Tukey posthoc failed: {e}")

        except Exception as e:
            print(f"Error in ANOVA for {metric}: {e}")

    if not data_df.empty:
        data_df['Angle'] = angle
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
            data_df.to_excel(writer, sheet_name='FE_Data', index=False)
        print(f"FE data saved to {output_file}")
    else:
        print("No FE data to save.")

    if all_anova_results:
        combined = pd.concat(all_anova_results, ignore_index=True)
        combined['Angle'] = angle
        return combined
    else:
        return None

def grab_fedata(directory, items, mt_map, cols):
    data = []

    qct_cols = ['age', 'sex', 'height_cm', 'mass_kg', 'Injured', 'mt_number',
                'CSA (mm^2)', 'CSI (g^2/cm^4)', 'BSI (cm^3)', 'average Ix (mg*cm^2)',
                'average Iy (mg*cm^2)', 'average J (mg*cm^2)', 'Imaxminratio avg',
                'Imin_mass50 (mg*cm^2)', 'Imax_mass50 (mg*cm^2)', 'Jo_mass50 (mg*cm^2)',
                'Imaxminratio50', 'Imin_area50 (mm^4)', 'Imax_area50 (mm^4)',
                'Jo_area50 (mm^4)', 'ImaxminratioA50', 'iBV (cm^3)', 'iBMC (mg)',
                'iBMD (mg/cm^3)']

    qct_file = os.path.join(directory, 'QCT_results_09_20_24.xlsx')
    qct = pd.read_excel(qct_file)

    for item in items:
        file = os.path.join(directory, item, 'abaqus_files', f'{item}_fe_data.csv')
        if os.path.isfile(file):
            ram = pd.read_csv(file)
            side_data = {'L': {}, 'R': {}}

            for _, r in ram.iterrows():
                segments = r['Step Key'].split('_')
                mtno = mt_map.get(segments[2], 0)
                side = 'L' if 'L' in segments[1] else 'R'
                load = int(segments[-1][:-1]) if 'N' in segments[-1] else segments[-1]
                angle = int(30) if ('bend' in segments[3] or 'bend' in segments[4]) else 0

                perturb = 0  # default baseline
                for s in segments:
                    if re.fullmatch(r'[pn]\d+', s):
                        sign = 1 if s[0] == 'p' else -1
                        perturb = sign * int(s[1:])
                        break

                if mtno not in side_data[side]:
                    side_data[side][mtno] = {}
                if perturb not in side_data[side][mtno]:
                    side_data[side][mtno][perturb] = {
                        'ID': item,
                        'MtNo': mtno,
                        'Side': side,
                        'Perturb': perturb,
                        'Load N': load
                    }

                row = side_data[side][mtno][perturb]

                matching_rows = qct[
                    (qct['mt_number'] == mtno) &
                    (qct['Sub'].str.contains(item, case=False)) &
                    (qct['Sub'].str.contains(f'_{side}_mt', case=False))
                ]

                if matching_rows.empty:
                    candidates = qct[qct['mt_number'] == mtno]
                    candidates = candidates[candidates['Sub'].str.contains(f'_{side}_mt', case=False)]
                    candidates = candidates[candidates['Sub'].str.contains(item[:5], case=False)]
                    if not candidates.empty:
                        matching_rows = candidates.iloc[[0]]

                angle_prefix = f'{angle}deg_'

                for col in qct_cols:
                    if col in matching_rows.columns:
                        val = matching_rows[col].values
                        if val.size > 0:
                            row[col] = val[0]

                if 'sex' not in row or pd.isna(row.get('sex')):
                    sub_rows = qct[qct['Sub'].str.contains(item, case=False)]
                    
                    if 'sex' in sub_rows.columns:
                        sx = sub_rows['sex'].dropna().values
                        if sx.size > 0:
                            s = str(sx[0]).strip().upper()
                            row['sex'] = 'M' if 'M' in s else 'F' if 'F' in s else np.nan
                    sub_rows = sub_rows[sub_rows['Sub'].str.contains(f'_{side}_mt', case=False)]

                for col in cols:
                    try:
                        col_val = float(r[col])
                    except:
                        col_val = np.nan

                    row[f'{angle_prefix}{col}'] = col_val

            for side in ['L', 'R']:
                for mtno in side_data[side]:
                    for perturb in side_data[side][mtno]:
                        data.append(side_data[side][mtno][perturb])

    return pd.DataFrame(data)

def plot_fedata(directory, items, mt_map, cols, data_df,
                anova_df=None, fig_size=(8, 5), fontsize=12, save=False, perturbs_to_plot=None):
    graphs_dir = os.path.join(directory, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    perturb_labels = [-4, -3, -2, -1, 1, 2, 3, 4]

    # Calculate the global axis limits for all metrics
    global_xlim = [min(perturb_labels), max(perturb_labels)]  # Fixed for perturbation range
    global_ylim = {}  # Store global y-limits per metric

    dodge = 0.2  # Horizontal dodge value

    # Precompute global y-limits and x-limits for all metrics (for consistent scaling)
    for metric in cols:
        global_ylim[metric] = (float('inf'), float('-inf'))  # Initialize y-limits for each metric

        for mt_num in [2, 3, 4]:
            # Filter data for the current metatarsal
            mt_data = data_df[data_df['MtNo'] == mt_num]

            means = []
            stds = []
            for p in perturb_labels:
                if perturbs_to_plot and p not in perturbs_to_plot:
                    continue
                for sex in ['M', 'F']:
                    df_p = mt_data[(mt_data['Perturb'] == p) & (mt_data['sex'] == sex)]
                    col = f'30deg_{metric}'
                    if col in df_p.columns:
                        vals = pd.to_numeric(df_p[col], errors='coerce').dropna()
                        if not vals.empty:
                            means.append(vals.mean())
                            stds.append(vals.std())

            if means:
                # Adjust the global ylim by considering the mean and the error bars (std)
                global_ylim[metric] = (
                    min(global_ylim[metric][0], min([mean - std for mean, std in zip(means, stds)])),
                    max(global_ylim[metric][1], max([mean + std for mean, std in zip(means, stds)]))
                )

    # Loop through each metatarsal number (2, 3, 4) and plot for each metric
    for mt_num in [2, 3, 4]:
        # Filter data for the current metatarsal
        mt_data = data_df[data_df['MtNo'] == mt_num]

        for metric in cols:
            means = []
            stds = []
            perturb_filtered = []
            sex_filtered = []  # Track the sex for each data point

            for p in perturb_labels:
                if perturbs_to_plot and p not in perturbs_to_plot:
                    continue
                for sex in ['M', 'F']:  # Include both M and F in the same plot
                    df_p = mt_data[(mt_data['Perturb'] == p) & (mt_data['sex'] == sex)]
                    col = f'30deg_{metric}'
                    if col in df_p.columns:
                        vals = pd.to_numeric(df_p[col], errors='coerce').dropna()
                        if not vals.empty:
                            perturb_filtered.append(p)
                            means.append(vals.mean())
                            stds.append(vals.std())
                            sex_filtered.append(sex)

            # Check if we have data to plot before proceeding
            if not means:
                print(f"No data for {metric} (MtNo: {mt_num}, Perturb: {perturb_labels})")
                continue  # Skip this metric if no data

            # Create a DataFrame for easier manipulation
            grouped_data = pd.DataFrame({
                'Perturb': perturb_filtered,
                'mean': means,
                'std': stds,
                'sex': sex_filtered
            })

            # Plotting
            plt.figure(figsize=fig_size)
            ax = plt.gca()  # Get current axes

            # Loop over the sexes (M/F) and plot them separately with dodging
            for sex in ['M', 'F']:
                sex_data = grouped_data[grouped_data['sex'] == sex]

                # Adjust x-positions based on dodge value for separation
                x_offset = dodge if sex == 'F' else -dodge
                color = 'red' if sex == 'F' else 'blue'

                # Plot the main data for the metric
                ax.plot(sex_data['Perturb'] + x_offset, sex_data['mean'], marker='o', linestyle='-', color=color, label=sex)
                ax.errorbar(sex_data['Perturb'] + x_offset, sex_data['mean'], yerr=sex_data['std'],
                            fmt='o', color=color, elinewidth=2, capsize=5)

            # Fatigue Life and regression line
            if metric == 'Max Principal Strain':
                def fatigue_life(e): return 10 ** (-139.6 * (e - 0.0569))

                # Colors
                sex_colors = {'F': 'red', 'M': 'blue'}

                # === Primary Plot ===
                ax.set_title(f'{metric} (MtNo: {mt_num}, Angle: 30°)', fontsize=fontsize)
                ax.set_xlabel('Perturbation', fontsize=fontsize)
                ax.set_ylabel(metric, fontsize=fontsize)
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.tick_params(axis='both', labelsize=fontsize)
                ax.legend(title='Sex', loc='upper center', fontsize=fontsize * 0.8, bbox_to_anchor=(0.5, 1.05))
                ax.set_xlim([
                    min(global_xlim[0], min(perturb_labels) - dodge * 2),
                    max(global_xlim[1], max(perturb_labels) + dodge * 2)
                ])
                ax.set_ylim(global_ylim[metric])
                plt.tight_layout()

                if save:
                    fname = f"{metric.replace(' ', '_').lower()}_mt{mt_num}_aggregated.png"
                    plt.savefig(os.path.join(graphs_dir, fname))

                                # === Secondary Plot (Fatigue Life) ===
                fig2, ax2 = plt.subplots()

                strain_vals = grouped_data['mean'].values
                #life_vals = fatigue_life(strain_vals)
                life_vals = strain_vals

                for sex in ['M', 'F']:
                    sex_data = grouped_data[grouped_data['sex'] == sex]
                    x = sex_data['Perturb'].values
                    y = np.log10(fatigue_life(sex_data['mean'].values))

                    slope, intercept, r, p, _ = stats.linregress(x, y)
                    y_fit = 10 ** (slope * x + intercept)

                    ax2.scatter(x, 10**y, marker='x', color=sex_colors[sex], label=f'Fatigue Life - {sex}')
                    ax2.plot(x, y_fit, linestyle='--', color=sex_colors[sex], label=f'{sex} Fit')

                    print(f'{sex} Regression for Fatigue Life: Slope = {slope:.2e}, R² = {r**2:.2f}, p-value = {p:.2e}')

                ax2.set_xlabel('Perturbation', fontsize=fontsize)
                ax2.set_ylabel('Fatigue Life', fontsize=fontsize)

                # Log scale for y-axis and configure minor ticks
                ax2.set_yscale('log')
                ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))  # Set minor ticks
                ax2.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False))  # Display minor tick labels

                # Set tick parameters for both major and minor ticks
                ax2.tick_params(axis='y', which='minor', labelsize=fontsize)  # Minor ticks labeled
                ax2.tick_params(axis='both', labelsize=fontsize)  # Major ticks labeled

                # Set y-axis limits
                ax2.set_ylim(0, 2e5)

                # Add grid lines for both major and minor ticks
                ax2.grid(True, linestyle='--', alpha=0.4, which='both')

                # Adjust layout to prevent clipping
                plt.tight_layout()

                # Save the figure
                if save:
                    fname = f"{metric.replace(' ', '_').lower()}_mt{mt_num}_fatigue_life.png"
                    plt.savefig(os.path.join(graphs_dir, fname))
                
            # Customize the plot appearance
            ax.set_title(f'{metric} (MtNo: {mt_num}, Angle: 30°)', fontsize=fontsize)
            ax.set_xlabel('Perturbation', fontsize=fontsize)
            ax.set_ylabel(metric, fontsize=fontsize)
            ax.grid(True, linestyle='--', alpha=0.4)

            # Set font size for tick labels
            ax.tick_params(axis='both', labelsize=fontsize)

            # Set legend font size
            ax.legend(title='Sex', loc='upper center', fontsize=fontsize * 0.8, bbox_to_anchor=(0.5, 1.05))

            # Set global axis limits for consistent scaling across all plots, including error bars
            xlim_min = min(global_xlim[0], min(perturb_labels) - dodge*2)  # Ensure left side has room
            xlim_max = max(global_xlim[1], max(perturb_labels) + dodge*2)  # Ensure right side has room
            ax.set_xlim([xlim_min, xlim_max])
            ax.tick_params(axis='both', labelsize=fontsize)

            # Set global y-limits for the metric, including error bars
            ax.set_ylim(global_ylim[metric])

            # Adjust the layout and save if necessary
            plt.tight_layout()

            if save:
                fname = f"{metric.replace(' ', '_').lower()}_mt{mt_num}_aggregated.png"
                plt.savefig(os.path.join(graphs_dir, fname))

            plt.show()


    # --- Total Aggregated Plots ---
    for metric in cols:
        means = []
        stds = []
        perturb_filtered = []
        sex_filtered = []

        for p in perturb_labels:
            if perturbs_to_plot and p not in perturbs_to_plot:
                continue
            for sex in ['M', 'F']:
                df_p = data_df[(data_df['Perturb'] == p) & (data_df['sex'] == sex)]
                col = f'30deg_{metric}'
                if col in df_p.columns:
                    vals = pd.to_numeric(df_p[col], errors='coerce').dropna()
                    if not vals.empty:
                        perturb_filtered.append(p)
                        means.append(vals.mean())
                        stds.append(vals.std())
                        sex_filtered.append(sex)

        if not means:
            print(f"No data for combined plot of {metric} (Perturb: {perturb_labels})")
            continue

        grouped_data = pd.DataFrame({
            'Perturb': perturb_filtered,
            'mean': means,
            'std': stds,
            'sex': sex_filtered
        })

        plt.figure(figsize=fig_size)
        ax = plt.gca()

        for sex in ['M', 'F']:
            sex_data = grouped_data[grouped_data['sex'] == sex]
            x_offset = dodge if sex == 'F' else -dodge
            color = 'red' if sex == 'F' else 'blue'

            ax.scatter(sex_data['Perturb'] + x_offset, sex_data['mean'], color=color, marker='o', label=sex)
            ax.errorbar(sex_data['Perturb'] + x_offset, sex_data['mean'], yerr=sex_data['std'],
                        fmt='o', color=color, elinewidth=2, capsize=5)

        ax.set_title(f'{metric} (All Metatarsals, Angle: 30°)', fontsize=fontsize)
        ax.set_xlabel('Perturbation', fontsize=fontsize)
        ax.set_ylabel(metric, fontsize=fontsize)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(title='Sex', loc='upper center', fontsize=fontsize * 0.8, bbox_to_anchor=(0.5, 1.05))

        xlim_min = min(global_xlim[0], min(perturb_labels) - dodge*2)
        xlim_max = max(global_xlim[1], max(perturb_labels) + dodge*2)
        ax.set_xlim([xlim_min, xlim_max])

        ax.set_ylim(global_ylim[metric])
        ax.tick_params(axis='both', labelsize=fontsize)

        plt.tight_layout()

        if save:
            fname = f"{metric.replace(' ', '_').lower()}_all_mt_aggregated.png"
            plt.savefig(os.path.join(graphs_dir, fname))

        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and plot FE data.')
    parser.add_argument('--subs', type=str, default='',
                        help='Comma separated subject IDs.')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help='Whether to generate plots.')
    parser.add_argument('-s', '--save', action='store_true', default=False,
                        help='Whether to save plots.')
    parser.add_argument('-f', '--fontsize', type=int, default=20,
                        help='Font size for plots')

    args = parser.parse_args(['-s','-p'])

    directory = 'Z:/_Current IRB Approved Studies/Karens_Metatarsal_Stress_Fractures/Subject Data/'
    items = [i for i in args.subs.split(',') if i]
    date = datetime.datetime.now().strftime("_%m_%d_%y")
    
    if not items:
        items = []
        for d in os.listdir(directory):
            full_path = os.path.join(directory, d)
            if os.path.isdir(full_path):
                if 'MTSFX' in d or 'R15BSI' in d:
                    items.append(d)
                    
    mt_map = {'MT2': 2, 'MT3': 3, 'MT4': 4}
    cols = ['Tsai-Wu Strained Volume', 'von Mises Strained Volume',
            'von Mises Stressed Volume', 'Max Principal Strain',
            'Min Principal Strain', 'Max von Mises Stress']

    output_path = os.path.join(directory, 'fe_data' + date + '.xlsx')
    
    if not os.path.exists(output_path):
        # If it doesn't exist, grab FE data (assuming you have implemented `grab_fedata`)
        data_df = grab_fedata(directory, items, mt_map, cols)
        
        # Save the FE data to an Excel file
        data_df.to_excel(output_path, index=False)
    else:
        # If the file exists, load the FE data
        data_df = pd.read_excel(output_path)

    # Call the function and capture the returned result
    anova_df = run_anova_to_excel(data_df, cols, angle=30, output_file=output_path)

    # Check if the ANOVA results were returned successfully
    if anova_df is not None:
        # Optionally save the results to CSV or perform additional operations
        if args.save:
            
            output_csv = os.path.join(directory, 'anova_results' + date + '.csv')
            anova_df.to_csv(output_csv, index=False)

        if args.plot:
            plot_fedata(directory, items, mt_map, cols, data_df,
                        anova_df=anova_df, fig_size=(10, 6), fontsize=24, save=args.save)
