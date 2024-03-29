# -*- coding: utf-8 -*-
#%% Initalize
"""
Created on Thu Sep 14 10:24:11 2023

@author: arwilzman
"""
import os
import pandas as pd
import numpy as np

def file_indexer(directory='Z:/_Current IRB Approved Studies/Jumping_Study/',
                study='JS', prefix=['S', 'F'], code=range(1, 21),trial_names=['DJump',
                'DropJump','SLJump'], trial_numbers=range(1, 21)):
    dfs = []
    for c in code:
        for p in prefix:
            a = study + '_' + p + str(c)
            height_file = os.path.join(directory, a, 'heights.txt')
            if os.path.exists(height_file):
                for name in trial_names:
                    for tno in trial_numbers:
                        grf_file = f"{directory}{a}/OpenSim_Data/GRF/{a}_{name}{tno}_grf.mot"
                        if os.path.exists(grf_file):
                            name_output = f"{a}_{name}{tno}"
                            last_float = None
                            with open(grf_file, "r") as grf:
                                # Read all lines into a list
                                grf_lines = grf.readlines()
                            # Update the first line
                            grf_lines[0] = f"name {name_output}.mot\n"
                            # Write the updated lines back to the filehi
                            with open(grf_file, "w") as grf:
                                grf.writelines(grf_lines)
                            with open(grf_file, "r") as grf:
                                for line in grf:
                                    if line.startswith("range"):
                                        values = line.strip().split()
                                        last_float = float(values[-1]) if values[-1] else None
                                        break
                            # Create a DataFrame for the current row
                            row_df = pd.DataFrame({'file': [name_output], 'endtime': [last_float]})
                            # Append the DataFrame to the list
                            dfs.append(row_df)
    # Concatenate all DataFrames in the list into a single DataFrame
    output = pd.concat(dfs, ignore_index=True)
    return output

def setuper(directory='Z:/_Current IRB Approved Studies/Jumping_Study/'):
    files = file_indexer(directory)
    bash_file = ['#!/bin/bash\n']
    for file in files['file']:
        bash_file.append('opensim-cmd run-tool '+file+'_SO_Setup.xml\n')
        bash_file.append('opensim-cmd run-tool '+file+'_Analyze_Setup.xml\n')
        endtime = str(float(files.loc[files['file'] == file, 'endtime'].iloc[0]))
        pt = '_'.join(file.split('_')[0:2])
        trial = file.split('_')[2]
        grf_content = ['<?xml version="1.0" encoding="UTF-8" ?>\n',
                       '<OpenSimDocument Version="40000">\n',
                       '\t<ExternalLoads name="externalloads">\n',
                       '\t\t<objects>\n',
                       '\t\t\t<ExternalForce name="externalforce">\n',
                       '\t\t\t\t<applied_to_body>calcn_r</applied_to_body>\n',
                       '\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>\n',
                       '\t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>\n',
                       '\t\t\t\t<force_identifier>ground_force_v</force_identifier>\n',
                       '\t\t\t\t<point_identifier>ground_force_p</point_identifier>\n',
                       '\t\t\t\t<torque_identifier>ground_torque_</torque_identifier>\n',
                       f'\t\t\t\t<data_source_name>{file}_grf.mot</data_source_name>\n',
                       '\t\t\t</ExternalForce>\n',
                       '\t\t\t<ExternalForce name="externalforce_0">\n',
                       '\t\t\t\t<applied_to_body>calcn_l</applied_to_body>\n',
                       '\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>\n',
                       '\t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>\n',
                       '\t\t\t\t<force_identifier>l_ground_force_v</force_identifier>\n',
                       '\t\t\t\t<point_identifier>l_ground_force_p</point_identifier>\n',
                       '\t\t\t\t<torque_identifier>l_ground_torque_</torque_identifier>\n',
                       f'\t\t\t\t<data_source_name>{file}_grf.mot</data_source_name>\n',
                       '\t\t\t</ExternalForce>\n',
                       '\t\t</objects>\n',
                       '\t\t<groups />\n',
                       f'\t\t<datafile>{file}_grf.mot</datafile>\n',
                       '\t</ExternalLoads>\n',
                       '</OpenSimDocument>']
        SO_content = ['<?xml version="1.0" encoding="UTF-8" ?>\n',
                      '<OpenSimDocument Version="40000">\n',
                      f'\t<AnalyzeTool name="{file}">\n',
                      f'\t\t<model_file>{pt}/OpenSim_Data/{pt}_scaled.osim</model_file>\n',
                      '\t\t<replace_force_set>false</replace_force_set>\n',
                      f'\t\t<force_set_files>{pt}/OpenSim_Data/gait2392_CMC_Actuators.xml</force_set_files>\n',
                      f'\t\t<results_directory>{pt}/OpenSim_Data/SO_Results</results_directory>\n',
                      '\t\t<output_precision>8</output_precision>\n',
                      '\t\t<initial_time>0</initial_time>\n',
                      f'\t\t<final_time>{endtime}</final_time>\n',
                      '\t\t<solve_for_equilibrium_for_auxiliary_states>true</solve_for_equilibrium_for_auxiliary_states>\n',
                      '\t\t<maximum_number_of_integrator_steps>20000</maximum_number_of_integrator_steps>\n',
                      '\t\t<maximum_integrator_step_size>1</maximum_integrator_step_size>\n',
                      '\t\t<minimum_integrator_step_size>1e-08</minimum_integrator_step_size>\n',
                      '\t\t<integrator_error_tolerance>1e-05</integrator_error_tolerance>\n',
                      '\t\t<AnalysisSet name="Analyses">\n',
                      '\t\t\t<objects>\n',
                      '\t\t\t\t<StaticOptimization name="StaticOptimization">\n',
                      '\t\t\t\t\t<on>true</on>\n',
                      '\t\t\t\t\t<start_time>0</start_time>\n',
                      f'\t\t\t\t\t<end_time>{endtime}</end_time>\n',
                      '\t\t\t\t\t<step_interval>1</step_interval>\n',
                      '\t\t\t\t\t<in_degrees>true</in_degrees>\n',
                      '\t\t\t\t\t<use_model_force_set>true</use_model_force_set>\n',
                      '\t\t\t\t\t<activation_exponent>2</activation_exponent>\n',
                      '\t\t\t\t\t<use_muscle_physiology>true</use_muscle_physiology>\n',
                      '\t\t\t\t\t<optimizer_convergence_criterion>0.0001</optimizer_convergence_criterion>\n',
                      '\t\t\t\t\t<optimizer_max_iterations>100</optimizer_max_iterations>\n',
                      '\t\t\t\t</StaticOptimization>\n',
                      '\t\t\t</objects>\n',
                      '\t\t\t<groups />\n',
                      '\t\t</AnalysisSet>\n',
                      '\t\t<ControllerSet name="Controllers">\n',
                      '\t\t\t<objects />\n',
                      '\t\t\t<groups />\n',
                      '\t\t</ControllerSet>\n',
                      f'\t\t<external_loads_file>{pt}/OpenSim_Data/GRF/{file}_grf.xml</external_loads_file>\n',
                      '\t\t<states_file />\n',
                      f'\t\t<coordinates_file>{pt}/OpenSim_Data/Input/{file}_input.mot</coordinates_file>\n',
                      '\t\t<speeds_file />\n',
                      '\t\t<lowpass_cutoff_frequency_for_coordinates>6</lowpass_cutoff_frequency_for_coordinates>\n',
                      '\t</AnalyzeTool>\n',
                      '</OpenSimDocument>']
        A_content = ['<?xml version="1.0" encoding="UTF-8" ?>\n',
                     '<OpenSimDocument Version="40000">\n',
                     f'\t<AnalyzeTool name="{file}">\n',
                     f'\t\t<model_file>{pt}/OpenSim_Data/{pt}_scaled.osim</model_file>\n',
                     '\t\t<replace_force_set>false</replace_force_set>\n',
                     f'\t\t<force_set_files>{pt}/OpenSim_Data/gait2392_CMC_Actuators.xml</force_set_files>\n',
                     f'\t\t<results_directory>{pt}/OpenSim_Results</results_directory>\n',
                     '\t\t<output_precision>8</output_precision>\n',
                     '\t\t<initial_time>0</initial_time>\n',
                     f'\t\t<final_time>{endtime}</final_time>\n',
                     '\t\t<solve_for_equilibrium_for_auxiliary_states>true</solve_for_equilibrium_for_auxiliary_states>\n',
                     '\t\t<maximum_number_of_integrator_steps>20000</maximum_number_of_integrator_steps>\n',
                     '\t\t<maximum_integrator_step_size>1</maximum_integrator_step_size>\n',
                     '\t\t<minimum_integrator_step_size>1e-08</minimum_integrator_step_size>\n',
                     '\t\t<integrator_error_tolerance>1e-05</integrator_error_tolerance>\n',
                     '\t\t<AnalysisSet name="Analyses">\n',
                     '\t\t\t<objects>\n',
                     '\t\t\t\t<JointReaction name="JointReaction">\n',
                     '\t\t\t\t\t<on>true</on>\n',
                     '\t\t\t\t\t<start_time>0</start_time>\n',
                     f'\t\t\t\t\t<end_time>{endtime}</end_time>\n',
                     '\t\t\t\t\t<step_interval>1</step_interval>\n',
                     '\t\t\t\t\t<in_degrees>true</in_degrees>\n',
                     f'\t\t\t\t\t<forces_file>{pt}/OpenSim_Data/SO_Results/{file}_StaticOptimization_force.sto</forces_file>\n',
                     '\t\t\t\t\t<joint_names> ALL</joint_names>\n',
                     '\t\t\t\t\t<apply_on_bodies> child</apply_on_bodies>\n',
                     '\t\t\t\t\t<express_in_frame> parent</express_in_frame>\n',
                     '\t\t\t\t</JointReaction>\n',
                     '\t\t\t</objects>\n',
                     '\t\t\t<groups />\n',
                     '\t\t</AnalysisSet>\n',
                     '\t\t<ControllerSet name="Controllers">\n',
                     '\t\t\t<objects>\n',
                     '\t\t\t\t<ControlSetController>\n',
                     f'\t\t\t\t\t<controls_file>{pt}/OpenSim_Data/SO_Results/{file}_StaticOptimization_controls.xml</controls_file>\n',
                     '\t\t\t\t</ControlSetController>\n',
                     '\t\t\t</objects>\n',
                     '\t\t\t<groups />\n',
                     '\t\t</ControllerSet>\n',
                     f'\t\t<external_loads_file>{pt}/OpenSim_Data/GRF/{file}_grf.xml</external_loads_file>\n',
                     '\t\t<states_file />\n',
                     f'\t\t<coordinates_file>{pt}/OpenSim_Data/Input/{file}_input.mot</coordinates_file>\n',
                     '\t\t<speeds_file />\n',
                     '\t\t<lowpass_cutoff_frequency_for_coordinates>6</lowpass_cutoff_frequency_for_coordinates>\n',
                     '\t</AnalyzeTool>\n',
                     '</OpenSimDocument>']
        with open(f"{directory}{pt}/OpenSim_Data/GRF/{file}_grf.xml", "w") as file2:
            file2.writelines(grf_content)
        with open(f"{directory}/{file}_SO_Setup.xml", "w") as file3:
            file3.writelines(SO_content)
        with open(f"{directory}/{file}_Analyze_Setup.xml", "w") as file4:
            file4.writelines(A_content)
    with open(f"{directory}/Run_StOp.sh", "w", newline='\n') as file5:
        file5.writelines(bash_file)
        
    