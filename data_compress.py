import numpy as np


import os
from os import path
from glob import glob
import sys
import csv
import dill
import pickle
sys.path.insert(0, path.abspath('./'))


import pandas as pd
import numpy as np

####################
#GLOBAL VARIABLES
num_points=500
#num_points=4
####################


parent=path.abspath('../actual/')
#generate the data pickles
path_data = parent+'/merged_directory'
path_configs = parent+'/configs'
path_output = parent+'/latent_pickled/'
path_output_exp = parent+'/exp/'
path_data_exp=parent+'/exp_input/0'
datasets = [
    'exp_19.6_05_eta_spectra', 'exp_200_05_y_spectra_piminus',
    'exp_19.6_05_integrated', 'exp_200_05_y_spectra_piplus',
    'exp_19.6_05_pT_spectra_kminus', 'exp_200_1525_eta_spectra',
    'exp_19.6_05_pT_spectra_kplus', 'exp_200_1525_phobos_v2_spectra',
    'exp_19.6_05_pT_spectra_p', 'exp_200_2030_integrated',
    'exp_19.6_05_pT_spectra_pbar', 'exp_200_2030_phenix_pT_v2_spectra',
    'exp_19.6_05_pT_spectra_piminus', 'exp_200_2030_phenix_pT_v3_spectra',
    'exp_19.6_05_pT_spectra_piplus', 'exp_200_2030_pT_spectra_kminus',
    'exp_19.6_1525_eta_spectra', 'exp_200_2030_pT_spectra_kplus','exp_200_2030_pT_spectra_p',
    'exp_19.6_2030_integrated', 'exp_200_2030_pT_spectra_pbar',
    'exp_19.6_2030_pT_spectra_kminus', 'exp_200_2030_pT_spectra_piminus',
    'exp_19.6_2030_pT_spectra_kplus', 'exp_200_2030_pT_spectra_piplus',
    'exp_19.6_2030_pT_spectra_p', 'exp_7.7_05_integrated',
    'exp_19.6_2030_pT_spectra_pbar', 'exp_7.7_05_pT_spectra_kminus',
    'exp_19.6_2030_pT_spectra_piminus', 'exp_7.7_05_pT_spectra_kplus',
    'exp_19.6_2030_pT_spectra_piplus', 'exp_7.7_05_pT_spectra_p',
    'exp_19.6_2030_star_v2_pT_spectra', 'exp_7.7_05_pT_spectra_pbar',
    'exp_200_05_eta_spectra', 'exp_7.7_05_pT_spectra_piminus',
    'exp_200_05_integrated', 'exp_7.7_05_pT_spectra_piplus',
    'exp_200_05_pT_spectra_kminus', 'exp_7.7_2030_integrated',
    'exp_200_05_pT_spectra_kplus', 'exp_7.7_2030_pT_spectra_kminus',
    'exp_200_05_pT_spectra_p', 'exp_7.7_2030_pT_spectra_kplus',
    'exp_200_05_pT_spectra_pbar', 'exp_7.7_2030_pT_spectra_p',
    'exp_200_05_pT_spectra_piminus', 'exp_7.7_2030_pT_spectra_pbar',
    'exp_200_05_pT_spectra_piplus', 'exp_7.7_2030_pT_spectra_piminus',
    'exp_200_05_y_spectra_kminus', 'exp_7.7_2030_pT_spectra_piplus',
    'exp_200_05_y_spectra_kplus', 'exp_7.7_2030_star_v2_pT_spectra'
]
integrated_values=[
    "dNdy_kminus",
    "dNdy_kplus",
    "dNdy_p",
    "dNdy_pbar",
    "dNdy_piminus",
    "dNdy_piplus",
    "meanpT_kminus",
    "meanpT_kplus",
    "meanpT_p",
    "meanpT_pbar",
    "meanpT_piminus",
    "meanpT_piplus",
    "star_v2",
    "star_v3"
]
def read_config(file):
    with open(file, 'r') as f:
        # Initialize an empty dictionary to store the values
        values = []
        # Read the file line by line
        for line in f:
            # If the line starts with a '#', it's a comment
            if line.startswith('#'):
                # Split the line at the colon
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                # The key is the part before the colon, stripped of leading/trailing whitespace and the '#'
                key = parts[0].strip().lstrip('#')
                # The value is the part after the colon, stripped of leading/trailing whitespace
                value = float(parts[1].strip())
                # Add the key-value pair to the dictionary
                values.append(value)
    return values

def pickle_data_in_from_to(path_output, start, end, eta_cut=None, star_pT_cut=None, data="all", system=None, energy=None, exp=False):
    dict_df_full = {}
    datasets_local = datasets.copy()
    if exp:
        start = 0
        end = 1
    if system is not None:
        if isinstance(system, list):
            datasets_local = [d for d in datasets_local if any(s in d for s in system)]
        else:
            datasets_local = [d for d in datasets_local if system in d]
    
    if energy is not None:
        if isinstance(energy, list):
            datasets_local = [d for d in datasets_local if any(e in d for e in energy)]
        else:
            datasets_local = [d for d in datasets_local if energy in d]
    match data:
        case "base-wo-phobos":
            datasets_local = [d for d in datasets_local if ("phobos" not in d and "eta" not in d and "y_spectra" not in d and "pT_spectra" not in d and "phenix" not in d and "star_v2_pT_spectra" not in d)]
        case "base":
           datasets_local = [d for d in datasets_local if ("y_spectra" not in d and "pT_spectra" not in d and "phenix" not in d and "star_v2_pT_spectra" not in d)]
        case "base-and-pT-spectra":
           datasets_local = [d for d in datasets_local if ("y_spectra" not in d and "phenix" not in d and "star_v2_pT_spectra" not in d)] 
        case "base-and-y-spectra":
            datasets_local = [d for d in datasets_local if ( "pT_spectra" not in d and "phenix" not in d and "star_v2_pT_spectra" not in d)]
        case "base-and-phenix":
            datasets_local = [d for d in datasets_local if ("y_spectra" not in d and "pT_spectra" not in d  and "star_v2_pT_spectra" not in d)]
        case "base-and-v2pT":
            datasets_local = [d for d in datasets_local if ("y_spectra" not in d and "pT_spectra" not in d and "phenix" not in d)  or ( "star_v2_pT_spectra" in d )]
        case "base-and-phenix-and-v2pT":
            datasets_local = [d for d in datasets_local if ("y_spectra" not in d and "pT_spectra" not in d ) or ( "star_v2_pT_spectra" in d )]
        case "base-and-phenix-and-pT":
            datasets_local = [d for d in datasets_local if ("y_spectra" not in d and "star_v2_pT_spectra" not in d )]
        case "base-and-v2pT-and-pT":
            datasets_local = [d for d in datasets_local if ("y_spectra" not in d and "phenix" not in d )]
        case "base-and-all-but-y":
            datasets_local = [d for d in datasets_local if ("y_spectra" not in d )]
        case "base-and-y-and-pT-spectra":
            datasets_local = [d for d in datasets_local if ("phenix" not in d and "star_v2_pT_spectra" not in d)]
        case "all":
            pass
        case _:
            raise ValueError("Unexpected case encountered")
    for i in range(start, end):
        i_str = str(i+1).zfill(3)
        dict_df={"obs": [], "parameter": [], "name":[], "lim":[0]}
        config_path = path_configs +"/config_sets_run_"+str(i+1).zfill(3)+".yaml"
        dict_df["parameter"]=read_config(config_path)
        obs_1 = []  # List for first entries of tuples
        obs_2 = []  # List for second entries of tuples
        for dataset in datasets_local:
            if "integrated" not in dataset:

                if not exp:
                    current_path = path_data +"/"+i_str+"/"+dataset.replace("exp", i_str)
                else:
                    current_path = path_data_exp +"/"+i_str+"/"+dataset
                
                df = pd.read_csv(current_path)
                if eta_cut is not None and ("eta" in dataset or "y_spectra" in dataset or "phobos" in dataset):
                    df_filtered = df[abs(df.iloc[:, -3]) < eta_cut]
                    new_obs_1 = df_filtered.iloc[:, -2].tolist()
                    new_obs_2 = df_filtered.iloc[:, -1].tolist()
                    obs_1.extend(new_obs_1)
                    obs_2.extend(new_obs_2)
                    dict_df["lim"].append(df_filtered.shape[0] + dict_df["lim"][-1])
                elif star_pT_cut is not None and "star_v2_pT_spectra" in dataset:
                    df_filtered = df[abs(df.iloc[:, -3]) < star_pT_cut]
                    new_obs_1 = df_filtered.iloc[:, -2].tolist()
                    new_obs_2 = df_filtered.iloc[:, -1].tolist()
                    obs_1.extend(new_obs_1)
                    obs_2.extend(new_obs_2)
                    dict_df["lim"].append(df_filtered.shape[0] + dict_df["lim"][-1])
                else:
                    new_obs_1 = df.iloc[:, -2].tolist()
                    new_obs_2 = df.iloc[:, -1].tolist()
                    obs_1.extend(new_obs_1)
                    obs_2.extend(new_obs_2)
                    dict_df["lim"].append(df.shape[0] + dict_df["lim"][-1])

                # Check for NaNs in the newly added part of obs_1 and print a warning if any are found
                if any(np.isnan(new_obs_1)):
                    print(f"Warning: NaN found in obs_1 for dataset {dataset} with parameter {dict_df['parameter']}")
                dict_df["name"].append(dataset)
                
            elif "integrated" in dataset:
                for key in integrated_values:
                    if not exp:
                        current_path = path_data +"/"+i_str+"/"+dataset.replace("exp", i_str)
                    else:
                        current_path = path_data_exp +"/"+i_str+"/"+dataset 
                    df = pd.read_csv(current_path)
                    obs_1.extend(df[key].values)
                    obs_2.extend(df[key+"_error"].values)
                    dict_df["name"].append(dataset+"_"+key)
                    dict_df["lim"].append(df.shape[0]+dict_df["lim"][-1])
        dict_df["obs"] = np.vstack((obs_1, obs_2))
        dict_df_full[i]=dict_df

    energy_str = "_".join(energy) if energy else "allenergies"
    system_str = "_".join(system) if system else "allsystems"
    eta_cut_str = str(eta_cut) if eta_cut is not None else "noetacut"
    star_pT_cut_str = str(star_pT_cut) if star_pT_cut is not None else "nostarptcut"
    
    outname = f"data_{energy_str}_{system_str}_{data}_{eta_cut_str}_{star_pT_cut_str}"
    if not exp:
        with open( path_output+outname+".pkl", 'wb') as f:
            pickle.dump(dict_df_full, f)
    else:
        with open( path_output_exp+outname+".pkl", 'wb') as f:
            pickle.dump(dict_df_full, f)


def parse_outname_and_generate_file(path_output, outname, start, end, exp=False):
    # Split the outname into its components
    parts = outname.split('_')
    
    # Extract the components
    energy_str = parts[1]
    system_str = parts[2]
    data = parts[3]
    eta_cut_str = parts[4]
    star_pT_cut_str = parts[5]
    
    # Convert the components to the appropriate types
    energy = energy_str.split('_') if energy_str != "allenergies" else None
    system = system_str.split('_') if system_str != "allsystems" else None
    eta_cut = float(eta_cut_str) if eta_cut_str != "noetacut" else None
    star_pT_cut = float(star_pT_cut_str) if star_pT_cut_str != "nostarptcut" else None
    
    # Check if the file already exists
    filepath = os.path.join(path_output, outname + ".pkl")

    if not os.path.exists(filepath):
        # Call the original function with the parsed arguments
        pickle_data_in_from_to(path_output, start, end, eta_cut, star_pT_cut, data, system, energy, exp)
    else:
        print(f"File {filepath} already exists.")



