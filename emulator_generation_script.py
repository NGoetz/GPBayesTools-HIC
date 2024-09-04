import numpy as np


import os
from os import path
from glob import glob
import sys
import csv
import dill
from data_compress import parse_outname_and_generate_file
sys.path.insert(0, path.abspath('./'))

from src import workdir, parse_model_parameter_file
from src.emulator_BAND import EmulatorBAND

#I draw data from latent train full. Then I need: full emulator and validation emulator which has last 15 entries missing.
# folder structure: full/validation below PCGP/PCSK below log/no_log below pca/no_pca
parent="../actual"
model_par = parent+'/config_AuAu_200_bulk_scan_central.yaml'
data_path = parent+"/latent_train_full/"
#data_path_val = parent+"/latent_train_val/"



training_set = ['data_7.7_05_base_noetacut_nostarptcut.pkl', 'data_7.7_05_base-and-pT-spectra_noetacut_nostarptcut.pkl']
for outname in training_set:
    # Remove the file extension to get the outname
    outname = outname.replace('.pkl', '')
    parse_outname_and_generate_file(data_path, outname, 0, 500)
output_file_list = []
for file in training_set:
    name=file.split('/')[-1].split('.p')[0]
    output_file_list.append(name)


import concurrent.futures

def train_emulator(tr_set, pcgp, log_transform, pca):
    path_output_full=parent+'/emulator_full/'
    path_output_val=parent+'/emulator_val/'
    tag=""
    if log_transform:
        tag=tag+"_log_"
        if pca:
            tag=tag+"pca_"
            path_output_val=path_output_val+'log/pca/'
            path_output_full=path_output_full+'log/pca/'
        else:
            tag=tag+"nopca_"
            path_output_val=path_output_val+'log/no_pca/'
            path_output_full=path_output_full+'log/no_pca/'
    else:
        tag=tag+"_nolog_"
        if pca:
            tag=tag+"pca_"
            path_output_val=path_output_val+'no_log/pca/'
            path_output_full=path_output_full+'no_log/pca/'
        else:
            tag=tag+"nopca_"
            path_output_val=path_output_val+'no_log/no_pca/'
            path_output_full=path_output_full+'no_log/no_pca/'  

    if pcgp:
        path_output_full=path_output_full+'PCGP/'
        path_output_val=path_output_val+'PCGP/'
        method='PCGP'
    else:
        path_output_full=path_output_full+'PCSK/'
        path_output_val=path_output_val+'PCSK/'
        method='PCSK'
    if not os.path.exists(path_output_val):
        os.makedirs(path_output_val)
    if not os.path.exists(path_output_full):
        os.makedirs(path_output_full)
    full_emulator_path = f"{path_output_full}/{tr_set.replace('.pkl','') + tag + method}"
    val_emulator_path = f"{path_output_val}/{tr_set.replace('.pkl','') + tag + method}"

    if not os.path.isfile(full_emulator_path):
        emu1 = EmulatorBAND(data_path+tr_set, model_par, method=method, logTrafo=log_transform, parameterTrafoPCA=pca)

        trainEventMask = [True]*emu1.nev
        emu1.trainEmulator(trainEventMask)
        
        with open(full_emulator_path, 'wb') as f:
            dill.dump(emu1, f)
    else:
        print("Full Emulator already exists for", tr_set,  " with log_transform =", log_transform, " and pca =", pca, " and method =", method)

    if not os.path.isfile(val_emulator_path):
        emu2 = EmulatorBAND(data_path+tr_set, model_par, method=method, logTrafo=log_transform, parameterTrafoPCA=pca)

        trainEventMask = [True]*(emu2.nev-1)
        trainEventMask.extend([False]*1)
        emu2.trainEmulator(trainEventMask)

        with open(val_emulator_path, 'wb') as f:
            dill.dump(emu2, f)
    else:
        print("Validation Emulator already exists for", tr_set,  " with log_transform =", log_transform, " and pca =", pca, " and method =", method)
    print("Generated emulators for", tr_set,  " with log_transform =", log_transform, " and pca =", pca, " and method =", method)


training_set = ['data_7.7_05_base_noetacut_nostarptcut.pkl', 'data_7.7_05_base-and-pT-spectra_noetacut_nostarptcut.pkl']
print("Start Generation")
with concurrent.futures.ThreadPoolExecutor() as executor:
    for tr_set in range(len(training_set)):
        executor.submit(train_emulator(tr_set,  True, False, False))
        executor.submit(train_emulator(tr_set,   True, True, False))
        executor.submit(train_emulator(tr_set, True, False, True))
        executor.submit(train_emulator(tr_set,  True, True, True))
        executor.submit(train_emulator(tr_set,  False, False, False))
        executor.submit(train_emulator(tr_set,   False, True, False))
        executor.submit(train_emulator(tr_set, False, False, True))
        executor.submit(train_emulator(tr_set,  False, True, True))
print("Generation done")
