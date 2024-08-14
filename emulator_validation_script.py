import numpy as np


import os
from os import path
from glob import glob
import sys
import csv
sys.path.insert(0, path.abspath('./'))

from src import workdir, parse_model_parameter_file
from src.emulator_BAND import EmulatorBAND

def rms_abs_prediction_err(emu_pred,vali_true):
    rms_abs_pred_err = np.zeros(emu_pred.shape[1])
    for obsIdx in range(emu_pred.shape[1]):
        quantity = np.zeros(emu_pred.shape[1])
        for testpoint in range(emu_pred.shape[0]):
            quantity[obsIdx] += ((emu_pred[testpoint,obsIdx] - vali_true[testpoint,obsIdx]) / vali_true[testpoint,obsIdx])**2.
        rms_abs_pred_err[obsIdx] = np.sqrt(quantity[obsIdx] / emu_pred.shape[0])
    return rms_abs_pred_err

def how_honest_is_GP(emu_pred,emu_pred_err,vali_true):
    rms_quantity = np.zeros(emu_pred.shape[1])
    for obsIdx in range(emu_pred.shape[1]):
        quantity = np.zeros(emu_pred.shape[1])
        for testpoint in range(emu_pred.shape[0]):
            quantity[obsIdx] += ((emu_pred[testpoint,obsIdx] - vali_true[testpoint,obsIdx]) / emu_pred_err[testpoint,obsIdx])**2.
        rms_quantity[obsIdx] = np.sqrt(quantity[obsIdx] / emu_pred.shape[0])
    return rms_quantity

def train_multiple_emulators(training_set, model_par, number_test_points, logFlag, parameterTrafoPCAFlag):
    emu1 = EmulatorBAND(training_set, model_par, method='PCGP', logTrafo=logFlag, parameterTrafoPCA=parameterTrafoPCAFlag)
    emu2 = EmulatorBAND(training_set, model_par, method='PCSK', logTrafo=logFlag, parameterTrafoPCA=parameterTrafoPCAFlag)
    #emu3 = Emulator(training_set, model_par, npc = 4, logTrafo=logFlag, parameterTrafoPCA=parameterTrafoPCAFlag)

    output_emu1 = emu1.testEmulatorErrors(number_test_points=number_test_points)
    emu_pred_1 = output_emu1[0]
    emu_pred_err_1 = output_emu1[1]
    vali_data_1 = output_emu1[2]
    vali_data_err_1 = output_emu1[3]

    output_emu2 = emu2.testEmulatorErrors(number_test_points=number_test_points)
    emu_pred_2 = output_emu2[0]
    emu_pred_err_2 = output_emu2[1]
    vali_data_2 = output_emu2[2]
    vali_data_err_2 = output_emu2[3]

    # output_emu3 = emu3.testEmulatorErrors(nTestPoints=number_test_points)
    # emu_pred_3 = output_emu3[0]
    # emu_pred_err_3 = output_emu3[1]
    # vali_data_3 = output_emu3[2]
    # vali_data_err_3 = output_emu3[3]

    nObs = vali_data_1.shape[1]  # Assuming all datasets have the same number of observables

    rms_abs_pred_err1 = rms_abs_prediction_err(emu_pred_1,vali_data_1)
    rms_abs_pred_err2 = rms_abs_prediction_err(emu_pred_2,vali_data_2)
    #rms_abs_pred_err3 = rms_abs_prediction_err(emu_pred_3,vali_data_3)
    honesty_1 = how_honest_is_GP(emu_pred_1,emu_pred_err_1,vali_data_1)
    honesty_2 = how_honest_is_GP(emu_pred_2,emu_pred_err_2,vali_data_2)
    #honesty_3 = how_honest_is_GP(emu_pred_3,emu_pred_err_3,vali_data_3)

    return (rms_abs_pred_err1,rms_abs_pred_err2), (honesty_1,honesty_2)

def write_output_to_csv_uncertainties(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow([row])

def train_multiple_emulators_and_write_to_csv(training_set, model_par, number_test_points, output_file, logFlag, parameterPCAFlag):
    (rms_abs_pred_err1, rms_abs_pred_err2), (honesty_1, honesty_2) = train_multiple_emulators(training_set, model_par, number_test_points, logFlag, parameterPCAFlag)

    # Write rms_abs_pred_err1, rms_abs_pred_err2, rms_abs_pred_err3 to CSV
    write_output_to_csv_uncertainties(output_file + f'_{number_test_points}_pred_err_obs_1.dat', rms_abs_pred_err1)
    write_output_to_csv_uncertainties(output_file + f'_{number_test_points}_pred_err_obs_2.dat', rms_abs_pred_err2)


    # Write honesty_1, honesty_2, honesty_3 to CSV
    write_output_to_csv_uncertainties(output_file + f'_{number_test_points}_GP_honesty_obs_1.dat', honesty_1)
    write_output_to_csv_uncertainties(output_file + f'_{number_test_points}_GP_honesty_obs_2.dat', honesty_2)

def read_emulator_file_errors(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(float(line.strip()))
    return data

def read_multiple_emulator_errors_files(number_test_points_list,foldername,filename):
    data_list1 = []
    data_list2 = []

    for i in number_test_points_list:
        filename1 = f"{foldername}/{filename}_{i}_pred_err_obs_1.dat"
        data1 = read_emulator_file_errors(filename1)
        data_list1.append(data1)

        filename2 = f"{foldername}/{filename}_{i}_pred_err_obs_2.dat"
        data2 = read_emulator_file_errors(filename2)
        data_list2.append(data2)

        # filename3 = f"./{foldername}/{filename}_{i}_pred_err_obs_3.dat"
        # data3 = read_emulator_file_errors(filename3)
        # data_list3.append(data3)
    
    data_list4 = []
    data_list5 = []

    for i in number_test_points_list:
        filename4 = f"./{foldername}/{filename}_{i}_GP_honesty_obs_1.dat"
        data4 = read_emulator_file_errors(filename4)
        data_list4.append(data4)

        filename5 = f"./{foldername}/{filename}_{i}_GP_honesty_obs_2.dat"
        data5 = read_emulator_file_errors(filename5)
        data_list5.append(data5)

    return (data_list1,data_list2), (data_list4,data_list5)

parent="../actual"
model_par = parent+'/config_AuAu_200_bulk_scan_central.yaml'
#TODO: this does not use the validation set. fix the training set and paths, think about the number of test points

model_par = parent+'/config_AuAu_200_bulk_scan_central.yaml'
path_output_pca = parent+'/validation_PCA/'
path_output_log = parent+'/validation_LOG/'
path_output_pca_log = parent+'/validation_PCA_LOG/'
path_output = parent+'/validation_no_pca/'
data_path = parent+"/latent_train/"
if not os.path.exists(path_output):
    os.makedirs(path_output)

if not os.path.exists(path_output_pca_log):
    os.makedirs(path_output_pca_log)

if not os.path.exists(path_output_log):
    os.makedirs(path_output_log)

if not os.path.exists(path_output_pca):
    os.makedirs(path_output_pca)


training_set = ['full_base_cut_3.pkl']
output_file_list = []
for file in training_set:
    name=file.split('/')[-1].split('.p')[0]
    output_file_list.append(name)


import concurrent.futures

def train_emulator(tr_set, i, output_path, log_transform, pca):
    train_multiple_emulators_and_write_to_csv(data_path + training_set[tr_set], model_par, i, output_path + output_file_list[tr_set], log_transform, pca)

with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in [15]:
        for tr_set in range(len(training_set)):
            executor.submit(train_emulator, tr_set, i, path_output_pca, False, True)
            executor.submit(train_emulator, tr_set, i, path_output, False, False)
            executor.submit(train_emulator, tr_set, i, path_output_pca_log, True, True)
            executor.submit(train_emulator, tr_set, i, path_output_log, True, False)
