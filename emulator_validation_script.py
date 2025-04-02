import numpy as np


import os
from os import path
from glob import glob
import sys
import csv
import dill
import argparse
from data_compress import parse_outname_and_generate_file
sys.path.insert(0, path.abspath('./'))

from src import workdir, parse_model_parameter_file
from src.emulator_BAND import EmulatorBAND

def main():
    parser = argparse.ArgumentParser(description='Process training set files.')
    parser.add_argument('--training_set', nargs='+', required=True, help='List of training set files')
    args = parser.parse_args()

    training_set = args.training_set

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
        training_set_name=training_set.split('/')[-1].split('.p')[0]
        emu1 = EmulatorBAND(training_set, model_par, method='PCGP', logTrafo=logFlag, parameterTrafoPCA=parameterTrafoPCAFlag)
        emu2 = EmulatorBAND(training_set, model_par, method='PCSK', logTrafo=logFlag, parameterTrafoPCA=parameterTrafoPCAFlag)
        #emu3 = Emulator(training_set, model_par, npc = 4, logTrafo=logFlag, parameterTrafoPCA=parameterTrafoPCAFlag)

        output_emu1 = emu1.testEmulatorErrors(number_test_points=number_test_points, randomize=True)
        emu_pred_1 = output_emu1[0]
        emu_pred_err_1 = output_emu1[1]
        vali_data_1 = output_emu1[2]
        vali_data_err_1 = output_emu1[3]

        output_emu2 = emu2.testEmulatorErrors(number_test_points=number_test_points, randomize=True)
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
        logstring="No log"
        PCAstring="No PCA"
        if logFlag:
            logstring="LOG"
        if parameterTrafoPCAFlag:
            PCAstring="Parameter PCA"

        rms_abs_pred_err1 = rms_abs_prediction_err(emu_pred_1,vali_data_1)

        rms_abs_pred_err2 = rms_abs_prediction_err(emu_pred_2,vali_data_2)

        #rms_abs_pred_err3 = rms_abs_prediction_err(emu_pred_3,vali_data_3)
        honesty_1 = how_honest_is_GP(emu_pred_1,emu_pred_err_1,vali_data_1)

        honesty_2 = how_honest_is_GP(emu_pred_2,emu_pred_err_2,vali_data_2)


        return (rms_abs_pred_err1,rms_abs_pred_err2), (honesty_1,honesty_2)


    def write_output_to_csv_uncertainties(filename, data):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow([row])
            average = sum(data) / len(data)
            writer.writerow(['Average', average])

    def train_multiple_emulators_and_write_to_csv(training_set, model_par, number_test_points, output_file, logFlag, parameterPCAFlag):
        if logFlag:
            tag="_log"
        else:
            tag="_nolog"
        if parameterPCAFlag:
            tag=tag+"_pca"
        else:
            tag=tag+"_nopca"
        output_file_pred_err1 = output_file + f'_{number_test_points}_pred_err_obs_PCGP' + tag + '.dat'
        output_file_pred_err2 = output_file + f'_{number_test_points}_pred_err_obs_PCSK' + tag + '.dat'
        output_file_honesty1 = output_file + f'_{number_test_points}_GP_honesty_obs_PCGP' + tag + '.dat'
        output_file_honesty2 = output_file + f'_{number_test_points}_GP_honesty_obs_PCSK' + tag + '.dat'

        if not (os.path.isfile(output_file_pred_err1) and os.path.isfile(output_file_pred_err2) and os.path.isfile(output_file_honesty1) and os.path.isfile(output_file_honesty2)):
            (rms_abs_pred_err1, rms_abs_pred_err2), (honesty_1, honesty_2) = train_multiple_emulators(training_set, model_par, number_test_points, logFlag, parameterPCAFlag)

            # Write rms_abs_pred_err1, rms_abs_pred_err2 to CSV
            write_output_to_csv_uncertainties(output_file_pred_err1, rms_abs_pred_err1)
            write_output_to_csv_uncertainties(output_file_pred_err2, rms_abs_pred_err2)

            # Write honesty_1, honesty_2 to CSV
            write_output_to_csv_uncertainties(output_file_honesty1, honesty_1)
            write_output_to_csv_uncertainties(output_file_honesty2, honesty_2)
        else:
            print("Output files already exist. Skipping training.")

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
            filename1 = f"{foldername}/{filename}_{i}_pred_err_obs_PCGP.dat"
            data1 = read_emulator_file_errors(filename1)
            data_list1.append(data1)

            filename2 = f"{foldername}/{filename}_{i}_pred_err_obs_PCSK.dat"
            data2 = read_emulator_file_errors(filename2)
            data_list2.append(data2)

            # filename3 = f"./{foldername}/{filename}_{i}_pred_err_obs_3.dat"
            # data3 = read_emulator_file_errors(filename3)
            # data_list3.append(data3)

        data_list4 = []
        data_list5 = []

        for i in number_test_points_list:
            filename4 = f"./{foldername}/{filename}_{i}_GP_honesty_obs_PCGP.dat"
            data4 = read_emulator_file_errors(filename4)
            data_list4.append(data4)

            filename5 = f"./{foldername}/{filename}_{i}_GP_honesty_obs_PCSK.dat"
            data5 = read_emulator_file_errors(filename5)
            data_list5.append(data5)

        return (data_list1,data_list2), (data_list4,data_list5)

    parent="../actual"
    model_par = parent+'/config_AuAu_200_bulk_scan_central.yaml'

    path_output_pca = parent+'/validation_PCA/'
    path_output_log = parent+'/validation_LOG/'
    path_output_pca_log = parent+'/validation_PCA_LOG/'
    path_output = parent+'/validation_no_pca/'
    data_path = parent+"/latent_train_full/"
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    if not os.path.exists(path_output_pca_log):
        os.makedirs(path_output_pca_log)

    if not os.path.exists(path_output_log):
        os.makedirs(path_output_log)

    if not os.path.exists(path_output_pca):
        os.makedirs(path_output_pca)


    for outname in training_set:
        # Remove the file extension to get the outname
        outname = outname.replace('.pkl', '')
        parse_outname_and_generate_file(data_path, outname, 0, 750)
    output_file_list = []
    for file in training_set:
        name=file.split('/')[-1].split('.p')[0]
        output_file_list.append(name)


    import concurrent.futures

    def train_emulator(tr_set, i, output_path, log_transform, pca):
        train_multiple_emulators_and_write_to_csv(data_path + training_set[tr_set], model_par, i, output_path + output_file_list[tr_set], log_transform, pca)
        print("Validated emulator for", training_set[tr_set], "with log_transform =", log_transform, "and pca =", pca)

    print("Start Validation")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in [15]:
            for tr_set in range(len(training_set)):
                #executor.submit(train_emulator(tr_set, i, path_output, False, False))
                executor.submit(train_emulator(tr_set, i, path_output_log, True, False))
                # executor.submit(train_emulator(tr_set, i, path_output_pca, False, True))
                # executor.submit(train_emulator(tr_set, i, path_output_pca_log, True, True))
    print("\n")
    print("Validated Emulators")

# save the average errors and honesty to file
# make program to generate data pkl if not exist
# make program to generate a bunch of emulators

if __name__ == "__main__":
    main()
