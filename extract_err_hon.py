import os
import csv

# Define the base directory where your folders are located
base_dir = '../actual'

# Initialize dictionaries to store data for errors and honesty
data_dict_errors = {}
data_dict_honesty = {}

# List the combinations to extract from the filenames
categories = ['no_pca', 'PCA', 'LOG', 'PCA_LOG']

# Traverse each folder
for category in categories:
    folder_path = os.path.join(base_dir, f'validation_{category}')

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.dat'):
            file_path = os.path.join(folder_path, filename)

            # Extract the dataset name from the filename
            dataset_name = filename.split('_15')[0] 

            # Determine the classifier from the filename
            classifier = 'PCGP' if 'PCGP' in filename else 'PCSK'

            # Determine the pca/nopca and log/nolog from the folder name
            pca_status = 'PCA' if 'PCA' in category else 'no_pca'
            log_status = 'log' if 'LOG' in category else 'nolog'

            # Read the file to get the average value
            with open(file_path, 'r') as file:
                lines = file.readlines()
                average_value = float(lines[-1].split(',')[-1].strip())  # Extract the last value after "Average,"

                # Create a key for the dataset and initialize if not already in dictionary
                if 'pred_err_obs' in filename:
                    if dataset_name not in data_dict_errors:
                        data_dict_errors[dataset_name] = {}
                    key = f'{pca_status}_{log_status}_{classifier}'
                    data_dict_errors[dataset_name][key] = average_value
                elif 'GP_honesty_obs' in filename:
                    if dataset_name not in data_dict_honesty:
                        data_dict_honesty[dataset_name] = {}
                    key = f'{pca_status}_{log_status}_{classifier}'
                    data_dict_honesty[dataset_name][key] = average_value

# Function to write data to CSV
def write_csv(data_dict, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        # Define the CSV columns
        fieldnames = ['Dataset'] + [f'{p}_{l}_{c}' for p in ['no_pca', 'PCA'] for l in ['nolog', 'log'] for c in ['PCGP', 'PCSK']]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the rows
        for dataset, values in data_dict.items():
            row = {'Dataset': dataset}
            row.update(values)
            writer.writerow(row)

# Write the data to separate CSV files
write_csv(data_dict_errors, 'validation_errors.csv')
write_csv(data_dict_honesty, 'validation_honesty.csv')
