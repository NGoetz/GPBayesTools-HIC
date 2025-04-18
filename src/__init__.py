""" Project initialization and common objects. """

import logging
import os
from pathlib import Path
import re
import sys
import yaml


logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)

workdir = Path(os.getenv('WORKDIR', '.'))

cachedir = workdir / 'cache'
cachedir.mkdir(parents=True, exist_ok=True)


# def parse_model_parameter_file(parfile):
#     pardict = {}
#     f = open(parfile, 'r')
#     for line in f:
#         par = line.split("#")[0]
#         if par != "":
#             par = par.split(":")
#             key = par[0]
#             val = [ival.strip() for ival in par[1].split(",")]
#             for i in range(1, 3):
#                 val[i] = float(val[i])
#             pardict.update({key: val})
#     return pardict



def parse_model_parameter_file(file):
    with open(file, 'r') as f:
        data = yaml.safe_load(f)

    pardict = {}
    for software in data:
        if 'Scan_parameters' in data[software]:
            parameters = data[software]['Scan_parameters']
            sorted_parameters = sorted(parameters, key=lambda s: s.lower())  # Sort parameters within this section
            for parameter in sorted_parameters:
                keys = parameter.split('.')
                temp_data = data[software]['Software_keys']
                for key in keys:
                    if key in temp_data:
                        temp_data = temp_data[key]
                    else:
                        break
                else:
                    if 'Scan' in temp_data:
                        range_list = temp_data['Scan']['Range']
                        range_list.insert(0, 0.0)  # Add a zero to the beginning of the list
                        pardict[parameter] = range_list

    # Create a sorted dictionary for each section based on section order
    sorted_pardict = {}
    for section in data:
        print(section)
        if 'Scan_parameters' in data[section]:
            # Sort parameters within the section but keep the section order
            section_parameters = {param: pardict[param] for param in sorted(data[section]['Scan_parameters'],key=lambda s: s.lower()) if param in pardict}
            sorted_pardict.update(section_parameters)

    print(sorted_pardict)
    return sorted_pardict
