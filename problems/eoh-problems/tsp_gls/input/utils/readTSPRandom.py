# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# https://github.com/FeiLiu36/EoH:
# Licensed under the MIT License.
#
# ===--------------------------------------------------------------------------------------===#
import pickle as pkl

def read_instance_all(instances_path):

    # Open the pickle file in read mode
    with open(instances_path, 'rb') as file:
        # Load the data from the pickle file
        data = pkl.load(file)

    # Access the individual data elements
    coords = data['coordinate']
    optimal_tour = data['optimal_tour']
    instances = data['distance_matrix']
    opt_costs = data['cost']
    
    return coords,instances,opt_costs

