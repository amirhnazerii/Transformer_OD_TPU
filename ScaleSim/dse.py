import pygad
import numpy as np
import scalesim  # Assuming you have a way to interact with ScaLeSim in Python
from configparser import ConfigParser
import os
import pandas as pd


def fitness_function(ga_instance, solution, solution_idx):    # Extract the parameters from the solution array
    array_height = int(solution[0])
    array_width = int(solution[1])
    ifmap_sram_sz_kb = int(solution[2])
    filter_sram_sz_kb = int(solution[3])
    ofmap_sram_sz_kb = int(solution[4])
    dataflow = solution[5]  # Make sure this is handled correctly in the simulation (e.g., mapping numeric to specific dataflow types)
    bandwidth = int(solution[6])
    memory_banks = int(solution[7])

    # Configure the simulator with these parameters
    config = {
        "ArrayHeight": array_height,
        "ArrayWidth": array_width,
        "IfmapSramSzkB": ifmap_sram_sz_kb,
        "FilterSramSzkB": filter_sram_sz_kb,
        "OfmapSramSzkB": ofmap_sram_sz_kb,
        "Dataflow": dataflows[int(dataflow)],  # Assume a mapping list for numeric to string conversion
        "Bandwidth": bandwidth,
        "MemoryBanks": memory_banks
    }

    # create a name for the experiment based on the configuration
    experiment_name = f"{array_height}_{array_width}_{ifmap_sram_sz_kb}_{filter_sram_sz_kb}_{ofmap_sram_sz_kb}_{dataflow}_{bandwidth}_{memory_banks}"

    # Read the dse_results/google.cfg file using the ConfigParser
    fileparser.read('dse_results/google.cfg')

    fileparser['general']['run_name'] = experiment_name
    # use the delite parser to update the configuration
    fileparser['architecture_presets']={
        'ArrayHeight': array_height,
        'ArrayWidth': array_width,
        'IfmapSramSzkB': ifmap_sram_sz_kb,
        'FilterSramSzkB': filter_sram_sz_kb,
        'OfmapSramSzkB': ofmap_sram_sz_kb,
        'IfmapOffset':    0,
        'FilterOffset':   10000000,
        'OfmapOffset':    20000000,
        'Dataflow': dataflows[int(dataflow)],
        'Bandwidth': bandwidth,
        'MemoryBanks': memory_banks
    }
    
    # Write the configuration to a file
    with open('dse_results/google.cfg', 'w') as configfile:
        fileparser.write(configfile)

    # build the command to run the simulation
    command = f"python scalesim/scale.py -c dse_results/google.cfg -t dse_results/detr.csv -p dse_results"
    # # run the command
    os.system(command)

    # read the .csv file with the results using the experiment name as pandas dataframe
    results = pd.read_csv(f'dse_results/{experiment_name}/COMPUTE_REPORT.csv')
    # the second column has the cycles for each layer
    # let's sum all the cycles
    cycles = results.iloc[:, 1].sum()

    print(f"Experiment {experiment_name} finished with {cycles} cycles")
   
    # The fitness function should minimize the number of cycles, hence we return negative cycles as we cannot minimize in PyGAD directly
    return -cycles

# define config parser
fileparser = ConfigParser()

# Define the dataflow mapping if necessary
dataflows = ["ws", "os", "is"]  # Example dataflows

# Genetic algorithm parameters
ga_instance = pygad.GA(
    num_generations=50,  # Number of generations
    num_parents_mating=4,  # Number of solutions to be selected as parents in the mating pool
    fitness_func=fitness_function,
    sol_per_pop=10,  # Number of solutions in the population
    num_genes=8,  # Number of parameters we are optimizing

    gene_space=[
        np.arange(260, 320, 20), # ArrayHeight
        np.arange(260, 320, 20),  # ArrayWidth
        np.arange(1024.0, 10*1024.0, 1024.0),  # IfmapSramSzkB
        np.arange(1024.0, 10*1024.0, 1024.0),  # FilterSramSzkB
        np.arange(512, 5*1024.0, 512),  # OfmapSramSzkB
        range(len(dataflows)),  # Dataflow
        np.arange(1.0, 10.0, 1.0),  # Bandwidth
        range(1, 4)  # MemoryBanks
    ],

    mutation_percent_genes=10,  # Mutation percentage
    mutation_type="adaptive",  # Type of mutation
    mutation_num_genes=[2, 1]  # Set to adaptive mutation with a range from 1 to 2 genes
    
)

# Running the GA to optimize the parameters
ga_instance.run()


# Getting the best solution after the end of the evolution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}")
print("Fitness value of the best solution = {solution_fitness}")

# You may want to decode the best solution back into the readable parameters
decoded_solution = {
    "ArrayHeight": int(solution[0]),
    "ArrayWidth": int(solution[1]),
    "IfmapSramSzkB": int(solution[2]),
    "FilterSramSzkB": int(solution[3]),
    "OfmapSramSzkB": int(solution[4]),
    "Dataflow": dataflows[int(solution[5])],
    "Bandwidth": int(solution[6]),
    "MemoryBanks": int(solution[7])
}
print("Decoded solution:", decoded_solution)
