import os
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
import yaml

def generate_run_dir_patterns(grid, experiment_name):
    """
    Generate expected run directory patterns based on the grid search parameters.

    Parameters:
    - grid (dict): A dictionary containing the grid search parameters.
    - experiment_name (str): The name of the experiment.

    Returns:
    - run_dir_patterns (list): A list of directory patterns for the runs.
    """
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    run_dir_patterns = []
    
    for experiment in experiments:
        parts = [f"{key}{value}" for key, value in experiment.items() if key != 'seed']
        if 'seed' in keys:
            dir_pattern = f"{experiment_name}_" + "_".join(parts) + f"_seed{experiment['seed']}_*"
        else:
            dir_pattern = f"{experiment_name}_" + "_".join(parts) + "_*"
        run_dir_patterns.append(dir_pattern)
    
    return run_dir_patterns

def get_best_model_per_seed(parent_dir, seeds, grid_params, run_patterns):
    """
    Get the best model per seed based on the highest median NSE score.

    Args:
        parent_dir (str): Parent directory containing the model run directories.
        seeds (list): List of seed values.
        grid_params (list): List of grid parameters.
        run_patterns (list, optional): List of run directory patterns. Defaults to None.

    Returns:
        dict: Dictionary containing the best model directory, best NSE score, best epoch, and best model config for each seed.
    """
    best_models_per_seed = {}

    # Get full run directories based on the run directory patterns
    run_dirs = []
    for run_pattern in run_patterns:
        run_dirs.extend(Path(parent_dir).rglob(run_pattern))
        
    # Iterate over each seed
    for seed in seeds:
        print(f"Evaluating repetition for seed {seed}.")
        
        best_median_nse = -float('inf')
        best_model = None
        best_model_config = None
        
        # Filter run directories for the current seed
        run_dirs_seed = [run_dir for run_dir in run_dirs if f'seed{seed}' in str(run_dir)]
        
        for model_dir in run_dirs_seed:
            model_path = str(model_dir)
            config_path = os.path.join(model_path, 'config.yml')
            validation_path = os.path.join(model_path, 'validation')

            if os.path.exists(config_path) and os.path.exists(validation_path):
                # Read the config file
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)

                # Collect all NSE scores across all epochs
                nse_scores = []

                # Get the list of epoch directories and sort them
                epoch_dirs = sorted(os.listdir(validation_path))

                # Iterate through all validation epochs
                for epoch_dir in epoch_dirs:
                    validation_metrics_path = os.path.join(validation_path, epoch_dir, 'validation_metrics.csv')
                    if os.path.exists(validation_metrics_path):
                        # Read the validation metrics file
                        metrics_df = pd.read_csv(validation_metrics_path)
                        nse_scores.extend(metrics_df['NSE'].tolist())
                
                # Get the highest median NSE score across all validation epochs
                if nse_scores:
                    median_nse = pd.Series(nse_scores).median()
                    
                # Check if this is the best NSE score for the current seed
                if median_nse > best_median_nse:
                    best_median_nse = median_nse
                    best_epoch = (np.argmax(nse_scores)+1)*config['validate_every']
                    best_model = model_dir
                    best_model_config = config
        
        best_models_per_seed[seed] = {
            'best_model_dir': best_model,
            'best_median_nse': best_median_nse,
            'best_epoch': best_epoch,
            'best_model_config': best_model_config
        }
    print('Finished.')
    
    return best_models_per_seed

def get_best_model(parent_dir, grid_params, run_patterns):
    """
    Get the best model based on the highest median NSE score.

    Args:
        parent_dir (str): Parent directory containing the model run directories.
        grid_params (list): List of grid parameters.
        run_patterns (list, optional): List of run directory patterns. Defaults to None.

    Returns:
        dict: Dictionary containing the best model directory, best NSE score, best epoch, and best model config for each seed.
    """
    best_model = {}

    # Get full run directories based on the run directory patterns
    run_dirs = []
    for run_pattern in run_patterns:
        run_dirs.extend(Path(parent_dir).rglob(run_pattern))
        
    # Iterate over each seed
    print(f"Evaluating.")
    
    best_median_nse = -float('inf')
    best_model = None
    best_model_config = None
    
    for model_dir in run_dirs:
        model_path = str(model_dir)
        config_path = os.path.join(model_path, 'config.yml')
        validation_path = os.path.join(model_path, 'validation')
        if os.path.exists(config_path) and os.path.exists(validation_path):
            # Read the config file
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            # Collect all NSE scores across all epochs
            nse_scores = []
            # Get the list of epoch directories and sort them
            epoch_dirs = sorted(os.listdir(validation_path))
            # Iterate through all validation epochs
            for epoch_dir in epoch_dirs:
                validation_metrics_path = os.path.join(validation_path, epoch_dir, 'validation_metrics.csv')
                if os.path.exists(validation_metrics_path):
                    # Read the validation metrics file
                    metrics_df = pd.read_csv(validation_metrics_path)
                    nse_scores.extend(metrics_df['NSE'].tolist())
            # find the best epoch
            best_epoch = np.argmax(nse_scores)
            
            # Get the highest median NSE score across all epochs
            if nse_scores:
                median_nse = pd.Series(nse_scores).median()
                
                # Check if this is the best NSE score for the current seed
                if median_nse > best_median_nse:
                    best_median_nse = median_nse
                    best_model = model_dir
                    best_model_config = config
    
    best_model = {
        'best_model_dir': best_model,
        'best_median_nse': best_median_nse,
        'best_epoch': best_epoch,
        'best_model_config': best_model_config
    }
    print('Finished.')
    
    return best_model