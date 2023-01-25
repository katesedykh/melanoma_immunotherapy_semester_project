import os
import sys
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from itertools import combinations
import networkx as nx
project_dir = str(Path(__file__).absolute().parents[2])
sys.path.append(project_dir)

import tqdm

from src.d00_utils.config import Config
from src.d00_utils.pandas_ops import *
from src.d00_utils.graph_ops import get_cells_from_edges
from src.d00_utils.clinical import get_slide_dir_paths
from src.d02_feature_extraction.feature_helper_functions import auc, g_cross_function
from src.d02_feature_extraction.tlo_features import *

# Define config & log filenames
config_filename = 'experiment_1_conf.json'
log_filename = 'experiment_1_log.txt'

# Configure logger
date_strftime_format = "%d-%m-%y %H:%M:%S"
message_format = "%(asctime)s -  %(levelname)s - %(message)s"
log_filepath = os.path.join(project_dir, 'logs', log_filename)
logging.basicConfig(
    format=message_format, datefmt=date_strftime_format, level=logging.DEBUG,
    handlers=[logging.FileHandler(log_filepath, mode='w'), logging.StreamHandler()]
)


def compute_graph_percentages(nodes_df, edges_df, config):
    phenotypes = config.get_list('phenotypes')
    regions = config.get_list('regions')
    lymphocytes = config.get_list('lymphocytes')

    if 'border' in regions:
        col_to_read = 'region'
    else:
        col_to_read = 'tissue_category'

    # Initialize features dictionary
    graph_percentages = {}

    # Calculate tissue percentage
    for region in regions:
        graph_percentages[f'cells_in_{region}'] = (nodes_df[col_to_read] == region).sum() / nodes_df.shape[0]

    # Calculate phenotype percentage
    for phenotype in phenotypes:
        graph_percentages[phenotype] = (nodes_df['phenotype'] == phenotype).sum() / nodes_df.shape[0]

    # Calculate regional phenotype percentage
    for region in regions:
        for phenotype in phenotypes:
            graph_percentages[f'{phenotype}_in_{region}'] = ((nodes_df['phenotype'] == phenotype) &
                                                             (nodes_df[col_to_read] == region)).sum() / \
                                                            (nodes_df[col_to_read] == region).sum()

    # Calculate lymphocyte infiltration percentages
    for region in regions:
        for lymphocyte in lymphocytes:
            graph_percentages[f'infil_{lymphocyte}_in_{region}'] = ((nodes_df['phenotype'] == lymphocyte) &
                                                                    (nodes_df[col_to_read] == region)).sum() / \
                                                                   (nodes_df['phenotype'] == lymphocyte).sum()

    # Calculate edge tissue percentages
    for region in regions:
        if region == 'border':
            region_mask = edges_df['on_border']
        else:
            region_mask = (edges_df['tissue_category_1'] == region) & (edges_df['tissue_category_2'] == region)
        graph_percentages[f'edges_in_{region}'] = edges_df[region_mask].shape[0] / edges_df.shape[0]

    # Calculate regional edge percentages
    for region in regions:
        if region == 'border':
            region_mask = edges_df['on_border']
        else:
            region_mask = (edges_df['tissue_category_1'] == region) & (edges_df['tissue_category_2'] == region)

        # Find unique combinations of phenotypes (since the graph is undirected)
        for phenotype_1, phenotype_2 in combinations(phenotypes, 2):
            graph_percentages[f'{phenotype_1}_{phenotype_2}_in_{region}'] = \
                (((edges_df[region_mask]['phenotype_1'] == phenotype_1) &
                  (edges_df[region_mask]['phenotype_2'] == phenotype_2)) |
                 ((edges_df[region_mask]['phenotype_1'] == phenotype_2) &
                  (edges_df[region_mask]['phenotype_2'] == phenotype_1))).sum() / edges_df[region_mask].shape[0]

    return graph_percentages


def compute_auc(edges_df, config):
    # Read config parameters
    regions = config.get_list('regions')
    auc_n_samples = int(config.get_value('auc_n_samples', as_bool=False))

    # Keep only border for experiment 3
    if regions == ['border']:
        edges_df = edges_df[edges_df['on_border']]

    # Computing AUC and adding it to nodes_stats
    min_distance = np.min(edges_df['distance'])
    max_distance = np.max(edges_df['distance'])

    distribution = g_cross_function(edges_df, min_distance, max_distance, auc_n_samples)

    return {"AUC": auc(distribution)}


def compute_tlo(nodes_df, edges_df, config):

    # Read config parameters
    a_type = config.get_value('tlo_A_type', as_bool=False)
    a_list = config.get_value('tlo_a_list', as_bool=False)
    a_list = np.array(a_list.split(',')).astype(np.int32)
    patch_dim_x = config.get_value('tlo_patch_dim_x', as_bool=False)
    patch_dim_x = float(patch_dim_x)
    patch_dim_y = config.get_value('tlo_patch_dim_y', as_bool=False)
    patch_dim_y = float(patch_dim_y)
    tlo_bins = config.get_value('tlo_bins', as_bool=False)
    min_sample_size = config.get_value('tlo_min_sample_size', as_bool=False)

    # Compute TLO features patch-wise
    tlo_results = compute_tlo_features_grid(
        cells_df=nodes_df,
        edges_df_raw=edges_df,
        a_list=a_list,
        a_type=a_type,
        patch_dim_x=patch_dim_x,
        patch_dim_y=patch_dim_y,
        min_sample_size=min_sample_size
    )

    tlo_values_patches = []
    for patch in tlo_results:
        tlo_values_patches.append(tlo_results[patch]['k_a_dict'][3])

    # Compute quantized version
    quantized, _ = np.histogram(tlo_values_patches, bins=tlo_bins, range=(0, 1))
    quantized = quantized / len(tlo_values_patches)

    tlo_quantized_dict = {}
    for i in range(tlo_bins):
        tlo_quantized_dict[f'tlo_{i}'] = quantized[i]
    return tlo_quantized_dict


def extract_exp1_features_single_graph(slide_dir_path, config):
    """
    Function that reads the preprocessed nodes and edges
    (nodes.csv) and (edges_{thresh}.csv), extracts features
    for experiment 1 and saves them to file.
    """

    slide_id = os.path.basename(slide_dir_path)

    # Read config parameters
    data_dir = config.get_path('data_dir')
    output_dir = config.get_path('output_dir')
    features_filename = config.get_value('features_filename', as_bool=False)

    # Initialize paths
    nodes_filename = 'nodes_grid_512_shift256_rot30.csv'
    edges_filename = config.get_value('edges_filename', as_bool=False)
    nodes_filepath = os.path.join(slide_dir_path, nodes_filename)
    edges_filepath = os.path.join(slide_dir_path, edges_filename)

    slide_output_dir = os.path.join(project_dir, data_dir, output_dir, slide_id)
    os.makedirs(slide_output_dir, exist_ok=True)

    # Load nodes & edges from file
    logging.info(f'==>> Loading nodes & edges for slide {slide_id}')
    nodes_df = pd.read_csv(nodes_filepath)
    edges_df = pd.read_csv(edges_filepath)

    # Initialize features dictionary
    all_features = {}

    for i in tqdm.tqdm(range(len(nodes_df.patch.unique()))): #iterate all patches
        patch_features = {}
        nodes_patch_df = nodes_df[nodes_df.patch==i]
        edges_patch_df = edges_df[edges_df.patch==i]

        #  Compute graph statistics
        logging.info('==>> Computing graph statistics')
        graph_percentages = compute_graph_percentages(nodes_patch_df, edges_patch_df, config)
        #print("--graph percentages--", graph_percentages)
        patch_features.update(graph_percentages)
        logging.info('==>> Done')

        #  Compute AUC
        logging.info('==>> Computing AUC')
        auc = compute_auc(edges_patch_df, config)
        #print("--auc--", auc)
        patch_features.update(auc)
        logging.info('==>> Done')

        #  Compute TLOs
        logging.info('==>> Computing TLOs')
        tlo_features = compute_tlo(nodes_patch_df, edges_patch_df, config)
        #print('--tlo--', tlo_features)
        patch_features.update(tlo_features)
        logging.info('==>> Done')

        all_features[f'patch_{i}'] = patch_features

    # Save features dictionary to file
    features_filepath = os.path.join(slide_output_dir, features_filename)
    logging.info(f'Saving features to {features_filepath}')
    pickle.dump(all_features, open(features_filepath, 'wb'))


if __name__ == '__main__':

    # Load config settings
    config_filepath = os.path.join(project_dir, 'config', config_filename)
    config = Config(config_filepath)

    # Get the paths of all slide directories
    slide_dir_paths = get_slide_dir_paths(project_dir, config)

    log_dict = {'failed': [], 'passed': []}
    for slide_dir_path in sorted(slide_dir_paths):
        slide_id = os.path.basename(slide_dir_path)

        logging.info('=======================================================')
        logging.info(f'EXTRACTING FEATURES FROM SLIDE {slide_id}...')

        start_time = time.time()

        try:
            extract_exp1_features_single_graph(slide_dir_path, config)
            log_dict['passed'].append(slide_id)
        except Exception as e:
            logging.error(f'Feature extraction for {slide_id} failed!!!')
            logging.error(f'Error message: {str(e)}')
            logging.info('Moving onto the next one...')
            log_dict['failed'].append(slide_id)

        end_time = time.time()

        elapsed = np.round(end_time - start_time, 2)
        logging.info(
            f'==>> Elapsed time for slide {slide_id} = {elapsed} seconds'
        )

    logging.info('=======================================================')
    logging.info('Successfully preprocessed slides:')
    logging.info(log_dict['passed'])
    logging.info('Failed to preprocess slides:')
    logging.info(log_dict['failed'])
    logging.info('=======================================================')