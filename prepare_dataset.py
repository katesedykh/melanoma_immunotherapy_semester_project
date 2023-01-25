import pickle
import pandas as pd
import numpy as np
import networkx as nx 
import random 
from tqdm import tqdm
import os
from collections import defaultdict

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from torch.utils.data import Dataset,TensorDataset,random_split,SubsetRandomSampler


FEATURES_DIR_PATH = '/home/jovyan/work/melanoma_immunotherapy/output/02_features'
DATA_PER_SLIDE_SAVEDIR = '/home/jovyan/work/melanoma_immunotherapy/output/final_features'

    
def get_response_dict(df, key_column, response_colum):
    slide_ids = list(df[key_column])
    responses = list(df[response_colum])

    slide_response_dict = {}
    for slide_id, response in zip(slide_ids, responses):
        slide_response_dict[slide_id] = response
    
    return slide_response_dict


def get_responses_to_labels_map(map_to_include=dict()):

    mapping = {
        'y': 1,    
        'n': 0,   
    }

    for key, val in map_to_include.items():
        mapping[key] = val

    mapping = defaultdict(lambda: -1, mapping)

    return mapping


def map_responses_to_labels(responses_dict, map_to_include):
    labels_dict = responses_dict.copy()
    mapping = get_responses_to_labels_map(map_to_include)
    for key, val in responses_dict.items():
        labels_dict[key] = mapping[val]
    
    return labels_dict



def get_slide_label_map(map_to_include=dict()):
    clindata_df_path = os.path.join("/home/sedykh/CHUVdata/clinical_data/deepMEL_clindata.2.4_noIPP.xlsx")
    clindata_df = pd.read_excel(clindata_df_path)

    slide_response_dict = get_response_dict(clindata_df, 'SLIDE ID', 'LONGTERMPFS_730D')
    slide_label_dict = map_responses_to_labels(
        slide_response_dict, map_to_include
        )

    return slide_label_dict
    

def get_grid_level_graph_df(cells_df, patch_col="patch"):
    try:
        cells_df = cells_df.set_index('cell_id')
    except KeyError:
        cells_df = cells_df.reset_index()
        cells_df = cells_df.set_index('cell_id')

    grid_level_graph_dict = {}
    for patch_id in cells_df[patch_col].unique():
        grid_level_graph_dict[patch_id] = {}
        
        grid_level_graph_dict[patch_id]['x'] = cells_df[
            cells_df[patch_col] == patch_id
            ]['patch_loc_x'] #maybe real??----------------------------------------------------------------------------

        grid_level_graph_dict[patch_id]['y'] = cells_df[
            cells_df[patch_col] == patch_id
            ]['patch_loc_y'] 

    grid_level_graph_df = pd.DataFrame(grid_level_graph_dict)\
        .transpose().reset_index().rename(
            columns={
            'index': 'cell_id', 'x': 'cell_x_position', 'y': 'cell_y_position'
            }
        )

    return grid_level_graph_df


def get_graph_from_grids_df(patches_nodes_df, patches_edges_df):
    patches_edges_tuples = []
    for i, row in patches_edges_df.iterrows():
        patches_edges_tuples.append(
            (row['node_id_1'], row['node_id_2'])
            )

    G_patches = nx.Graph()
    for i, row in patches_edges_df.iterrows(): #######------nodes-------------------------
        G_patches.add_node(
            #row['patch'], 
            #x=row['patch_loc_x'], y=row['patch_loc_y']
            row['node_id_1'], 
            x=row['patch_loc_x_1'], y=row['patch_loc_y_1']
            )

    G_patches.add_edges_from(patches_edges_tuples)
    print(G_patches)
    return G_patches


def get_data_per_slide(features_dir_path, suffix): 
    
    slide_dirs_abs_paths = []
    for slide_dir in os.listdir(features_dir_path): #--------------------------------------------------------------------------
        slide_dir_abs_path = os.path.join(features_dir_path, slide_dir)
        
        if not os.path.isfile(slide_dir_abs_path) \
        and len(os.listdir(slide_dir_abs_path)) > 0:
            slide_dirs_abs_paths.append(slide_dir_abs_path)

    data_per_slide = {}

    for i, slide_dir_abs_path in enumerate(slide_dirs_abs_paths):

        if os.path.isfile(slide_dir_abs_path):
            continue
        
        slide_id = slide_dir_abs_path.split('/')[-1]
        print(f'{slide_id} ; ({i} / {len(slide_dirs_abs_paths)})')

        grid_based_features_path = \
            os.path.join(slide_dir_abs_path, 'exp_1_features'+suffix+'.pkl')

        print(grid_based_features_path)

        if not os.path.isfile(grid_based_features_path):
            continue

        cells_df_path = os.path.join(
            slide_dir_abs_path.replace("02_features", "01_preprocessed"), 
            'nodes_grid'+suffix+'.csv'
            )
        if not os.path.isfile(cells_df_path):
            continue
        # Read the grid-based features (TLOs, node_stats, sizes)
        grid_based_features = pickle.load(open(grid_based_features_path, 'rb'))

        # Read the cells dataframe with grid information
        cells_df = pd.read_csv(cells_df_path)

        # Iterate through the grids, add TLOs and sizes to node_stats DFs,
        # and finally merge all the per-grid DFs in order to fill NaNs and make
        # sure that the sizes are consistent. We'll probably unpack it afterwards.
        features_all_together = pd.DataFrame()
        
        if len(grid_based_features) > 1700:  #just skip some large grids
            print("skipping", len(grid_based_features))
            continue  
        for grid_id in tqdm(grid_based_features):


            features_all_together = pd.concat([
                features_all_together, 
                pd.DataFrame(grid_based_features[grid_id], index=grid_based_features.keys())
                ])

        features_all_together = features_all_together.fillna(0).drop_duplicates()

        # Creating the grid-level graph in nx
        grids_df = cells_df #get_grid_level_graph_df(cells_df, 'patch')

        patch_edges_df_path = os.path.join(
            slide_dir_abs_path.replace("02_features", "01_preprocessed"), 
            'patch_grid_edges'+suffix+'.csv'
            )
        patches_edges_df = pd.read_csv(patch_edges_df_path)
        #grids_edges_df = delaunay_triangulation(grids_df, keep_fewer_cols=True)
        G_grids = get_graph_from_grids_df(grids_df, patches_edges_df)

        # Adding it to the per-slide dictionary
        data_per_slide[slide_id] = {
            'graph': G_grids, 'features': features_all_together
            }
        
    return data_per_slide

def data_per_slide_to_pyg_dataset(data_per_slide):
    dataset_pyg = []

    # Iterate over patient slides
    for slide_id in data_per_slide.keys():
        curr_data = data_per_slide[slide_id] 

        # Iterate over grids and prepare them for the Data class
        all_grid_ids = list(curr_data['graph'].nodes())
        removed_cnt = 0
        for grid_id in all_grid_ids:

            # Remove old node attributes (x, y, size)
            attribute_keys = list(curr_data['graph'].nodes()[grid_id].keys())
            for key in attribute_keys:
                del curr_data['graph'].nodes()[grid_id][key] 

            # Convert the corresponding row from the DataFrame to numpy 
            # and assign it as node attributes (x)
            try:
                curr_data['graph'].nodes()[grid_id]['x'] = \
                    curr_data['features'].iloc[int(grid_id)].values
                
            except KeyError:
                removed_cnt += 1
                curr_data['graph'].remove_node(grid_id)
            except IndexError:
                removed_cnt += 1
                curr_data['graph'].remove_node(grid_id)

        if removed_cnt > 0:
            print(f'{slide_id} - removed {removed_cnt} out of {len(all_grid_ids)}')

        # Convert the nx graph (with important attributes stored in x) 
        # into a torch geometric Data object
        data_pyg = from_networkx(curr_data['graph'])

        # Add the slide label
        data_pyg.y = curr_data['label']

        # Add the Data object of the current slide to dataset_pyg
        dataset_pyg.append(data_pyg)   
    
    return dataset_pyg

def main(suffix = ""):
    # convert
    data_per_slide = get_data_per_slide(FEATURES_DIR_PATH, suffix)

    print(len(data_per_slide))

    os.makedirs(os.path.join(DATA_PER_SLIDE_SAVEDIR, 'experiment_1'+suffix), exist_ok=True)

    data_per_slide_savepath = os.path.join(
        DATA_PER_SLIDE_SAVEDIR, 'experiment_1'+suffix, 'patches_data' +suffix + '_per_slide.pkl'
        )

    pickle.dump(data_per_slide, open(data_per_slide_savepath, 'wb'))
    
    
    # sth features
    set_of_features = set()
    for slide, data in data_per_slide.items():
        set_of_features = set.union(set_of_features, set(data['features'].columns))

    print(f'Number of all features = {len(set_of_features)}')

    # Adding null columns to pad all slides to 70 columns
    for slide_id in data_per_slide:
        curr_features = data_per_slide[slide_id]['features'].columns
        features_diff = set_of_features - set(curr_features)
        data_per_slide[slide_id]['features'][list(features_diff)] = 0

    for slide_id in data_per_slide:
        if data_per_slide[slide_id]['features'].shape[1] != len(set_of_features):
            print(f'{slide_id} NOT padded !')

    # Discarding missing-related columns
    set_of_non_missing_features = \
        [feature for feature in set_of_features if 'missing' not in feature.lower()]

    print(f'Number of non-missing features = {len(set_of_non_missing_features)}')

    for slide_id in data_per_slide:
        data_per_slide[slide_id]['features'] = \
            data_per_slide[slide_id]['features'][set_of_non_missing_features]
        
        
    #map patients and slides
    slide_label_dict = get_slide_label_map({'DR': 0})

    slide_label_dict_expanded = {}
    for slide_id, label in slide_label_dict.items():
        slide_label_dict_expanded[slide_id] = label 
        slide_label_dict_expanded[f'DEEPMEL_{slide_id}'] = label

    for slide_id in data_per_slide:
        try:
            data_per_slide[slide_id]['label'] = slide_label_dict_expanded[slide_id]
        except Exception:
            data_per_slide[slide_id]['label'] = -1

    # Keep only rows with labels 0 or 1
    data_per_slide_0_1 = {}
    for slide_id, curr_data in data_per_slide.items():
        if curr_data['label'] in (0, 1):
            data_per_slide_0_1[slide_id] = curr_data
        else:
            print(curr_data['label'], slide_id)

    print(len(data_per_slide), len(data_per_slide_0_1))
    
    dataset_pyg = data_per_slide_to_pyg_dataset(data_per_slide_0_1)

    print(len(dataset_pyg)) 
    
    #save data
    dataset_pyg_savepath = os.path.join(
        DATA_PER_SLIDE_SAVEDIR, 'experiment_1'+suffix, 'patches'+suffix+'_dataset_pyg.pkl'
        )

    os.makedirs(os.path.join(DATA_PER_SLIDE_SAVEDIR, 'experiment_1'+suffix), exist_ok=True)

    pickle.dump(dataset_pyg, open(dataset_pyg_savepath, 'wb'))


if __name__ == "__main__":
    main("_512_rot30")