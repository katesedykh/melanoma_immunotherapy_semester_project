import os
import pandas as pd
import numpy as np
import math
import tqdm
import matplotlib.pyplot as plt

def create_grid(df_nodes, patch_size = 1024, cell_count_threshold = 0, rotate=0, shift = 0, draw_plot = False):
    """
    patch size: size of each patch to extract
    cell_count_threshold: minimum number of cells to count patch as valid
    rotate: to augment data, rotation angle in degrees
    shift: to augment data, value of vertical and horizontal shift
    """
    edges_graph = pd.DataFrame({'node_id_1':[], 'node_id_2':[], 'patch_loc_x_1':[], 'patch_loc_y_1':[], 'patch_loc_x_2':[], 'patch_loc_y_2':[]})
    
    graph = df_nodes.copy()
    graph['patch'] = np.nan
    graph['patch_loc_x'] = np.nan
    graph['patch_loc_y'] = np.nan
    graph['patch_real_loc_x'] = np.nan
    graph['patch_real_loc_y'] = np.nan
    
    xs = graph.cell_x_position
    ys = graph.cell_y_position
     
    
    if rotate:
        RotRad = math.radians(rotate)

        # 2D rotation matrix, clockwise
        RotMatrix = np.array([[np.cos(RotRad),  np.sin(RotRad)],
                              [-np.sin(RotRad), np.cos(RotRad)]])
        rotated = np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([xs, ys]))
        graph.cell_x_position = rotated[0][0]
        graph.cell_y_position = rotated[1][0]

    
    # Get min max border coordinates
    min_border_x = graph.cell_x_position.min()
    min_border_y = graph.cell_y_position.min()
    
    # Shift to 0,0
    graph.cell_x_position = graph.cell_x_position - min_border_x
    graph.cell_y_position = graph.cell_y_position - min_border_y
    
    max_border_x = graph.cell_x_position.max()
    max_border_y = graph.cell_y_position.max()
    
        
    # Get mesh grip
    xspan = np.arange(-patch_size, max_border_x+patch_size, patch_size)
    yspan = np.arange(-patch_size, max_border_y+patch_size, patch_size)
    
    
    #mesh_x, mesh_y = meshgrid_rotate(xspan, yspan, degrees = rotate)
    mesh_x, mesh_y  = np.meshgrid(xspan, yspan) 
    
    if shift:
        mesh_x = mesh_x + shift
        mesh_y = mesh_y + shift
        
    coordsx = mesh_x[0]
    coordsy = np.transpose(mesh_y)[0]
    coordsx = [0] + coordsx
    coordsy = [0] + coordsy

    ip = 0 # patch number
    for ix,xmesh in enumerate(coordsx[:-1]):
        for iy, ymesh in enumerate(coordsy[:-1]):
            df_nodes_patch = graph.copy()
            
            # Check if inside the patch        
            indexlist = df_nodes_patch.index[(df_nodes_patch['cell_x_position'] >coordsx[ix]) 
                                 & (df_nodes_patch['cell_x_position'] < coordsx[ix+1])
                                 &(df_nodes_patch['cell_y_position'] >coordsy[iy]) 
                                 & (df_nodes_patch['cell_y_position'] < coordsy[iy+1])]
            
            if len(indexlist) >= cell_count_threshold: # if number of cells in patch > X -> valid patch
                graph["patch"].iloc[indexlist] = ip
                ip += 1
                
                mid_x = (coordsx[ix] +coordsx[ix+1]) / 2
                mid_y = (coordsy[iy] +coordsy[iy+1]) / 2
                graph["patch_real_loc_x"].iloc[indexlist] = mid_x
                graph["patch_real_loc_y"].iloc[indexlist] = mid_y
                graph["patch_loc_x"].iloc[indexlist] = ix 
                graph["patch_loc_y"].iloc[indexlist] = iy 
                
            

    if draw_plot == True:
        plt.figure(figsize=(12,12))
        plt.plot(mesh_x, mesh_y, color='k')
        plt.plot(np.transpose(mesh_x), np.transpose(mesh_y), color='k')
        scat = sns.scatterplot(data=graph, x='cell_x_position', y='cell_y_position', hue='phenotype', s=3) #tissue_category
        sns.scatterplot(data=graph, x='patch_loc_x', y='patch_loc_y', s=35, color = "#888EBA")
        plt.show()
        
    graph = graph[graph['patch'].notna()]
    
    loc_nodes = graph.groupby(['patch_loc_x','patch_loc_y','patch_real_loc_x','patch_real_loc_y']).size().reset_index()

    xs =  loc_nodes.patch_loc_x.unique()
    ys =  loc_nodes.patch_loc_y.unique()
    valid_nodes = graph.set_index(['patch_loc_x','patch_loc_y']).index.unique()
    edges = []

    for ix,x in enumerate(xs):
        for iy,y in enumerate(ys):
            if (xs[ix],ys[iy]) in valid_nodes:
                if (xs[ix]+1,ys[iy]) in valid_nodes: #check right neighbour
                    patch_id1 = graph[(graph.patch_loc_x==xs[ix]) & (graph.patch_loc_y==ys[iy])]["patch"].iloc[0]
                    patch_id2 = graph[(graph.patch_loc_x==xs[ix]+1) & (graph.patch_loc_y==ys[iy])]["patch"].iloc[0]
                    edges_graph.loc[len(edges_graph.index)] = [patch_id1,patch_id2,xs[ix],ys[iy],xs[ix]+1,ys[iy]]
                if (xs[ix],ys[iy]+1) in valid_nodes: #check top neighbour
                    patch_id1 = graph[(graph.patch_loc_x==xs[ix]) & (graph.patch_loc_y==ys[iy])]["patch"].iloc[0]
                    patch_id2 = graph[(graph.patch_loc_x==xs[ix]) & (graph.patch_loc_y==ys[iy]+1)]["patch"].iloc[0]
                    edges_graph.loc[len(edges_graph.index)] = [patch_id1,patch_id2,xs[ix],ys[iy],xs[ix],ys[iy]+1]

    return graph, edges_graph


data_dir = '~/CHUVdata/IFQuant/version_2/output/01_preprocessed/'
out_dir = '~/melanoma_immunotherapy/output/01_preprocessed/' 

for folder in tqdm.tqdm(os.listdir(data_dir)):
    try:
        os.mkdir(out_dir + folder)
    except:
        print("exists")
    # Read data
    path_nodes = data_dir + folder + "/nodes.csv"
    path_edges = data_dir + folder + "/edges_30.csv"
    df_nodes = pd.read_csv(path_nodes)
    df_edges = pd.read_csv(path_edges)

    # Run and save
    patches_df, patches_edges_df = create_grid(df_nodes, patch_size = 512, cell_count_threshold = 50, rotate = 30, shift = 256, draw_plot=False)
    patches_df.to_csv(out_dir + folder + "/nodes_grid_512_shift256_rot30.csv")   
    patches_edges_df.to_csv(out_dir + folder + "/patch_grid_edges_512_shift256_rot30.csv")  #connections between nodes
    
    # Add data to edges files
    df_nodes_patches = patches_df
    df_edges_patches = pd.DataFrame()
    for i in range(len(df_nodes_patches.patch.unique())): #iterate all patches
        df_patch = df_nodes_patches[df_nodes_patches.patch==i]
        df_edges_patch1 = df_edges[df_edges['cell_id_1'].isin(df_patch['cell_id'])]
        df_edges_patch2 = df_edges[df_edges['cell_id_2'].isin(df_patch['cell_id'])]
        df_edges_patch = df_edges_patch1.merge(df_edges_patch2)
        df_edges_patch["patch"] = i
        df_edges_patches = pd.concat([df_edges_patches,df_edges_patch])

    df_edges_patches.to_csv(out_dir + folder + "/edges_grid_30_512_shift256_rot30.csv")  
    print("saved: "+out_dir + folder + "/edges_grid_30_512_shift256_rot30.csv" )
