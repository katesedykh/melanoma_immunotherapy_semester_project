# melanoma_immunotherapy_semester_project

Steps to create grid graphs from nodes.csv, edges_30.csv files and prepare for training:
1. `generate_grid_csv.py` Creates new csv files with grid graph data from all files in set folder
Possible parameters to change: 
patch size: size of each patch to extract
cell_count_threshold: minimum number of cells to count patch as valid
rotate: to augment data, rotation angle in degrees
shift: to augment data, value of vertical and horizontal shift

Example: patch_size = 512, cell_count_threshold = 50, rotate = 30, shift = 256

2. Extract features from created csv files `experiment_1_patch.py`. Change names of csv files as it was in base code (for nodes in this script, for edges in config)

3. Run `prepare_dataset.py` to convert graphs with features to dataset (only valid for one experiment, for example, 512 ptches with 30 degrees rotation, hence run for different data separately). To run on specific data, change main() argument suffix, e.g. '512_shift256_rot30' - how previous data was saved

4. `train.py` Train model with one or multiple created datasets