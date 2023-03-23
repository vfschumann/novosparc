import novosparc
import time
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import random
import pickle
import os
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

if __name__ == '__main__':


    ######################################
    # 1. Set the data and output paths ###
    ######################################
    print("reading in data paths and parameter")

    data_dir = 'data'
    dataset_path = os.path.join(data_dir, 'TNBC_sn_integrated_dge.zip')  # use pickle to load
    target_space_path = os.path.join(data_dir, 'TNBC_6w_fc51_4_coordinates.csv')  # location coordinates
    atlas_path = os.path.join(data_dir, 'TNBC_atlas_OvrlbAtlasMarkerSnGenes.zip') # use pickle to load
    output_folder = os.path.join(data_dir, 'output')  # folder to save the results, plots etc.

    # 1.1. set subsample parameter when used
    min_num_cells = 4000
    max_num_cells = 5000

    n_loc_atlas = 2000

    # for linear assumption
    marker_subset = 400
    alpha_parameter = 0.8

    # for neighbor assumption
    num_neighbors_s = 5
    num_neighbors_t = 5

    # for plotting
    gene_list_to_plot = None

    # output
    filename_tissue = f"tissue_maxCells{max_num_cells}_loc{n_loc_atlas}_run4.pkl"
    filename_tissue_gw = f"tissue_gwmaxCells{max_num_cells}_loc{n_loc_atlas}_run4.pkl"
    selected_cells_name = f"tissue_maxCells{max_num_cells}_loc{n_loc_atlas}_run4_cellsselected.csv"

    #######################################
    # 2. Read the dataset and subsample ###
    #######################################

    # Read the dge. this assumes the file formatted in a way that genes are columns and cells are rows.
    # If the data is the other way around, transpose the dataset object (e.g dataset=dataset.T)
    print("Reading in dge")
    dataset_rdn = pd.read_pickle(dataset_path)
    print("convert dge to anndata")
    dataset = ad.AnnData(dataset_rdn)

    gene_names = dataset.var.index.tolist()
    num_cells, num_genes = dataset.shape
    print('number of cells total: %d' % num_cells)
    print('number of genes: %d' % num_genes)

    # Optional: downsample number of cells.
    print("Downsampling number of cells")
    cells_selected, dataset = novosparc.pp.subsample_dataset(dataset,
                                                             min_num_cells=min_num_cells,
                                                             max_num_cells=max_num_cells)

    num_cells, num_genes = dataset.shape
    print('number of cells used: %d' % num_cells)

    # Load the location coordinates from file if it exists
    print("Reading in locations")
    locations = pd.read_csv(target_space_path, sep=',')
    num_locations = locations.shape[0]  # coming from the spatial data #
    locations_apriori = locations[:num_locations][['xcoord', 'ycoord']].values
    locations = locations_apriori

    # Read the atlas
    print("Reading in atlas")
    atlas_rdn = pd.read_pickle(atlas_path)
    atlas = ad.AnnData(atlas_rdn)

    atlas_genes = atlas.var.index.tolist()
    atlas.obsm['spatial'] = locations

    print('number of locations total: %d' % len(locations))

    # Optional: downsample the number of locations
    print("Downsampling atlas")
    sc.pp.subsample(atlas, n_obs=n_loc_atlas)
    locations = atlas.obsm['spatial']
    locations_apriori = locations
    atlas_genes = atlas.var.index.tolist()
    print('number of locations used: %d' % len(locations_apriori))

    #########################################
    # 3. Setup and spatial reconstruction ###
    #########################################
    tissue = novosparc.cm.Tissue(dataset=dataset, locations=locations,
                                 output_folder=output_folder) # create a tissue object

    # Optional: use marker genes
    print("Getting marker genes for reconstruction atlas")
    # params for linear cost
    markers = list(set(atlas_genes).intersection(gene_names))

    # Optional: subset marker
   # markers = random.sample(markers, marker_subset)

    # build marker list
    atlas_matrix = atlas.to_df()[markers].values
    markers_idx = pd.DataFrame({'markers_idx': np.arange(num_genes)}, index=gene_names)
    markers_to_use = np.concatenate(markers_idx.loc[markers].values)

    # reconstruction using both assumptions
    tissue.setup_reconstruction(atlas_matrix=atlas_matrix,
                                markers_to_use=markers_to_use,
                                num_neighbors_s=num_neighbors_s,
                                num_neighbors_t=num_neighbors_t)

    # alpha parameter controls the reconstruction. Set 0 for de novo, between
    # 0 and 1 in case markers are available.
    print("Start reconstruction")
    tissue.reconstruct(alpha_linear=alpha_parameter)  # reconstruct with the given alpha value

    #############################################
    # 4. Save the results and plot some genes ###
    #############################################

    # save the tissue and tissue.gw to file
    print("save the tissue obj")
    # create a pickle file
    tissue_file = os.path.join(output_folder, filename_tissue)
    picklefile = open(tissue_file, 'wb')
    # pickle the dictionary and write it to file
    pickle.dump(tissue, picklefile)
    # close the file
    picklefile.close()

    print("save the tissue.gw only")
    # create a pickle file
    tissue_gw_file = os.path.join(output_folder, filename_tissue_gw)
    picklefile = open(tissue_gw_file, 'wb')
    # pickle the dictionary and write it to file
    pickle.dump(tissue.gw, picklefile)
    # close the file
    picklefile.close()

    print("save selected cells to txt")
    np.savetxt(selected_cells_name, cells_selected, delimiter=',')

    # Todo: add a txt file output with cell,loc and gene numbers

    # plot some genes and save them

    if gene_list_to_plot is not None:
        print("make some plots and save them")
        novosparc.pl.embedding(atlas, gene_list_to_plot)


    print("done")