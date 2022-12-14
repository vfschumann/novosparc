import importlib.util
import sys
# TODO I kinda hate to do it like this, bc that's not dynamic. Have to ask sometime how Enes does this....
spec = importlib.util.spec_from_file_location("novosparc",
                                              "/novosparc/__init__.py")
# spec = importlib.util.spec_from_file_location("novosparc",
#                                               "/home/vschuma/PycharmProjects/novosparc/novosparc/__init__.py")
novosparc = importlib.util.module_from_spec(spec)
sys.modules["novosparc"] = novosparc
spec.loader.exec_module(novosparc)

import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
plt.viridis()
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import zscore
import sklearn
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from sim_score_funs import mean_spearman
from sim_score_funs import mean_ssim
from sim_score_funs import box_swarm_label

# Data
# Read in Single Cell data

# Reading expression data to scanpy AnnData (cells x genes)
data_dir = '../../../novosparc/datasets/drosophila_scRNAseq/'
data_path = os.path.join(data_dir, 'dge_normalized.txt')
dataset = sc.read(data_path).T
gene_names = dataset.var.index.tolist()

num_cells, num_genes = dataset.shape # 1297 cells x 8924 genes

print('number of cells: %d' % num_cells)
print('number of genes: %d' % num_genes)
# optional: subset cells
num_cells = 1000
sc.pp.subsample(dataset, n_obs=num_cells)
# dge_rep mode
dge_rep = None # a representation of cells gene expression
# highly var genes
sc.pp.highly_variable_genes(dataset)
is_var_gene = dataset.var['highly_variable']
var_genes = list(is_var_gene.index[is_var_gene])

## Read in atlas
# atlas settings
atlas_dir = '../../../novosparc/datasets/bdtnp/'
target_space_path = os.path.join(atlas_dir, 'geometry.txt')
locations = pd.read_csv(target_space_path, sep=' ')
type(locations)
num_locations = 3039 # coming from the spatial data
locations_apriori = locations[:num_locations][['xcoord', 'zcoord']].values
locations_apriori
locations = locations_apriori
locations
atlas_path = os.path.join(atlas_dir, 'dge.txt')
atlas = sc.read(atlas_path)
atlas_genes = atlas.var.index.tolist()
atlas.obsm['spatial'] = locations

# Reconstruction linear and smooth, alpha = 0.8, neighbours = 5
# calculate cost matrix
# params for smooth cost # only needed when/for the part where you don't use the atlas!
num_neighbors_s = num_neighbors_t = 5

# params for linear cost
markers = list(set(atlas_genes).intersection(gene_names))
atlas_matrix = atlas.to_df()[markers].values
markers_idx = pd.DataFrame({'markers_idx': np.arange(num_genes)}, index=gene_names)
markers_to_use = np.concatenate(markers_idx.loc[markers].values)

# construct tissue object
tissue = novosparc.cm.Tissue(dataset=dataset, locations=locations_apriori)

# setup smooth
num_neighbors_s = num_neighbors_t = 5

# alternative 1: setup both assumptions
tissue.setup_reconstruction(atlas_matrix=atlas_matrix,
                            markers_to_use=markers_to_use,
                            num_neighbors_s=num_neighbors_s,
                            num_neighbors_t=num_neighbors_t)

# compute optimal transport of cells to locations
alpha_linear = 0.8
epsilon = 5e-3
# tissue.dge = sparse.csr_matrix(tissue.dge)
tissue.reconstruct(alpha_linear=alpha_linear, epsilon=epsilon)

# reconstructed expression of individual genes
sdge = tissue.sdge

dataset_reconst = sc.AnnData(pd.DataFrame(sdge.T, columns=gene_names))
dataset_reconst.obsm['spatial'] = locations

# GMM
tissue.cleaning_expression_data(dataset_reconst,tissue.sdge.T,normalization='zscore', selected_genes=atlas_genes)
# reconstructed expression of individual genes
sdge_postcleaned = tissue.cleaned_dge
dataset_reconst_postcleaned = sc.AnnData(pd.DataFrame(sdge_postcleaned, columns=gene_names))
dataset_reconst_postcleaned.obsm['spatial'] = locations

# Set expression matrices and scale
# truth
subset_cols = []
for i, gene in enumerate(markers):
    if gene in dataset_reconst.var_names:
        subset_cols.append(np.asarray(atlas[:, gene].X).reshape(-1, 1))
exprmtrx_truth = np.concatenate(subset_cols, axis=1)
exprmtrx_truth_scaled = zscore(exprmtrx_truth)
# reconstruction
subset_cols = []
for i, gene in enumerate(atlas_genes):
    if gene in dataset_reconst.var_names:
        subset_cols.append(np.asarray(dataset_reconst[:, gene].X).reshape(-1, 1))
exprmtrx_simplerecon = np.concatenate(subset_cols, axis=1)
exprmtrx_simplerecon_scaled = zscore(exprmtrx_simplerecon)
# cleaning
subset_cols = []
for i, gene in enumerate(atlas_genes):
    if gene in dataset_reconst_postcleaned.var_names:
        subset_cols.append(np.asarray(dataset_reconst_postcleaned[:, gene].X).reshape(-1, 1))
exprmtrx_recon_defaultGMM = np.concatenate(subset_cols, axis=1)
exprmtrx_recon_defaultGMM_scaled = zscore(exprmtrx_recon_defaultGMM)

# score calculation

plt.viridis()
# spearman
method_recon = ["normal reconstruct"] * len(atlas_genes)
r_values_recon, mean_corre_recon = mean_spearman(exprmtrx_truth_scaled, exprmtrx_simplerecon_scaled)
spearman_corre_reconstruction = pd.DataFrame(list(zip(atlas_genes, r_values_recon, method_recon)),
                                    columns=["gene", "r score (spearman)", "method"])

method_clean = ["cleaned reconstruct"] * len(atlas_genes)
r_values_clean, mean_corre_clean = mean_spearman(exprmtrx_truth_scaled, exprmtrx_recon_defaultGMM_scaled)
spearman_corre_cleaned = pd.DataFrame(list(zip(atlas_genes, r_values_clean, method_clean)),
                                    columns=["gene", "r score (spearman)", "method"])

print(f"median spearman; recon: {np.median(r_values_recon)}, clean:{np.median(r_values_clean)}")
spearman_corre_plt = pd.concat([spearman_corre_reconstruction, spearman_corre_cleaned])
box_swarm_label(spearman_corre_plt,
                "r score (spearman)",
                [spearman_corre_reconstruction, spearman_corre_cleaned])

plt.savefig("Droso_spearman.png")

# ssim - per gene

method_recon = ["normal reconstruct"] * len(atlas_genes)
ssim_values_recon, mean_ssim_recon = mean_ssim(exprmtrx_truth_scaled, exprmtrx_simplerecon_scaled)

ssim_reconstruction = pd.DataFrame(list(zip(atlas_genes, ssim_values_recon, method_recon)),
                                   columns=["gene", "ssim score", "method"])

method_clean = ["cleaned reconstruct"] * len(atlas_genes)
ssim_values_clean, mean_ssim_clean = mean_ssim(exprmtrx_truth_scaled, exprmtrx_recon_defaultGMM_scaled)
ssim_cleaned = pd.DataFrame(list(zip(atlas_genes, ssim_values_clean, method_clean)),
                            columns=["gene", "ssim score", "method"])

print(f"median ssim gene-wise; recon: {np.median(ssim_values_recon)}, clean:{np.median(ssim_values_clean)}")
ssim_plt = pd.concat([ssim_reconstruction, ssim_cleaned])
box_swarm_label(ssim_plt,
                "ssim score",
                [ssim_reconstruction, ssim_cleaned])

plt.savefig("Droso_ssim.png")
