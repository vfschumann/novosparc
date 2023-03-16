import os
import numpy as np
import pandas as pd
import pickle
import scanpy as sc
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls

def merge_meta_gw(gw_matrix, meta_matrix): # checked, add tests
    """
    gw_matrix - pd.DataFrame, from tissue object (tissue.gw)
    meta_matrix - pd.DataFrame, meta data (x,1), row length (cells?) must match gw_matrix
    output: pd.DataFrame with gw matrix and the meta data labels as column names
    """

    # ToDo: Implement Check for matching dimensions

    # use one-hot encoding for meta-data labels
    meta_encode = pd.get_dummies(meta_matrix)
    # get unique lsit of labels
    unque_lbs = list(meta_matrix.iloc[:,0].unique())
    # merge matrixes
    meta_gw_merge = np.dot(gw_matrix.T,meta_encode) # transpose to get locations as rows
    # dataframe with column names
    meta_gw_df = pd.DataFrame(meta_gw_merge, columns = unque_lbs)

    return meta_gw_df

def meta_to_tissue(list_of_metagw_names, list_of_metagw_df, tissue): # checked,
    """
    list_of_metagw_names - list containing strings of names of meta_gw matrices to be added to the tissue obj, ordering has to match ordering in list_of_metagw_df
    list_of_metagw_df - list containing dataframes to add, ordering has to match ordering in list_of_metagw_df
    tissue - tissue object, output from novosparc.reconstruct
    """

    # ToDo: implement check for matching length of the both lists

    metadata_mapped = dict()

    for df_name in list_of_metagw_names:
        for df in list_of_metagw_df: # this assumes the usage of only a few metadata sets, not really a scalable approach
            metadata_mapped[df_name] = df

    tissue.metadata = metadata_mapped
    print("tissue object modified with set of metadata dataframes")

def get_highest_prop_lbl(meta_array, plot_anndata): # checked, add tests
    """
    meta_array = pd.Df of meta_gw_merged
    plot_anndata = anndata version of the metadata matrix for plotting (compare w dataset_tissue)
    output: pd.DF, 3 columns: 1)mapping_prop_lbl - mapping probability value for the label at the location,
                          2)mapped_lbl_idx - column number index of the label in the meta_gw_merge matrix,
                          3)mapped_lbl - label name
    """
    # ToDo: check if meta_array is a float array with string value column names

    # find max value in the row
    out_array = np.amax(np.array(meta_array), axis=1)
    # find the values column name - get the index
    out_array = np.vstack((out_array, np.argmax(np.array(meta_array), axis=1)))
    # transform to df to enable multiple data types
    out_df = pd.DataFrame(out_array.T)
    # write value and column name into the new df
    # can I add a new column like this? - yes, but it takes already quite some time
    # ToDo: Implement speed up versions Enes wrote to you
    out_df[2] = out_df[1].apply(lambda x: meta_array.columns[x])
    out_df.columns = ["mapping_prop_lbl", "mapped_lbl_idx", "mapped_lbl"]

    # add annotation to the dataset anndata
    for col in out_df.columns:
        plot_anndata.obsm[col] = out_df[col].to_numpy().reshape(-1,1)
    print(f"updated anndata objc by adding {out_df.columns} as obsm")

    return out_df

def plot_high_prop_label_no_opac(plot_anndata,width=1000,height=800,plot_bgcolor="black",columnname="label"): # checked, add tests
    """
    plot_anndata - anndata version of the metadata matrix for plotting (compare w dataset_tissue) after
    "get_highest_prop_lbl" processing
    """

    # ToDo: Implement check for existence of "mapped_lbl" property
    # ToDo: maybe also check if only expected input was given
    # ToDo: add option for plot title

    # set coordinates
    xy = plot_anndata.obsm['spatial']
    x = xy[:, 1]
    y = xy[:, 0] if xy.shape[1] > 1 else np.ones_like(x)

    # set values
    values = plot_anndata.obsm["mapped_lbl"]
    plot_df = pd.DataFrame(values, columns=[columnname])

    # set figure
    fig = px.scatter(plot_df, x=x, y=y, color=columnname,
                 width=width, height=height)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_layout(plot_bgcolor=plot_bgcolor)
    # ToDo: add color palette option (especially important when doing plots for multiple metadata sets)
    fig.show()

    return fig


def flatten(l):
    return [item for sublist in l for item in sublist]


def labels_to_legend(plot_anndata, out_df, labelname="label"):
    """
    todo
    output:
        lbl_mtching: dict [label index : lable string]
        lbl_idx: pd.Series, as many rows as location, index per location
    """

    # ToDo: Implement check for assumptions (e.g. "mapped_lbl")

    # labels as discrete colors with strings as label
    meta_labels = plot_anndata.obsm["mapped_lbl"]
    lbl_idx = out_df["mapped_lbl_idx"]
    plot_df = pd.DataFrame({labelname: plot_anndata.obsm["mapped_lbl"].tolist(),
                            "lbl_idx": plot_anndata.obsm["mapped_lbl_idx"].tolist()
                            })
    # get labels as string list
    lbl_str = flatten(plot_df[labelname])
    # match the lable string wiht their index number (for legend)
    lbl_mtchng = {k: str(v) for k, v in zip(tuple(lbl_idx), tuple(lbl_str))}

    return lbl_mtchng, lbl_idx


def plot_high_prop_label_var_opac_2(plot_anndata, out_df, lbl_idx, lbl_mtchng, labelname="label",
                                    width=1000, height=800):
    """
    todo
    """

    # ToDo: checks for assumptions and naming conventions
    # ToDo: add option for plot title
    # plot frame
    xy = plot_anndata.obsm['spatial']
    x = xy[:, 1]
    df = pd.DataFrame({'x_lv': xy[:, 1],
                       'y_lv': xy[:, 0] if xy.shape[1] > 1 else np.ones_like(x),
                       'color': lbl_idx,
                       'alpha': out_df["mapping_prop_lbl"]
                       })

    # color legend
    clrs_dscrt = px.colors.qualitative.Alphabet[:len(df['color'].unique())]
    idx_clr = {idx: str(clr) for idx, clr in zip(df['color'].unique(), set(clrs_dscrt))}

    # plot
    fig = go.Figure()
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    fig.layout.height = height
    fig.layout.width = width

    for c in df['color'].unique():
        df_color = df[df['color'] == c]
        # normalize alpha values
        alpha_raw = df_color['alpha'].to_numpy().reshape(-1, 1)
        alpha_norm = (alpha_raw - alpha_raw.min()) / (alpha_raw.max() - alpha_raw.min())
        # plot label color
        fig.add_trace(
            go.Scatter(
                x=df_color['x_lv'],
                y=df_color['y_lv'],
                name=lbl_mtchng[c],
                mode="markers",
                # text = out_df[\ mapped_ct\ ], # todo this could be an array of joint string with cell type + probability,
                showlegend=True,
                marker=go.scatter.Marker(
                    color=idx_clr[c],
                    size=11,
                    opacity=alpha_norm)  # I think this has to be another column in that dataframe then
            ))
        fig.update_layout(legend=dict(
            bordercolor=idx_clr[c]))
    fig.show()
    return fig

def save_plots_to_html(list_of_plots, filename='plotly_graphs.html', full_html=False, include_plotlyjs='cdn'):
    """
    list of plots - list of plots in plotly format (convert mpl plots with tls.mpl_to_plotly(plot) first)
    :returns: -
    """

    # Todo: check format assumptions
    with open(filename, 'a') as f:
        for plot in list_of_plots:
            f.write(plot.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs))
    print(f"successfully saved plots to {filename}")
