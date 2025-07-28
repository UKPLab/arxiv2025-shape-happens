from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from umap import UMAP
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
from time_stuff.utils import SupervisedMDS
from time_stuff.utils import ActivationDataset
from time_stuff.utils import farthest_point_sampling
from pycolormap_2d import ColorMap2DBremm, ColorMap2DSteiger, ColorMap2DZiegler, BaseColorMap2D, ColorMap2DCubeDiagonal
from matplotlib import patheffects as pe
import tempfile
from typing import Tuple
from skimage.color import rgb2lab, lab2rgb

def plot_activations(ad: ActivationDataset, label_col: str, reduction_method, target_col='correct_answer', layers=None, components=(0,1),
                     label_col_str=None, n_components=2, manifold='discrete_circular', title=None, save_path=None, plots_per_row=4,
                     annotations='random',  filter_incorrect=True, orthonormal=False, max_samples=500, plot_test=True,
                     preprocess_func=None, annotation_preprocess_func=None, postprocess_func=None,
                     return_fig=False):
    
    if layers is None:
        # Get number of layers from activation shape
        activations, labels = ad.get_slice(target_name=target_col, columns=label_col, preprocess_funcs=preprocess_func, filter_incorrect=filter_incorrect)
        layers = range(activations.shape[1])

    num_layers = len(layers)
    if num_layers < plots_per_row:
        plots_per_row = num_layers

    if manifold in ['discrete_circular', 'circular']:
        palette = 'twilight'
    elif manifold in ['log_linear', 'euclidean', 'semicircular', 'log_semicircular']:
        palette = 'flare'
    elif manifold in ['cluster']:
        palette = 'tab10'
    # elif len(np.unique(labels)) < 3:
    #     palette = ['#3A4CC0', '#B40426']
    else:
        palette = 'viridis'

    rows = int(np.ceil(num_layers / plots_per_row))
    scaling_factor = 6 if num_layers > 1 else 8
    fig, axs = plt.subplots(rows, plots_per_row, figsize=(scaling_factor * plots_per_row, scaling_factor * rows), constrained_layout=True)

    axs = np.atleast_1d(axs).flatten()  # Flatten to make indexing easier

    for i, layer in enumerate(layers):
        ax = axs[i]

        plot_activations_single(
            ad=ad,
            label_col=label_col,
            reduction_method=reduction_method,
            layer=layer,
            ax=ax,
            target_col=target_col,
            components=components,
            label_col_str=label_col_str,
            n_components=n_components,
            plot_test=plot_test,
            manifold=manifold,
            annotations=annotations,
            filter_incorrect=filter_incorrect,
            orthonormal=orthonormal,
            preprocess_func=preprocess_func,
            annotation_preprocess_func=annotation_preprocess_func,
            postprocess_func=postprocess_func,
            palette=palette,
            title=title,
            max_samples=max_samples,
        )

        ax.set_title(f"Layer {layer}")

    # Set the super title
    if title is not None:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle(f"{reduction_method} - {ad.model_name} - {ad.dataset_name}", fontsize=20)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if return_fig:
        return fig, axs
    plt.show()
    plt.close(fig)

def plot_activations_single(ad: ActivationDataset, label_col: str, reduction_method, layer, ax, target_col='correct_answer',  components=(0,1),
                     label_col_str=None, n_components=2, manifold='discrete_circular', filter_incorrect=True, orthonormal=False, 
                     palette='viridis', title=None, save_path=None, plots_per_row=4, s=40, linewidth=0.8, flip_axes=False,
                     annotations=None, plot_test=True, annotation_offset=(0,0), annotation_fontsize=14, n_annotations=10,
                     preprocess_func=None, annotation_preprocess_func=None, postprocess_func=None, palette_column=None,
                     max_samples=500):

    normalizer = Normalizer()

    if reduction_method == 'PCA':
        rmodel = PCA(n_components=n_components)
    elif reduction_method == 'tSNE':
        rmodel = TSNE(n_components=n_components)
    elif reduction_method == 'Isomap':
        rmodel = Isomap(n_components=n_components)
    elif reduction_method == 'PLS':
        rmodel = PLSRegression(n_components=n_components)
    elif reduction_method == 'LDA':
        rmodel = LinearDiscriminantAnalysis(n_components=n_components)
    elif reduction_method == 'SMDS':
        rmodel = SupervisedMDS(n_components=n_components, manifold=manifold, orthonormal=orthonormal)
    elif reduction_method == 'UMAP':
        rmodel = UMAP(n_components=n_components)
    elif reduction_method == 'MDS':
        rmodel = MDS(n_components=n_components)

    activations, labels = ad.get_slice(target_name=target_col, columns=label_col, preprocess_funcs=preprocess_func, filter_incorrect=filter_incorrect)
    labels = np.squeeze(labels)

    if len(labels[0]) > 1:
        # If labels are multi-dimensional, vstack to a 2D array
        labels = np.vstack(labels)

    df = ad.get_metadata_df(filter_incorrect=filter_incorrect)

    max_samples = min(max_samples, len(activations)) # Limit to 500 samples  
    activations = activations[:max_samples]
    labels = labels[:max_samples]
    df = df.iloc[:max_samples].reset_index(drop=True)

    split = 0.5
    idx_split = int(len(activations) * split)
    activations_train = activations[:idx_split]
    activations_test = activations[idx_split:]
    labels_train = labels[:idx_split]
    labels_test = labels[idx_split:]
    df_train = df.iloc[:idx_split].reset_index(drop=True)
    df_test = df.iloc[idx_split:].reset_index(drop=True)

    if reduction_method in ['PLS'] and labels_train.ndim == 1:
        min_label = labels_train.min()
        max_label = labels_train.max()
        labels_train = (labels_train - min_label) / (max_label - min_label)
        labels_test = (labels_test - min_label) / (max_label - min_label)

    # fig, axs = plt.subplots(int(np.ceil(len(layers) / plots_per_row)), plots_per_row,
    #                         figsize=(scaling_factor * plots_per_row, scaling_factor * len(layers) // plots_per_row),
    #                         constrained_layout=True)

    act_train = activations_train[:, layer]
    act_test = activations_test[:, layer]

    if reduction_method in ['PCA']:
        act_train = normalizer.fit_transform(act_train)
        act_test = normalizer.transform(act_test)

    rmodel.fit(act_train, labels_train)
    if plot_test:
        act_plot_red = rmodel.transform(act_test)
        plotted_labels = labels_test
    else:
        act_plot_red = rmodel.transform(act_train)
        plotted_labels = labels_train

    if flip_axes:
        # flip_axes is a tuple of length n_components
        for i, flip in enumerate(flip_axes):
            act_plot_red[:, i] = act_plot_red[:, i] * flip

    if reduction_method not in ['UMAP', 'MDS']:
        print(f"Layer: {layer} - Score: {rmodel.score(act_test, labels_test):.4f}")

    if postprocess_func is not None:
        plotted_labels = postprocess_func(plotted_labels)

    if ax == None: # Early return
        return rmodel, act_plot_red, plotted_labels

    if labels.ndim > 1 and palette_column is None:
        # If labels are multi-dimensional, map them to 0-1 range
        # cmap_labels_test = (plotted_labels - plotted_labels.min(axis=0)) / (plotted_labels.max(axis=0) - plotted_labels.min(axis=0))
        # Instead of 0-1 mapping, standardize them
        cmap_labels_test = (plotted_labels - plotted_labels.mean(axis=0)) / (plotted_labels.std(axis=0) + 1e-8)
        
        # Use ColorMap2DZiegler to get bidimensional colors
        cmap = ColorMap2DSet1()
        hues = [cmap(l1, l2) / 255.0 for l1, l2 in cmap_labels_test]

        # Plot the data
        # sns.set_style("whitegrid")
        if len(components) == 2:
            ax.scatter(act_plot_red[:, components[0]], act_plot_red[:, components[1]], c=hues, s=25)
        elif len(components) == 3:
            ax.scatter(act_plot_red[:, components[0]], act_plot_red[:, components[1]],
                       act_plot_red[:, components[2]])
    else:
        if palette_column is not None:
            # Use the specified column for coloring
            plotted_labels = df_test[palette_column].values if plot_test else df_train[palette_column].values
        else:
            palette = palette if len(np.unique(labels)) > 2 else ['#3A4CC0', '#B40426']
        sns.scatterplot(
            x=act_plot_red[:, components[0]], 
            y=act_plot_red[:, components[1]],
            hue=plotted_labels,
            ax=ax,
            palette=palette,
            alpha=1.0,
            s=s,
            linewidth=linewidth,
        )

    if label_col_str is not None and annotations is not None:
        # Preprocess all labels first (if needed)
        preprocessed_labels = df_test[label_col_str].values
        if annotation_preprocess_func is not None:
            preprocessed_labels = np.array([annotation_preprocess_func(t) for t in preprocessed_labels])

        if annotations == 'random':
            indices = np.random.choice(len(activations_test), size=n_annotations, replace=False)
            points = act_plot_red[indices]
            txt = preprocessed_labels[indices]

        elif annotations == 'uniform':
            # indices = farthest_point_sampling(act_test_red, 10, center_bias=annotation_center_bias)
            unique_labels = np.unique(preprocessed_labels)
            if unique_labels.shape[0] > n_annotations:
                # unique_labels = np.random.choice(unique_labels, size=16, replace=False)
                indices = np.linspace(0, len(unique_labels)-1, n_annotations, dtype=int)
            else:
                indices = np.arange(len(unique_labels))
            unique_labels = unique_labels[indices]
            points = act_plot_red[indices]
            txt = preprocessed_labels[indices]

        elif annotations == 'class':
            unique_labels = df_test[label_col_str].unique()
            indices = [df_test[df_test[label_col_str] == label].index[0] for label in unique_labels]
            points = act_plot_red[indices]
            txt = preprocessed_labels[indices]

        elif annotations == 'centroids':
            unique_labels = np.unique(plotted_labels)
            if unique_labels.shape[0] > n_annotations:
                # unique_labels = np.random.choice(unique_labels, size=16, replace=False)
                indices = np.linspace(0, len(unique_labels)-1, n_annotations, dtype=int)
                unique_labels = unique_labels[indices]

            points = []
            txt = []
            for label in unique_labels:
                indices = np.where(plotted_labels == label)[0]
                centroid = np.mean(act_plot_red[indices], axis=0)
                points.append(centroid)

                # Get the first index of the label to extract its preprocessed text
                txt.append(preprocessed_labels[indices[0]])

            points = np.array(points)

        else:
            raise ValueError("Invalid annotations value. Use 'random', 'class' or 'centroids'.")

        # Plot and annotate
        ax.scatter(points[:, components[0]], points[:, components[1]], color='red', edgecolor='k', s=20)

        # for j, label in enumerate(txt):
        #     ax.annotate(
        #         label,
        #         (points[j, components[0]] + annotation_offset[0],
        #          points[j, components[1]] + annotation_offset[1]),
        #         fontsize=annotation_fontsize,
        #         path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        #     )
        texts = []
        for j, label in enumerate(txt):
            x = points[j, components[0]] + annotation_offset[0]
            y = points[j, components[1]] + annotation_offset[1]
            texts.append(ax.text(
                x, y, label,
                fontsize=annotation_fontsize,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
            ))

        # After all texts are added:
        adjust_text(texts, ax=ax, expand_points=(1.2, 1.2), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # ax.set_title(f"Layer {layer}")
    if ax.get_legend() is not None:
        ax.get_legend().set_visible(False)

    # if title is not None:
    #     fig.suptitle(title, fontsize=20)
    # else:
    #     fig.suptitle(f"{reduction_method} - {ad.model_name} - {ad.dataset_name}", fontsize=20)

    # if return_fig:
    #     return fig, axs
    # if not save_path:
    #     plt.show()
    # else:
    #     plt.savefig(save_path, bbox_inches='tight')
    #     plt.close(fig)
    return rmodel, act_plot_red, plotted_labels

def plot_activations_plotly(
    ad,
    label_col: str,
    reduction_method,
    layer: int,
    target_col='correct_answer',
    label_col_str=None,
    n_components=3,
    components=(0, 1, 2),
    manifold='discrete_circular',
    annotations=None,
    filter_incorrect=True,
    orthonormal=False,
    plot_test=True,
    preprocess_func=None,
    annotation_preprocess_func=None,
    postprocess_func=None,
    max_samples=500,
    title=None,
    color_by=None,
):
    normalizer = Normalizer()

    # Setup reducer
    if reduction_method == 'PCA':
        rmodel = PCA(n_components=n_components)
    elif reduction_method == 'tSNE':
        rmodel = TSNE(n_components=n_components)
    elif reduction_method == 'Isomap':
        rmodel = Isomap(n_components=n_components)
    elif reduction_method == 'PLS':
        rmodel = PLSRegression(n_components=n_components)
    elif reduction_method == 'LDA':
        rmodel = LinearDiscriminantAnalysis(n_components=n_components)
    elif reduction_method == 'SMDS':
        rmodel = SupervisedMDS(n_components=n_components, manifold=manifold, orthonormal=orthonormal)
    elif reduction_method == 'UMAP':
        rmodel = UMAP(n_components=n_components)
    elif reduction_method == 'MDS':
        rmodel = MDS(n_components=n_components)

    activations, labels = ad.get_slice(target_name=target_col, columns=label_col, preprocess_funcs=preprocess_func, filter_incorrect=filter_incorrect)
    labels = np.squeeze(labels)
    df = ad.get_metadata_df(filter_incorrect=filter_incorrect)

    max_samples = min(max_samples, len(activations))
    activations = activations[:max_samples]
    labels = labels[:max_samples]
    df = df.iloc[:max_samples].reset_index(drop=True)

    # Split train/test
    split = 0.5
    idx_split = int(len(activations) * split)
    act_train, act_test = activations[:idx_split], activations[idx_split:]
    labels_train, labels_test = labels[:idx_split], labels[idx_split:]
    df_train, df_test = df.iloc[:idx_split].reset_index(drop=True), df.iloc[idx_split:].reset_index(drop=True)

    act_train = act_train[:, layer]
    act_test = act_test[:, layer]

    if reduction_method in ['PCA']:
        act_train = normalizer.fit_transform(act_train)
        act_test = normalizer.transform(act_test)

    if reduction_method in ['PLS']:
        min_label, max_label = labels_train.min(), labels_train.max()
        labels_train = (labels_train - min_label) / (max_label - min_label)
        labels_test = (labels_test - min_label) / (max_label - min_label)

    rmodel.fit(act_train, labels_train)
    act_plot_red = rmodel.transform(act_test if plot_test else act_train)
    plotted_labels = labels_test if plot_test else labels_train
    df_plot = df_test if plot_test else df_train

    score = rmodel.score(act_test, labels_test) if plot_test else rmodel.score(act_train, labels_train)
    print(f"Layer: {layer} - Score: {score:.4f}")
 
    if postprocess_func:
        plotted_labels = postprocess_func(plotted_labels)

    # Prepare plot
    if len(components) == 3:
        fig = px.scatter_3d(
            x=act_plot_red[:, components[0]],
            y=act_plot_red[:, components[1]],
            z=act_plot_red[:, components[2]],
            color=df_plot[color_by] if color_by else plotted_labels,
            hover_name=df_plot[label_col_str] if label_col_str else None,
            title=title or f"{reduction_method} Visualization - Layer {layer}",
            labels={'x': 'Component 0', 'y': 'Component 1', 'z': 'Component 2'},
        )
    else:
        fig = px.scatter(
            x=act_plot_red[:, components[0]],
            y=act_plot_red[:, components[1]],
            color=df_plot[color_by] if color_by else plotted_labels,
            hover_name=df_plot[label_col_str] if label_col_str else None,
            title=title or f"{reduction_method} Visualization - Layer {layer}",
            labels={'x': 'Component 0', 'y': 'Component 1'},
        )

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        scene=dict(
            xaxis_title='Component 0',
            yaxis_title='Component 1',
            zaxis_title='Component 2',
        ),
        scene_camera=dict(
            eye=dict(x=0.8, y=0.8, z=0.8),
            up=dict(x=0, y=0, z=1)
        ),
        legend_title_text='City',
    )
    return fig, rmodel, act_plot_red, plotted_labels

# sunset_continuous = ['#41476BFF', '#675478FF', '#9E6374FF', '#C67B6FFF', '#DE9B71FF', '#EFBC82FF', '#FBDFA2FF']
# base_colors = [
#     '#41476BFF', '#675478FF', '#9E6374FF',
#     '#C67B6FFF', '#DE9B71FF', '#EFBC82FF', '#FBDFA2FF'
# ]
# looping_colors = base_colors + base_colors[-2::-1]  # exclude the last to avoid a hard repeat

# sunset_base = LinearSegmentedColormap.from_list("sunset_base", base_colors)
# sunset_looping = LinearSegmentedColormap.from_list("sunset_looping", looping_colors)
# sunset_categorical = sns.color_palette([
#     '#41476B',  # Deep indigo (base anchor)
#     '#FBDFA2',  # Pale gold (base anchor)
#     '#008080',  # Teal
#     '#D81B60',  # Vivid magenta
#     '#1E88E5',  # Vivid blue
#     '#FFC107',  # Bright amber
#     '#43A047',  # Emerald green
#     '#E53935',  # Vivid red
# ])

class ColorMap2DSet1(BaseColorMap2D):
    """
    2D colormap interpolating between first 4 Set1 colors from seaborn.

    Interpolation: Top-left, top-right, bottom-left, bottom-right.
    """

    def __init__(self,
                 range_x: Tuple[float, float] = (0.0, 1.0),
                 range_y: Tuple[float, float] = (0.0, 1.0),
                 resolution: int = 256) -> None:

        # Step 1: Get first 4 Set1 colors (TL, TR, BL, BR)
        base_colors = np.array(sns.color_palette("muted", 9)[:4])
        # base_colors = np.array(sns.color_palette([
        #     '#E41A1C',  # Red
        #     '#377EB8',  # Blue
        #     '#4DAF4A',  # Green
        #     '#FF7F00'   # Orange
        # ]))

        # Step 2: Generate bilinear interpolation
        def generate_bilinear_cmap(colors, res=256):
            """
            Generate a 2D RGB color map using bilinear interpolation in CIELAB space
            between 4 RGB colors.

            :param colors: List of 4 RGB colors (floats between 0 and 1), ordered as:
                        [top-left, top-right, bottom-left, bottom-right]
            :param res: Resolution (size) of the 2D color map.
            :return: (res, res, 3) array of dtype uint8
            """
            # Convert corners to Lab
            tl, tr, bl, br = [rgb2lab(np.array(c).reshape(1, 1, 3)) for c in colors]

            # Interpolation grid
            x = np.linspace(0, 1, res)
            y = np.linspace(0, 1, res)
            xv, yv = np.meshgrid(x, y)
            xv = xv[..., None]  # shape (res, res, 1)
            yv = yv[..., None]

            # Bilinear interpolation in Lab space
            lab = (
                (1 - xv) * (1 - yv) * tl +
                xv * (1 - yv) * tr +
                (1 - xv) * yv * bl +
                xv * yv * br
            )

            # Convert back to RGB
            rgb = lab2rgb(lab)

            # Clamp and convert to uint8
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            return rgb_uint8

        cmap_data = generate_bilinear_cmap(base_colors, resolution)

        # Step 3: Save to temporary file
        tmp_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        np.save(tmp_file, cmap_data)
        tmp_file.close()

        # Step 4: Call super().__init__ with path to tmp_file
        super().__init__(colormap_npy_loc=tmp_file.name,
                         range_x=range_x, range_y=range_y)

    def sample(self, x, y):
        return super()._sample(x, y)

