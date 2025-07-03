from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
                     annotations='random',  filter_incorrect=True, orthonormal=False,
                     preprocess_func=None, annotation_preprocess_func=None, postprocess_func=None,
                     return_fig=False):
    if len(layers) < plots_per_row:
        plots_per_row = len(layers)

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

    df = ad.get_metadata_df(filter_incorrect=filter_incorrect)

    # Split train and test sets
    split = 0.5
    activations_train = activations[:int(len(activations)*split)]
    activations_dev = activations[int(len(activations)*split):]

    labels_train = labels[:int(len(labels)*split)]
    labels_dev = labels[int(len(labels)*split):]

    df_train = df.iloc[:int(len(df)*split)].reset_index(drop=True)
    df_dev = df.iloc[int(len(df)*split):].reset_index(drop=True)

    # Standardize labels to 0-1 range
    if reduction_method in ['PLS']:
        min_label = labels_train.min()
        max_label = labels_train.max()
        labels_train = (labels_train-min_label)/(max_label-min_label)
        labels_dev = (labels_dev-min_label)/(max_label-min_label)

    if layers is None:
        layers = range(activations.shape[1])

    # Plot the data
    scaling_factor = 6 if len(layers) > 1 else 8
    fig, axs = plt.subplots(int(np.ceil(len(layers)/plots_per_row)), plots_per_row, figsize=(scaling_factor*plots_per_row, scaling_factor*len(layers)//plots_per_row), constrained_layout=True)

    for i, layer in tqdm(enumerate(layers)):
        if plots_per_row > 1 and len(layers) > plots_per_row:
            ax = axs[i//plots_per_row][i%plots_per_row]
        elif len(layers) > 1:
            ax = axs[i]
        else:
            ax = axs

        activations_layer_train = activations_train[:, layer]
        activations_layer_dev = activations_dev[:, layer]

        if reduction_method in ['PCA']:
            activations_layer_train = normalizer.fit_transform(activations_layer_train)
            activations_layer_dev = normalizer.transform(activations_layer_dev)
        
        if reduction_method in ['MDS']:
            activations_reduced_dev = rmodel.fit_transform(activations_layer_dev)
        else:
            # Fit the model
            rmodel.fit(activations_layer_train, labels_train)
            # Transform the data
            activations_reduced_train = rmodel.transform(activations_layer_train)
            activations_reduced_dev = rmodel.transform(activations_layer_dev)
        
        if reduction_method == 'LDA' and activations_reduced_dev.shape[1] <= 1:
            print(f"Layer {layer} has collinear centroids. Skipping.")
            continue
        
        if postprocess_func is not None:
            plotted_labels = postprocess_func(labels_dev)
        else:
            plotted_labels = labels_dev

        if labels_dev.ndim > 1:
            # If labels are multi-dimensional, map them to 0-1 range
            cmap_labels_dev = (plotted_labels - plotted_labels.min(axis=0)) / (plotted_labels.max(axis=0) - plotted_labels.min(axis=0))
            # Use ColorMap2DZiegler to get bidimensional colors
            # cmap = ColorMap2DZiegler()
            cmap = ColorMap2DSet1()
            hues = [cmap(l1,l2) / 255.0 for l1, l2 in cmap_labels_dev]

            # Plot the data
            if len(components) == 2:
                ax.scatter(activations_reduced_dev[:, components[0]], activations_reduced_dev[:, components[1]], c=hues, s=20)
            elif len(components) == 3:
                ax.scatter(activations_reduced_dev[:, components[0]], activations_reduced_dev[:, components[1]], 
                           activations_reduced_dev[:, components[2]], c=hues, s=20)

        else:
            hues = plotted_labels
            # Plot the data
            if manifold in ['discrete_circular', 'circular']:
                palette = 'twilight'
            elif manifold in ['log_linear', 'linear', 'euclidean', 'semicircular', 'log_semicircular']:
                # Use a continuous colormap for linear manifolds
                palette = 'flare'
            elif manifold in ['cluster']:
                palette = 'tab10'
            elif len(np.unique(hues)) < 3:
                palette = ['#3A4CC0', '#B40426']#['blue', 'red']

            if len(components) == 2:
                sns.scatterplot(x=activations_reduced_dev[:, components[0]], y=activations_reduced_dev[:, components[1]], 
                                hue=hues, ax=ax, palette=palette, alpha=1.0)
            elif len(components) == 3:
                ax.scatter(activations_reduced_dev[:, components[0]], activations_reduced_dev[:, components[1]],
                           activations_reduced_dev[:, components[2]], c=hues, s=20, cmap=palette)
            # ax.get_legend().set_visible(False)

        # Set title
        ax.set_title(f"Layer {layer}")

        if reduction_method not in ['UMAP', 'MDS']:
            print(f"Layer: {layer} Score: {rmodel.score(activations_layer_dev, labels_dev)}")

        if label_col_str is not None:
            if annotations == 'random':
                # Select a few random sentences
                indices = np.random.choice(len(activations_layer_dev), size=10, replace=False)
                rnd_activations = activations_reduced_dev[indices]
                txt = df_dev.iloc[indices][label_col_str].values

                points = rnd_activations

            elif annotations == 'uniform':
                indices = farthest_point_sampling(activations_reduced_dev, 15)
                rnd_activations = activations_reduced_dev[indices]
                txt = df_dev.iloc[indices][label_col_str].values

                points = rnd_activations

            elif annotations == 'class':
                # Select sentences so that they represent one item per class in label_col_str
                unique_labels = df_dev[label_col_str].unique()
                indices = []
                for label in unique_labels:
                    # Get the index of the first occurrence of the label
                    idx = df_dev[df_dev[label_col_str] == label].index[0]
                    indices.append(idx)
                
                # Get the activations and sentences for the selected indices
                rnd_activations = activations_reduced_dev[indices]
                txt = df_dev.iloc[indices][label_col_str].values

                points = rnd_activations


            elif annotations=='centroids':
                # Compute the centroids of each class
                centroids = []
                unique_labels = np.unique(labels_dev)
                
                # Cap the number of centroids to 12
                if unique_labels.shape[0] > 16:
                    unique_labels = np.random.choice(unique_labels, size=16, replace=False)

                for label in unique_labels:
                    # Get the indices of the samples with the current label
                    indices = np.where(labels_dev == label)[0]
                    # Compute the centroid of the samples with the current label
                    centroid = np.mean(activations_reduced_dev[indices], axis=0)
                    centroids.append(centroid)
                centroids = np.array(centroids)

                # Compute the corresponding txt
                txt = []
                for label in unique_labels:
                    # Get the indices of the samples with the current label
                    indices = np.where(labels_dev == label)[0]
                    # Get the first sample with the current label
                    txt.append(df_dev.iloc[indices[0]][label_col_str])
                points = centroids

            else:
                raise ValueError("Invalid annotations value. Use 'random', 'class' or 'centroids'.")

            # Plot the points
            ax.scatter(points[:, components[0]], points[:, components[1]], color='red', edgecolor='k', s=20)

            if annotation_preprocess_func is not None:
                # Preprocess the text
                txt = [annotation_preprocess_func(t) for t in txt]

            # Annotate the points
            for j, txt in enumerate(txt):
                ax.annotate(txt, (points[j, components[0]], points[j, components[1]]), fontsize=16,  path_effects=[pe.withStroke(linewidth=2, foreground="white")])


    # Set the title
    if title is not None:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle(f"{reduction_method} - {ad.model_name} - {ad.dataset_name}", fontsize=20)

    
    # plt.tight_layout()
    if return_fig:
        return fig, axs
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()        
    plt.close(fig)

def plot_activations_single(ad: ActivationDataset, label_col: str, reduction_method, layer, ax, target_col='correct_answer',  components=(0,1),
                     label_col_str=None, n_components=2, manifold='discrete_circular', palette='viridis', title=None, save_path=None, plots_per_row=4,
                     annotations=None,  filter_incorrect=True, orthonormal=False,
                     preprocess_func=None, annotation_preprocess_func=None, postprocess_func=None,
                     return_fig=False, max_samples=500, annotation_center_bias=0.1, annotation_offset=(0,0)):

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

    # if postprocess_func is not None:
    #     labels = postprocess_func(labels)

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

    if reduction_method in ['PLS']:
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

    act_train_red = rmodel.transform(act_train)
    act_test_red = rmodel.transform(act_test)

    if postprocess_func is not None:
        plotted_labels = postprocess_func(labels_test)
    else:
        plotted_labels = labels_test

    if labels.ndim > 1:
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
            ax.scatter(act_test_red[:, components[0]], act_test_red[:, components[1]], c=hues, s=25)
        elif len(components) == 3:
            ax.scatter(act_test_red[:, components[0]], act_test_red[:, components[1]],
                       act_test_red[:, components[2]], c=hues, s=20)
    else:
        palette = palette if len(np.unique(labels)) > 2 else ['#3A4CC0', '#B40426']
        sns.scatterplot(
            x=act_test_red[:, components[0]], 
            y=act_test_red[:, components[1]],
            hue=plotted_labels,
            ax=ax,
            palette=palette,
            alpha=1.0,
            s=40
        )

    if label_col_str is not None and annotations is not None:
        # Preprocess all labels first (if needed)
        preprocessed_labels = df_test[label_col_str].values
        if annotation_preprocess_func is not None:
            preprocessed_labels = np.array([annotation_preprocess_func(t) for t in preprocessed_labels])

        if annotations == 'random':
            indices = np.random.choice(len(activations_test), size=10, replace=False)
            points = act_test_red[indices]
            txt = preprocessed_labels[indices]

        elif annotations == 'uniform':
            # indices = farthest_point_sampling(act_test_red, 10, center_bias=annotation_center_bias)
            unique_labels = np.unique(labels_test)
            if unique_labels.shape[0] > 16:
                # unique_labels = np.random.choice(unique_labels, size=16, replace=False)
                indices = np.linspace(0, len(unique_labels)-1, 16, dtype=int)
                unique_labels = unique_labels[indices]
            points = act_test_red[indices]
            txt = preprocessed_labels[indices]

        elif annotations == 'class':
            unique_labels = df_test[label_col_str].unique()
            indices = [df_test[df_test[label_col_str] == label].index[0] for label in unique_labels]
            points = act_test_red[indices]
            txt = preprocessed_labels[indices]

        elif annotations == 'centroids':
            unique_labels = np.unique(labels_test)
            if unique_labels.shape[0] > 16:
                # unique_labels = np.random.choice(unique_labels, size=16, replace=False)
                indices = np.linspace(0, len(unique_labels)-1, 16, dtype=int)
                unique_labels = unique_labels[indices]

            points = []
            txt = []
            for label in unique_labels:
                indices = np.where(labels_test == label)[0]
                centroid = np.mean(act_test_red[indices], axis=0)
                points.append(centroid)

                # Get the first index of the label to extract its preprocessed text
                txt.append(preprocessed_labels[indices[0]])

            points = np.array(points)

        else:
            raise ValueError("Invalid annotations value. Use 'random', 'class' or 'centroids'.")

        # Plot and annotate
        ax.scatter(points[:, components[0]], points[:, components[1]], color='red', edgecolor='k', s=20)

        for j, label in enumerate(txt):
            ax.annotate(
                label,
                (points[j, components[0]] + annotation_offset[0],
                 points[j, components[1]] + annotation_offset[1]),
                fontsize=14,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
            )

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

