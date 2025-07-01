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
from sklearn.manifold import SupervisedMDS
from time_stuff.utils import ActivationDataset
from time_stuff.utils import ColorMap2DZiegler, farthest_point_sampling
from matplotlib import patheffects as pe


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
    if postprocess_func is not None:
        labels = postprocess_func(labels)

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
        
        if labels_dev.ndim > 1:
            # If labels are multi-dimensional, map them to 0-1 range
            cmap_labels_dev = (labels_dev - labels_dev.min(axis=0)) / (labels_dev.max(axis=0) - labels_dev.min(axis=0))
            # Use ColorMap2DZiegler to get bidimensional colors
            cmap = ColorMap2DZiegler()
            hues = [cmap(l1,l2) / 255.0 for l1, l2 in cmap_labels_dev]

            # Plot the data
            if len(components) == 2:
                ax.scatter(activations_reduced_dev[:, components[0]], activations_reduced_dev[:, components[1]], c=hues, s=20)
            elif len(components) == 3:
                ax.scatter(activations_reduced_dev[:, components[0]], activations_reduced_dev[:, components[1]], 
                           activations_reduced_dev[:, components[2]], c=hues, s=20)

        else:
            hues = labels_dev
            # Plot the data
            palette = 'viridis' if len(np.unique(hues)) > 2 else ['#3A4CC0', '#B40426']#['blue', 'red']

            if len(components) == 2:
                sns.scatterplot(x=activations_reduced_dev[:, components[0]], y=activations_reduced_dev[:, components[1]], 
                                hue=hues, ax=ax, palette=palette, alpha=1.0)
            elif len(components) == 3:
                ax.scatter(activations_reduced_dev[:, components[0]], activations_reduced_dev[:, components[1]],
                           activations_reduced_dev[:, components[2]], c=hues, s=20, cmap=palette)
            ax.get_legend().set_visible(False)
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
    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

def plot_activations_single(ad: ActivationDataset, label_col: str, reduction_method, layer, ax, target_col='correct_answer',  components=(0,1),
                     label_col_str=None, n_components=2, manifold='discrete_circular', palette='viridis', title=None, save_path=None, plots_per_row=4,
                     annotations='random',  filter_incorrect=True, orthonormal=False,
                     preprocess_func=None, annotation_preprocess_func=None, postprocess_func=None,
                     return_fig=False):

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

    if postprocess_func is not None:
        labels = postprocess_func(labels)

    df = ad.get_metadata_df(filter_incorrect=filter_incorrect)

    max_samples = min(500, len(activations)) # Limit to 500 samples  
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

    if labels.ndim > 1:
        raise NotImplementedError("Multi-dimensional labels not supported in this version.")

    palette = palette if len(np.unique(labels)) > 2 else ['#3A4CC0', '#B40426']

    # Plot training data (lighter)
    # sns.scatterplot(
    #     x=act_train_red[:, components[0]], 
    #     y=act_train_red[:, components[1]],
    #     hue=labels_train,
    #     ax=ax,
    #     palette=palette,
    #     alpha=0.5,
    #     marker='o'
    # )

    # Plot test data (full opacity)
    sns.scatterplot(
        x=act_test_red[:, components[0]], 
        y=act_test_red[:, components[1]],
        hue=labels_test,
        ax=ax,
        palette=palette,
        alpha=1.0,
        # marker='X',
        s=60
    )


    # ax.set_title(f"Layer {layer}")
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
