from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import random
from time_stuff.utils import ActivationDataset, SupervisedMDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import Normalizer
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS
from umap import UMAP
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.patheffects as pe
from pycolormap_2d import ColorMap2DZiegler
from sklearn.model_selection import KFold


# General script for plotting activations
def datetime_to_dayofyear(x):
    # Convert to datetime
    x = pd.to_datetime(x)
    # Get the day of the year
    return x.day_of_year

def datetime_to_month(x):
    # Convert to datetime
    x = pd.to_datetime(x)
    # Get the month
    return x.month

def farthest_point_sampling(X, k, noise=0.1):
    n_points = X.shape[0]
    selected_indices = [np.random.randint(n_points)]
    distances = np.full(n_points, np.inf)

    for _ in range(1, k):
        last_selected = X[selected_indices[-1]]
        dist_to_last = np.linalg.norm(X - last_selected, axis=1)
        distances = np.minimum(distances, dist_to_last)
        # Add noise to distances proportional to their magnitude
        distances += noise * np.abs(distances) * np.random.rand(n_points)
        next_index = np.argmax(distances)
        selected_indices.append(next_index)

    return selected_indices

def explode_dict(d):
    # Turn every value in the dictionary into a list if it is not already
    d = {k: v if isinstance(v, list) else [v] for k, v in d.items()}
    # Create combinations
    keys = d.keys()
    values = d.values()
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations

# Define a wrapper for parallel execution
def process_setting(setting):
    return score_activations(**setting)

# Parallel processing
def parallel_score(settings, max_workers=None):
    df_list = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_setting, s): s for s in settings}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            scores_df = future.result()
            df_list.append(scores_df)

    return pd.concat(df_list, ignore_index=True)

def score_activations(path: str, label_col: str, reduction_method: str, k=5, target_columns=None, layers=None,
                     n_components=2, manifold=None, preprocess_func=None, label_shift=0, max_samples=None):
    ad = ActivationDataset.load(path)
    
    if layers is None: # Use all layers save for the first
        layers = range(1, ad.activations['correct_answer'].shape[1])

    if target_columns is None: # Use all columns
        target_columns = ['correct_answer', 'last_prompt_token'] + ad.global_metadata['extra_columns']
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    norm = Normalizer()

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
        rmodel = SupervisedMDS(n_components=n_components, manifold=manifold)
    else:
        raise ValueError(f"Unknown reduction method: {reduction_method}")
    
    if isinstance(preprocess_func, str):
        preprocess_func = [preprocess_func]
    
    preprocess_func_lambdas = []
    for i, func in enumerate(preprocess_func):
        if func == 'datetime_to_dayofyear':
            preprocess_func_lambdas.append(lambda x: pd.to_datetime(x).day_of_year)
        elif func == 'datetime_to_month':
            preprocess_func_lambdas.append(lambda x: pd.to_datetime(x).month)
        elif func == 'datetime_to_year':
            preprocess_func_lambdas.append(lambda x: pd.to_datetime(x).year + label_shift)
        elif func == 'datetime_to_hour':
            preprocess_func_lambdas.append(lambda x: pd.to_datetime(x).hour)
        elif func == 'log':
            preprocess_func_lambdas.append(lambda x: np.log(x + 1))

    scores = []
    for target_col in target_columns:
        activations, labels = ad.get_slice(target_name=target_col, columns=label_col, preprocess_funcs=preprocess_func_lambdas, filter_incorrect=True)
        labels = np.squeeze(labels)

        if max_samples is not None and activations.shape[0] > max_samples:
            activations = activations[:max_samples]
            labels = labels[:max_samples]

        for i, layer in tqdm(enumerate(layers), desc=f"Processing layer {target_col}"):
            try:
                # Define a K-Fold cross-validation generator
                kf = KFold(n_splits=k, random_state=42, shuffle=True)
                fold_scores = []
                for fold, (train_index, test_index) in enumerate(kf.split(activations)):
                    activations_layer_train = activations[train_index, layer]
                    activations_layer_dev = activations[test_index, layer]
                    labels_train = labels[train_index]
                    labels_dev = labels[test_index]

                    if reduction_method in ['PCA']:
                        activations_layer_train = norm.fit_transform(activations_layer_train)
                        activations_layer_dev = norm.transform(activations_layer_dev)

                    # Fit the model
                    rmodel.fit(activations_layer_train, labels_train)
                    # Transform the data
                    activations_reduced_train = rmodel.transform(activations_layer_train)
                    activations_reduced_dev = rmodel.transform(activations_layer_dev)

                    if reduction_method == 'LDA' and activations_reduced_dev.shape[1] <= 1:
                        print(f"Layer {layer} has collinear centroids. Skipping.")
                        continue

                    fold_scores.append(rmodel.score(activations_layer_dev, labels_dev))

                # Average the scores across folds
                avg_score = np.mean(fold_scores)
                score = {
                    'preprocess_func': preprocess_func,
                    'n_components': n_components,  
                    'k': k,
                    'manifold': manifold,
                    'layer': layer,
                    'target_col': target_col,
                    'reduction_method': reduction_method,
                    'score': avg_score
                }
                score.update(ad.global_metadata)
                scores.append(score)
            except Exception as e:
                print(f"Something went wrong with layer {layer} and target column {target_col}: {e}")
                score = {
                    'preprocess_func': preprocess_func,
                    'n_components': n_components,  
                    'k': k,
                    'manifold': manifold,
                    'layer': layer,
                    'target_col': target_col,
                    'reduction_method': reduction_method,
                    'score': None
                }
                score.update(ad.global_metadata)
                scores.append(score)

    return pd.DataFrame(scores)

if __name__ == "__main__":
    # (path: str, label_col: str, reduction_method: str, k=5, target_columns=None, layers=None,
    #                  n_components=2, manifold=None, preprocess_func=None, label_shift=0, max_samples=None)
    global_scoring_settings = {
        'reduction_method': 'SMDS',
        'k': 5,
        'target_columns': None,  # Use all columns
        'layers': None,  # Use all layers
        'n_components': 2,
        'manifold': ['euclidean', 'circular', 'semicircular', 'log_linear', 'log_semicircular'],
        'max_samples': 20
    }
    scoring_settings = [
        {'path': 'results/gemma-2-2b-it/date_3way.pt', 'label_col': 'correct_date', 'preprocess_func': 'datetime_to_dayofyear', },
        {'path': 'results/gemma-2-2b-it/date_3way_season.pt', 'label_col': 'correct_season_label'},
        {'path': 'results/gemma-2-2b-it/date_3way_temperature.pt', 'label_col': 'correct_temperature_label',},
        {'path': 'results/gemma-2-2b-it/time_of_day_3way.pt', 'label_col': 'correct_time', 'preprocess_func': 'datetime_to_hour', },
        {'path': 'results/gemma-2-2b-it/time_of_day_3way_phase.pt', 'label_col': 'correct_phase_label'},
        {'path': 'results/gemma-2-2b-it/duration_3way.pt', 'label_col': ['correct_duration_length', 'correct_date'], 'preprocess_func':['log', 'datetime_to_dayofyear'], },
        {'path': 'results/gemma-2-2b-it/periodic_3way.pt', 'label_col': 'correct_period_length', 'preprocess_func': 'log', },
        {'path': 'results/gemma-2-2b-it/notable_3way.pt', 'label_col': 'correct_date', 'preprocess_func': 'datetime_to_year', 'label_shift': -2023},

        {'path': 'results/Llama-3.1-8B-Instruct/date_3way.pt', 'label_col': 'correct_date', 'preprocess_func': 'datetime_to_dayofyear', },
        {'path': 'results/Llama-3.1-8B-Instruct/date_3way_season.pt', 'label_col': 'correct_season_label'},
        {'path': 'results/Llama-3.1-8B-Instruct/date_3way_temperature.pt', 'label_col': 'correct_temperature_label',},
        {'path': 'results/Llama-3.1-8B-Instruct/time_of_day_3way.pt', 'label_col': 'correct_time', 'preprocess_func': 'datetime_to_hour', },
        {'path': 'results/Llama-3.1-8B-Instruct/time_of_day_3way_phase.pt', 'label_col': 'correct_phase_label'},
        {'path': 'results/Llama-3.1-8B-Instruct/duration_3way.pt', 'label_col': ['correct_duration_length', 'correct_date'], 'preprocess_func':['log', 'datetime_to_dayofyear'], },
        {'path': 'results/Llama-3.1-8B-Instruct/periodic_3way.pt', 'label_col': 'correct_period_length', 'preprocess_func': 'log', },
        {'path': 'results/Llama-3.1-8B-Instruct/notable_3way.pt', 'label_col': 'correct_date', 'preprocess_func': 'datetime_to_year', 'label_shift': -2023},

        {'path': 'results/Qwen2.5-3B-Instruct/date_3way.pt', 'label_col': 'correct_date', 'preprocess_func': 'datetime_to_dayofyear', },
        {'path': 'results/Qwen2.5-3B-Instruct/date_3way_season.pt', 'label_col': 'correct_season_label'},
        {'path': 'results/Qwen2.5-3B-Instruct/date_3way_temperature.pt', 'label_col': 'correct_temperature_label',},
        {'path': 'results/Qwen2.5-3B-Instruct/time_of_day_3way.pt', 'label_col': 'correct_time', 'preprocess_func': 'datetime_to_hour', },
        {'path': 'results/Qwen2.5-3B-Instruct/time_of_day_3way_phase.pt', 'label_col': 'correct_phase_label'},
        {'path': 'results/Qwen2.5-3B-Instruct/duration_3way.pt', 'label_col': ['correct_duration_length', 'correct_date'], 'preprocess_func':['log', 'datetime_to_dayofyear'], },
        {'path': 'results/Qwen2.5-3B-Instruct/periodic_3way.pt', 'label_col': 'correct_period_length', 'preprocess_func': 'log', },
        {'path': 'results/Qwen2.5-3B-Instruct/notable_3way.pt', 'label_col': 'correct_date', 'preprocess_func': 'datetime_to_year', 'label_shift': -2023},
    ]

    # Explode the global settings
    global_scoring_settings_exploded = explode_dict(global_scoring_settings)
    # Make all pairs of global and scoring settings
    all_settings = []
    for global_settings in global_scoring_settings_exploded:
        for scoring_setting in scoring_settings:
            combined_settings = {**global_settings, **scoring_setting}
            if 'duration' in combined_settings['path'] and 'manifold' != 'euclidean':
                continue
            all_settings.append(combined_settings)

    # Run the scoring in parallel
    scores_df = parallel_score(all_settings, max_workers=16)
    # Save the scores to a CSV file
    scores_df.to_csv("results/scores.csv", header=True, index=False)