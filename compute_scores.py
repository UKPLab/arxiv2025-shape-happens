from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from time_stuff.utils import ActivationDataset, SupervisedMDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import Normalizer
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
from tqdm import tqdm
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

def process_layer(args):
    (layer, label_col, target_col, activations, labels, reduction_method, n_components,
     manifold, k, preprocess_func, global_metadata) = args
    
    if preprocess_func is not None and isinstance(preprocess_func, list) and len(preprocess_func) == 1:
        preprocess_func = preprocess_func[0]
    
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

    try:
        kf = KFold(n_splits=k, random_state=42, shuffle=True)
        fold_scores = []

        for train_index, test_index in kf.split(activations):
            train_acts = activations[train_index, layer]
            test_acts = activations[test_index, layer]
            train_labels = labels[train_index]
            test_labels = labels[test_index]

            if reduction_method == 'PCA':
                train_acts = norm.fit_transform(train_acts)
                test_acts = norm.transform(test_acts)

            rmodel.fit(train_acts, train_labels)
            reduced_test = rmodel.transform(test_acts)

            if reduction_method == 'LDA' and reduced_test.shape[1] <= 1:
                return None  # Skip collinear case

            fold_scores.append(rmodel.score(test_acts, test_labels))

        return {
            'preprocess_func': preprocess_func,
            'n_samples': len(labels),
            'n_components': n_components,
            'k': k,
            'manifold': manifold,
            'layer': layer,
            'target_col': target_col,
            'reduction_method': reduction_method,
            'score': float(np.mean(fold_scores)),
            'label_col': label_col,
            **global_metadata
        }

    except Exception as e:
        print(f"Error in layer {layer}, target_col {target_col}: {e}")
        return {
            'preprocess_func': preprocess_func,
            'n_components': n_components,
            'k': k,
            'manifold': manifold,
            'layer': layer,
            'target_col': target_col,
            'reduction_method': reduction_method,
            'score': None,
            **global_metadata
        }

def score_activations(path: str, label_col: str, reduction_method: str, k=5, target_columns=None, layers=None,
                      n_components=2, manifold=None, preprocess_func=None, label_shift=0, max_samples=None, max_workers=4):

    ad = ActivationDataset.load(path)

    if layers is None:
        layers = range(1, ad.activations['correct_answer'].shape[1])

    if target_columns is None:
        target_columns = ['correct_answer', 'last_prompt_token'] + ad.global_metadata['extra_columns']
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    if isinstance(preprocess_func, str):
        preprocess_func = [preprocess_func]

    if preprocess_func is None:
        preprocess_func_lambdas = None
    else:
        preprocess_func_lambdas = []
        for func in preprocess_func or []:
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

    all_scores = []

    for target_col in target_columns:
        activations, labels = ad.get_slice(
            target_name=target_col,
            columns=label_col,
            preprocess_funcs=preprocess_func_lambdas,
            filter_incorrect=True
        )
        labels = np.squeeze(labels)

        if max_samples is not None and activations.shape[0] > max_samples:
            activations = activations[:max_samples]
            labels = labels[:max_samples]

        # Prepare args list
        args_list = [
            (
                layer,
                label_col,
                target_col,
                activations,
                labels,
                reduction_method,
                n_components,
                manifold,
                k,
                preprocess_func,
                ad.global_metadata
            )
            for layer in layers
        ]

        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_layer, args_list)

            for result in tqdm(results, total=len(args_list), desc=f"Target: {target_col}"):
                if result is not None:
                    all_scores.append(result)

    return pd.DataFrame(all_scores)

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
        'max_samples': 500
    }
    scoring_settings = [
        {'path': 'results/gemma-2-2b-it/date_3way.pt', 'label_col': 'correct_date', 'preprocess_func': 'datetime_to_dayofyear', },
        {'path': 'results/gemma-2-2b-it/date_3way_season.pt', 'label_col': 'correct_season_label',},
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

    # Run the scoring
    all_scores = []
    for setting in all_settings:
        print(f"Scoring with settings: {setting}")
        scores_df = score_activations(**setting)
        if scores_df is not None and not scores_df.empty:
            all_scores.append(scores_df)
    # Concatenate all scores into a single DataFrame
    if all_scores:
        scores_df = pd.concat(all_scores, ignore_index=True)
    else:
        scores_df = pd.DataFrame()
            
    # Save the scores to a CSV file
    print("Saving scores to results/scores.csv")
    scores_df.to_csv("results/scores.csv", header=True, index=False)