import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap
from sklearn.model_selection import KFold
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

from shape_happens import Runner
from shape_happens.utils import ActivationDataset, SupervisedMDS


def process_layer(args):
    (control, layer, label_col, target_col, activations, labels, reduction_method, n_components,
     manifold, k, repetitions, preprocess_func, global_metadata) = args
    
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
        rep_results = []
        for i in range(repetitions):
            kf = KFold(n_splits=k, random_state=i, shuffle=True)
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

            rep_results.append({
                'preprocess_func': preprocess_func,
                'n_samples': len(labels),
                'n_components': n_components,
                'k': k,
                'repetition_id': i,
                'manifold': manifold,
                'layer': layer,
                'target_col': target_col,
                'reduction_method': reduction_method,
                'score': float(np.mean(fold_scores)), # TODO: log all fold scores to get error bars
                'fold_scores': fold_scores,
                'label_col': label_col,
                'control': control,
                **global_metadata
            })

        return rep_results  

    except Exception as e:
        print(f"Error in layer {layer}, target_col {target_col}: {e}")
        return {
            'preprocess_func': preprocess_func,
            'n_components': n_components,
            'k': k,
            'repetition_id': i,
            'manifold': manifold,
            'layer': layer,
            'target_col': target_col,
            'reduction_method': reduction_method,
            'score': None,
            'fold_scores': None,
            'label_col': label_col,
            'control': control,
            'error': str(e),
            **global_metadata
        }


class ScoreRunner(Runner):
    def __init__(self, config_path=None, save_path="results/scores/combined_scores.csv",
                 overwrite=False):
        self.save_path = save_path
        self.overwrite = overwrite
        super().__init__(config_path=config_path)

    def score_activations(self, **kwargs):
        id = self.hash_args(kwargs)

        path = kwargs["path"]
        label_col = kwargs["label_col"]
        reduction_method = kwargs["reduction_method"]

        model_name = kwargs.get("model_name", None)
        k = kwargs.get("k", 5)
        repetitions = kwargs.get("repetitions", 1)
        target_columns = kwargs.get("target_columns", None)
        layers = kwargs.get("layers", None)
        n_components = kwargs.get("n_components", 2)
        manifold = kwargs.get("manifold", None)
        preprocess_func = kwargs.get("preprocess_func", None)
        label_shift = kwargs.get("label_shift", 0)
        max_samples = kwargs.get("max_samples", None)
        control = kwargs.get("control", False)
        
        print(f"Scoring activations for {kwargs}")

        ad = ActivationDataset.load(path, model_name=model_name)

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
                    preprocess_func_lambdas.append(lambda x: np.abs(pd.to_datetime(x).year + label_shift))
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

            if control: # If this is a control task, shuffle the labels
                labels = np.random.permutation(labels)

            # Prepare args list
            args_list = [
                (
                    control,
                    layer,
                    label_col,
                    target_col,
                    activations,
                    labels,
                    reduction_method,
                    n_components,
                    manifold,
                    k,
                    repetitions,
                    preprocess_func,
                    ad.global_metadata
                )
                for layer in layers
            ]
            for args in tqdm(args_list, total=len(args_list), desc=f"Target: {target_col}"):
                results = process_layer(args)
                all_scores.extend(results)

        return pd.DataFrame(all_scores).to_csv(f"results/scores/{id}.csv", header=True, index=False)

    def run_experiment(self, args):
        return self.score_activations(**args)

    def combine_results(self, results_args):
        # Combine results from multiple experiments (e.g., aggregate metrics)
        print("Combining results...")
        ids = [self.hash_args(args) for args in results_args]
        combined_df = pd.concat([pd.read_csv(f"results/scores/{id}.csv") for id in ids], ignore_index=True)
        combined_df.to_csv(self.save_path, index=False)
        print(f"Results combined and saved to {self.save_path}")

    def results_exist(self, args):
        # Check if the results for the given args already exist
        if self.overwrite:
            return False
        id = self.hash_args(args)
        return os.path.exists(f"results/scores/{id}.csv")
    
    def validate_args(self, args):
        if 'duration' in args['path'] and args['manifold'] != 'euclidean' and isinstance(args['label_col'], list):
            print(f"Skipping {args['path']} with manifold {args['manifold']} and label_col {args['label_col']}")
            return False
        return super().validate_args(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file containing global, grid, and local configs.")
    parser.add_argument("--save_path", type=str, default="results/scores/combined_scores.csv",
                        help="Path to save the results.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing results.")
    args = parser.parse_args()

    runner = ScoreRunner(config_path=args.config, save_path=args.save_path, overwrite=args.overwrite)
    runner.run_all(multiprocessing=False)