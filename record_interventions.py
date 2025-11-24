import argparse
import gc
import os
from multiprocessing import set_start_method

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from shape_happens import (ActivationDataset, activate_eval_intervene)
from shape_happens.config import Runner
from shape_happens.utils import SupervisedMDS


def datetime_to_dayofyear(x):
    x = pd.to_datetime(x)
    return x.day_of_year


def train_smds(activations_train, labels_train, n_components=2, manifold='discrete_circular'):
    rmodel = SupervisedMDS(n_components=n_components, manifold=manifold)
    rmodel.fit(activations_train, labels_train)
    if labels_train.ndim > 1:
        raise NotImplementedError("Multi-dimensional labels not supported.")
    return rmodel


class InterventionRunner(Runner):

    def run_experiment(self, args):
        model_name = args['model_name']
        intervention_layer = args['intervention_layer']
        intervention_type = args['intervention_type']
        noise_scale = args.get('noise_scale', 1)
        dataset = args['dataset']
        label_column = args['label_column']
        target_column = args['target_column']
        preprocess_func = args.get('preprocess_func', None)
        label_shift = args.get('label_shift', 0)
        debug = args.get('debug', False)
        dataset_name = os.path.splitext(os.path.basename(dataset))[0]
        manifold = args['manifold']
        n_components = args['n_components']
        delta_token = args['delta_token']
        k = args.get('k', 5)
        frac = args.get('frac', 1)

        if intervention_type in ['replace', 'rand']:
            save_path = f"results/intervention_{intervention_type}/{model_name.split('/')[-1]}/{dataset_name}/{dataset_name}_n{n_components}_s{noise_scale}.pt"
        elif intervention_type in ['full']:
            save_path = f"results/intervention_{intervention_type}/{model_name.split('/')[-1]}/{dataset_name}/{dataset_name}_s{noise_scale}.pt"
        elif intervention_type in ['denoise']:
            save_path = f"results/intervention_{intervention_type}/{model_name.split('/')[-1]}/{dataset_name}/{dataset_name}_n{n_components}.pt"

        if os.path.exists(save_path):
            print(f"Results already exist for {save_path}. Skipping.")
            return

        print(f"Running model={model_name} dataset={dataset_name} int_type={intervention_type} noise_scale={noise_scale} n_components={n_components}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = HookedTransformer.from_pretrained_no_processing(model_name, device="cuda", dtype=torch.bfloat16)
        model.eval().to('cuda')

        with torch.no_grad():
            ad_base = ActivationDataset.load(f"results/{model_name.split('/')[-1]}/{dataset_name}.pt")
            df = ad_base.get_metadata_df().copy()
            
            if preprocess_func == 'datetime_to_dayofyear':
                preprocess_func = lambda x: pd.to_datetime(x).day_of_year
            elif preprocess_func == 'datetime_to_month':
                preprocess_func = lambda x: pd.to_datetime(x).month
            elif preprocess_func == 'datetime_to_hour':
                preprocess_func = lambda x: pd.to_datetime(x).hour
            elif preprocess_func == 'datetime_to_year':
                preprocess_func = lambda x: np.abs(pd.to_datetime(x).year + label_shift)
            elif preprocess_func == 'log':
                preprocess_func = lambda x: np.log(x + 1)
            elif preprocess_func == None:
                preprocess_func = None
            else:
                raise ValueError(f"Unknown preprocess_func: {preprocess_func}")

            # TODO: Add kfold logic here
            kf = KFold(n_splits=k, shuffle=True, random_state=0)

            # extract activations & labels once
            activations, labels = ad_base.get_slice(
                target_name=target_column,
                columns=label_column,
                preprocess_funcs=preprocess_func,
                filter_incorrect=False
            )
            activations = activations[:, intervention_layer, :]

            all_smds = []
            all_splits = []

            for train_idx, test_idx in kf.split(activations):
                act_train, act_test = activations[train_idx], activations[test_idx]
                labels_train, labels_test = labels[train_idx], labels[test_idx]
                df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

                smds = (
                    train_smds(act_train, labels_train, n_components, manifold)
                    if intervention_type in ['replace', 'denoise']
                    else None
                )
                all_smds.append(smds)

                all_splits.append(dict(
                    df_test=df_test,
                    A_test=act_test,
                    L_test=labels_test,
                ))

            # downstream evaluation — now giving *all* SMDS and *all* test splits
            adf = activate_eval_intervene(
                all_splits,
                dataset_name,
                model,
                tokenizer,
                smds_splits=all_smds,
                intervention_layer=intervention_layer,
                intervention_type=intervention_type,
                noise_scale=noise_scale,
                n_components=n_components,
                target_column=target_column,
                label_columns=label_column,
                extra_columns=[target_column],
                delta_token=delta_token,
            )

            print(f"Model: {model_name}, Dataset: {dataset_name}, Accuracy: {adf.get_accuracy()}")


            print(f"Model: {model_name}, Dataset: {dataset_name}, Accuracy: {adf.get_accuracy()}")

            gc.collect()
            torch.cuda.empty_cache()

            if debug:
                adf.save("results/debug.pt")
            elif frac < 1:
                adf.save(f"results/test_check/{model_name.split('/')[-1]}/{dataset_name}.pt")
            else:
                adf.save(save_path)

            del adf
            print(f"Saved results for {model_name} on {dataset_name}")

        print(f"Finished model: {model_name}")

    def combine_results(self, results):
        pass

    def results_exist(self, args):
        pass

if __name__ == "__main__":
    set_start_method("spawn") # Required for CUDA

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file containing global, grid, and local configs.")
    args = parser.parse_args()

    runner = InterventionRunner(config_path=args.config)
    runner.run_all(multiprocessing=True)
