import itertools
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from multiprocessing import Process
import os
import time
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModel, AutoTokenizer, GPT2Config
import torch
import transformers
from matplotlib import cm
import datetime
from tqdm import tqdm
import numpy as np
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import LlamaConfig, AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from tqdm import tqdm
import ast
from time_stuff import clean, activate_eval, ActivationDataset, activate_eval_intervene
import gc
from time_stuff.utils import SupervisedMDS


def explode_dict(d):
    # Turn every value in the dictionary into a list if it is not already
    d = {k: v if isinstance(v, list) else [v] for k, v in d.items()}
    # Create combinations
    keys = d.keys()
    values = d.values()
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations


def datetime_to_dayofyear(x):
    # Convert to datetime
    x = pd.to_datetime(x)
    # Get the day of the year
    return x.day_of_year


def train_smds(activations_train, labels_train,
                     n_components=2, manifold='discrete_circular',
                     ):
    
    rmodel = SupervisedMDS(n_components=n_components, manifold=manifold)

    rmodel.fit(activations_train, labels_train)

    if labels_train.ndim > 1:
        raise NotImplementedError("Multi-dimensional labels not supported in this version.")

    return rmodel


configs = [
    {
        'model_name': 'meta-llama/Llama-3.2-3B-Instruct', 
        'intervention_layer': 1,
        'manifold': 'circular',
        'dataset': 'datasets/templates/date_3way.csv',
        'label_column': 'correct_date',
        'target_column': 'correct_date_str',
        'preprocess_func': datetime_to_dayofyear,
        'delta_token': 0,
    },
    {
        'model_name': 'meta-llama/Llama-3.2-3B-Instruct', 
        'intervention_layer': 1,
        'manifold': 'cluster',
        'dataset': 'datasets/templates/date_3way_season.csv',
        'label_column': 'correct_season_label',
        'target_column': 'correct_date_str',
        'delta_token': 0,
    },
    {
        'model_name': 'meta-llama/Llama-3.2-3B-Instruct', 
        'intervention_layer': 1,
        'manifold': 'cluster',
        'dataset': 'datasets/templates/time_of_day_3way_phase.csv',
        'label_column': 'correct_phase_label',
        'target_column': 'correct_time_expr',
        'delta_token': 0,
    },
]

explode_settings = {
    'noise_scale': [0, 0.5, 1, 5, 10, 50],
    'n_components': [2, 5, 10, 50, 100, 500],
    # 'n_components': [2],
    'intervention_type': ['full', 'rand', 'replace'],
    # 'noise_scale': [10],
}

settings = {
    'delta_token': 0,
    'debug': False,
    'frac': 1,  # Fraction of the dataset to use
}

def run_model(config):
    model_name = config['model_name']
    intervention_layer = config['intervention_layer']
    intervention_type = config['intervention_type']
    noise_scale = config.get('noise_scale', 1)  # Default to 0 if not specified
    dataset = config['dataset']
    label_column = config['label_column']
    target_column = config['target_column']
    preprocess_func = config.get('preprocess_func', None)
    debug = config.get('debug', False)
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]

    manifold = config['manifold']
    n_components = config['n_components']
    delta_token = config['delta_token']

    frac = settings.get('frac', 1)
    if intervention_type == 'replace':
        save_path = f"results/intervention_{intervention_type}/{model_name.split('/')[-1]}/{dataset_name}/{dataset_name}_n{n_components}_s{noise_scale}.pt"
    else:
        save_path = f"results/intervention_{intervention_type}/{model_name.split('/')[-1]}/{dataset_name}/{dataset_name}_s{noise_scale}.pt"
    if os.path.exists(save_path):
        print(f"Results already exist for {save_path}. Skipping.")
        return

    print(f"Running model={model_name} dataset={dataset_name} int_type={intervention_type} noise_scale={noise_scale} n_components={n_components}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        device="cuda",
        dtype=torch.bfloat16,
    )
    model.eval()
    model.to('cuda')


    with torch.no_grad():
        print(f"Running dataset: {dataset}")

        # Retrieve dataset from existing activation dataset
        ad_base = ActivationDataset.load(f"results/{model_name.split('/')[-1]}/{dataset_name}.pt")
        df = ad_base.get_metadata_df().copy()

        # Do a train-test split of activations, labels and dataset
        activations, labels = ad_base.get_slice(target_name=target_column, columns=label_column, preprocess_funcs=preprocess_func, filter_incorrect=False)
        activations = activations[:, intervention_layer, :]  # Select the specified layer's activations
        activations_train = activations[:int(len(activations) * 0.5)]
        activations_test = activations[int(len(activations) * 0.5):]
        labels_train = labels[:int(len(labels) * 0.5)]
        labels_test = labels[int(len(labels) * 0.5):]
        df_train = df[:int(len(df) * 0.5)]
        df_test = df[int(len(df) * 0.5):]

        if intervention_type == 'replace':
            # Train a specific SMDS on the activations
            smds = train_smds(
                activations_train,
                labels_train,
                n_components=n_components,
                manifold=manifold,
            )
        else:
            smds = None  # No SMDS needed for 'full' intervention type
        # Re-run the model on the dataset using SMDS to intervene
        adf = activate_eval_intervene(df_test, dataset_name, model, tokenizer,
                            smds=smds,
                            intervention_layer=intervention_layer,
                            intervention_type=intervention_type,
                            noise_scale=noise_scale,
                            n_components=n_components,
                            target_column=target_column,
                            label_columns=label_column,
                            extra_columns=[target_column],
                            delta_token=delta_token,)

        # Print accuracy
        print(f"Model: {model_name}, Dataset: {dataset_name}, Accuracy: {adf.get_accuracy()}")
        gc.collect()
        torch.cuda.empty_cache()
        if debug:
            adf.save("results/debug.pt")
            # adf.save(f"results/{model_name.split('/')[-1]}/{dataset_name}.pt")
        elif frac < 1:
            adf.save(f"results/test_check/{model_name.split('/')[-1]}/{dataset_name}.pt")
        else:
            adf.save(save_path)
        del adf

        print(f"Saved results for {model_name} on {dataset_name}")

    print(f"Finished model: {model_name}")

# Main loop
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # safer for CUDA

    # Explode the settings for each model
    for c in configs:
        c.update(explode_settings)
    exp_configs = []
    for c in configs:
        exp_configs.extend(explode_dict(c))
    configs = exp_configs

    for c in configs:
        # run_model_on_datasets(model_name, datasets, delta_token=settings['delta_token'], debug=settings.get('debug', False))
        p = Process(target=run_model, args=(c,))
        p.start()
        p.join()  # Wait for one model to finish before starting next
        time.sleep(1)