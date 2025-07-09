import argparse
import itertools
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from multiprocessing import Process, set_start_method
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
from time_stuff.config import Runner


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
        debug = args.get('debug', False)
        dataset_name = os.path.splitext(os.path.basename(dataset))[0]
        manifold = args['manifold']
        n_components = args['n_components']
        delta_token = args['delta_token']
        frac = args.get('frac', 1)

        if intervention_type in ['replace', 'rand']:
            save_path = f"results/intervention_{intervention_type}/{model_name.split('/')[-1]}/{dataset_name}/{dataset_name}_n{n_components}_s{noise_scale}.pt"
        else:
            save_path = f"results/intervention_{intervention_type}/{model_name.split('/')[-1]}/{dataset_name}/{dataset_name}_s{noise_scale}.pt"

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
            activations, labels = ad_base.get_slice(target_name=target_column, columns=label_column, preprocess_funcs=preprocess_func, filter_incorrect=False)
            activations = activations[:, intervention_layer, :]
            mid = int(len(activations) * 0.5)
            activations_train, activations_test = activations[:mid], activations[mid:]
            labels_train, labels_test = labels[:mid], labels[mid:]
            df_train, df_test = df[:mid], df[mid:]

            smds = train_smds(activations_train, labels_train, n_components, manifold) if intervention_type == 'replace' else None

            adf = activate_eval_intervene(df_test, dataset_name, model, tokenizer,
                                          smds=smds,
                                          intervention_layer=intervention_layer,
                                          intervention_type=intervention_type,
                                          noise_scale=noise_scale,
                                          n_components=n_components,
                                          target_column=target_column,
                                          label_columns=label_column,
                                          extra_columns=[target_column],
                                          delta_token=delta_token)

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
