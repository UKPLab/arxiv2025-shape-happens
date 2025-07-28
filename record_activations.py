import argparse
import gc
import os
from multiprocessing import set_start_method
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

from shape_happens import ActivationDataset, activate_eval
from shape_happens.config import Runner
from shape_happens.utils import activate_eval


class ActivationRunner(Runner):
    def run_experiment(self, args):
        model_name = args['model_name']
        datasets = args['datasets']
        delta_token = args.get('delta_token', 0)
        frac = args.get('frac', 1)
        debug = args.get('debug', False)
        use_quantization = args.get('use_quantization', False)

        print(f"Running model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if use_quantization:
            print("Using 4-bit quantization via bitsandbytes.")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        model.eval()

        with torch.no_grad():
            for dataset in tqdm(datasets):
                print(f"Running dataset: {dataset['filename']}")
                df = pd.read_csv(dataset['filename'])
                df = df.sample(frac=frac, random_state=42)
                dataset_name = os.path.splitext(os.path.basename(dataset['filename']))[0]
                base_path = f"results/debug" if debug else f"results/{model_name.split('/')[-1]}/{dataset_name}"

                # Try to load existing results, if succeeds, skip evaluation
                try:
                    if os.path.exists(base_path + '.pt'):
                        adf = ActivationDataset.load(base_path + '.pt')
                    elif os.path.exists(base_path):
                        adf = ActivationDataset.load(base_path)
                    else:
                        raise FileNotFoundError(f"No existing results found for {model_name} on {dataset_name}")
                    print(f"Found existing results for {model_name} on {dataset_name}, skipping evaluation.")
                    del adf  # Free memory
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                except FileNotFoundError:
                    print(f"No existing results found for {model_name} on {dataset_name}, proceeding with evaluation.")


                try:
                    adf = activate_eval(
                        df, dataset_name, model, tokenizer,
                        delta_token=delta_token,
                        label_columns=dataset['label_columns'],
                        extra_columns=dataset.get('extra_columns', []),
                        debug=debug
                    )
                except ValueError as e:
                    if not str(e).startswith("need at least one array to stack"):
                        raise e
                    print(f"Retrying {dataset_name} with constrained generation.")
                    try:
                        adf = activate_eval(
                            df, dataset_name, model, tokenizer,
                            label_columns=dataset['label_columns'],
                            extra_columns=dataset.get('extra_columns', []),
                            constrained_generation=True
                        )
                    except ValueError as e:
                        if str(e).startswith("Alternatives column not found"):
                            print(f"Skipping {dataset_name} due to missing alternatives column.")
                            continue

                print(f"Model: {model_name}, Dataset: {dataset_name}, Accuracy: {adf.get_accuracy()}")

                extension = '.pt'
                if next(iter(adf.activations.values())).nbytes > 1e9:
                    print("Activations too large, sharding result.")
                    extension = ''

                adf.save(base_path + extension, samples_per_shard=100)
                del adf
                gc.collect()
                torch.cuda.empty_cache()

        print(f"Finished model: {model_name}")

    def results_exist(self, args):
        model_name = args['model_name']
        datasets = args['datasets']
        debug = args.get('debug', False)

        for dataset in datasets:
            dataset_name = os.path.splitext(os.path.basename(dataset['filename']))[0]
            base_path = f"results/debug" if debug else f"results/{model_name.split('/')[-1]}/{dataset_name}"
            if not (Path(base_path).with_suffix(".pt").exists() or Path(base_path).exists()):
                return False  # At least one missing
        return True

    def combine_results(self, results_args):
        pass

    def validate_args(self, args):
        return True

if __name__ == "__main__":
    set_start_method("spawn") # Required for CUDA

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file containing global, grid, and local configs.")
    args = parser.parse_args()

    runner = ActivationRunner(config_path=args.config)
    runner.run_all(multiprocessing=True)
