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
from tqdm import tqdm
import ast
from time_stuff import clean, activate_eval, ActivationDataset
import gc


models = [
    'Qwen/Qwen2.5-3B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'gpt2',
    'google/gemma-2-2b-it',

    ### Not instructed ###
    # 'meta-llama/Llama-3.2-3B',
    # 'google/gemma-2-2b',
    

    ### Chonky bois ###
    # 'meta-llama/Llama-3.1-8B-Instruct',
    # 'meta-llama/Llama-3.1-8B',
    # 'google/gemma-7b-it',
    # 'Qwen/Qwen3-8B',

]

datasets = [
    # {'filename': 'datasets/templates/time_of_day_HH:MM.csv', 'label_columns': ['correct_hour', 'correct_answer'],},
    {'filename': 'datasets/templates/time_of_day_3way.csv', 'label_columns': ['correct_time', 'correct_answer', 'correct_time_diff'], 'extra_columns': ['correct_time_expr']},
    {'filename': 'datasets/templates/time_of_day_3way_phase.csv', 'label_columns': ['correct_time', 'correct_answer', 'correct_phase', 'correct_phase_label'], 'extra_columns': ['correct_time_expr']},

    {'filename': 'datasets/templates/duration_3way.csv', 'label_columns': ['correct_duration_length', 'correct_date', 'correct_end_date', 'correct_month'], 'extra_columns': ['correct_duration_str', 'correct_date_expr']},

    {'filename': 'datasets/templates/date_3way.csv', 'label_columns': ['correct_date'], 'extra_columns': ['correct_date_str']},
    # {'filename': 'datasets/templates/date_1way.csv', 'label_columns': ['correct_date'], 'extra_columns': ['correct_date_str']},
    # {'filename': 'datasets/templates/date_3way_month.csv', 'label_columns': ['correct_date', 'correct_month_label', 'correct_month'],},
    {'filename': 'datasets/templates/date_3way_season.csv', 'label_columns': ['correct_season', 'correct_season_label', 'correct_date'],},
    {'filename': 'datasets/templates/date_3way_temperature.csv', 'label_columns': ['correct_temperature', 'correct_temperature_label', 'correct_date'],},
        
    {'filename': 'datasets/templates/periodic_3way.csv', 'label_columns': ['correct_period_length', 'period_type'], 'extra_columns': ['correct_period_str']},

    {'filename': 'datasets/templates/notable_3way.csv', 'label_columns': ['correct_date']}
]

settings = {
    'delta_token': 0,
    'debug': False,
    'frac': 1,  # Fraction of the dataset to use
}

def run_model_on_datasets(model_name, datasets, delta_token=0, frac=1, debug=False):
    print(f"Running model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    model.to('cuda')
    with torch.no_grad():
        for dataset in tqdm(datasets):
            print(f"Running dataset: {dataset['filename']}")
            df = pd.read_csv(dataset['filename'])
            df = df.sample(frac=frac, random_state=42)
            dataset_name = os.path.splitext(os.path.basename(dataset['filename']))[0]
            try:
                adf = activate_eval(df, dataset_name, model, tokenizer, delta_token=delta_token,
                                    label_columns=dataset['label_columns'],
                                    extra_columns=dataset.get('extra_columns', []), debug=debug,)
            except ValueError as e:
                # If ValueError: need at least one array to stack, repeat the process forcing constrained generation
                if not e.args[0].startswith("need at least one array to stack"):
                    raise e
                print(f"ValueError: model {model_name} on dataset {dataset_name} achieved zero accuracy. Retrying with constrained generation.")
                try:
                    adf = activate_eval(df, dataset_name, model, tokenizer,
                            label_columns=dataset['label_columns'],
                            extra_columns=dataset.get('extra_columns', []),
                            constrained_generation=True)
                except ValueError as e:
                    if e.args[0].startswith("Alternatives column not found in dataset."):
                        print(f"Alternatives column not found in dataset. Skipping {dataset_name}.")
                        continue

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
                adf.save(f"results/{model_name.split('/')[-1]}/{dataset_name}.pt")
            del adf

            print(f"Saved results for {model_name} on {dataset_name}")

    print(f"Finished model: {model_name}")

# Main loop
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # safer for CUDA

    for model_name in models:
        # run_model_on_datasets(model_name, datasets, delta_token=settings['delta_token'], debug=settings.get('debug', False))
        p = Process(target=run_model_on_datasets, args=(model_name, datasets, settings['delta_token'], settings.get('frac', 1), settings.get('debug', False)))
        p.start()
        p.join()  # Wait for one model to finish before starting next
        time.sleep(5)