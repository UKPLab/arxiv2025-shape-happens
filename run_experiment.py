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
from numba import cuda

models = [
    ## Done ##
    'meta-llama/Llama-3.2-3B-Instruct',
    # 'gpt2',
    # 'meta-llama/Llama-3.2-3B',
    # 'google/gemma-2-2b',
    'google/gemma-2-2b-it',

    # NOTE: Tokenizer changed, need to repeat experiments https://github.com/volcengine/verl/issues/1580#issuecomment-2894381339
    # 'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-4B',
    ##########
    # 'meta-llama/Llama-3.1-8B-Instruct',
    # 'meta-llama/Llama-3.1-8B',

]

datasets = [
    # {'filename': 'datasets/templates/date_3way.csv', 'label_columns': ['correct_date']},
    # {'filename': 'datasets/templates/date_3way_2hop.csv', 'label_columns': ['correct_date', 'correct_month_label', 'correct_month'],},
    # {'filename': 'datasets/templates/date_3way_month.csv', 'label_columns': ['correct_date', 'correct_month_label', 'correct_month'],},
    # {'filename': 'datasets/templates/season_3way.csv', 'label_columns': ['correct_season', 'correct_season_label', 'correct_date'],},
    
    {'filename': 'datasets/templates/duration_3way.csv', 'label_columns': ['correct_duration_days', 'correct_date', 'correct_end_date', 'correct_month'], 'extra_columns': ['correct_duration_str', 'correct_date_expr']},
    # {'filename': 'datasets/templates/duration_1way_0hop.csv', 'label_columns': ['correct_duration_days', 'correct_date', 'correct_month'], 'extra_columns': ['correct_duration_str', 'correct_date_expr']},
    
    # {'filename': 'datasets/templates/time_of_day_HH:MM.csv', 'label_columns': ['correct_hour', 'correct'],},
    # {'filename': 'datasets/templates/time_of_day_3way.csv', 'label_columns': ['correct_time', 'correct', 'correct_time_diff'], 'extra_columns': ['correct_time_expr']},
    # {'filename': 'datasets/templates/time_of_day_3way_phase.csv', 'label_columns': ['correct_time', 'correct', 'correct_phase', 'correct_phase_label'], 'extra_columns': ['correct_time_expr']},
]

settings = {
    'delta_token': 0,
    'bypass_eval': False,
}

def run_model_on_datasets(model_name, datasets, delta_token=0, bypass_eval=False):
    print(f"Running model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    model.to('cuda')

    for dataset in tqdm(datasets):
        print(f"Running dataset: {dataset['filename']}")
        df = pd.read_csv(dataset['filename'])
        df = df.sample(frac=1, random_state=42)
        dataset_name = os.path.splitext(os.path.basename(dataset['filename']))[0]
        try:
            adf = activate_eval(df, dataset_name, model, tokenizer, delta_token=delta_token,
                                bypass_eval=bypass_eval,
                                label_columns=dataset['label_columns'],
                                extra_columns=dataset.get('extra_columns', []))
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
                # If ValueError: need at least one array to stack, repeat the process bypassing eval
                if not e.args[0].startswith("need at least one array to stack"):
                    raise e
                print(f"ValueError: model {model_name} on dataset {dataset_name} achieved zero accuracy. Retrying without eval.")
                adf = activate_eval(df, dataset_name, model, tokenizer,
                        label_columns=dataset['label_columns'],
                        extra_columns=dataset.get('extra_columns', []),
                        bypass_eval=True, delta_token=-1)
        # Print accuracy
        print(f"Model: {model_name}, Dataset: {dataset_name}, Accuracy: {adf.get_accuracy()}")
        adf.save(f"results/{model_name.split('/')[-1]}/{dataset_name}.pt")

    print(f"Finished model: {model_name}")

# Main loop
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # safer for CUDA

    for model_name in models:
        p = Process(target=run_model_on_datasets, args=(model_name, datasets, settings['delta_token'], settings['bypass_eval']))
        p.start()
        p.join()  # Wait for one model to finish before starting next



# for model_name in models:
#     print(f"Running model: {model_name}")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype="auto",
#         device_map="auto"
#     )
#     model.eval()
#     model.to('cuda')
#     for dataset in tqdm(datasets):
#         print(f"Running dataset: {dataset['filename']}")
#         df = pd.read_csv(dataset['filename'])
#         df = df.sample(frac=0.01, random_state=42)
#         dataset_name = dataset['filename'].split('/')[-1].split('.')[0]
#         adf = activate_eval(df, dataset_name, model, tokenizer,
#                             label_columns=dataset['label_columns'],
#                             extra_columns=dataset.get('extra_columns', []))
#         # adf.save(f"results/{model_name.split('/')[-1]}/{dataset_name}.pt")

#     print(f"Finished model: {model_name}")
