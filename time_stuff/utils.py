import ast
import os
import pickle
import requests

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.linalg import eigh
import torch
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList
from typing import Callable, Union
import pandas as pd
from transformers import AutoTokenizer
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
import matplotlib.patheffects as pe
from pycolormap_2d import ColorMap2DZiegler
from sklearn.model_selection import KFold


def prompt_ollama(prompt, model="llama3.1:latest", api_url="http://10.167.31.201:11434/api/generate"):
    """
    Queries the Ollama API.
    
    Args:
        prompt (str): The prompt to send to the model.
        model (str): The name of the model to use.
        api_url (str): The endpoint for the Ollama API.
    
    Returns:
        string: The response from the model.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        output = result.get("response", "")
        return output
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama API: {e}")
        return []


class ActivationDataset:
    def __init__(self, global_metadata: dict, activations: dict[str, np.ndarray],
                 sample_metadata: list[dict]):
        """
        Initializes the ActivationDataset with the given parameters.

        Parameters:
            dataset_name (str): The name of the dataset.
            model_name (str): The name of the model.
            global_metadata (dict): Global metadata for the dataset.
            activations (dict[str, np.ndarray]): Activations for selected columns.
            sample_metadata (list[dict]): Metadata for each sample.
        """
        self.dataset_name = global_metadata['dataset_name']
        self.model_name = global_metadata['model_name']
        self.global_metadata = global_metadata
        self.activations = activations
        self.n_samples, self.n_layers, self.embedding_size = activations['correct_answer'].shape
        self.n_tokens = len(activations) # Number of saved activations
        self.sample_metadata = sample_metadata
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name) # Necessary for slicing
        self._df = pd.DataFrame(sample_metadata)

    def save(self, path: str):
        """
        Saves the ActivationDataset to a file.

        Parameters:
            path (str): The path to save the dataset.
        """
        # If path does not exist, create it
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving ActivationDataset to {path}...")
        activations_names = list(self.activations.keys())
        activations = list(self.activations.values())
        # Convert activations to a tensor
        activations = np.stack(activations, axis=1)  # Shape: (n_samples, n_tokens, n_layers, embedding_size)
        activations = torch.from_numpy(activations)
        print(f"Activations shape: {activations.shape}, names: {activations_names}")
        torch.save({
            'global_metadata': self.global_metadata,
            'activations': activations,
            'activations_names': activations_names,
            'sample_metadata': self.get_metadata_df()
        }, path)

    @classmethod
    def load(cls, path: str):
        """
        Loads the ActivationDataset from a file.

        Parameters:
            path (str): The path to load the dataset from.

        Returns:
            ActivationDataset: The loaded ActivationDataset.
        """
        data = torch.load(path, weights_only=False)

        # Unstack activations along axis 1 and map them back to names
        activations_tensor = data['activations']  # Shape: (n_samples, n_tokens, n_layers, embedding_size)
        activations_names = data['activations_names']
        activations = {name: activations_tensor[:, i].detach().cpu().numpy()
                    for i, name in enumerate(activations_names)}
        
        if 'model_name' not in data['global_metadata']:
            # Get it from the path as the last folder
            data['global_metadata']['model_name'] = os.path.basename(os.path.dirname(path))
        if 'dataset_name' not in data['global_metadata']:
            # Get it from the path as the filename without extension
            data['global_metadata']['dataset_name'] = os.path.splitext(os.path.basename(path))[0]

        return cls(
            global_metadata=data['global_metadata'],
            activations=activations,
            sample_metadata=data['sample_metadata'].to_dict(orient='records')
        )

    
    def get_metadata_df(self, filter_incorrect=False) -> pd.DataFrame:
        """
        Returns the metadata as a pandas DataFrame.

        Returns:
            pd.DataFrame: The metadata DataFrame.
        """
        metadata_df = self._df
        if filter_incorrect:
            # Filter out samples with incorrect answers
            metadata_df = metadata_df[metadata_df['correct'] == True]
        return metadata_df

    def get_target_activations(self, target_column: str):
        """
        Returns the activation at the target token for all samples.

        Parameters:
            target_column (str): The name of the column containing the target token index in the activation metadata.
        Returns:
            np.ndarray: The activations at the target token. Shape: (n_samples, n_layers, embedding_size).
        """
        return self.activations[target_column]
        # n_samples, n_tokens, n_layers, embedding_size = self.activations[target_column].shape

        # # First check if the first item in sample_metadata is a string
        # # if it is, use find_token_idx to get the index int
        # # then, use the int to get the activations
        # if isinstance(self.sample_metadata[0][target_column], str):
        #     target_indices = [find_token_idx(self._tokenizer, meta['decoded'], meta[target_column])
        #                       + len(self._tokenizer.encode(meta['sentence']))
        #                       for meta in self.sample_metadata]
        # elif isinstance(self.sample_metadata[0][target_column], int):
        #     target_indices = [meta[target_column] for meta in self.sample_metadata]
        
        # if len(target_indices) != self.n_samples:
        #     # Log warning if not all samples have the target column
        #     print(f"Warning: Not all samples have the target column '{target_column}'.")
        
        # # Create batch indices (0, 1, ..., n_samples-1)
        # batch_indices = np.arange(n_samples)

        # # Select the target activations
        # target_activations = self.activations[batch_indices, :, target_indices, :]
        
        # return target_activations

    def get_metadata_column(self, column_name: str):
        """
        Returns the specified column from the activation metadata.

        Parameters:
            column_name (str): The name of the column to retrieve.
        Returns:
            list: The values in the specified column.
        """
        return [meta[column_name] for meta in self.sample_metadata]

    def get_slice(self, target_name: str = 'correct_answer', columns: Union[str, list[str]] = None,
                  preprocess_funcs: Union[Callable, list[Callable]] = None, filter_incorrect: bool = True):
        """
        Returns a slice of the dataset consisting of target tokens and one or more metadata columns.

        Parameters:
            target_name (str): The name of the column of target tokens in the activation metadata.
            columns (Union[str, list[str]]): The column(s) to include in the slice. If None, all columns are included.
            preprocess_funcs (Union[Callable, list[Callable]]): Functions to preprocess the columns.
            filter_incorrect (bool): If True, filters out samples with incorrect answers.

        Returns:
            np.ndarray, pd.DataFrame: The sliced activations and the metadata DataFrame.
        """
        target_activations = self.get_target_activations(target_name)
        metadata_df = self.get_metadata_df(filter_incorrect=False) # Need all metadata to filter later

        # Filter out incorrect samples if specified
        if filter_incorrect:
            correct_mask = metadata_df['correct'] == True
            target_activations = target_activations[correct_mask]
            metadata_df = metadata_df[correct_mask]

        if columns is None:
            columns = metadata_df.columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]

        metadata_df = metadata_df[columns]

        if preprocess_funcs is not None:
            if isinstance(preprocess_funcs, Callable):
                preprocess_funcs = [preprocess_funcs] * len(columns)
            elif len(preprocess_funcs) != len(columns):
                raise ValueError(f"Length of preprocess_funcs {len(preprocess_funcs)} does not match provided columns {columns}.") 
            for func, col in zip(preprocess_funcs, columns):
                metadata_df[col] = metadata_df[col].apply(func)

        return target_activations, metadata_df.to_numpy().squeeze()
    
    def get_accuracy(self):
        """
        Returns the accuracy of the model on the dataset.

        Returns:
            float: The accuracy of the model.
        """
        return self.global_metadata.get('accuracy', 0.0)


class SupervisedMDS(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, manifold='trivial', alpha=0.1, orthonormal=False):
        """
        Parameters:
            n_components: int
                Target embedding dimension.
            manifold: str or callable
                If 'trivial', d_ij = 0 if y_i == y_j else 1.
                If callable, should return a (n x n) ideal distance matrix given y.
        """
        self.n_components = n_components
        self.manifold = manifold
        self.W_ = None
        self.alpha = alpha
        self.orthonormal = orthonormal
        self._X_mean = None
        self._Y_mean = None
        if orthonormal and alpha != 0:
            print("Warning: orthonormal=True and alpha!=0. alpha will be ignored.")

    def _compute_ideal_distances(self, y, threshold=2):
        n = len(y)
        D = np.zeros((n, n))

        if self.manifold in ['trivial', 'cluster']: # Retrocompatibility
            for i in range(n):
                for j in range(n):
                    D[i, j] = 0.0 if y[i] == y[j] else 1.0
        elif self.manifold == 'euclidean':
            for i in range(n):
                for j in range(n):
                    D[i, j] = np.linalg.norm(y[i] - y[j])
        elif self.manifold == 'log_linear':
            log_y = np.log(y + 1)
            D = np.abs(log_y[:, None] - log_y[None, :])
        elif self.manifold == 'circular':
            max_y = np.max(y)
            min_y = np.min(y)

            # Normalize y to [0, 1]
            y_norm = (y - min_y) / (max_y - min_y)

            # Compute pairwise circular distances
            delta = np.abs(y_norm[:, None] - y_norm[None, :])
            delta = np.minimum(delta, 1 - delta)  # Wrap around the circle
            D = 2 * np.sin(np.pi * delta)  # Full circle version
        elif self.manifold == 'helix':
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))  # shape: (n,)

            # Map to 3D spiral
            theta = 2 * np.pi * y_norm  # angle around the circle
            x = np.cos(theta)
            y_circle = np.sin(theta)
            z = y_norm  # vertical component

            spiral_coords = np.stack([x, y_circle, z], axis=1)  # shape: (n, 3)

            # Compute pairwise Euclidean distances in spiral space
            diffs = spiral_coords[:, None, :] - spiral_coords[None, :, :]
            D = np.linalg.norm(diffs, axis=2)  # shape: (n, n)

        elif self.manifold == 'discrete_circular':
            max_y = np.max(y)
            for i in range(n):
                for j in range(n):
                    D[i, j] = min(np.abs(y[i] - y[j]), max_y + 1 - np.abs(y[i] - y[j]))
        elif self.manifold == 'chain':
            max_y = np.max(y)
            for i in range(n):
                for j in range(n):
                    dist = min(np.abs(y[i] - y[j]), max_y + 1 - np.abs(y[i] - y[j]))
                    D[i, j] = dist if dist < threshold else -1
        elif self.manifold == 'semicircular':
            max_y = np.max(y)
            min_y = np.min(y)
            for i in range(n):
                for j in range(n):
                    y_i_norm = (y[i] - min_y) / (max_y - min_y)
                    y_j_norm = (y[j] - min_y) / (max_y - min_y)
                    D[i, j] = 2 * np.sin(np.pi/2 * np.abs(y_i_norm - y_j_norm)) 
        elif self.manifold == 'log_semicircular':
            max_y = np.max(y)
            min_y = np.min(y)
            for i in range(n):
                for j in range(n):
                    y_i_norm = (y[i] - min_y) / (max_y - min_y)
                    y_j_norm = (y[j] - min_y) / (max_y - min_y)
                    D[i, j] = 2 * np.sin(np.pi/2 * np.abs(np.log(y_i_norm + 1) - np.log(y_j_norm + 1)))

        elif callable(self.manifold):
            D = self.manifold(y)
        else:
            raise ValueError("Invalid manifold specification.")
        
        return D

    def _classical_mds(self, D):
        # Step 1: square distances
        D2 = D ** 2

        # Step 2: double centering
        n = D2.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D2 @ H

        # Step 3: eigen-decomposition
        eigvals, eigvecs = eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx][:self.n_components]
        eigvecs = eigvecs[:, idx][:, :self.n_components]

        # Step 4: compute embedding
        Y = eigvecs * np.sqrt(np.maximum(eigvals, 0))
        return Y

    def _masked_loss(self, W_flat, X, D, mask):
        W = W_flat.reshape((self.n_components, X.shape[1]))
        X_proj = (W @ X.T).T
        D_pred = np.linalg.norm(X_proj[:, None, :] - X_proj[None, :, :], axis=-1)
        loss = (D_pred - D)[mask]
        return np.sum(loss ** 2)


    def fit(self, X, y):
        """
        Fit the linear transformation W to match distances induced by labels y.
        Uses classical MDS + closed-form when all distances are defined,
        and switches to optimization if some distances are undefined (negative).
        """
        X = np.asarray(X)
        y = np.asarray(y).squeeze()  # Ensure y is 1D
        D = self._compute_ideal_distances(y)

        if np.any(D < 0):
            # Raise warning if any distances are negative
            print("Warning: Distance matrix is incomplete. Using optimization to fit W.")
            mask = D >= 0
            rng = np.random.default_rng(42)
            W0 = rng.normal(scale=0.01, size=(self.n_components, X.shape[1]))

            result = minimize(
                self._masked_loss,
                W0.ravel(),
                args=(X, D, mask),
                method='L-BFGS-B'
            )
            self.W_ = result.x.reshape((self.n_components, X.shape[1]))
        else:
            # Use classical MDS + closed-form least squares
            Y = self._classical_mds(D)

            self._X_mean = X.mean(axis=0)  # Centering
            self._Y_mean = Y.mean(axis=0)  # Centering Y
            X_centered = X - X.mean(axis=0)
            Y_centered = Y - Y.mean(axis=0)
            if self.orthonormal:
                # Orthogonal Procrustes
                M = Y_centered.T @ X_centered
                U, _, Vt = np.linalg.svd(M)
                self.W_ = U @ Vt
            else:
                if self.alpha == 0:
                    self.W_ = Y_centered.T @ np.linalg.pinv(X_centered.T)
                else:
                    XtX = X_centered.T @ X_centered
                    self.W_ = Y_centered.T @ X_centered @ np.linalg.inv(XtX + self.alpha * np.eye(XtX.shape[0]))


        return self


    def transform(self, X):
        """
        Apply the learned transformation to X.
        """
        if self.W_ is None:
            raise RuntimeError("You must fit the model before calling transform.")
        X = np.asarray(X)
        if self._X_mean is not None:
            # Center X using the same logic as during fit
            X_centered = X - self._X_mean
        else:
            X_centered = X
        return (self.W_ @ X_centered.T).T
        
    def _truncated_pinv(self, W, tol=1e-5):
        U, S, VT = np.linalg.svd(W, full_matrices=False)
        S_inv = np.array([1/s if s > tol else 0 for s in S])
        return VT.T @ np.diag(S_inv) @ U.T

    def _regularized_pinv(self, W, lambda_=1e-5):
        return np.linalg.inv(W.T @ W + lambda_ * np.eye(W.shape[1])) @ W.T


    def inverse_transform(self, X_proj):
        """
        Reconstruct the original input X from its low-dimensional projection.
        
        Parameters:
            X_proj: array-like of shape (n_samples, n_components)
                The low-dimensional representation of the input data.
        
        Returns:
            X_reconstructed: array of shape (n_samples, original_n_features)
                The reconstructed data in the original space.
        """
        if self.W_ is None:
            raise RuntimeError("You must fit the model before calling inverse_transform.")
        
        X_proj = np.asarray(X_proj)

        # Use pseudo-inverse in case W_ is not square or full-rank
        # W_pinv = np.linalg.pinv(self.W_)
        # Use regularized pseudo-inverse to avoid numerical issues
        # W_pinv = self._regularized_pinv(self.W_)
        W_pinv = self._truncated_pinv(self.W_)

        X_centered = (W_pinv @ X_proj.T).T

        if hasattr(self, '_X_mean') and self._X_mean is not None:
            return X_centered + self._X_mean
        else:
            return X_centered


    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def score(self, X, y):
        """
        Compute how well the transformed distances match ideal distances.

        Returns:
            A score between -∞ and 1. Higher is better.
        """
        if self.W_ is None:
            raise RuntimeError("Model must be fit before scoring.")

        D_true = self._compute_ideal_distances(y)
        X_proj = self.transform(X)

        # Compute predicted pairwise distances
        n = X_proj.shape[0]
        D_pred = np.linalg.norm(X_proj[:, np.newaxis, :] - X_proj[np.newaxis, :, :], axis=-1)

        # Compute stress and normalize
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        stress = np.sum((D_pred[mask] - D_true[mask]) ** 2)
        denom = np.sum(D_true[mask] ** 2)

        return 1 - stress / denom if denom > 0 else -np.inf
    
    def save(self, filepath):
        """
        Save the model to disk, including learned weights.
        """
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Load a model from disk.
        Returns:
            An instance of SupervisedMDS.
        """
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj


def clean(x):
    return ast.literal_eval(x)

class ConstrainedPrefixLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_seqs, tokenizer):
        self.tokenizer = tokenizer
        # Add a space at the start of each sequence
        self.allowed_seqs = [f" {seq}" for seq in allowed_seqs]
        # Add period to the allowed sequences
        self.allowed_seqs.append('.')
        self.allowed_seqs.append('<|end_of_text|>')

        self.allowed_token_seqs = [tokenizer(seq, add_special_tokens=False)['input_ids'] for seq in self.allowed_seqs]
        # Add end of sentence token to the allowed sequences
        self.allowed_token_seqs.append([tokenizer.eos_token_id])
         
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        current_len = input_ids.shape[-1]
        
        allowed_next_tokens = set()
        for seq in self.allowed_token_seqs:
            if len(seq) > current_len:
                if torch.equal(input_ids[0][-len(seq)+1:], torch.tensor(seq[:-1], device=input_ids.device)):
                    allowed_next_tokens.add(seq[len(input_ids[0]) - len(seq)])

        if not allowed_next_tokens:
            # Start of generation, allow first tokens of all candidate sequences
            allowed_next_tokens = {seq[0] for seq in self.allowed_token_seqs}

        mask = torch.full_like(scores, float("-inf"))
        for token_id in allowed_next_tokens:
            mask[0, token_id] = scores[0, token_id]
        return mask


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


def find_token_idx(tokenizer, text, target, start=0):
    """
    Find the index of the last token corresponding to a target substring in a text using the tokenizer,
    starting the search from a given character index.
        
    Args:
        tokenizer: A HuggingFace tokenizer.
        text (str): The input text to tokenize.
        target (str): The substring to find in the tokenized input.
        start (int): The character index in text to start searching from.
    
    Returns:
        int: Index of the last token corresponding to the target substring in the tokenized input.
             Returns -1 if the target is not found.
    """
    # Tokenize with offset mappings
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding['offset_mapping']
    input_ids = encoding['input_ids']

    # Find the character span of the target in the text, starting at 'start'
    target_start = text.find(target, start)
    if target_start == -1:
        return -1  # Target substring not found
    target_end = target_start + len(target)

    # Find token indices that overlap with the target span
    matched_token_idxs = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end == 0:
            continue  # Likely a special/control token
        if not (tok_end <= target_start or tok_start >= target_end):
            matched_token_idxs.append(idx)

    if not matched_token_idxs:
        return -1

    return matched_token_idxs[-1]  # Return the last matching token index




# def find_token_idx(tokenizer, text, target):
#     """
#     Find the index of a target token in a text using the tokenizer.
#     """
#     # Find the token in the text
#     start_token_idx = text.find(target)

#     # If the token is not found or if it is at the end of the text, return -1 
#     if start_token_idx == -1 or start_token_idx + len(target) >= len(text):
#         return -1
#     end_token_idx = start_token_idx + len(target)
#     # Tokenize the text to compute the token index
#     tokenized_text = tokenizer(text[:end_token_idx], add_special_tokens=False)
#     return len(tokenized_text['input_ids']) - 1
    

# Does both the evaluation and storing of activations
def activate_eval(df, dataset_name, model, tokenizer, label_columns, question_column='question', answer_column='correct_answer', context_column='context', 
                  extra_columns=None, constrained_generation=False, delta_token=0, template="{context}", debug=False):
    model.eval()
    # Silence warnings
    model.config.top_p = None
    model.config.top_k = None
    model.config.temperature = 1.0

    global_metadata = {
        'dataset_name': dataset_name,
        'model_name': model.config._name_or_path,
        'model_type': model.config.model_type,
        'model_size': model.num_parameters(),
        'question_column': question_column,
        'answer_column': answer_column,
        'context_column': context_column,
        'label_columns': label_columns,
        'extra_columns': extra_columns,
        'constrained_generation': constrained_generation,
        'template': template,
        'delta_token': delta_token,
    }
    
    sample_metadata = []
    outputs = []
    logits_processor = None
    correct_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        extra_cols_dict = {}
        labels = {}
        activations_idxs = []
        for label_column in label_columns:
            labels[label_column] = row[label_column] if label_column in row else None

        context = row[context_column] if context_column in row else row['sentence']
        question = row[question_column] if question_column in row else None
        answer = str(row[answer_column]) if answer_column in row else None
        sentence = template.format(context=context, question=question, answer=answer)

        input_ids = tokenizer(sentence, return_tensors="pt").to("cuda")

        if constrained_generation:
            if 'alternatives' not in row:
                raise ValueError("Alternatives column not found in dataset.")
            logits_processor = LogitsProcessorList()
            logits_processor.append(ConstrainedPrefixLogitsProcessor(
                allowed_seqs=row['alternatives'],
                tokenizer=tokenizer
            ))

        # Generate the answer to the question, no sampling
        # NOTE: generate only outputs the hidden states of the generation, not the input
        generations = model.generate(**input_ids, max_new_tokens=8, return_dict_in_generate=True, output_hidden_states=True, 
                                     do_sample=False, logits_processor=logits_processor, pad_token_id=tokenizer.eos_token_id)
        # Decode the generated answer
        gen_decoded = tokenizer.decode(generations.sequences[0], skip_special_tokens=True)

        correct_answer_idx = find_token_idx(tokenizer, gen_decoded, answer, start=len(sentence))
        # If it's the last token, consider it incorrect as no hidden state is produced
        if correct_answer_idx == len(generations.sequences[0]) - 1:
            correct_answer_idx = -1
        
        correct = correct_answer_idx != -1
        correct_count += correct
        if not correct:
            # Defaults to the first generated token
            correct_answer_idx = 0
        correct_answer_idx + delta_token
        activations_idxs.append(correct_answer_idx)

        last_prompt_idx = len(input_ids['input_ids'][0]) - 1 
        activations_idxs.append(last_prompt_idx)

        if extra_columns is not None:
            for extra_column in extra_columns:
                if extra_column in row:
                    extra_cols_dict[extra_column] = row[extra_column]
                    extra_col_idx = find_token_idx(tokenizer, gen_decoded, extra_cols_dict[extra_column])
                    if extra_col_idx == -1:
                        raise ValueError(f"Token '{extra_cols_dict[extra_column]}' from extra column '{extra_column}' not found in generation.")
                    # extra_col_idx = extra_col_idx + len(input_ids['input_ids'][0])
                    activations_idxs.append(extra_col_idx)
                else:
                    raise ValueError(f"Extra column '{extra_column}' not found in dataset.")

        # The full length of tokens is = prompt size + n_generated_tokens + n_system_tokens
        hidden_states = torch.cat([torch.cat(hs, dim=0) for hs in generations.hidden_states], dim=1)
        hidden_states = hidden_states[:,activations_idxs]
        hidden_states = hidden_states.type(torch.float64)
        hidden_states_np = hidden_states.cpu().detach().numpy()

        # hidden_states = generations.hidden_states[gen_idx]
        # hidden_states = [h.type(torch.float16)[0,idx_end,:] for h in hidden_states]

        outputs.append(hidden_states_np)
        row_metadata = {
            'context': context,
            'question': question,
            'answer': answer,
            'sentence': sentence,
            'correct_answer_idx': correct_answer_idx,
            'decoded': gen_decoded,
            'correct': correct,
        }
        row_metadata.update(labels)
        row_metadata.update(extra_cols_dict)
        sample_metadata.append(row_metadata)

        if debug:
            break
    
    outputs = np.stack(outputs, axis=0)

    if debug: # Debug contains an int
        # Create a dummy output for debugging with that length by multiplying all the outputs
        outputs = torch.randn((debug, outputs.shape[1], outputs.shape[2], outputs.shape[3]))
        sample_metadata = sample_metadata * debug 

    # Turn hidden_states from an array to a dict of col_name: array
    activation_cols = ['correct_answer', 'last_prompt_token'] + extra_columns
    activations = {col: outputs[:, :, i, :] for i, col in enumerate(activation_cols)}
    
    accuracy = correct_count / len(df)
    global_metadata['accuracy'] = accuracy

    # Create ActivationDataset object
    dataset = ActivationDataset(
        global_metadata=global_metadata,
        activations=activations,
        sample_metadata=sample_metadata,
    )

    return dataset

def generate_with_hooks(model, input_ids, max_new_tokens, hook_layer, edit_hook, extract_hook):
    model.reset_hooks()
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Run with hooks only on the current input
        logits = model.run_with_hooks(
            generated,
            fwd_hooks = (
                            [(f"blocks.{hook_layer}.hook_resid_post", edit_hook)] +
                            [(f"blocks.{i}.hook_resid_post", extract_hook) for i in range(model.cfg.n_layers)]
                        )
        )

        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    return generated

def activate_eval_intervene(df, dataset_name, model, tokenizer, label_columns, intervention_layer, target_column, smds=None, intervention_type='replace', noise_scale=1, n_components=None, question_column='question', answer_column='answer', context_column='context', 
                             extra_columns=None, constrained_generation=False, delta_token=0, debug=False, max_new_tokens=8):
    template = "{context}"
    model.eval()

    if not isinstance(label_columns, list):
        label_columns = [label_columns]

    global_metadata = {
        'dataset_name': dataset_name,
        'model_name': model.cfg.tokenizer_name, # model_name doesn't have model family
        'model_type': None,
        'model_size': None,
        'question_column': question_column,
        'answer_column': answer_column,
        'context_column': context_column,
        'label_columns': label_columns,
        'target_column': target_column,
        'extra_columns': extra_columns,
        'constrained_generation': constrained_generation,
        'template': template,
        'delta_token': delta_token,
        'intervention_layer': intervention_layer,
        'noise_scale': noise_scale,
        'smds': smds,
        'n_components': n_components,
        'intervention_type': intervention_type,
    }
    
    sample_metadata = []
    outputs = []
    logits_processor = None
    correct_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        extra_cols_dict = {}
        labels = {}
        activations_idxs = []
        hidden_states = []
        for label_column in label_columns:
            labels[label_column] = row[label_column] if label_column in row else None

        context = row[context_column] if context_column in row else row['sentence']
        question = row[question_column] if question_column in row else None
        answer = str(row[answer_column]) if answer_column in row else None
        sentence = template.format(context=context, question=question, answer=answer)
        target_string = row[target_column]
        
        input_ids = tokenizer(sentence, return_tensors="pt").to("cuda")

        if constrained_generation:
            if 'alternatives' not in row:
                raise ValueError("Alternatives column not found in dataset.")
            logits_processor = LogitsProcessorList()
            logits_processor.append(ConstrainedPrefixLogitsProcessor(
                allowed_seqs=row['alternatives'],
                tokenizer=tokenizer
            ))
        
        target_index = find_token_idx(tokenizer, sentence, target_string)
        last_prompt_idx = len(input_ids['input_ids'][0]) - 1 

        if intervention_type == 'replace':
            def edit_hook(value, hook):
                activation = value[:, target_index, :].float().detach().cpu().numpy()
                subspace = smds.transform(activation)
                # Summing and subtracting subspace is necessary to cancel out mean
                noise = subspace + noise_scale * np.random.normal(0, 1, subspace.shape)  # Add noise
                patch = activation - smds.inverse_transform(subspace) + smds.inverse_transform(noise)
                value[:, target_index, :] = torch.tensor(patch, device=value.device, dtype=value.dtype)
                return value
        elif intervention_type == 'rand':
            def edit_hook(value, hook):
                activation = value[:, target_index, :]  # (batch, d_model)

                dtype = activation.dtype
                activation_f32 = activation.to(torch.float32)

                subspace_mapper = torch.randn(activation.shape[-1], n_components, device=value.device, dtype=torch.float32)
                subspace_inv_mapper = torch.linalg.pinv(subspace_mapper)  # safe with float32
                subspace = activation_f32 @ subspace_mapper
                noise = subspace + noise_scale * torch.randn_like(subspace)

                patch = activation_f32 - (subspace @ subspace_inv_mapper) + (noise @ subspace_inv_mapper)
                patch = patch.to(dtype)  # cast back to original dtype (e.g., bfloat16)

                value[:, target_index, :] = patch
                return value


        elif intervention_type == 'full':
            def edit_hook(value, hook):
                activation = value[:, target_index, :]
                noise = noise_scale * torch.randn(activation.shape, device=value.device, dtype=value.dtype)
                value[:, target_index, :] = activation + noise
                return value

        def extract_hook(value, hook):
            nonlocal hidden_states # Uses variable defined in eval loop
            if value.shape[1] == len(input_ids['input_ids'][0]) + max_new_tokens - 1:
                hidden_states.append(value)
            

        generations = generate_with_hooks(
            model,
            input_ids['input_ids'],
            max_new_tokens=max_new_tokens,
            hook_layer=intervention_layer,
            edit_hook=edit_hook,
            extract_hook=extract_hook
        )
        
        # Decode the generated answer
        gen_decoded = tokenizer.decode(generations[0], skip_special_tokens=True)

        correct_answer_idx = find_token_idx(tokenizer, gen_decoded, answer, start=len(sentence))
        # If it's the last token, consider it incorrect as no hidden state is produced
        if correct_answer_idx == len(generations[0]) - 1:
            correct_answer_idx = -1
        
        correct = correct_answer_idx != -1
        correct_count += correct
        if not correct:
            # Defaults to the first generated token
            correct_answer_idx = 0
        correct_answer_idx + delta_token
        activations_idxs.append(correct_answer_idx)

        last_prompt_idx = len(input_ids['input_ids'][0]) - 1 
        activations_idxs.append(last_prompt_idx)

        if extra_columns is not None:
            for extra_column in extra_columns:
                if extra_column in row:
                    extra_cols_dict[extra_column] = row[extra_column]
                    extra_col_idx = find_token_idx(tokenizer, gen_decoded, extra_cols_dict[extra_column])
                    if extra_col_idx == -1:
                        raise ValueError(f"Token '{extra_cols_dict[extra_column]}' from extra column '{extra_column}' not found in generation.")
                    # extra_col_idx = extra_col_idx + len(input_ids['input_ids'][0])
                    activations_idxs.append(extra_col_idx)
                else:
                    raise ValueError(f"Extra column '{extra_column}' not found in dataset.")

        # The full length of tokens is = prompt size + n_generated_tokens + n_system_tokens
        hidden_states = torch.cat(hidden_states, dim=0)
        hidden_states = hidden_states[:,activations_idxs]
        hidden_states = hidden_states.type(torch.float64)
        hidden_states_np = hidden_states.cpu().detach().numpy()

        # hidden_states = generations.hidden_states[gen_idx]
        # hidden_states = [h.type(torch.float16)[0,idx_end,:] for h in hidden_states]

        outputs.append(hidden_states_np)
        row_metadata = {
            'context': context,
            'question': question,
            'answer': answer,
            'sentence': sentence,
            'correct_answer_idx': correct_answer_idx,
            'decoded': gen_decoded,
            'correct': correct,
        }
        row_metadata.update(labels)
        row_metadata.update(extra_cols_dict)
        sample_metadata.append(row_metadata)

        if debug:
            break
    
    outputs = np.stack(outputs, axis=0)

    if debug: # Debug contains an int
        # Create a dummy output for debugging with that length by multiplying all the outputs
        outputs = torch.randn((debug, outputs.shape[1], outputs.shape[2], outputs.shape[3]))
        sample_metadata = sample_metadata * debug 

    # Turn hidden_states from an array to a dict of col_name: array
    activation_cols = ['correct_answer', 'last_prompt_token'] + extra_columns
    activations = {col: outputs[:, :, i, :] for i, col in enumerate(activation_cols)}
    
    accuracy = correct_count / len(df)
    global_metadata['accuracy'] = accuracy

    # Create ActivationDataset object
    dataset = ActivationDataset(
        global_metadata=global_metadata,
        activations=activations,
        sample_metadata=sample_metadata,
    )

    return dataset


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
