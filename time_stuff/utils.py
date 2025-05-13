import os
import requests

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.linalg import eigh


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

class LinearMetricEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, graph='trivial'):
        """
        Parameters:
            n_components: int
                Target embedding dimension.
            graph: str or callable
                If 'trivial', d_ij = 0 if y_i == y_j else 1.
                If callable, should return a (n x n) ideal distance matrix given y.
        """
        self.n_components = n_components
        self.graph = graph
        self.W_ = None

    def _compute_ideal_distances(self, y, threshold=2):
        n = len(y)
        d_ij = np.zeros((n, n))

        if self.graph == 'trivial':
            for i in range(n):
                for j in range(n):
                    d_ij[i, j] = 0.0 if y[i] == y[j] else 1.0
        elif self.graph == 'euclidean':
            for i in range(n):
                for j in range(n):
                    d_ij[i, j] = np.linalg.norm(y[i] - y[j])
        elif self.graph == 'circular':
            max_y = np.max(y)
            for i in range(n):
                for j in range(n):
                    d_ij[i, j] = min(np.abs(y[i] - y[j]), max_y + 1 - np.abs(y[i] - y[j]))
        elif self.graph == 'chain':
            max_y = np.max(y)
            for i in range(n):
                for j in range(n):
                    dist = min(np.abs(y[i] - y[j]), max_y + 1 - np.abs(y[i] - y[j]))
                    d_ij[i, j] = dist if dist < threshold else -1

        elif callable(self.graph):
            d_ij = self.graph(y)
        else:
            raise ValueError("Invalid graph specification.")
        
        return d_ij

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
        D = self._compute_ideal_distances(y)

        if np.any(D < 0):
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
            X_centered = X - X.mean(axis=0)
            Y_centered = Y - Y.mean(axis=0)
            self.W_ = Y_centered.T @ np.linalg.pinv(X_centered.T)

        return self


    def transform(self, X):
        """
        Apply the learned transformation to X.
        """
        if self.W_ is None:
            raise RuntimeError("You must fit the model before calling transform.")
        X = np.asarray(X)
        X_centered = X - X.mean(axis=0)  # Important: center using same logic as during fit
        return (self.W_ @ X_centered.T).T

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