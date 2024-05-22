import numpy as np
import math
import os
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class Identity(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight

    def forward(self):
        return self.weight

def dynm_fun(f):
    """A wrapper for the dynamical function"""

    def wrapper(self, t, x):
        new_fun = lambda t, x: f(self, t, x)
        return new_fun(t, x)

    return wrapper

def cholesky_decomposition(A):
    """
    This function returns the cholesky decomposition of a matrix A.
    :param A: The matrix to be decomposed.
    :return: The cholesky decomposition of A.
    """
    L = torch.linalg.cholesky(A)
    return L

def check_accuracy(model, loader, device):
    """
    Returns the fraction of correct predictions for a given model on a given dataset.
    """
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x, y_true in loader:
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            correct += (y_pred.argmax(1) == y_true).type(torch.float).sum().item()
    return correct / len(loader.dataset)


def get_activation(activation, name):
    def hook(model, input, output):
        activation[name] = output

    return hook


def pca(data, n_components=2):
    """
    Returns the principal components of a given dataset.
    """
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.transform(data)


def cosine_similarity_across_classes(data, labels, num_classes=10):
    """
    Returns the mean cosine similarity between the embeddings of each class.
    """
    cos_sim = cosine_similarity(data)
    avg_cos_sim = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            avg_cos_sim[i, j] = np.mean(cos_sim[labels == i, :][:, labels == j])
    return avg_cos_sim


def process_activations(activation, not_keys):
    """
    Detatches the activations of the layers which are not in not_keys and
    converts them into the shape (batch_size, number of neurons).
    """
    for key, item in activation.items():
        if key not in not_keys:
            activation[key] = activation[key].detach().cpu()
            activation[key] = (
                activation[key].view(activation[key].size()[0], -1).numpy()
            )
    return activation
