# embedding_viz.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any


def compute_analogy_and_similarity(embeddings: Dict[str, np.ndarray], 
                                   a: str, b: str, c: str, target: str) -> float:
    """
    Compute vector analogy: a - b + c, then return cosine similarity with target.
    Example: king - man + woman vs queen
    """
    vec_a = embeddings[a]
    vec_b = embeddings[b]
    vec_c = embeddings[c]
    result_vector = vec_a - vec_b + vec_c
    
    target_vector = embeddings[target]
    
    sim = cosine_similarity(
        result_vector.reshape(1, -1),
        target_vector.reshape(1, -1)
    )[0][0]
    
    return result_vector, sim


def plot_analogy_2d(embeddings: Dict[str, np.ndarray], 
                    a: str, b: str, c: str, d: str,
                    result_vector: np.ndarray = None,
                    title: str = "Vector Analogy Visualization",
                    save_path: str = None):
    """
    Visualize word embeddings and analogies in 2D using PCA.
    
    Args:
        embeddings: dict of {word: embedding_vector}
        a, b, c, d: words for analogy a - b + c ≈ d
        result_vector: optional precomputed result vector
        title: plot title
        save_path: optional path to save image
    """
    # Prepare data
    labels = [a.title(), b.title(), c.title(), d.title()]
    vectors = np.array([embeddings[a], embeddings[b], embeddings[c], embeddings[d]])

    if result_vector is not None:
        vectors = np.vstack([vectors, result_vector])
        labels.append('Result')

    # Apply PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'blue', 'red', 'red'] + (['purple'] if result_vector is not None else [])
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=colors, s=120, edgecolor='k')

    # Annotate points
    for i, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]),
                     textcoords="offset points", xytext=(8, 5), fontsize=12, ha='left')

    # Draw analogy arrows
    idx_map = {a: 0, b: 1, c: 2, d: 3}
    if result_vector is not None:
        idx_map['result'] = 4

    # Arrow: b -> a  (man → king)
    plt.arrow(vectors_2d[idx_map[b], 0], vectors_2d[idx_map[b], 1],
              vectors_2d[idx_map[a], 0] - vectors_2d[idx_map[b], 0],
              vectors_2d[idx_map[a], 1] - vectors_2d[idx_map[b], 1],
              head_width=0.02, fc='gray', ec='gray', length_includes_head=True, linestyle='-', linewidth=1)

    # Arrow: c -> d  (woman → queen)
    plt.arrow(vectors_2d[idx_map[c], 0], vectors_2d[idx_map[c], 1],
              vectors_2d[idx_map[d], 0] - vectors_2d[idx_map[c], 0],
              vectors_2d[idx_map[d], 1] - vectors_2d[idx_map[c], 1],
              head_width=0.02, fc='gray', ec='gray', length_includes_head=True, linestyle='-', linewidth=1)

    # Optional dashed arrow: a -> result
    if result_vector is not None:
        plt.arrow(vectors_2d[idx_map[a], 0], vectors_2d[idx_map[a], 1],
                  vectors_2d[idx_map['result'], 0] - vectors_2d[idx_map[a], 0],
                  vectors_2d[idx_map['result'], 1] - vectors_2d[idx_map[a], 1],
                  head_width=0.02, fc='orange', ec='orange', length_includes_head=True,
                  linestyle='--', linewidth=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='k', label='Male'),
        Patch(facecolor='red', edgecolor='k', label='Female'),
        Patch(facecolor='purple', edgecolor='k', label='Prediction') if result_vector is not None else None,
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Observed vector'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='Computed step')
    ]
    legend_elements = [el for el in legend_elements if el is not None]

    plt.legend(handles=legend_elements, loc='best')
    plt.title(title, fontsize=14)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
