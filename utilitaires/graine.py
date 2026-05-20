"""
Module de fixation de la graine aléatoire pour la reproductibilité.

POURQUOI LA REPRODUCTIBILITÉ EST CRUCIALE ?
════════════════════════════════════════════

Dans un projet de recherche, la reproductibilité permet :
1. Comparer des expériences équitablement (même initialisation)
2. Déboguer des comportements non-déterministes
3. Publier des résultats reproductibles par d'autres chercheurs
4. Identifier si une amélioration vient du modèle ou du hasard

Ce qu'il faut fixer :
    - Python random (mélange des listes)
    - NumPy (opérations sur tableaux)
    - PyTorch (initialisation des poids, dropout)
    - cuDNN (algorithmes non-déterministes GPU)
    - Hash de Python (PYTHONHASHSEED)
"""

import os
import random

import numpy as np
import torch


def fixer_graine_aleatoire(graine: int = 42) -> None:
    """
    Fixe toutes les graines aléatoires pour la reproductibilité complète.

    Args:
        graine : Valeur de la graine. 42 = convention de la communauté ML.

    Example:
        >>> fixer_graine_aleatoire(42)
        >>> # Maintenant tous les runs produisent les mêmes résultats
    """
    # Python built-in
    random.seed(graine)

    # Variable d'environnement pour le hash Python
    os.environ["PYTHONHASHSEED"] = str(graine)

    # NumPy
    np.random.seed(graine)

    # PyTorch CPU
    torch.manual_seed(graine)

    # PyTorch GPU (tous les GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(graine)
        torch.cuda.manual_seed_all(graine)

        # cuDNN déterministe
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[Graine] Graine aléatoire fixée à {graine} — Reproductibilité activée")
