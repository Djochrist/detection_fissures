"""
Module de détection et configuration du dispositif de calcul.

Compatibilité :
    - NVIDIA GPU (CUDA) : performance maximale
    - Apple Silicon (MPS) : accélération Metal
    - CPU : fallback universel (lent mais fonctionnel)

Détection de l'environnement :
    - Google Colab : détecté via la présence de /content
    - Kaggle : détecté via la variable d'environnement KAGGLE_KERNEL_RUN_TYPE
    - Local : par défaut
"""

import os
import platform
from typing import Optional

import torch


def detecter_dispositif(forcer: Optional[str] = None) -> torch.device:
    """
    Détecte et retourne le meilleur dispositif de calcul disponible.

    Priorité de sélection :
        1. CUDA (NVIDIA GPU) : performance maximale pour l'entraînement
        2. MPS (Apple Silicon) : accélération Metal sur Mac M1/M2/M3
        3. CPU : fallback universel

    Args:
        forcer : Forcer un dispositif spécifique ('cuda', 'mps', 'cpu').
                 None = détection automatique.

    Returns:
        Objet torch.device configuré.
    """
    if forcer is not None:
        return torch.device(forcer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def afficher_info_dispositif(dispositif: torch.device) -> None:
    """
    Affiche les informations détaillées sur le dispositif sélectionné.

    Inclut :
    - Type de dispositif (CUDA / MPS / CPU)
    - Nom du GPU (si CUDA)
    - Mémoire disponible
    - Version CUDA
    - Environnement d'exécution

    Args:
        dispositif : Dispositif de calcul détecté.
    """
    print(f"\n{'═'*55}")
    print(f"  CONFIGURATION DU DISPOSITIF DE CALCUL")
    print(f"{'═'*55}")
    print(f"  Système           : {platform.system()} {platform.release()}")
    print(f"  Python            : {platform.python_version()}")
    print(f"  PyTorch           : {torch.__version__}")
    print(f"  Dispositif choisi : {dispositif}")

    if dispositif.type == "cuda":
        idx_gpu = dispositif.index or 0
        nom_gpu = torch.cuda.get_device_name(idx_gpu)
        memoire_totale = torch.cuda.get_device_properties(idx_gpu).total_memory
        memoire_go = memoire_totale / (1024 ** 3)
        print(f"  GPU               : {nom_gpu}")
        print(f"  Mémoire GPU       : {memoire_go:.1f} Go")
        print(f"  Version CUDA      : {torch.version.cuda}")
        print(f"  cuDNN             : {torch.backends.cudnn.version()}")
        print(f"  Précision mixte   : Disponible (float16)")

    elif dispositif.type == "mps":
        print(f"  GPU               : Apple Silicon (Metal)")
        print(f"  Précision mixte   : Limitée (bfloat16)")

    else:
        print(f"  Mode CPU          : Entraînement lent")
        nb_threads = torch.get_num_threads()
        print(f"  Threads CPU       : {nb_threads}")
        print(f"  Recommandation    : Utiliser Google Colab (GPU gratuit)")

    # Détection de l'environnement
    env = detecter_environnement()
    print(f"  Environnement     : {env}")
    print(f"{'═'*55}\n")


def detecter_environnement() -> str:
    """
    Détecte l'environnement d'exécution actuel.

    Returns:
        Description de l'environnement ('Colab', 'Kaggle', 'Local').
    """
    # Google Colab
    try:
        import google.colab
        return "Google Colab"
    except ImportError:
        pass

    # Kaggle
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "Kaggle"

    # Répertoire /content typique de Colab
    if os.path.exists("/content"):
        return "Google Colab (probable)"

    return "Local"


def configurer_cudnn_pour_reproductibilite() -> None:
    """
    Configure cuDNN pour la reproductibilité (au détriment d'un peu de performance).

    Par défaut, cuDNN choisit l'algorithme le plus rapide de façon non-déterministe.
    Pour la reproductibilité des expériences, on désactive ce comportement.

    NOTE : Désactiver benchmark réduit légèrement la vitesse mais rend
    les résultats identiques d'une exécution à l'autre.
    """
    if torch.cuda.is_available():
        # Désactiver le choix non-déterministe des algorithmes cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[Dispositif] cuDNN configuré pour la reproductibilité")
