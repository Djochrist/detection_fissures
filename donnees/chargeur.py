"""
Module de création des DataLoaders PyTorch.

Assemble les jeux de données et DataLoaders pour les phases
entraînement, validation et test.

Particularité du collate_fn :
    Mask R-CNN reçoit une LISTE de tuples (image, cible) car chaque image
    peut avoir un nombre différent de fissures annotées.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .jeu_donnees_coco import JeuDonneesFissuresCOCO


def collate_fn_detection(
    lot: List[Tuple[torch.Tensor, Dict]]
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Fonction de collation adaptée à Mask R-CNN.

    Retourne des listes plutôt qu'un tenseur empiché car chaque image
    peut avoir un nombre différent d'annotations.
    """
    images = [element[0] for element in lot]
    cibles = [element[1] for element in lot]
    return images, cibles


def creer_chargeurs_donnees(
    chemin_train: str | Path,
    chemin_valid: str | Path,
    chemin_test: str | Path,
    annotations_train: str | Path,
    annotations_valid: str | Path,
    annotations_test: str | Path,
    taille_lot: int = 4,
    taille_image: int = 384,
    nombre_workers: int = 2,
    epingler_memoire: bool = True,
) -> Dict[str, DataLoader]:
    """
    Crée et retourne les trois DataLoaders (train / valid / test).

    Args:
        chemin_train       : Dossier images d'entraînement.
        chemin_valid       : Dossier images de validation.
        chemin_test        : Dossier images de test.
        annotations_train  : Fichier _annotations.coco.json entraînement.
        annotations_valid  : Fichier _annotations.coco.json validation.
        annotations_test   : Fichier _annotations.coco.json test.
        taille_lot         : Nombre d'images par lot.
        taille_image       : Résolution cible (carré, en pixels).
        nombre_workers     : Processus parallèles de chargement.
        epingler_memoire   : Accélère les transferts CPU→GPU.

    Returns:
        Dictionnaire avec clés 'entrainement', 'validation', 'test'.
    """
    jeu_entrainement = JeuDonneesFissuresCOCO(
        chemin_images=chemin_train,
        chemin_annotations=annotations_train,
        taille_image=taille_image,
    )

    jeu_validation = JeuDonneesFissuresCOCO(
        chemin_images=chemin_valid,
        chemin_annotations=annotations_valid,
        taille_image=taille_image,
    )

    jeu_test = JeuDonneesFissuresCOCO(
        chemin_images=chemin_test,
        chemin_annotations=annotations_test,
        taille_image=taille_image,
    )

    chargeur_entrainement = DataLoader(
        jeu_entrainement,
        batch_size=taille_lot,
        shuffle=True,
        num_workers=nombre_workers,
        pin_memory=epingler_memoire,
        collate_fn=collate_fn_detection,
        drop_last=True,
    )

    chargeur_validation = DataLoader(
        jeu_validation,
        batch_size=taille_lot,
        shuffle=False,
        num_workers=nombre_workers,
        pin_memory=epingler_memoire,
        collate_fn=collate_fn_detection,
        drop_last=False,
    )

    chargeur_test = DataLoader(
        jeu_test,
        batch_size=1,
        shuffle=False,
        num_workers=nombre_workers,
        pin_memory=epingler_memoire,
        collate_fn=collate_fn_detection,
        drop_last=False,
    )

    print(f"\n{'═'*55}")
    print(f"  CHARGEURS DE DONNÉES INITIALISÉS")
    print(f"{'═'*55}")
    print(f"  Entraînement : {len(jeu_entrainement):>6} images | {len(chargeur_entrainement):>4} lots")
    print(f"  Validation   : {len(jeu_validation):>6} images | {len(chargeur_validation):>4} lots")
    print(f"  Test         : {len(jeu_test):>6} images | {len(chargeur_test):>4} lots")
    print(f"  Taille lot   : {taille_lot}")
    print(f"  Résolution   : {taille_image}×{taille_image}")
    print(f"  Workers      : {nombre_workers}")
    print(f"{'═'*55}\n")

    return {
        "entrainement": chargeur_entrainement,
        "validation": chargeur_validation,
        "test": chargeur_test,
    }
