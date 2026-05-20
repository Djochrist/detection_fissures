"""
Fonctions image partagées par le dataset et l'inférence.

Les modèles de détection torchvision attendent des tenseurs float dans [0, 1].
La normalisation ImageNet est appliquée ensuite par le transform interne du
modèle Mask R-CNN, ce qui évite une double normalisation côté projet.
"""

from pathlib import Path
from typing import Any

import cv2
import torch


def charger_image_rgb(chemin_image: str | Path) -> tuple[Any, int, int]:
    """Charge une image avec OpenCV et retourne RGB, largeur originale, hauteur originale."""
    chemin_image = Path(chemin_image)
    image_bgr = cv2.imread(str(chemin_image))
    if image_bgr is None:
        raise FileNotFoundError(f"Image illisible : {chemin_image}")

    hauteur_orig, largeur_orig = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, largeur_orig, hauteur_orig


def redimensionner_image_carree(image_rgb: Any, taille_image: int) -> Any:
    """Redimensionne une image RGB en carré avec l'interpolation adaptée aux images."""
    return cv2.resize(
        image_rgb,
        (taille_image, taille_image),
        interpolation=cv2.INTER_LINEAR,
    )


def image_rgb_vers_tenseur(image_rgb: Any) -> torch.Tensor:
    """Convertit une image RGB numpy [H, W, C] en tenseur float [C, H, W] dans [0, 1]."""
    return torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
