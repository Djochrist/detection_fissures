"""
Package des architectures de modèles de segmentation.

Modèle unique — segmentation d'INSTANCE :
    Mask R-CNN ResNet50-FPN-V2  (torchvision)
    → Détecte et segmente chaque fissure séparément
    → Format COCO natif, analyse géométrique par instance
"""
from .masque_rcnn import (
    construire_modele_masque_rcnn,
    geler_backbone,
    degeler_couches_superieures,
    degeler_backbone_complet,
    compter_parametres,
    afficher_resume_modele,
)

__all__ = [
    "construire_modele_masque_rcnn",
    "geler_backbone",
    "degeler_couches_superieures",
    "degeler_backbone_complet",
    "compter_parametres",
    "afficher_resume_modele",
]
