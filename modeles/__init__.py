"""
Package des architectures de modèles de segmentation.

Modèles — segmentation d'INSTANCE :
    Mask R-CNN ResNet50-FPN-V2  (torchvision, défaut)
    Mask R-CNN ResNet50-FPN     (torchvision, benchmark alternatif)
    → Détecte et segmente chaque fissure séparément
    → Format COCO natif, analyse géométrique par instance
"""
from .masque_rcnn import (
    ARCHITECTURES_MASK_RCNN,
    ARCHITECTURE_MASKRCNN_RESNET50_FPN,
    ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2,
    construire_modele_masque_rcnn,
    geler_backbone,
    degeler_couches_superieures,
    degeler_backbone_complet,
    compter_parametres,
    afficher_resume_modele,
)

__all__ = [
    "ARCHITECTURES_MASK_RCNN",
    "ARCHITECTURE_MASKRCNN_RESNET50_FPN",
    "ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2",
    "construire_modele_masque_rcnn",
    "geler_backbone",
    "degeler_couches_superieures",
    "degeler_backbone_complet",
    "compter_parametres",
    "afficher_resume_modele",
]
