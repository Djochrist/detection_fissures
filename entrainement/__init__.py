"""Package d'entraînement : boucle, pertes, métriques — Mask R-CNN uniquement."""
from .entraineur import Entraineur
from .pertes import (
    PerteCombineeMaskRCNN,
    PerteUnifiedFocal,
    PerteMasqueRCNNUnifiedFocal,
    calculer_perte_totale_masque_rcnn,
)
from .metriques import (
    calculer_metriques_segmentation,
    calculer_precision_rappel_f1,
    calculer_iou_masques,
    afficher_tableau_metriques,
)

__all__ = [
    "Entraineur",
    "PerteCombineeMaskRCNN",
    "PerteUnifiedFocal",
    "PerteMasqueRCNNUnifiedFocal",
    "calculer_perte_totale_masque_rcnn",
    "calculer_metriques_segmentation",
    "calculer_precision_rappel_f1",
    "calculer_iou_masques",
    "afficher_tableau_metriques",
]
