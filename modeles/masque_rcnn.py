"""
Module de construction du modèle Mask R-CNN pour la détection de fissures.

POURQUOI MASK R-CNN POUR CE PROJET ?
═════════════════════════════════════

Contexte : dataset COCO avec segmentation d'INSTANCE (4700 images, 384×384)

Modèles retenus pour ce projet :
──────────────────────────────────────────

1. MASK R-CNN RESNET50-FPN-V2 ✓
   Architecture : ResNet-50 + FPN + RPN + ROI Align + masque
   Forces :
     - Segmentation d'instance nativement (chaque fissure = objet séparé)
     - Format COCO directement supporté par torchvision
     - Transfer learning depuis COCO detection (très efficace)
     - FPN = détecte fissures à différentes échelles simultanément
     - Bien étudié, peu de surprises, documentation abondante
   Faiblesses :
     - Plus lent que YOLO (non critique ici, pas de temps-réel)
     - Peut manquer des fissures très fines (< 2px) → géré par FPN

2. YOLO-seg ✓
   Forces :
     - Temps-réel, léger, facile à déployer
     - Très utile pour comparer vitesse/précision face à Mask R-CNN
   Point d'attention :
     - Nécessite une conversion COCO vers YOLO-seg, gérée par entrainer_yolo.py

Aucun autre modèle n'est exposé par le code d'entraînement.

ARCHITECTURE MASK R-CNN DÉTAILLÉE :
─────────────────────────────────────

Image [3, 384, 384]
    ↓
ResNet-50 (backbone)
    ↓ features multi-échelles [P2, P3, P4, P5, P6]
FPN (Feature Pyramid Network)
    ↓ proposals de régions
RPN (Region Proposal Network)
    ↓ ROIs sélectionnées
ROI Align (extraction de features par région)
    ↓ features [N, 256, 7, 7] pour chaque ROI
┌──────────────────┬─────────────────────┐
│   Tête Detection  │    Tête Masque      │
│   (classes + bbox)│  (masque 28×28)     │
└──────────────────┴─────────────────────┘
    ↓
Prédictions : boîtes + scores + masques binaires
"""

from typing import Any, Dict

import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    MaskRCNN,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from ..configuration.parametres import (
    ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2,
    ARCHITECTURES_MODELES_SEGMENTATION,
)

ARCHITECTURES_MASK_RCNN = ARCHITECTURES_MODELES_SEGMENTATION


def _resoudre_poids_mask_rcnn(
    architecture: str,
    poids_preentraine: str | Any | None,
) -> Any | None:
    """Convertit l'option utilisateur vers l'énumération officielle torchvision."""
    if poids_preentraine is None:
        return None

    poids_par_defaut = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT

    if not isinstance(poids_preentraine, str):
        return poids_preentraine
    if poids_preentraine.upper() == "DEFAULT":
        return poids_par_defaut
    if poids_preentraine.upper() in {"NONE", "AUCUN", "FALSE"}:
        return None
    raise ValueError(
        "poids_preentraine doit valoir 'DEFAULT', 'NONE' ou une valeur "
        "de weights torchvision compatible avec l'architecture choisie."
    )


def _remplacer_tetes_prediction(modele: MaskRCNN, nombre_classes: int) -> None:
    """Remplace les têtes COCO par des têtes adaptées aux classes du projet."""
    dimension_entree_bbox = modele.roi_heads.box_predictor.cls_score.in_features
    modele.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=dimension_entree_bbox,
        num_classes=nombre_classes,
    )

    dimension_entree_masque = modele.roi_heads.mask_predictor.conv5_mask.in_channels
    modele.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=dimension_entree_masque,
        dim_reduced=256,
        num_classes=nombre_classes,
    )


def _creer_generateur_ancres_fissures(
    tailles: tuple[tuple[int, ...], ...],
    ratios: tuple[tuple[float, ...], ...],
) -> AnchorGenerator:
    """Crée des ancres RPN adaptées aux fissures longues et minces."""
    if len(tailles) != len(ratios):
        raise ValueError(
            "tailles_ancres_rpn et ratios_ancres_rpn doivent avoir le même "
            f"nombre de niveaux FPN, reçu {len(tailles)} et {len(ratios)}."
        )
    return AnchorGenerator(sizes=tailles, aspect_ratios=ratios)


def _adapter_ancres_rpn(
    modele: MaskRCNN,
    tailles: tuple[tuple[int, ...], ...],
    ratios: tuple[tuple[float, ...], ...],
) -> None:
    """
    Remplace les ancres RPN sans casser les poids préentraînés.

    Le nombre d'ancres par position reste identique aux poids COCO pour garder
    la tête RPN préentraînée, mais les tailles/ratios ciblent mieux les fissures.
    """
    generateur = _creer_generateur_ancres_fissures(tailles, ratios)
    nb_ancres_nouveau = generateur.num_anchors_per_location()
    nb_ancres_courant = modele.rpn.anchor_generator.num_anchors_per_location()
    if nb_ancres_nouveau != nb_ancres_courant:
        raise ValueError(
            "La configuration RPN changerait le nombre d'ancres par position "
            f"({nb_ancres_courant} -> {nb_ancres_nouveau}). "
            "Gardez le même nombre de ratios pour conserver la tête RPN préentraînée."
        )
    modele.rpn.anchor_generator = generateur


def construire_modele_masque_rcnn(
    nombre_classes: int = 2,
    architecture: str = ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2,
    poids_preentraine: str | Any | None = "DEFAULT",
    nom_backbone: str = "resnet50",
    seuil_score_detection: float = 0.5,
    seuil_iou_nms: float = 0.3,
    detections_max_par_image: int = 100,
    taille_image_min: int = 512,
    taille_image_max: int = 512,
    tailles_ancres_rpn: tuple[tuple[int, ...], ...] = (
        (8,),
        (16,),
        (32,),
        (64,),
        (128,),
    ),
    ratios_ancres_rpn: tuple[tuple[float, ...], ...] = (
        (0.25, 1.0, 4.0),
        (0.25, 1.0, 4.0),
        (0.25, 1.0, 4.0),
        (0.25, 1.0, 4.0),
        (0.25, 1.0, 4.0),
    ),
) -> MaskRCNN:
    """
    Construit et retourne un modèle Mask R-CNN adapté à la détection de fissures.

    Stratégie de transfer learning en 3 phases :
    ─────────────────────────────────────────────
    Phase 1 (époques 1-5) : Backbone gelé
        → On entraîne SEULEMENT les têtes de détection et masque
        → Le backbone garde ses features ImageNet
        → Convergence rapide des têtes (moins de paramètres)

    Phase 2 (époques 5-15) : Dégelage progressif du backbone
        → On dégèle les couches layer4 + layer3 du backbone
        → Fine-tuning des features de haut niveau
        → Taux d'apprentissage 10× plus bas pour le backbone

    Phase 3 (époques 15+) : Dégelage complet
        → Toutes les couches entraînables
        → Fine-tuning fin complet
        → Early stopping surveille la validation mAP

    Args:
        nombre_classes : Nombre de classes + 1 fond. Ici : 2 (fond + fissure).
        architecture : Doit valoir maskrcnn_resnet50_fpn_v2.
        poids_preentraine : "DEFAULT" = poids COCO préentraînés.
        nom_backbone : Backbone ResNet. 'resnet50' recommandé.
        seuil_score_detection : Confiance minimale pour conserver une détection.
        seuil_iou_nms : Seuil IoU pour la suppression non-maximale.
        detections_max_par_image : Limite de détections par image.
        taille_image_min : Taille minimale image d'entrée.
        taille_image_max : Taille maximale image d'entrée.
        tailles_ancres_rpn : Tailles d'ancres par niveau FPN.
        ratios_ancres_rpn : Ratios hauteur/largeur par niveau FPN.

    Returns:
        Modèle MaskRCNN initialisé avec poids COCO et têtes adaptées.
    """
    if architecture not in ARCHITECTURES_MASK_RCNN:
        architectures = ", ".join(ARCHITECTURES_MASK_RCNN)
        raise ValueError(
            f"Architecture non autorisée : {architecture}. "
            f"Le projet expose uniquement : {architectures}"
        )

    if nom_backbone != "resnet50":
        raise ValueError("Les architectures disponibles utilisent le backbone 'resnet50'.")

    poids_modele = _resoudre_poids_mask_rcnn(architecture, poids_preentraine)

    modele = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=poids_modele,
        # Paramètres de détection adaptés aux fissures
        box_score_thresh=seuil_score_detection,
        box_nms_thresh=seuil_iou_nms,
        box_detections_per_img=detections_max_par_image,
        min_size=taille_image_min,
        max_size=taille_image_max,
    )

    _remplacer_tetes_prediction(modele, nombre_classes)
    _adapter_ancres_rpn(modele, tailles_ancres_rpn, ratios_ancres_rpn)
    modele.nom_architecture_detection = architecture

    return modele


def geler_backbone(modele: MaskRCNN) -> None:
    """
    Gèle toutes les couches du backbone (Phase 1 d'entraînement).

    Pendant les premières époques, seules les têtes de détection
    et de masque sont entraînées. Cela permet :
    - Une convergence rapide des nouvelles têtes
    - D'éviter la destruction des features préentraînées
    - Une stabilisation avant le fine-tuning fin

    Args:
        modele : Modèle Mask R-CNN.
    """
    for parametre in modele.backbone.parameters():
        parametre.requires_grad = False
    print("[Modele] Backbone gelé — entraînement têtes uniquement")


def degeler_couches_superieures(modele: MaskRCNN) -> None:
    """
    Dégèle les couches supérieures du backbone (Phase 2).

    On dégèle layer3 et layer4 de ResNet50 (features de haut niveau)
    tout en gardant layer1 et layer2 gelés (features bas niveau = texture,
    bords → déjà génériques depuis ImageNet).

    Pourquoi cette approche progressive ?
        Les couches profondes apprennent des patterns spécifiques au domaine
        (fissures). Les couches peu profondes (bords, textures) sont déjà
        optimales depuis ImageNet. Les regeler économise du calcul et
        réduit l'overfitting.

    Args:
        modele : Modèle Mask R-CNN avec backbone gelé.
    """
    # Dégeler layer3 et layer4 du ResNet (couches profondes)
    couches_a_degeler = ["layer3", "layer4"]
    for nom_couche, module in modele.backbone.body.named_children():
        if nom_couche in couches_a_degeler:
            for parametre in module.parameters():
                parametre.requires_grad = True
    print(f"[Modele] Couches dégelées : {couches_a_degeler}")


def degeler_backbone_complet(modele: MaskRCNN) -> None:
    """
    Dégèle toutes les couches du backbone (Phase 3 - fine-tuning complet).

    Utilisé après stabilisation des têtes pour un ajustement fin
    de l'ensemble des paramètres du modèle.

    Args:
        modele : Modèle Mask R-CNN.
    """
    for parametre in modele.backbone.parameters():
        parametre.requires_grad = True
    print("[Modele] Backbone complètement dégelé — fine-tuning complet")


def compter_parametres(modele: nn.Module) -> Dict[str, int]:
    """
    Compte les paramètres entraînables et non-entraînables du modèle.

    Utile pour comprendre la capacité du modèle et diagnostiquer
    les problèmes de gélification/dégelification.

    Args:
        modele : Modèle PyTorch.

    Returns:
        Dictionnaire avec le nombre de paramètres.
    """
    parametres_entrainables = sum(
        p.numel() for p in modele.parameters() if p.requires_grad
    )
    parametres_totaux = sum(p.numel() for p in modele.parameters())
    parametres_geles = parametres_totaux - parametres_entrainables

    return {
        "total": parametres_totaux,
        "entrainables": parametres_entrainables,
        "geles": parametres_geles,
        "pourcentage_entraine": round(100 * parametres_entrainables / parametres_totaux, 1),
    }


def afficher_resume_modele(modele: nn.Module) -> None:
    """
    Affiche un résumé concis du modèle dans le terminal.

    Args:
        modele : Modèle PyTorch.
    """
    stats = compter_parametres(modele)
    print(f"\n{'═'*55}")
    architecture = getattr(modele, "nom_architecture_detection", "Mask R-CNN")
    print(f"  RÉSUMÉ DU MODÈLE : {architecture}")
    print(f"{'═'*55}")
    print(f"  Paramètres totaux     : {stats['total']:>12,}")
    print(f"  Paramètres entraîn.   : {stats['entrainables']:>12,}")
    print(f"  Paramètres gelés      : {stats['geles']:>12,}")
    print(f"  Taux d'entraînement   : {stats['pourcentage_entraine']:>11.1f}%")
    print(f"{'═'*55}\n")
