"""
Module de construction du modèle Mask R-CNN pour la détection de fissures.

POURQUOI MASK R-CNN POUR CE PROJET ?
═════════════════════════════════════

Contexte : dataset COCO avec segmentation d'INSTANCE (4700 images, 384×384)

Comparaison des architectures 2025-2026 :
──────────────────────────────────────────

1. MASK R-CNN (NOTRE CHOIX) ✓
   Architecture : ResNet-50 + FPN + RPN + ROI Align + Masque
   Forces :
     - Segmentation d'instance nativement (chaque fissure = objet séparé)
     - Format COCO directement supporté par torchvision
     - Transfer learning depuis COCO detection (très efficace)
     - FPN = détecte fissures à différentes échelles simultanément
     - Bien étudié, peu de surprises, documentation abondante
   Faiblesses :
     - Plus lent que YOLO (non critique ici, pas de temps-réel)
     - Peut manquer des fissures très fines (< 2px) → géré par FPN

2. YOLO11-seg ✗ (pour notre usage)
   Forces : Temps-réel, léger, facile à déployer
   Faiblesses pour nous :
     - Tête de segmentation moins précise (masks 28×28 upsampled)
     - Moins adapté à l'analyse géométrique fine post-segmentation
     - Nécessite format YOLO (conversion depuis COCO nécessaire)

3. Mask2Former ✗
   Forces : État de l'art 2024, query-based, excellente précision
   Faiblesses pour nous :
     - Nécessite ~100k images pour bien converger
     - Très lourd (300M+ paramètres), risque d'overfitting sur 4700 images
     - Complexité d'implémentation élevée

4. SegFormer ✗
   Forces : Segmentation sémantique légère et efficace
   Faiblesses pour nous :
     - Segmentation SÉMANTIQUE seulement (pas d'instances séparées)
     - Transformers nécessitent beaucoup de données
     - Ne supporte pas directement le format COCO d'instance

5. SAM3 ✗  (successeur de SAM2, Meta 2025)
   Forces : Segmentation vidéo + image généraliste, qualité masque élevée
   Faiblesses pour nous :
     - AutomaticMaskGenerator sur-segmente tout sans supervision (pas spécifique aux fissures)
     - Pas d'entraînement supervisé fin sur dataset custom → précision/rappel non contrôlés
     - Aucune métrique COCO native, pas de pipeline d'entraînement structuré
     - Adapté à la segmentation promptée interactive, pas à la détection autonome

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

from typing import Any, Callable, Dict

import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    MaskRCNN,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from ..configuration.parametres import (
    ARCHITECTURE_MASKRCNN_RESNET50_FPN,
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

    poids_par_defaut = {
        ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2: MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        ARCHITECTURE_MASKRCNN_RESNET50_FPN: MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
    }[architecture]

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


def construire_modele_masque_rcnn(
    nombre_classes: int = 2,
    architecture: str = ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2,
    poids_preentraine: str | Any | None = "DEFAULT",
    nom_backbone: str = "resnet50",
    seuil_score_detection: float = 0.5,
    seuil_iou_nms: float = 0.3,
    detections_max_par_image: int = 100,
    taille_image_min: int = 384,
    taille_image_max: int = 384,
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
        architecture : Variante Mask R-CNN torchvision à instancier.
        poids_preentraine : "DEFAULT" = poids COCO préentraînés.
        nom_backbone : Backbone ResNet. 'resnet50' recommandé.
        seuil_score_detection : Confiance minimale pour conserver une détection.
        seuil_iou_nms : Seuil IoU pour la suppression non-maximale.
        detections_max_par_image : Limite de détections par image.
        taille_image_min : Taille minimale image d'entrée.
        taille_image_max : Taille maximale image d'entrée.

    Returns:
        Modèle MaskRCNN initialisé avec poids COCO et têtes adaptées.
    """
    if architecture not in ARCHITECTURES_MASK_RCNN:
        architectures = ", ".join(ARCHITECTURES_MASK_RCNN)
        raise ValueError(f"Architecture inconnue : {architecture}. Choix : {architectures}")

    if nom_backbone != "resnet50":
        raise ValueError("Les architectures disponibles utilisent le backbone 'resnet50'.")

    constructeurs: dict[str, Callable[..., MaskRCNN]] = {
        ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2: torchvision.models.detection.maskrcnn_resnet50_fpn_v2,
        ARCHITECTURE_MASKRCNN_RESNET50_FPN: torchvision.models.detection.maskrcnn_resnet50_fpn,
    }
    poids_modele = _resoudre_poids_mask_rcnn(architecture, poids_preentraine)

    modele = constructeurs[architecture](
        weights=poids_modele,
        # Paramètres de détection adaptés aux fissures
        box_score_thresh=seuil_score_detection,
        box_nms_thresh=seuil_iou_nms,
        box_detections_per_img=detections_max_par_image,
        min_size=taille_image_min,
        max_size=taille_image_max,
    )

    _remplacer_tetes_prediction(modele, nombre_classes)
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
