"""
Module de métriques d'évaluation pour la segmentation d'instance (Mask R-CNN).

MÉTRIQUES CALCULÉES
═══════════════════

┌─────────────────────┬────────────────────────────────────────────────────┐
│ Métrique            │ Description                                        │
├─────────────────────┼────────────────────────────────────────────────────┤
│ mAP [0.5:0.95]      │ Standard COCO universelle sur 10 seuils IoU        │
│                     │ Mesure la qualité de segmentation la plus complète │
├─────────────────────┼────────────────────────────────────────────────────┤
│ mAP @ IoU=0.50      │ Métrique principale — tolérance IoU modérée        │
├─────────────────────┼────────────────────────────────────────────────────┤
│ mAP @ IoU=0.75      │ Métrique stricte — masques précis requis           │
├─────────────────────┼────────────────────────────────────────────────────┤
│ Précision           │ VP / (VP + FP) — peu de fausses alarmes            │
├─────────────────────┼────────────────────────────────────────────────────┤
│ Rappel              │ VP / (VP + FN) — PRIORITÉ SÉCURITÉ STRUCTURELLE    │
│                     │ Manquer une fissure = danger > fausse alarme       │
├─────────────────────┼────────────────────────────────────────────────────┤
│ F1 Score            │ 2·P·R / (P+R) — compromis harmonique              │
└─────────────────────┴────────────────────────────────────────────────────┘

SEUILS DE PERFORMANCE CIBLES (fissures structurelles)
══════════════════════════════════════════════════════
    Rappel    > 0.85 : priorité — ne pas manquer de fissures critiques
    Précision > 0.80 : éviter trop d'inspections inutiles
    F1        > 0.82 : bon équilibre global
    mAP@0.5   > 0.65 : bonne qualité de segmentation d'instance
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision


def calculer_metriques_segmentation(
    predictions: List[Dict[str, torch.Tensor]],
    cibles: List[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """
    Calcule les métriques de détection et segmentation d'instance (Mask R-CNN).

    Utilise le protocole COCO via torchmetrics.detection.MeanAveragePrecision.

    Format attendu des prédictions :
        {"boxes": FloatTensor[N,4], "scores": FloatTensor[N],
         "labels": IntTensor[N], "masks": BoolTensor[N,H,W]}

    Format attendu des cibles :
        {"boxes": FloatTensor[M,4], "labels": IntTensor[M],
         "masks": BoolTensor[M,H,W]}

    Args:
        predictions : Liste de dicts de prédictions par image.
        cibles      : Liste de dicts de vérités terrain par image.

    Returns:
        Dictionnaire avec mAP, mAP_50, mAP_75, précision, rappel, F1.
    """
    calculateur_map = MeanAveragePrecision(
        iou_type="segm",
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=None,
        class_metrics=False,
        extended_summary=False,
    )

    predictions_converties = []
    for pred in predictions:
        pred_conv = dict(pred)
        if "masks" in pred_conv and pred_conv["masks"].dtype != torch.bool:
            pred_conv["masks"] = pred_conv["masks"].bool()
        predictions_converties.append(pred_conv)

    cibles_converties = []
    for cible in cibles:
        cible_conv = dict(cible)
        if "masks" in cible_conv and cible_conv["masks"].dtype != torch.bool:
            cible_conv["masks"] = cible_conv["masks"].bool()
        cibles_converties.append(cible_conv)

    calculateur_map.update(predictions_converties, cibles_converties)
    resultats = calculateur_map.compute()

    metriques = {
        "map":     float(resultats.get("map", 0.0)),
        "map_50":  float(resultats.get("map_50", 0.0)),
        "map_75":  float(resultats.get("map_75", 0.0)),
        "mar_1":   float(resultats.get("mar_1", 0.0)),
        "mar_10":  float(resultats.get("mar_10", 0.0)),
        "mar_100": float(resultats.get("mar_100", 0.0)),
    }

    # Calcul pixel-level Précision, Rappel, F1 depuis masques d'instance
    precision_list, rappel_list, f1_list = [], [], []
    for pred, cible in zip(predictions_converties, cibles_converties):
        if len(pred.get("masks", [])) > 0 and len(cible.get("masks", [])) > 0:
            masque_predit = pred["masks"][0].cpu().numpy().astype(bool)
            masque_verite = cible["masks"][0].cpu().numpy().astype(bool)

            if masque_predit.shape == masque_verite.shape:
                p, r, f1 = calculer_precision_rappel_f1(masque_predit, masque_verite)
                precision_list.append(p)
                rappel_list.append(r)
                f1_list.append(f1)

    metriques["precision"] = float(np.mean(precision_list)) if precision_list else 0.0
    metriques["rappel"]    = float(np.mean(rappel_list))    if rappel_list    else 0.0
    metriques["f1_score"]  = float(np.mean(f1_list))        if f1_list        else 0.0

    return metriques


def calculer_iou_masques(
    masque_predit: np.ndarray,
    masque_verite: np.ndarray,
) -> float:
    """
    Calcule l'IoU pixel-à-pixel entre deux masques binaires.

    IoU = Intersection / Union = |P ∩ G| / (|P| + |G| - |P ∩ G|)

    Args:
        masque_predit : Masque binaire numpy [H, W].
        masque_verite : Masque binaire ground truth [H, W].

    Returns:
        Valeur IoU dans [0, 1].
    """
    masque_predit = masque_predit.astype(bool)
    masque_verite = masque_verite.astype(bool)

    intersection = np.logical_and(masque_predit, masque_verite).sum()
    union = np.logical_or(masque_predit, masque_verite).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection) / float(union)


def calculer_precision_rappel_f1(
    masque_predit: np.ndarray,
    masque_verite: np.ndarray,
    epsilon: float = 1e-8,
) -> Tuple[float, float, float]:
    """
    Calcule Précision, Rappel et F1-Score au niveau pixel.

    Définitions :
        VP (Vrais Positifs)  : pixel prédit fissure ET est fissure
        FP (Faux Positifs)   : pixel prédit fissure MAIS est fond
        FN (Faux Négatifs)   : pixel prédit fond MAIS est fissure

        Précision = VP / (VP + FP)
        Rappel    = VP / (VP + FN)
        F1        = 2 × Précision × Rappel / (Précision + Rappel)

    REMARQUE SÉCURITÉ STRUCTURELLE :
        Pour les fissures, le RAPPEL est plus important que la Précision.
        Manquer une fissure (FN) est plus dangereux qu'une fausse alarme (FP).
        Un modèle avec Rappel=0.92 et Précision=0.78 est préférable à
        Rappel=0.78 et Précision=0.92 dans un contexte de génie civil.

    Args:
        masque_predit : Masque binaire numpy [H, W].
        masque_verite : Masque binaire ground truth [H, W].
        epsilon       : Stabilité numérique (évite division par zéro).

    Returns:
        Tuple (precision, rappel, f1).
    """
    predit_bool = masque_predit.astype(bool)
    verite_bool = masque_verite.astype(bool)

    vp = int(np.logical_and(predit_bool, verite_bool).sum())
    fp = int(np.logical_and(predit_bool, ~verite_bool).sum())
    fn = int(np.logical_and(~predit_bool, verite_bool).sum())

    precision = vp / (vp + fp + epsilon)
    rappel    = vp / (vp + fn + epsilon)
    f1        = 2.0 * precision * rappel / (precision + rappel + epsilon)

    return precision, rappel, f1


def afficher_tableau_metriques(metriques: Dict[str, float]) -> None:
    """
    Affiche les métriques dans un tableau formaté dans le terminal.

    Args:
        metriques : Dictionnaire de métriques calculées.
    """
    print(f"\n{'═'*55}")
    print(f"  MÉTRIQUES — MASK R-CNN (SEGMENTATION D'INSTANCE)")
    print(f"{'═'*55}")

    correspondance_labels = [
        ("map",       "mAP [0.5:0.95]  (Standard COCO)"),
        ("map_50",    "mAP @ IoU=0.50  (Principal)    "),
        ("map_75",    "mAP @ IoU=0.75  (Précis)       "),
        ("mar_100",   "mAR @ 100 dét.  (Rappel COCO)  "),
        ("precision", "Précision       (VP / VP+FP)   "),
        ("rappel",    "Rappel          (VP / VP+FN)   "),
        ("f1_score",  "F1 Score        (2PR / P+R)    "),
    ]

    for cle, label in correspondance_labels:
        valeur = metriques.get(cle)
        if valeur is None:
            continue
        barre = "█" * int(valeur * 20)
        print(f"  {label} : {valeur:.4f}  {barre}")

    print(f"{'═'*55}\n")
