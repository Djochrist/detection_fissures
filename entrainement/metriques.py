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

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torchmetrics.detection import MeanAveragePrecision


def _decrire_valeur(valeur: Any) -> str:
    """Retourne une description courte et utile d'une valeur reçue."""
    if isinstance(valeur, torch.Tensor):
        return (
            f"Tensor(shape={tuple(valeur.shape)}, dtype={valeur.dtype}, "
            f"device={valeur.device})"
        )
    if isinstance(valeur, np.ndarray):
        return f"ndarray(shape={valeur.shape}, dtype={valeur.dtype})"
    if isinstance(valeur, dict):
        return f"dict(keys={list(valeur.keys())})"
    if isinstance(valeur, list):
        types = [type(item).__name__ for item in valeur[:3]]
        suffixe = "..." if len(valeur) > 3 else ""
        return f"list(len={len(valeur)}, types={types}{suffixe})"
    if isinstance(valeur, tuple):
        types = [type(item).__name__ for item in valeur[:3]]
        suffixe = "..." if len(valeur) > 3 else ""
        return f"tuple(len={len(valeur)}, types={types}{suffixe})"
    return f"{type(valeur).__name__}({valeur!r})"


def _decrire_detection(detection: Any) -> str:
    """Décrit une prédiction/cible sans supposer son format interne."""
    if not isinstance(detection, dict):
        return _decrire_valeur(detection)

    morceaux = [f"{cle}={_decrire_valeur(valeur)}" for cle, valeur in detection.items()]
    return "{" + ", ".join(morceaux) + "}"


def _est_rle_coco(masque: Any) -> bool:
    """Vérifie strictement la forme minimale d'un masque COCO RLE."""
    return (
        isinstance(masque, dict)
        and isinstance(masque.get("size"), (list, tuple))
        and len(masque["size"]) == 2
        and "counts" in masque
    )


def _normaliser_rle_coco(rle: dict[str, Any], contexte: str) -> dict[str, Any]:
    """Normalise un RLE COCO pour décodage via pycocotools."""
    assert _est_rle_coco(rle), f"{contexte}: masque RLE COCO invalide"

    hauteur, largeur = int(rle["size"][0]), int(rle["size"][1])
    if hauteur <= 0 or largeur <= 0:
        raise ValueError(f"{contexte}: taille RLE invalide : {rle['size']}")

    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    elif isinstance(counts, np.ndarray):
        counts = counts.tolist()
    elif not isinstance(counts, (str, list)):
        raise TypeError(
            f"{contexte}: counts RLE doit être str, bytes ou list, reçu "
            f"{type(counts).__name__}"
        )

    return {"size": [hauteur, largeur], "counts": counts}


def _decoder_rle_vers_bool(rle: dict[str, Any], contexte: str) -> np.ndarray:
    """Décode un RLE COCO en masque booléen [H, W]."""
    rle_normalise = _normaliser_rle_coco(rle, contexte)
    rle_pycoco = dict(rle_normalise)
    if isinstance(rle_pycoco["counts"], str):
        rle_pycoco["counts"] = rle_pycoco["counts"].encode("ascii")

    masque = mask_utils.decode(rle_pycoco)
    if masque.ndim == 3:
        masque = np.any(masque, axis=2)
    assert masque.ndim == 2, f"{contexte}: RLE décodé avec ndim={masque.ndim}"
    return masque.astype(bool)


def _normaliser_tableau_masques(masques: Any, contexte: str) -> np.ndarray:
    """
    Convertit tout format de masque supporté vers un tableau bool [N, H, W].

    Formats supportés :
    - Tensor/ndarray [H, W], [N, H, W] ou [N, 1, H, W]
    - dict RLE COCO
    - list/tuple de dicts RLE COCO ou de masques 2D
    """
    if isinstance(masques, torch.Tensor):
        tableau = masques.detach().cpu().numpy()
    elif isinstance(masques, np.ndarray):
        tableau = masques
    elif _est_rle_coco(masques):
        return _decoder_rle_vers_bool(masques, contexte)[None, :, :]
    elif isinstance(masques, (list, tuple)):
        if len(masques) == 0:
            return np.zeros((0, 0, 0), dtype=bool)

        masques_decodes = []
        for index, masque in enumerate(masques):
            sous_contexte = f"{contexte}.masks[{index}]"
            if _est_rle_coco(masque):
                masques_decodes.append(_decoder_rle_vers_bool(masque, sous_contexte))
            elif isinstance(masque, torch.Tensor):
                tableau_masque = masque.detach().cpu().numpy()
                if tableau_masque.ndim == 3 and tableau_masque.shape[0] == 1:
                    tableau_masque = tableau_masque[0]
                if tableau_masque.ndim != 2:
                    raise ValueError(
                        f"{sous_contexte}: masque tensor/list doit être 2D, reçu "
                        f"{tableau_masque.shape}"
                    )
                masques_decodes.append(tableau_masque.astype(bool))
            elif isinstance(masque, np.ndarray):
                tableau_masque = masque
                if tableau_masque.ndim == 3 and tableau_masque.shape[0] == 1:
                    tableau_masque = tableau_masque[0]
                if tableau_masque.ndim != 2:
                    raise ValueError(
                        f"{sous_contexte}: masque ndarray/list doit être 2D, reçu "
                        f"{tableau_masque.shape}"
                    )
                masques_decodes.append(tableau_masque.astype(bool))
            else:
                raise TypeError(
                    f"{sous_contexte}: format de masque non supporté : "
                    f"{_decrire_valeur(masque)}"
                )

        formes = {masque.shape for masque in masques_decodes}
        if len(formes) != 1:
            raise ValueError(f"{contexte}: masques avec tailles incohérentes : {formes}")
        return np.stack(masques_decodes).astype(bool)
    else:
        raise TypeError(
            f"{contexte}: format de masks non supporté : {_decrire_valeur(masques)}"
        )

    if tableau.ndim == 2:
        tableau = tableau[None, :, :]
    elif tableau.ndim == 3:
        pass
    elif tableau.ndim == 4 and tableau.shape[1] == 1:
        tableau = tableau[:, 0, :, :]
    else:
        raise ValueError(
            f"{contexte}: masks doit être [H,W], [N,H,W], [N,1,H,W] ou RLE, "
            f"reçu shape={tableau.shape}"
        )

    if tableau.shape[0] == 0:
        hauteur = int(tableau.shape[-2]) if tableau.ndim >= 2 else 0
        largeur = int(tableau.shape[-1]) if tableau.ndim >= 2 else 0
        return np.zeros((0, hauteur, largeur), dtype=bool)

    return tableau > 0.5


def _convertir_masques_vers_tensor(masques: Any, contexte: str) -> torch.Tensor:
    """Convertit les masques vers le Tensor [N, H, W] attendu par TorchMetrics."""
    tableau = _normaliser_tableau_masques(masques, contexte)
    if tableau.ndim != 3:
        raise ValueError(
            f"{contexte}: masks normalisé doit être [N,H,W], reçu {tableau.shape}"
        )
    return torch.as_tensor(tableau.astype(np.uint8), dtype=torch.uint8)


def _normaliser_vecteur_tensor(
    valeur: Any,
    longueur_attendue: int,
    contexte: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convertit et valide boxes/labels/scores avant TorchMetrics."""
    if isinstance(valeur, torch.Tensor):
        tenseur = valeur.detach().cpu().to(dtype=dtype)
    elif isinstance(valeur, np.ndarray):
        tenseur = torch.as_tensor(valeur, dtype=dtype)
    elif isinstance(valeur, (list, tuple)):
        tenseur = torch.as_tensor(valeur, dtype=dtype)
    else:
        raise TypeError(f"{contexte}: type non supporté : {_decrire_valeur(valeur)}")

    if tenseur.ndim == 0:
        tenseur = tenseur.reshape(1)
    if len(tenseur) != longueur_attendue:
        raise ValueError(
            f"{contexte}: longueur {len(tenseur)} incompatible avec "
            f"{longueur_attendue} masque(s)"
        )
    return tenseur


def _normaliser_boites(
    valeur: Any,
    longueur_attendue: int,
    contexte: str,
) -> torch.Tensor:
    """Convertit et valide les boîtes [N, 4]."""
    if isinstance(valeur, torch.Tensor):
        boites = valeur.detach().cpu().to(dtype=torch.float32)
    elif isinstance(valeur, np.ndarray):
        boites = torch.as_tensor(valeur, dtype=torch.float32)
    elif isinstance(valeur, (list, tuple)):
        boites = torch.as_tensor(valeur, dtype=torch.float32)
    else:
        raise TypeError(f"{contexte}: type non supporté : {_decrire_valeur(valeur)}")

    if boites.numel() == 0 and longueur_attendue == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    if len(boites) != longueur_attendue:
        raise ValueError(
            f"{contexte}: longueur {len(boites)} incompatible avec "
            f"{longueur_attendue} masque(s)"
        )
    if boites.ndim != 2 or boites.shape[1] != 4:
        raise ValueError(f"{contexte}: boxes doit être [N, 4], reçu {tuple(boites.shape)}")
    return boites


def _normaliser_detection_pour_map(
    detection: Any,
    contexte: str,
    prediction: bool,
) -> dict[str, Any]:
    """Valide et convertit une prédiction/cible pour MeanAveragePrecision(segm)."""
    if not isinstance(detection, dict):
        raise TypeError(f"{contexte}: dict attendu, reçu {_decrire_valeur(detection)}")

    champs_requis = {"boxes", "labels", "masks"}
    if prediction:
        champs_requis.add("scores")

    manquants = sorted(champs_requis - set(detection))
    if manquants:
        raise KeyError(f"{contexte}: champs manquants : {manquants}")

    masques_tensor = _convertir_masques_vers_tensor(detection["masks"], contexte)
    nombre_masques = int(masques_tensor.shape[0])

    sortie: dict[str, Any] = {
        "boxes": _normaliser_boites(
            detection["boxes"],
            nombre_masques,
            f"{contexte}.boxes",
        ),
        "labels": _normaliser_vecteur_tensor(
            detection["labels"],
            nombre_masques,
            f"{contexte}.labels",
            torch.int64,
        ),
        "masks": masques_tensor,
    }

    if prediction:
        sortie["scores"] = _normaliser_vecteur_tensor(
            detection["scores"],
            nombre_masques,
            f"{contexte}.scores",
            torch.float32,
        )

    assert isinstance(sortie["masks"], torch.Tensor), f"{contexte}: masks doit être un Tensor"
    assert sortie["masks"].ndim == 3, f"{contexte}: masks doit être [N,H,W]"
    return sortie


def _journaliser_structures_map(
    predictions_originales: list[Any],
    cibles_originales: list[Any],
    predictions_converties: list[dict[str, Any]],
    cibles_converties: list[dict[str, Any]],
    max_elements: int,
) -> None:
    """Affiche les structures avant l'appel critique à map.update()."""
    print("\n[TorchMetrics] Préparation MeanAveragePrecision(iou_type='segm')")
    print(
        f"[TorchMetrics] Images : predictions={len(predictions_originales)}, "
        f"cibles={len(cibles_originales)}"
    )

    for nom, originaux, convertis in (
        ("prediction", predictions_originales, predictions_converties),
        ("cible", cibles_originales, cibles_converties),
    ):
        for index, (original, converti) in enumerate(zip(originaux, convertis)):
            if index >= max_elements:
                restant = len(originaux) - max_elements
                if restant > 0:
                    print(f"[TorchMetrics] ... {restant} {nom}(s) non affichée(s)")
                break

            print(f"[TorchMetrics] {nom}[{index}] original : {_decrire_detection(original)}")
            print(
                f"[TorchMetrics] {nom}[{index}] converti : "
                f"boxes={_decrire_valeur(converti['boxes'])}, "
                f"labels={_decrire_valeur(converti['labels'])}, "
                f"masks={_decrire_valeur(converti['masks'])}"
            )
            if "scores" in converti:
                print(
                    f"[TorchMetrics] {nom}[{index}] "
                    f"scores={_decrire_valeur(converti['scores'])}"
                )


def calculer_metriques_segmentation(
    predictions: List[Dict[str, torch.Tensor]],
    cibles: List[Dict[str, torch.Tensor]],
    journaliser: bool = True,
    max_elements_journal: int = 3,
) -> Dict[str, float]:
    """
    Calcule les métriques de détection et segmentation d'instance (Mask R-CNN).

    Utilise le protocole COCO via torchmetrics.detection.MeanAveragePrecision.

    Formats acceptés des prédictions :
        {"boxes": FloatTensor[N,4], "scores": FloatTensor[N],
         "labels": IntTensor[N], "masks": Tensor/ndarray/list/RLE}

    Formats acceptés des cibles :
        {"boxes": FloatTensor[M,4], "labels": IntTensor[M],
         "masks": Tensor/ndarray/list/RLE}

    Avant map.update(), tous les masques sont convertis en Tensor binaire [N,H,W],
    le format attendu par TorchMetrics pour MeanAveragePrecision(iou_type="segm").

    Args:
        predictions : Liste de dicts de prédictions par image.
        cibles      : Liste de dicts de vérités terrain par image.

    Returns:
        Dictionnaire avec mAP, mAP_50, mAP_75, précision, rappel, F1.
    """
    if not isinstance(predictions, list) or not isinstance(cibles, list):
        raise TypeError(
            "calculer_metriques_segmentation attend deux listes : "
            f"predictions={_decrire_valeur(predictions)}, cibles={_decrire_valeur(cibles)}"
        )
    if len(predictions) != len(cibles):
        raise ValueError(
            f"Nombre d'images incohérent : {len(predictions)} prédiction(s) "
            f"pour {len(cibles)} cible(s)"
        )

    calculateur_map = MeanAveragePrecision(
        iou_type="segm",
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=None,
        class_metrics=False,
        extended_summary=False,
    )

    predictions_converties = [
        _normaliser_detection_pour_map(pred, f"predictions[{index}]", prediction=True)
        for index, pred in enumerate(predictions)
    ]
    cibles_converties = [
        _normaliser_detection_pour_map(cible, f"cibles[{index}]", prediction=False)
        for index, cible in enumerate(cibles)
    ]

    if journaliser:
        _journaliser_structures_map(
            predictions_originales=predictions,
            cibles_originales=cibles,
            predictions_converties=predictions_converties,
            cibles_converties=cibles_converties,
            max_elements=max_elements_journal,
        )

    try:
        calculateur_map.update(predictions_converties, cibles_converties)
    except Exception as exc:
        print("\n[TorchMetrics][ERREUR] map.update() a échoué.")
        print(f"[TorchMetrics][ERREUR] Type : {type(exc).__name__}")
        print(f"[TorchMetrics][ERREUR] Message : {exc}")
        _journaliser_structures_map(
            predictions_originales=predictions,
            cibles_originales=cibles,
            predictions_converties=predictions_converties,
            cibles_converties=cibles_converties,
            max_elements=len(predictions),
        )
        raise RuntimeError(
            "Échec MeanAveragePrecision(iou_type='segm') après conversion Tensor "
            "des masques. "
            "Les structures détaillées ont été affichées juste avant cette exception."
        ) from exc

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
    for index, (pred, cible) in enumerate(zip(predictions, cibles)):
        try:
            masques_pred = _normaliser_tableau_masques(pred["masks"], f"precision.predictions[{index}]")
            masques_cible = _normaliser_tableau_masques(cible["masks"], f"precision.cibles[{index}]")
        except Exception as exc:
            print(f"[Métriques pixel] Image {index} ignorée : {type(exc).__name__}: {exc}")
            continue

        if len(masques_pred) > 0 and len(masques_cible) > 0:
            masque_predit = masques_pred[0].astype(bool)
            masque_verite = masques_cible[0].astype(bool)

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
