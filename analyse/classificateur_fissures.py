"""
Module de classification des fissures après inférence Mask R-CNN.

Ce module analyse les masques de segmentation prédits par le modèle
pour déterminer le TYPE de chaque fissure selon deux axes :

1. ORIENTATION (analyse PCA sur les pixels du masque)
   ─────────────────────────────────────────────────────
   La PCA (Analyse en Composantes Principales) appliquée aux coordonnées
   des pixels du masque donne l'axe principal de la fissure.
   L'angle de cet axe détermine l'orientation :

       Horizontale  : |angle| < SEUIL_HORIZONTAL (20°)
       Verticale    : |angle| > SEUIL_VERTICAL    (70°)
       Inclinée     : entre les deux

2. LOCALISATION / PROFONDEUR (analyse morphologique du masque)
   ─────────────────────────────────────────────────────────────
   La largeur moyenne d'une fissure est estimée via la DISTANCE TRANSFORM
   (chaque pixel du masque reçoit la valeur de sa distance au bord le plus
   proche → la valeur médiane × 2 = largeur estimée en pixels).

       Superficielle  : fissure fine → largeur_moy < SEUIL_SUPERFICIELLE
                        Affect uniquement la couche de surface (enduit, peinture)
                        Danger structurel : FAIBLE

       Profonde       : fissure large → largeur_moy > SEUIL_PROFONDE
                        Pénètre dans le matériau porteur
                        Danger structurel : ÉLEVÉ

       Transversale   : fissure traversant la majorité de l'élément structurel
                        Proxy visuel : bbox couvre > SEUIL_TRAVERSEE de la
                        dimension image, ET longueur relative élevée
                        Danger structurel : CRITIQUE (peut indiquer séparation)

NOTE SUR LES SEUILS :
    Les seuils en pixels dépendent de la résolution et de la distance caméra.
    Pour une calibration physique, multiplier par la résolution mm/pixel.
    Les valeurs par défaut sont adaptées aux images 640×640px (dataset Roboflow YOLOv11).

USAGE :
    from detection_fissures.analyse.classificateur_fissures import classifier_predictions
    resultats = classifier_predictions(predictions, largeur_image=640, hauteur_image=640)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


# ──────────────────────────────────────────────────────────────────────────────
# SEUILS DE CLASSIFICATION (ajustables)
# ──────────────────────────────────────────────────────────────────────────────

SEUIL_ANGLE_HORIZONTAL: float = 20.0   # Degrés : en dessous → horizontale
SEUIL_ANGLE_VERTICAL: float   = 70.0   # Degrés : au dessus → verticale

SEUIL_LARGEUR_SUPERFICIELLE: float = 6.0   # Pixels : en dessous → superficielle (640×640px)
SEUIL_LARGEUR_PROFONDE: float      = 12.0  # Pixels : au dessus → profonde (640×640px)

SEUIL_TRAVERSEE: float = 0.65   # Ratio (0-1) : bbox > 65% de la dim. image → transversale


# ──────────────────────────────────────────────────────────────────────────────
# TYPES ÉNUMÉRÉS
# ──────────────────────────────────────────────────────────────────────────────

class ClassificationOrientation(str, Enum):
    """Classification de la fissure selon son orientation angulaire."""
    HORIZONTALE = "horizontale"
    VERTICALE   = "verticale"
    INCLINEE    = "inclinée"
    INCONNUE    = "inconnue"   # Masque trop petit pour analyser


class ClassificationLocalisation(str, Enum):
    """
    Classification selon la profondeur/localisation de la fissure.

    SUPERFICIELLE : Fine, n'affecte que la surface — peu dangereuse.
    PROFONDE      : Large, pénètre dans le matériau porteur — dangereuse.
    TRANSVERSALE  : Traverse tout l'élément structurel — critique.
    INCONNUE      : Information insuffisante.
    """
    SUPERFICIELLE = "superficielle"
    PROFONDE      = "profonde"
    TRANSVERSALE  = "transversale"
    INCONNUE      = "inconnue"


# ──────────────────────────────────────────────────────────────────────────────
# RÉSULTAT DE CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResultatClassification:
    """
    Résultat complet de la classification d'une fissure.

    Attributs :
        orientation          : Type d'orientation (horizontale/verticale/inclinée)
        localisation         : Type de localisation (superficielle/profonde/transversale)
        angle_degres         : Angle précis en degrés [0, 90]
        largeur_moy_pixels   : Largeur moyenne estimée en pixels
        ratio_couverture     : Ratio de la bbox sur la dimension de l'image [0, 1]
        aire_pixels          : Aire totale du masque en pixels
        score_detection      : Score de confiance du modèle (si disponible)
        indice_danger        : Score de danger composite [0, 1]
    """
    orientation:        ClassificationOrientation = ClassificationOrientation.INCONNUE
    localisation:       ClassificationLocalisation = ClassificationLocalisation.INCONNUE
    angle_degres:       float = 0.0
    largeur_moy_pixels: float = 0.0
    ratio_couverture:   float = 0.0
    aire_pixels:        int   = 0
    score_detection:    float = 1.0
    indice_danger:      float = 0.0

    def __str__(self) -> str:
        return (
            f"Orientation : {self.orientation.value:15s} | "
            f"Angle : {self.angle_degres:5.1f}°\n"
            f"Localisation : {self.localisation.value:13s} | "
            f"Largeur moy. : {self.largeur_moy_pixels:.1f}px | "
            f"Danger : {self.indice_danger:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# FONCTIONS D'ANALYSE GÉOMÉTRIQUE
# ──────────────────────────────────────────────────────────────────────────────

def _calculer_angle_pca(masque: np.ndarray) -> Optional[float]:
    """
    Calcule l'angle de l'axe principal de la fissure par PCA.

    La PCA appliquée aux coordonnées (x, y) des pixels du masque donne
    le vecteur propre dominant = direction principale de la fissure.

    Pourquoi PCA plutôt que boite englobante ?
        La boite englobante est sensible aux courbures et ramifications.
        La PCA donne l'axe de VARIANCE MAXIMALE = orientation réelle,
        même pour les fissures courbes ou ramifiées.

    Args:
        masque : Masque binaire numpy [H, W] avec valeurs 0 ou 1.

    Returns:
        Angle en degrés [0, 90] ou None si masque insuffisant.
    """
    indices_y, indices_x = np.where(masque > 0)

    if len(indices_x) < 10:
        return None

    # Construire la matrice de coordonnées centrées
    coords = np.column_stack([indices_x, indices_y]).astype(float)
    coords -= coords.mean(axis=0)

    # Matrice de covariance 2×2
    cov = np.cov(coords.T)

    # Vecteurs propres → le premier = axe principal
    valeurs_propres, vecteurs_propres = np.linalg.eigh(cov)
    axe_principal = vecteurs_propres[:, np.argmax(valeurs_propres)]

    # Angle en degrés [0, 180)
    angle_rad = np.arctan2(axe_principal[1], axe_principal[0])
    angle_deg = np.degrees(angle_rad) % 180.0

    # Ramener dans [0, 90] (symétrie : fissure à 100° = fissure à 80°)
    if angle_deg > 90.0:
        angle_deg = 180.0 - angle_deg

    return float(angle_deg)


def _calculer_largeur_moyenne(masque: np.ndarray) -> float:
    """
    Estime la largeur moyenne d'une fissure via la distance transform.

    Méthode distance transform (Rosenfeld & Pfaltz, 1966) :
        Pour chaque pixel du masque, la distance euclidienne au bord le
        plus proche (fond) est calculée. Cette valeur = rayon local de
        la fissure. La largeur locale = valeur × 2.
        La largeur moyenne = médiane des largeurs locales (robuste aux outliers).

    Avantage vs méthode boite englobante :
        Mesure locale → précise même pour les fissures non-rectilignes.
        La médiane est robuste aux extrémités élargies de la fissure.

    Args:
        masque : Masque binaire numpy [H, W].

    Returns:
        Largeur moyenne en pixels (diamètre, pas rayon).
    """
    if masque.sum() == 0:
        return 0.0

    masque_bool = masque.astype(bool)
    dist = distance_transform_edt(masque_bool)

    rayons = dist[masque_bool]
    if len(rayons) == 0:
        return 0.0

    # × 2 : rayon → diamètre (largeur)
    return float(np.median(rayons)) * 2.0


def _calculer_ratio_couverture(
    masque: np.ndarray,
    largeur_image: int,
    hauteur_image: int,
) -> float:
    """
    Calcule le ratio de couverture de la fissure par rapport à l'image.

    Proxy pour la classification TRANSVERSALE :
        Une fissure qui traverse tout l'élément structurel aura une boite
        englobante couvrant une grande fraction de la dimension de l'image.

    Formule :
        ratio = max(bbox_width / image_width, bbox_height / image_height)

    Un ratio > SEUIL_TRAVERSEE (0.65) indique une fissure transversale probable.

    Args:
        masque        : Masque binaire [H, W].
        largeur_image : Largeur de l'image en pixels.
        hauteur_image : Hauteur de l'image en pixels.

    Returns:
        Ratio de couverture maximal dans [0, 1].
    """
    indices_y, indices_x = np.where(masque > 0)
    if len(indices_x) == 0:
        return 0.0

    bbox_largeur = float(indices_x.max() - indices_x.min() + 1)
    bbox_hauteur = float(indices_y.max() - indices_y.min() + 1)

    ratio_x = bbox_largeur / largeur_image
    ratio_y = bbox_hauteur / hauteur_image

    return float(max(ratio_x, ratio_y))


def _calculer_indice_danger(
    orientation: ClassificationOrientation,
    localisation: ClassificationLocalisation,
    largeur_moy: float,
    ratio_couverture: float,
) -> float:
    """
    Calcule un indice de danger composite [0, 1] basé sur les classifications.

    Formule pondérée :

        indice = w_loc * score_localisation
               + w_ori * score_orientation
               + w_lar * score_largeur_normalisee
               + w_cov * ratio_couverture

    Poids (génie civil, priorités) :
        Localisation  : 0.45 — profonde/transversale = danger principal
        Orientation   : 0.20 — verticale = plus dangereuse que horizontale
        Largeur       : 0.20 — largeur normalisée sur 20px max
        Couverture    : 0.15 — étendue de la fissure

    Args:
        orientation      : Type d'orientation.
        localisation     : Type de localisation.
        largeur_moy      : Largeur moyenne en pixels.
        ratio_couverture : Ratio bbox/image.

    Returns:
        Indice de danger dans [0, 1].
    """
    scores_localisation = {
        ClassificationLocalisation.SUPERFICIELLE: 0.20,
        ClassificationLocalisation.PROFONDE:      0.75,
        ClassificationLocalisation.TRANSVERSALE:  1.00,
        ClassificationLocalisation.INCONNUE:      0.30,
    }

    scores_orientation = {
        ClassificationOrientation.HORIZONTALE: 0.30,
        ClassificationOrientation.INCLINEE:    0.60,
        ClassificationOrientation.VERTICALE:   0.85,
        ClassificationOrientation.INCONNUE:    0.40,
    }

    score_loc = scores_localisation.get(localisation, 0.3)
    score_ori = scores_orientation.get(orientation, 0.4)
    score_lar = min(largeur_moy / 20.0, 1.0)
    score_cov = ratio_couverture

    indice = (
        0.45 * score_loc
        + 0.20 * score_ori
        + 0.20 * score_lar
        + 0.15 * score_cov
    )

    return float(np.clip(indice, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE : CLASSIFICATION D'UN MASQUE
# ──────────────────────────────────────────────────────────────────────────────

def classifier_fissure(
    masque: np.ndarray,
    largeur_image: int,
    hauteur_image: int,
    score_detection: float = 1.0,
    seuil_angle_horizontal: float = SEUIL_ANGLE_HORIZONTAL,
    seuil_angle_vertical: float   = SEUIL_ANGLE_VERTICAL,
    seuil_largeur_superficielle: float = SEUIL_LARGEUR_SUPERFICIELLE,
    seuil_largeur_profonde: float      = SEUIL_LARGEUR_PROFONDE,
    seuil_traversee: float             = SEUIL_TRAVERSEE,
) -> ResultatClassification:
    """
    Classifie une fissure à partir de son masque de segmentation.

    Effectue une analyse géométrique complète en 4 étapes :
        1. PCA sur les pixels → angle de l'axe principal → orientation
        2. Distance transform → largeur médiane → profondeur/localisation
        3. Ratio bbox/image → détection transversale
        4. Calcul de l'indice de danger composite

    Args:
        masque                    : Masque binaire numpy [H, W] (0 ou 1).
        largeur_image             : Largeur de l'image d'origine en pixels.
        hauteur_image             : Hauteur de l'image d'origine en pixels.
        score_detection           : Score de confiance du modèle [0, 1].
        seuil_angle_horizontal    : Angle max pour classification horizontale.
        seuil_angle_vertical      : Angle min pour classification verticale.
        seuil_largeur_superficielle : Largeur max (px) pour superficielle.
        seuil_largeur_profonde    : Largeur min (px) pour profonde.
        seuil_traversee           : Ratio min bbox/image pour transversale.

    Returns:
        ResultatClassification avec toutes les propriétés calculées.
    """
    masque_np = np.asarray(masque, dtype=np.uint8)
    aire = int(masque_np.sum())

    resultat = ResultatClassification(score_detection=score_detection, aire_pixels=aire)

    if aire < 30:
        return resultat

    # ── 1. Classification de l'orientation (PCA) ──────────────────────────────
    angle = _calculer_angle_pca(masque_np)

    if angle is not None:
        resultat.angle_degres = angle

        if angle < seuil_angle_horizontal:
            resultat.orientation = ClassificationOrientation.HORIZONTALE
        elif angle > seuil_angle_vertical:
            resultat.orientation = ClassificationOrientation.VERTICALE
        else:
            resultat.orientation = ClassificationOrientation.INCLINEE

    # ── 2. Largeur moyenne (distance transform) ────────────────────────────────
    largeur_moy = _calculer_largeur_moyenne(masque_np)
    resultat.largeur_moy_pixels = largeur_moy

    # ── 3. Ratio de couverture (détection transversale) ───────────────────────
    ratio = _calculer_ratio_couverture(masque_np, largeur_image, hauteur_image)
    resultat.ratio_couverture = ratio

    # ── 4. Classification de la localisation ─────────────────────────────────
    if ratio >= seuil_traversee:
        # Fissure traversant la majorité de l'élément structurel
        resultat.localisation = ClassificationLocalisation.TRANSVERSALE

    elif largeur_moy <= seuil_largeur_superficielle:
        # Fissure fine → superficielle
        resultat.localisation = ClassificationLocalisation.SUPERFICIELLE

    elif largeur_moy >= seuil_largeur_profonde:
        # Fissure large → profonde
        resultat.localisation = ClassificationLocalisation.PROFONDE

    else:
        # Zone intermédiaire → on utilise le ratio pour trancher
        if ratio > 0.40:
            resultat.localisation = ClassificationLocalisation.PROFONDE
        else:
            resultat.localisation = ClassificationLocalisation.SUPERFICIELLE

    # ── 5. Indice de danger composite ─────────────────────────────────────────
    resultat.indice_danger = _calculer_indice_danger(
        resultat.orientation,
        resultat.localisation,
        largeur_moy,
        ratio,
    )

    return resultat


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION DE TOUTES LES PRÉDICTIONS D'UNE IMAGE
# ──────────────────────────────────────────────────────────────────────────────

def classifier_predictions(
    predictions: List[Dict],
    largeur_image: int,
    hauteur_image: int,
    seuil_score_min: float = 0.3,
) -> List[ResultatClassification]:
    """
    Classifie toutes les fissures détectées par Mask R-CNN sur une image.

    Filtre les détections avec un score inférieur à seuil_score_min,
    puis classifie chaque masque restant.

    Format attendu des prédictions (sortie de Mask R-CNN) :
        [
            {
                "masks"  : Tensor[N, 1, H, W]  (float, 0..1)
                "scores" : Tensor[N]
                "labels" : Tensor[N]
            },
            ...
        ]

    Args:
        predictions    : Liste de dicts de prédictions par image (format torchvision).
        largeur_image  : Largeur des images en pixels.
        hauteur_image  : Hauteur des images en pixels.
        seuil_score_min : Score de confiance minimal pour analyser une détection.

    Returns:
        Liste de ResultatClassification, une par fissure détectée.

    Example:
        >>> resultats = classifier_predictions(predictions, 640, 640)
        >>> for r in resultats:
        ...     print(r)
    """
    resultats = []

    for pred in predictions:
        masques = pred.get("masks")
        scores  = pred.get("scores")

        if masques is None or len(masques) == 0:
            continue

        # Conversion tenseur → numpy si nécessaire
        if hasattr(masques, "numpy"):
            masques_np = masques.squeeze(1).cpu().numpy()  # [N, H, W]
        else:
            masques_np = np.array(masques)

        if hasattr(scores, "numpy"):
            scores_np = scores.cpu().numpy()
        else:
            scores_np = np.array(scores) if scores is not None else np.ones(len(masques_np))

        for i, masque in enumerate(masques_np):
            score = float(scores_np[i]) if i < len(scores_np) else 1.0

            if score < seuil_score_min:
                continue

            # Binariser le masque (seuil 0.5 standard Mask R-CNN)
            masque_binaire = (masque > 0.5).astype(np.uint8)

            r = classifier_fissure(
                masque=masque_binaire,
                largeur_image=largeur_image,
                hauteur_image=hauteur_image,
                score_detection=score,
            )
            resultats.append(r)

    return resultats


# ──────────────────────────────────────────────────────────────────────────────
# AFFICHAGE DES RÉSULTATS
# ──────────────────────────────────────────────────────────────────────────────

def afficher_resultats_classification(
    resultats: List[ResultatClassification],
    nom_image: str = "",
) -> None:
    """
    Affiche un tableau récapitulatif de la classification de toutes les fissures.

    Args:
        resultats  : Liste de résultats de classification.
        nom_image  : Nom de l'image pour l'en-tête (optionnel).
    """
    if not resultats:
        print("  Aucune fissure détectée.")
        return

    entete = f"  CLASSIFICATION DES FISSURES"
    if nom_image:
        entete += f" — {nom_image}"

    print(f"\n{'═'*68}")
    print(entete)
    print(f"{'═'*68}")
    print(
        f"  {'#':>3} | {'Orientation':13} | {'Localisation':13} | "
        f"{'Angle':>7} | {'Largeur':>8} | {'Danger':>6} | {'Score':>6}"
    )
    print(f"  {'-'*3}-+-{'-'*13}-+-{'-'*13}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")

    for i, r in enumerate(resultats, start=1):
        icone_danger = "▓▓▓" if r.indice_danger > 0.65 else ("▓▓░" if r.indice_danger > 0.35 else "▓░░")
        print(
            f"  {i:>3} | {r.orientation.value:13} | {r.localisation.value:13} | "
            f"{r.angle_degres:>6.1f}° | {r.largeur_moy_pixels:>7.1f}px | "
            f"{r.indice_danger:>5.2f} {icone_danger} | {r.score_detection:>5.2f}"
        )

    print(f"{'─'*68}")

    # Statistiques par type
    ori_counts: Dict[str, int] = {}
    loc_counts: Dict[str, int] = {}
    for r in resultats:
        ori_counts[r.orientation.value] = ori_counts.get(r.orientation.value, 0) + 1
        loc_counts[r.localisation.value] = loc_counts.get(r.localisation.value, 0) + 1

    print(f"\n  Répartition par orientation :")
    for k, v in sorted(ori_counts.items()):
        barre = "█" * v
        print(f"    {k:15s} : {v:3d}  {barre}")

    print(f"\n  Répartition par localisation :")
    for k, v in sorted(loc_counts.items()):
        barre = "█" * v
        print(f"    {k:15s} : {v:3d}  {barre}")

    danger_max = max(r.indice_danger for r in resultats)
    danger_moy = sum(r.indice_danger for r in resultats) / len(resultats)
    print(f"\n  Indice de danger  → Moyen : {danger_moy:.2f}  |  Max : {danger_max:.2f}")
    print(f"{'═'*68}\n")


def generer_rapport_classification(
    resultats: List[ResultatClassification],
    nom_image: str = "",
) -> Dict:
    """
    Génère un dictionnaire structuré des résultats pour export JSON.

    Args:
        resultats  : Liste de résultats de classification.
        nom_image  : Nom de l'image analysée.

    Returns:
        Dictionnaire sérialisable (JSON) avec statistiques et détails.
    """
    if not resultats:
        return {"image": nom_image, "fissures": [], "statistiques": {}}

    fissures = [
        {
            "id":           i + 1,
            "orientation":  r.orientation.value,
            "localisation": r.localisation.value,
            "angle_degres": round(r.angle_degres, 2),
            "largeur_moy_pixels": round(r.largeur_moy_pixels, 2),
            "ratio_couverture":   round(r.ratio_couverture, 3),
            "aire_pixels":        r.aire_pixels,
            "indice_danger":      round(r.indice_danger, 3),
            "score_detection":    round(r.score_detection, 3),
        }
        for i, r in enumerate(resultats)
    ]

    statistiques = {
        "nombre_fissures":    len(resultats),
        "danger_moyen":       round(sum(r.indice_danger for r in resultats) / len(resultats), 3),
        "danger_maximum":     round(max(r.indice_danger for r in resultats), 3),
        "orientation": {
            v.value: sum(1 for r in resultats if r.orientation == v)
            for v in ClassificationOrientation
            if v != ClassificationOrientation.INCONNUE
        },
        "localisation": {
            v.value: sum(1 for r in resultats if r.localisation == v)
            for v in ClassificationLocalisation
            if v != ClassificationLocalisation.INCONNUE
        },
    }

    return {
        "image":       nom_image,
        "fissures":    fissures,
        "statistiques": statistiques,
    }
