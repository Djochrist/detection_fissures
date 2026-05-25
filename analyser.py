"""
Script d'analyse et classification des fissures après entraînement.

Charge un modèle entraîné, effectue l'inférence sur un dossier d'images
et classifie chaque fissure détectée selon :
    - Son orientation  : horizontale / verticale / inclinée
    - Sa localisation  : superficielle / profonde / transversale

Usage :
    python analyser.py --modele sorties_yolo/entrainements/yolo_seg_fissures/weights/best.pt --images chemin/images/
    python analyser.py --modele sorties/modeles/meilleur_modele.pth --backend maskrcnn --images photos/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection_fissures.analyse.classificateur_fissures import (
    classifier_predictions,
    afficher_resultats_classification,
    generer_rapport_classification,
)
from detection_fissures.configuration.parametres import (
    ARCHITECTURES_MODELES_SEGMENTATION,
    ConfigurationGlobale,
)

EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def analyser_arguments() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    configuration = ConfigurationGlobale()
    analyseur = argparse.ArgumentParser(
        description="Classification des fissures après inférence YOLO-seg ou Mask R-CNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyseur.add_argument(
        "--modele",
        type=str,
        required=True,
        help="Chemin vers le checkpoint entraîné (.pt YOLO-seg ou .pth Mask R-CNN)",
    )
    analyseur.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "yolo", "maskrcnn"],
        help="Moteur d'inférence. auto utilise YOLO pour .pt et Mask R-CNN pour .pth",
    )
    analyseur.add_argument(
        "--images",
        type=str,
        required=True,
        help="Dossier contenant les images à analyser",
    )
    analyseur.add_argument(
        "--taille-image",
        type=int,
        default=configuration.modele.taille_image_min,
        help="Résolution d'entrée du modèle (doit correspondre à l'entraînement)",
    )
    analyseur.add_argument(
        "--architecture",
        type=str,
        default=configuration.modele.architecture,
        choices=ARCHITECTURES_MODELES_SEGMENTATION,
        help="Architecture Mask R-CNN utilisée pour reconstruire le modèle .pth",
    )
    analyseur.add_argument(
        "--seuil",
        type=float,
        default=0.4,
        help="Score de confiance minimal pour conserver une détection [0, 1]",
    )
    analyseur.add_argument(
        "--dispositif",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Dispositif de calcul",
    )
    analyseur.add_argument(
        "--sortie",
        type=str,
        default="",
        help="Fichier JSON de sortie (optionnel). Ex : resultats.json",
    )
    analyseur.add_argument(
        "--nombre-classes",
        type=int,
        default=configuration.modele.nombre_classes,
        help="Nombre de classes du modèle (doit correspondre à l'entraînement)",
    )
    return analyseur.parse_args()


def charger_modele(
    chemin_pth: str | Path,
    nombre_classes: int,
    taille_image: int,
    architecture: str,
    dispositif: object,
) -> object:
    """
    Charge le modèle Mask R-CNN depuis un checkpoint .pth.

    Args:
        chemin_pth    : Chemin vers le fichier checkpoint.
        nombre_classes : Nombre de classes (doit correspondre au modèle sauvegardé).
        taille_image  : Taille d'image utilisée à l'entraînement.
        architecture  : Architecture Mask R-CNN utilisée.
        dispositif    : Dispositif de calcul.

    Returns:
        Modèle chargé en mode évaluation.
    """
    chemin_pth = Path(chemin_pth)
    if not chemin_pth.exists():
        raise FileNotFoundError(f"Modèle introuvable : {chemin_pth}")

    import torch

    from detection_fissures.modeles.masque_rcnn import construire_modele_masque_rcnn

    checkpoint = torch.load(chemin_pth, map_location=dispositif, weights_only=False)
    architecture_checkpoint = (
        checkpoint.get("architecture_modele", architecture)
        if isinstance(checkpoint, dict)
        else architecture
    ) or architecture

    modele = construire_modele_masque_rcnn(
        nombre_classes=nombre_classes,
        architecture=architecture_checkpoint,
        taille_image_min=taille_image,
        taille_image_max=taille_image,
    )

    # Le checkpoint peut contenir "etat_modele" (format Entraineur) ou les poids directs
    if isinstance(checkpoint, dict) and "etat_modele" in checkpoint:
        modele.load_state_dict(checkpoint["etat_modele"])
        epoque = checkpoint.get("epoque", "?")
        map_50 = checkpoint.get("metriques", {}).get("map_50", 0.0)
        print(f"[Modele] Checkpoint chargé — époque {epoque}, mAP@0.5 = {map_50:.4f}")
    else:
        modele.load_state_dict(checkpoint)
        print("[Modele] Poids chargés directement.")

    modele.to(dispositif)
    modele.eval()
    return modele


def charger_modele_yolo(chemin_pt: str | Path) -> object:
    """
    Charge un checkpoint YOLO-seg Ultralytics.

    Args:
        chemin_pt : Chemin vers un fichier .pt entraîné par Ultralytics.

    Returns:
        Modèle YOLO chargé.
    """
    chemin_pt = Path(chemin_pt)
    if not chemin_pt.exists():
        raise FileNotFoundError(f"Modèle YOLO introuvable : {chemin_pt}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Le paquet 'ultralytics>=8.4.0' est requis pour analyser avec YOLO-seg. "
            "Installez-le avec : pip install -U ultralytics"
        ) from exc

    modele = YOLO(str(chemin_pt))
    print(f"[Modele] Checkpoint YOLO-seg chargé : {chemin_pt}")
    return modele


def _backend_depuis_arguments(args: argparse.Namespace) -> str:
    """Résout le backend d'inférence depuis --backend et l'extension du checkpoint."""
    if args.backend != "auto":
        return args.backend

    suffixe = Path(args.modele).suffix.lower()
    if suffixe == ".pt":
        return "yolo"
    if suffixe == ".pth":
        return "maskrcnn"

    raise ValueError(
        "Impossible de détecter le backend automatiquement. "
        "Utilisez --backend yolo ou --backend maskrcnn."
    )


def _device_ultralytics(dispositif: str) -> str | None:
    """Traduit le nom de dispositif du projet vers Ultralytics."""
    if dispositif == "auto":
        return None
    if dispositif == "cuda":
        return "0"
    return dispositif


def predire_yolo(
    modele: object,
    chemin_image: str | Path,
    taille_image: int,
    seuil: float,
    dispositif: str,
) -> tuple[list[dict], int, int]:
    """
    Lance YOLO-seg sur une image et convertit la sortie au format du classificateur.

    Le classificateur existant attend des dictionnaires proches de torchvision :
    masks, scores, labels, boxes. Cette fonction fait le pont depuis Ultralytics.
    """
    resultats = modele.predict(
        source=str(chemin_image),
        task="segment",
        imgsz=taille_image,
        conf=seuil,
        device=_device_ultralytics(dispositif),
        verbose=False,
    )

    if not resultats:
        return [], taille_image, taille_image

    resultat = resultats[0]
    hauteur_image, largeur_image = getattr(resultat, "orig_shape", (taille_image, taille_image))
    masques = getattr(resultat, "masks", None)
    boites = getattr(resultat, "boxes", None)

    if masques is None or getattr(masques, "data", None) is None:
        return [{"masks": [], "scores": [], "labels": [], "boxes": []}], largeur_image, hauteur_image

    masques_tensor = masques.data
    if getattr(masques_tensor, "ndim", 0) == 3:
        masques_tensor = masques_tensor.unsqueeze(1)
    hauteur_masque = int(masques_tensor.shape[-2])
    largeur_masque = int(masques_tensor.shape[-1])

    scores = boites.conf if boites is not None and getattr(boites, "conf", None) is not None else None
    labels = boites.cls if boites is not None and getattr(boites, "cls", None) is not None else None
    boxes_xyxy = boites.xyxy if boites is not None and getattr(boites, "xyxy", None) is not None else None

    return [
        {
            "masks": masques_tensor,
            "scores": scores,
            "labels": labels,
            "boxes": boxes_xyxy,
        }
    ], largeur_masque, hauteur_masque


def preparer_image(
    chemin_image: str | Path,
    taille_image: int,
    dispositif: object,
) -> tuple[object, int, int]:
    """
    Charge et prépare une image pour l'inférence.

    Applique les mêmes transformations que le Dataset d'entraînement :
        1. Chargement BGR → RGB
        2. Redimensionnement à taille_image × taille_image
        3. Conversion en tenseur float [0, 1]

    Args:
        chemin_image : Chemin de l'image.
        taille_image : Résolution cible.
        dispositif   : Dispositif de calcul.

    Returns:
        Tuple (tenseur image, largeur_originale, hauteur_originale).
    """
    from detection_fissures.utilitaires.images import (
        charger_image_rgb,
        image_rgb_vers_tenseur,
        redimensionner_image_carree,
    )

    image_rgb, largeur_orig, hauteur_orig = charger_image_rgb(chemin_image)
    image_redim = redimensionner_image_carree(image_rgb, taille_image)
    tenseur = image_rgb_vers_tenseur(image_redim)
    tenseur = tenseur.unsqueeze(0).to(dispositif)

    return tenseur, largeur_orig, hauteur_orig


def main() -> None:
    """
    Point d'entrée principal du script d'analyse.

    Flux d'exécution :
    1. Chargement du modèle entraîné
    2. Listage des images dans le dossier cible
    3. Pour chaque image : inférence → classification → affichage
    4. Export JSON du rapport complet (si --sortie spécifié)
    """
    args = analyser_arguments()
    backend = _backend_depuis_arguments(args)

    import torch

    from detection_fissures.utilitaires.dispositif import detecter_dispositif

    # ── Dispositif ────────────────────────────────────────────────────────────
    dispositif = detecter_dispositif(
        forcer=None if args.dispositif == "auto" else args.dispositif
    )
    print(f"[Dispositif] {dispositif}")
    print(f"[Backend] {backend}")

    # ── Chargement du modèle ─────────────────────────────────────────────────
    if backend == "yolo":
        modele = charger_modele_yolo(args.modele)
    else:
        modele = charger_modele(
            chemin_pth=args.modele,
            nombre_classes=args.nombre_classes,
            taille_image=args.taille_image,
            architecture=args.architecture,
            dispositif=dispositif,
        )

    # ── Listage des images ────────────────────────────────────────────────────
    dossier_images = Path(args.images)
    if not dossier_images.exists():
        print(f"[Erreur] Dossier introuvable : {dossier_images}")
        sys.exit(1)

    chemins_images = sorted([
        p for p in dossier_images.iterdir()
        if p.suffix.lower() in EXTENSIONS_IMAGES
    ])

    if not chemins_images:
        print(f"[Erreur] Aucune image trouvée dans : {dossier_images}")
        sys.exit(1)

    print(f"\n[Analyse] {len(chemins_images)} images à traiter...")

    # ── Analyse image par image ───────────────────────────────────────────────
    rapport_global = []

    with torch.no_grad():
        for chemin in chemins_images:
            try:
                if backend == "yolo":
                    predictions_filtrees, largeur_image, hauteur_image = predire_yolo(
                        modele=modele,
                        chemin_image=chemin,
                        taille_image=args.taille_image,
                        seuil=args.seuil,
                        dispositif=args.dispositif,
                    )
                else:
                    tenseur, largeur_image, hauteur_image = preparer_image(
                        chemin_image=chemin,
                        taille_image=args.taille_image,
                        dispositif=dispositif,
                    )

                    predictions = modele(tenseur)

                    # Filtrer par seuil de confiance
                    predictions_filtrees = []
                    for pred in predictions:
                        masque_score = pred["scores"] >= args.seuil
                        predictions_filtrees.append({
                            "masks":  pred["masks"][masque_score],
                            "scores": pred["scores"][masque_score],
                            "labels": pred["labels"][masque_score],
                            "boxes":  pred["boxes"][masque_score],
                        })

                    largeur_image = args.taille_image
                    hauteur_image = args.taille_image

                # Classification
                resultats = classifier_predictions(
                    predictions=predictions_filtrees,
                    largeur_image=largeur_image,
                    hauteur_image=hauteur_image,
                    seuil_score_min=0.0,   # Déjà filtré ci-dessus
                )

                # Affichage
                afficher_resultats_classification(resultats, nom_image=chemin.name)

                # Rapport JSON
                if args.sortie:
                    rapport = generer_rapport_classification(resultats, nom_image=chemin.name)
                    rapport_global.append(rapport)

            except Exception as e:
                print(f"[Erreur] {chemin.name} : {e}")
                continue

    # ── Export JSON ───────────────────────────────────────────────────────────
    if args.sortie:
        chemin_sortie = Path(args.sortie)
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

        with open(chemin_sortie, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "parametres": {
                        "modele":       args.modele,
                        "backend":      backend,
                        "seuil":        args.seuil,
                        "taille_image": args.taille_image,
                    },
                    "resultats": rapport_global,
                    "resume": {
                        "images_analysees": len(rapport_global),
                        "total_fissures":   sum(r["statistiques"].get("nombre_fissures", 0) for r in rapport_global),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n[Rapport] Sauvegardé : {chemin_sortie}")

    print("\nAnalyse terminée.")


if __name__ == "__main__":
    main()
