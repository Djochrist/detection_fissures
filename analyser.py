"""
Analyse et classification des fissures après entraînement.

Charge un modèle entraîné, effectue l'inférence sur un dossier d'images
et classifie chaque fissure détectée :
    - Orientation  : horizontale / verticale / inclinée
    - Localisation : superficielle / profonde / transversale
    - Danger       : indice composite [0.0 = faible → 1.0 = critique]

═══════════════════════════════════════════════════════════════════════
  COMMANDES EXACTES SELON LE MODÈLE ENTRAÎNÉ
═══════════════════════════════════════════════════════════════════════

  Après entraînement YOLO11-seg
  ─────────────────────────────
    python analyser.py \\
      --modele sorties_yolo/entrainements/yolo11_seg_fissures/weights/best.pt \\
      --backend yolo \\
      --images dataset/test/

    # Avec export JSON des résultats
    python analyser.py \\
      --modele sorties_yolo/entrainements/yolo11_seg_fissures/weights/best.pt \\
      --backend yolo \\
      --images dataset/test/ \\
      --sortie resultats_yolo.json

    # Seuil plus bas pour détecter plus de fissures (éviter les manques)
    python analyser.py \\
      --modele sorties_yolo/entrainements/yolo11_seg_fissures/weights/best.pt \\
      --backend yolo \\
      --images dataset/test/ \\
      --seuil 0.10

  Après entraînement Mask R-CNN
  ──────────────────────────────
    python analyser.py \\
      --modele sorties/modeles/meilleur_modele.pth \\
      --backend maskrcnn \\
      --images dataset/test/

    # Avec export JSON des résultats
    python analyser.py \\
      --modele sorties/modeles/meilleur_modele.pth \\
      --backend maskrcnn \\
      --images dataset/test/ \\
      --sortie resultats_maskrcnn.json

═══════════════════════════════════════════════════════════════════════
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

BACKEND_YOLO     = "yolo"
BACKEND_MASKRCNN = "maskrcnn"

TAILLE_IMAGE_YOLO_DEFAUT    = 640
SEUIL_YOLO_DEFAUT           = 0.25
IOU_YOLO_DEFAUT             = 0.45
MAX_DET_YOLO_DEFAUT         = 300
LOT_YOLO_DEFAUT             = 8

TAILLE_IMAGE_MASKRCNN_DEFAUT = 384
SEUIL_MASKRCNN_DEFAUT        = 0.40


def analyser_arguments() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    configuration = ConfigurationGlobale()
    analyseur = argparse.ArgumentParser(
        description=(
            "Classification des fissures après inférence YOLO11-seg ou Mask R-CNN.\n\n"
            "YOLO  : --modele best.pt     --backend yolo\n"
            "Mask  : --modele best.pth    --backend maskrcnn"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    analyseur.add_argument(
        "--modele",
        type=str,
        required=True,
        help=(
            "Checkpoint entraîné. "
            "YOLO : .pt (ex: sorties_yolo/.../weights/best.pt). "
            "Mask R-CNN : .pth (ex: sorties/modeles/meilleur_modele.pth)"
        ),
    )
    analyseur.add_argument(
        "--backend",
        type=str,
        default=BACKEND_YOLO,
        choices=[BACKEND_YOLO, BACKEND_MASKRCNN],
        help="Moteur d'inférence : 'yolo' pour YOLO11-seg, 'maskrcnn' pour Mask R-CNN",
    )
    analyseur.add_argument(
        "--images",
        type=str,
        required=True,
        help="Dossier contenant les images à analyser (ex: dataset/test/)",
    )
    analyseur.add_argument(
        "--taille-image",
        type=int,
        default=None,
        help=(
            f"Résolution d'entrée. Défaut : {TAILLE_IMAGE_YOLO_DEFAUT}px pour YOLO, "
            f"{TAILLE_IMAGE_MASKRCNN_DEFAUT}px pour Mask R-CNN"
        ),
    )
    analyseur.add_argument(
        "--architecture",
        type=str,
        default=configuration.modele.architecture,
        choices=ARCHITECTURES_MODELES_SEGMENTATION,
        help="Architecture Mask R-CNN (uniquement avec --backend maskrcnn)",
    )
    analyseur.add_argument(
        "--seuil",
        type=float,
        default=None,
        help=(
            f"Score de confiance minimal. "
            f"Défaut : {SEUIL_YOLO_DEFAUT} pour YOLO, {SEUIL_MASKRCNN_DEFAUT} pour Mask R-CNN. "
            "Réduire pour détecter plus de fissures (moins d'omissions)."
        ),
    )
    analyseur.add_argument(
        "--iou-yolo",
        type=float,
        default=IOU_YOLO_DEFAUT,
        help="IoU NMS YOLO : seuil de suppression des doublons [0, 1]",
    )
    analyseur.add_argument(
        "--max-det-yolo",
        type=int,
        default=MAX_DET_YOLO_DEFAUT,
        help="Nombre maximal de fissures détectées par image (YOLO)",
    )
    analyseur.add_argument(
        "--lot-yolo",
        type=int,
        default=LOT_YOLO_DEFAUT,
        help="Taille du batch d'inférence YOLO",
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
        help=(
            "Fichier JSON de sortie pour le rapport complet (optionnel). "
            "Ex : --sortie resultats.json"
        ),
    )
    analyseur.add_argument(
        "--nombre-classes",
        type=int,
        default=configuration.modele.nombre_classes,
        help="Nombre de classes du modèle Mask R-CNN (doit correspondre à l'entraînement)",
    )
    return analyseur.parse_args()


def _backend_depuis_arguments(args: argparse.Namespace) -> str:
    """Valide la cohérence entre le backend choisi et le checkpoint."""
    suffixe = Path(args.modele).suffix.lower()

    if args.backend == BACKEND_YOLO:
        if suffixe != ".pt":
            raise ValueError(
                f"Le backend YOLO attend un fichier .pt, reçu : {args.modele}\n"
                "Utilisez --backend maskrcnn pour un checkpoint .pth."
            )
        return BACKEND_YOLO

    if args.backend == BACKEND_MASKRCNN:
        if suffixe != ".pth":
            raise ValueError(
                f"Le backend Mask R-CNN attend un fichier .pth, reçu : {args.modele}\n"
                "Utilisez --backend yolo pour un checkpoint .pt."
            )
        return BACKEND_MASKRCNN

    raise ValueError(f"Backend non supporté : {args.backend}")


def _taille_image_depuis_backend(backend: str, taille_image: int | None) -> int:
    """Applique la résolution par défaut adaptée au backend."""
    if taille_image is not None:
        return taille_image
    return (
        TAILLE_IMAGE_YOLO_DEFAUT
        if backend == BACKEND_YOLO
        else TAILLE_IMAGE_MASKRCNN_DEFAUT
    )


def _seuil_depuis_backend(backend: str, seuil: float | None) -> float:
    """Applique le seuil de confiance par défaut adapté au backend."""
    if seuil is not None:
        return seuil
    return SEUIL_YOLO_DEFAUT if backend == BACKEND_YOLO else SEUIL_MASKRCNN_DEFAUT


def _device_ultralytics(dispositif: str) -> str | None:
    """Traduit le nom de dispositif vers le format Ultralytics."""
    if dispositif == "auto":
        return None
    if dispositif == "cuda":
        return "0"
    return dispositif


def valider_parametres_inference(
    args: argparse.Namespace,
    backend: str,
    taille_image: int,
) -> None:
    """Bloque les réglages incohérents avant de lancer l'inférence."""
    if backend == BACKEND_YOLO:
        if taille_image % 32 != 0:
            raise ValueError(
                f"--taille-image={taille_image} n'est pas un multiple de 32 "
                "(requis par YOLO). Ex : 640, 672, 704..."
            )
        if args.lot_yolo < 1:
            raise ValueError("--lot-yolo doit être ≥ 1.")
        if args.max_det_yolo < 1:
            raise ValueError("--max-det-yolo doit être ≥ 1.")
        if not 0.0 <= args.iou_yolo <= 1.0:
            raise ValueError("--iou-yolo doit être dans [0, 1].")


def charger_modele_maskrcnn(
    chemin_pth: str | Path,
    nombre_classes: int,
    taille_image: int,
    architecture: str,
    dispositif: object,
) -> object:
    """
    Charge le modèle Mask R-CNN depuis un checkpoint .pth.

    Le checkpoint peut contenir :
      - Format Entraineur : dict avec clé "etat_modele" (standard de ce projet)
      - Format direct     : state_dict des poids uniquement

    Args:
        chemin_pth     : Chemin vers le fichier .pth.
        nombre_classes : Doit correspondre à l'entraînement (par défaut 2).
        taille_image   : Résolution d'entrée (même que l'entraînement).
        architecture   : Architecture Mask R-CNN.
        dispositif     : Dispositif de calcul.

    Returns:
        Modèle en mode évaluation.
    """
    chemin_pth = Path(chemin_pth)
    if not chemin_pth.exists():
        raise FileNotFoundError(
            f"Modèle Mask R-CNN introuvable : {chemin_pth}\n"
            "Vérifiez que l'entraînement s'est terminé correctement."
        )

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

    if isinstance(checkpoint, dict) and "etat_modele" in checkpoint:
        modele.load_state_dict(checkpoint["etat_modele"])
        epoque  = checkpoint.get("epoque", "?")
        map_50  = checkpoint.get("metriques", {}).get("map_50", 0.0)
        print(f"  Checkpoint chargé — époque {epoque}, mAP@0.5 validation = {map_50:.4f}")
    else:
        modele.load_state_dict(checkpoint)
        print("  Poids chargés directement depuis le state_dict.")

    modele.to(dispositif)
    modele.eval()
    return modele


def charger_modele_yolo(chemin_pt: str | Path) -> object:
    """
    Charge un checkpoint YOLO11-seg Ultralytics.

    Args:
        chemin_pt : Chemin vers un fichier .pt entraîné par Ultralytics.

    Returns:
        Modèle YOLO chargé.
    """
    chemin_pt = Path(chemin_pt)
    if not chemin_pt.exists():
        raise FileNotFoundError(
            f"Modèle YOLO introuvable : {chemin_pt}\n"
            "Vérifiez que l'entraînement s'est terminé correctement."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Le paquet 'ultralytics' est requis.\n"
            "Installez-le avec : pip install -U ultralytics"
        ) from exc

    modele = YOLO(str(chemin_pt))
    print(f"  Checkpoint YOLO-seg chargé : {chemin_pt}")
    return modele


def preparer_image(
    chemin_image: str | Path,
    taille_image: int,
    dispositif: object,
) -> tuple[object, int, int]:
    """
    Charge et prépare une image pour l'inférence Mask R-CNN.

    Applique les mêmes transformations que pendant l'entraînement :
        1. Chargement BGR → RGB
        2. Redimensionnement à taille_image × taille_image
        3. Conversion en tenseur float [0, 1]

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


def predire_yolo_batch(
    modele: object,
    chemins_images: list[Path],
    taille_image: int,
    seuil: float,
    dispositif: str,
    iou: float,
    max_det: int,
    lot: int,
) -> list[tuple[Path, list[dict], int, int]]:
    """
    Inférence YOLO-seg en batch avec masques haute résolution.

    retina_masks=True : conserve des masques à la résolution de l'image originale,
    ce qui améliore la précision de l'analyse géométrique (orientation, largeur).
    """
    resultats = modele.predict(
        source=[str(c) for c in chemins_images],
        task="segment",
        imgsz=taille_image,
        conf=seuil,
        iou=iou,
        max_det=max_det,
        batch=lot,
        retina_masks=True,
        device=_device_ultralytics(dispositif),
        verbose=False,
    )

    sorties = []
    for chemin, resultat in zip(chemins_images, resultats):
        hauteur_image, largeur_image = getattr(resultat, "orig_shape", (taille_image, taille_image))
        masques = getattr(resultat, "masks", None)
        boites  = getattr(resultat, "boxes", None)

        if masques is None or getattr(masques, "data", None) is None:
            sorties.append((
                chemin,
                [{"masks": [], "scores": [], "labels": [], "boxes": []}],
                largeur_image,
                hauteur_image,
            ))
            continue

        masques_tensor = masques.data
        if getattr(masques_tensor, "ndim", 0) == 3:
            masques_tensor = masques_tensor.unsqueeze(1)

        hauteur_masque = int(masques_tensor.shape[-2])
        largeur_masque = int(masques_tensor.shape[-1])

        scores    = boites.conf   if boites is not None and getattr(boites, "conf", None)  is not None else None
        labels    = boites.cls    if boites is not None and getattr(boites, "cls",  None)  is not None else None
        boxes_xyxy = boites.xyxy  if boites is not None and getattr(boites, "xyxy", None)  is not None else None

        sorties.append((
            chemin,
            [{"masks": masques_tensor, "scores": scores, "labels": labels, "boxes": boxes_xyxy}],
            largeur_masque,
            hauteur_masque,
        ))

    return sorties


def predire_maskrcnn(
    modele: object,
    chemin_image: Path,
    taille_image: int,
    seuil: float,
    dispositif: object,
) -> tuple[list[dict], int, int]:
    """Inférence Mask R-CNN sur une image avec filtrage par score."""
    tenseur, largeur_image, hauteur_image = preparer_image(
        chemin_image=chemin_image,
        taille_image=taille_image,
        dispositif=dispositif,
    )

    predictions = modele(tenseur)
    predictions_filtrees = []
    for pred in predictions:
        masque_score = pred["scores"] >= seuil
        predictions_filtrees.append({
            "masks":  pred["masks"][masque_score],
            "scores": pred["scores"][masque_score],
            "labels": pred["labels"][masque_score],
            "boxes":  pred["boxes"][masque_score],
        })

    return predictions_filtrees, taille_image, taille_image


def afficher_banniere(
    backend: str,
    chemin_modele: str,
    dossier_images: str,
    taille_image: int,
    seuil: float,
    nb_images: int,
) -> None:
    """Affiche un résumé de la configuration d'analyse."""
    print("\n" + "═" * 65)
    print("  ANALYSE DE FISSURES — SEGMENTATION D'INSTANCE")
    print("═" * 65)
    print(f"  Backend        : {backend.upper()}")
    print(f"  Modèle         : {chemin_modele}")
    print(f"  Images         : {dossier_images}")
    print(f"  Nb images      : {nb_images}")
    print(f"  Résolution     : {taille_image}px")
    print(f"  Seuil confiance: {seuil:.2f}")
    print()
    print("  RÉSULTATS ATTENDUS PAR FISSURE :")
    print("    Orientation   → horizontale / verticale / inclinée")
    print("    Localisation  → superficielle / profonde / transversale")
    print("    Danger [0→1]  → 0.0=faible | 0.5=modéré | 1.0=critique")
    print("═" * 65 + "\n")


def main() -> None:
    """
    Point d'entrée principal du script d'analyse.

    Flux d'exécution :
    1. Chargement du modèle entraîné (YOLO ou Mask R-CNN)
    2. Listage des images dans le dossier cible
    3. Inférence → masques de segmentation
    4. Classification géométrique de chaque masque :
         - PCA → orientation (horizontale / verticale / inclinée)
         - Distance transform → largeur → localisation (superficielle / profonde)
         - Ratio bbox/image → détection transversale
    5. Affichage des résultats image par image
    6. Export JSON du rapport complet (si --sortie spécifié)
    """
    args = analyser_arguments()
    backend = _backend_depuis_arguments(args)
    taille_image = _taille_image_depuis_backend(backend, args.taille_image)
    seuil = _seuil_depuis_backend(backend, args.seuil)
    valider_parametres_inference(args, backend, taille_image)

    import torch
    from detection_fissures.utilitaires.dispositif import detecter_dispositif

    dispositif = detecter_dispositif(
        forcer=None if args.dispositif == "auto" else args.dispositif
    )

    # ── Chargement du modèle ──────────────────────────────────────────────────
    print("\n[Chargement] Chargement du modèle...")
    if backend == BACKEND_YOLO:
        modele = charger_modele_yolo(args.modele)
    else:
        modele = charger_modele_maskrcnn(
            chemin_pth=args.modele,
            nombre_classes=args.nombre_classes,
            taille_image=taille_image,
            architecture=args.architecture,
            dispositif=dispositif,
        )

    # ── Listage des images ────────────────────────────────────────────────────
    dossier_images = Path(args.images)
    if not dossier_images.exists():
        print(f"[Erreur] Dossier introuvable : {dossier_images}")
        sys.exit(1)

    chemins_images = sorted([
        p for p in dossier_images.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS_IMAGES
    ])

    if not chemins_images:
        print(f"[Erreur] Aucune image trouvée dans : {dossier_images}")
        print(f"  Extensions acceptées : {', '.join(sorted(EXTENSIONS_IMAGES))}")
        sys.exit(1)

    afficher_banniere(
        backend=backend,
        chemin_modele=args.modele,
        dossier_images=str(dossier_images),
        taille_image=taille_image,
        seuil=seuil,
        nb_images=len(chemins_images),
    )

    # ── Analyse image par image ───────────────────────────────────────────────
    rapport_global = []
    nb_sans_fissure = 0
    nb_avec_fissure = 0

    with torch.no_grad():

        if backend == BACKEND_YOLO:
            try:
                sorties_predictions = predire_yolo_batch(
                    modele=modele,
                    chemins_images=chemins_images,
                    taille_image=taille_image,
                    seuil=seuil,
                    dispositif=args.dispositif,
                    iou=args.iou_yolo,
                    max_det=args.max_det_yolo,
                    lot=args.lot_yolo,
                )
            except Exception as exc:
                print(f"[Erreur] Inférence YOLO : {exc}")
                sys.exit(1)

        else:
            sorties_predictions = []
            for chemin in chemins_images:
                try:
                    preds, larg, haut = predire_maskrcnn(
                        modele=modele,
                        chemin_image=chemin,
                        taille_image=taille_image,
                        seuil=seuil,
                        dispositif=dispositif,
                    )
                    sorties_predictions.append((chemin, preds, larg, haut))
                except Exception as exc:
                    print(f"[Erreur] {chemin.name} : {exc}")

        for chemin, predictions, largeur_image, hauteur_image in sorties_predictions:
            try:
                resultats = classifier_predictions(
                    predictions=predictions,
                    largeur_image=largeur_image,
                    hauteur_image=hauteur_image,
                    seuil_score_min=0.0,  # Déjà filtré par seuil ci-dessus
                )

                if resultats:
                    nb_avec_fissure += 1
                else:
                    nb_sans_fissure += 1

                afficher_resultats_classification(resultats, nom_image=chemin.name)

                if args.sortie:
                    rapport = generer_rapport_classification(resultats, nom_image=chemin.name)
                    rapport_global.append(rapport)

            except Exception as exc:
                print(f"[Erreur] {chemin.name} : {exc}")

    # ── Résumé global ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RÉSUMÉ DE L'ANALYSE")
    print("═" * 65)
    print(f"  Images analysées       : {len(sorties_predictions)}")
    print(f"  Images avec fissures   : {nb_avec_fissure}")
    print(f"  Images sans fissures   : {nb_sans_fissure}")
    total_fissures = sum(
        r["statistiques"].get("nombre_fissures", 0) for r in rapport_global
    ) if rapport_global else sum(
        1 for _, preds, larg, haut in sorties_predictions
        for pred in preds
        if pred.get("masks") is not None and hasattr(pred["masks"], "__len__") and len(pred["masks"]) > 0
    )
    print(f"  Total fissures détectées: {total_fissures}")
    print("═" * 65)

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
                        "seuil":        seuil,
                        "taille_image": taille_image,
                        "iou_yolo":     args.iou_yolo if backend == BACKEND_YOLO else None,
                        "max_det_yolo": args.max_det_yolo if backend == BACKEND_YOLO else None,
                    },
                    "resultats": rapport_global,
                    "resume": {
                        "images_analysees":  len(rapport_global),
                        "images_avec_fissures": nb_avec_fissure,
                        "images_sans_fissures": nb_sans_fissure,
                        "total_fissures": sum(
                            r["statistiques"].get("nombre_fissures", 0)
                            for r in rapport_global
                        ),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n[Rapport] Rapport JSON sauvegardé : {chemin_sortie}")
        print(f"          Ouvrez-le avec un éditeur ou parsez-le avec Python.")

    print("\n[Analyse] Terminée.\n")


if __name__ == "__main__":
    main()
