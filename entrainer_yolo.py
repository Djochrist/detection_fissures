"""
Entraînement YOLOv11-seg pour la détection de fissures.

Le dataset source reste au format COCO Roboflow. Ce script le convertit en
YOLO segmentation dans le dossier de sortie, puis lance Ultralytics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RACINE_PROJET = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE_PROJET.parent))

from detection_fissures.configuration.parametres import (
    MODELE_YOLOV11_SEG_DEFAUT,
    MODELES_YOLOV11_SEG_OFFICIELS,
    ConfigurationGlobale,
)


def analyser_arguments() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    configuration = ConfigurationGlobale()
    analyseur = argparse.ArgumentParser(
        description="Entraînement YOLOv11-seg — Détection de fissures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyseur.add_argument(
        "--donnees",
        type=str,
        default=str(RACINE_PROJET / configuration.chemins.donnees_racine),
        help="Répertoire COCO racine contenant train/, valid/ et test/",
    )
    analyseur.add_argument(
        "--sorties",
        type=str,
        default="sorties_yolo",
        help="Répertoire de sortie YOLO",
    )
    analyseur.add_argument(
        "--modele",
        type=str,
        default=MODELE_YOLOV11_SEG_DEFAUT,
        help="Poids YOLOv11 segmentation : yolo11n/s/m/l/x-seg.pt",
    )
    analyseur.add_argument("--epoques", type=int, default=100)
    analyseur.add_argument("--lot", type=int, default=8, help="Batch size")
    analyseur.add_argument("--taille-image", type=int, default=384, help="imgsz YOLO")
    analyseur.add_argument("--lr", type=float, default=1e-4, help="lr0 Ultralytics")
    analyseur.add_argument("--patience", type=int, default=20)
    analyseur.add_argument("--workers", type=int, default=2)
    analyseur.add_argument(
        "--dispositif",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Dispositif de calcul",
    )
    analyseur.add_argument(
        "--nom",
        type=str,
        default="yolo11_seg_fissures",
        help="Nom de l'expérience Ultralytics",
    )
    analyseur.add_argument(
        "--copier-images",
        action="store_true",
        help="Copie les images au lieu de créer des liens symboliques",
    )
    analyseur.add_argument(
        "--convertir-seulement",
        action="store_true",
        help="Convertit le dataset COCO vers YOLO sans entraîner",
    )
    analyseur.add_argument(
        "--exist-ok",
        action="store_true",
        help="Autorise Ultralytics à réutiliser un dossier d'expérience existant",
    )
    return analyseur.parse_args()


def _device_ultralytics(dispositif: str) -> str | None:
    """Traduit le nom de dispositif du projet vers Ultralytics."""
    if dispositif == "auto":
        return None
    if dispositif == "cuda":
        return "0"
    return dispositif


def verifier_modele_yolov11_seg(chemin_modele: str) -> None:
    """Bloque les poids YOLO qui ne sont pas des modèles YOLOv11-seg."""
    nom_modele = Path(chemin_modele).name
    if nom_modele in MODELES_YOLOV11_SEG_OFFICIELS:
        return

    if nom_modele.startswith("yolo11") and "-seg" in Path(nom_modele).stem:
        return

    choix = ", ".join(MODELES_YOLOV11_SEG_OFFICIELS)
    raise ValueError(
        f"Modèle YOLO non autorisé : {chemin_modele}. "
        f"Ce projet accepte uniquement YOLOv11-seg ({choix}) "
        "ou un checkpoint YOLOv11-seg nommé explicitement."
    )


def main() -> None:
    """Convertit le dataset puis entraîne YOLO11-seg."""
    args = analyser_arguments()
    verifier_modele_yolov11_seg(args.modele)
    racine_sorties = Path(args.sorties).expanduser().resolve()
    racine_dataset_yolo = racine_sorties / "dataset_yolo"

    from detection_fissures.donnees.conversion_yolo import convertir_dataset_coco_vers_yolo

    chemin_yaml = convertir_dataset_coco_vers_yolo(
        racine_coco=args.donnees,
        racine_yolo=racine_dataset_yolo,
        copier_images=args.copier_images,
    )
    print(f"[YOLO] Dataset converti : {chemin_yaml}")

    if args.convertir_seulement:
        print("[YOLO] Conversion terminée, entraînement ignoré.")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Le paquet 'ultralytics' est requis pour YOLOv11. "
            "Installez-le avec : pip install -U ultralytics"
        ) from exc

    modele = YOLO(args.modele)
    resultats = modele.train(
        data=str(chemin_yaml),
        task="segment",
        epochs=args.epoques,
        imgsz=args.taille_image,
        batch=args.lot,
        lr0=args.lr,
        patience=args.patience,
        device=_device_ultralytics(args.dispositif),
        workers=args.workers,
        project=str(racine_sorties / "entrainements"),
        name=args.nom,
        exist_ok=args.exist_ok,
    )

    print(f"[YOLO] Entraînement terminé : {resultats.save_dir}")

    metriques_test = modele.val(
        data=str(chemin_yaml),
        split="test",
        imgsz=args.taille_image,
        batch=1,
        device=_device_ultralytics(args.dispositif),
        project=str(racine_sorties / "evaluations"),
        name=f"{args.nom}_test",
        exist_ok=True,
    )
    print(f"[YOLO] Évaluation test terminée : {metriques_test.save_dir}")


if __name__ == "__main__":
    main()
