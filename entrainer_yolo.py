"""
Entraînement YOLOv11-seg pour la détection de fissures.

Le dataset source reste au format COCO Roboflow. Ce script le convertit en
YOLO segmentation dans le dossier de sortie, puis lance Ultralytics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
        "--save-period",
        type=int,
        default=5,
        help="Sauvegarde un checkpoint YOLO toutes les N époques (-1 désactive)",
    )
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
    analyseur.add_argument(
        "--silencieux",
        action="store_true",
        help="Réduit les journaux YOLO au minimum",
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


def _compter_lignes_labels(dossier_labels: Path) -> tuple[int, int]:
    """Retourne (images annotees, instances) pour un dossier labels YOLO."""
    fichiers = sorted(dossier_labels.glob("*.txt"))
    images_annotees = 0
    instances = 0
    for fichier in fichiers:
        lignes = [
            ligne
            for ligne in fichier.read_text(encoding="utf-8").splitlines()
            if ligne.strip()
        ]
        if lignes:
            images_annotees += 1
            instances += len(lignes)
    return images_annotees, instances


def afficher_resume_dataset_yolo(racine_dataset_yolo: Path) -> None:
    """Affiche un résumé compact du dataset YOLO converti."""
    print("\n" + "═" * 55)
    print("  DATASET YOLO-SEG CONVERTI")
    print("═" * 55)
    for split in ("train", "valid", "test"):
        dossier_images = racine_dataset_yolo / "images" / split
        dossier_labels = racine_dataset_yolo / "labels" / split
        nb_images = sum(1 for chemin in dossier_images.iterdir() if chemin.is_file() or chemin.is_symlink())
        images_annotees, instances = _compter_lignes_labels(dossier_labels)
        print(
            f"  {split:<5} : {nb_images:5d} image(s) | "
            f"{images_annotees:5d} avec fissure(s) | {instances:6d} instance(s)"
        )
    print("═" * 55 + "\n")


def _valeur_metrique_yolo(resultats: Any, fragments_cles: tuple[str, ...]) -> float | None:
    """Récupère une métrique Ultralytics depuis results_dict si disponible."""
    dictionnaire = getattr(resultats, "results_dict", None)
    if not isinstance(dictionnaire, dict):
        return None

    for cle, valeur in dictionnaire.items():
        cle_min = str(cle).lower()
        if all(fragment.lower() in cle_min for fragment in fragments_cles):
            try:
                return float(valeur)
            except (TypeError, ValueError):
                return None
    return None


def _calculer_f1(precision: float | None, rappel: float | None) -> float | None:
    """Calcule le F1 score depuis précision et rappel si les deux existent."""
    if precision is None or rappel is None:
        return None
    denominateur = precision + rappel
    if denominateur <= 0:
        return 0.0
    return 2.0 * precision * rappel / denominateur


def afficher_metriques_yolo(resultats: Any, titre: str) -> None:
    """Affiche les métriques YOLO principales quand Ultralytics les expose."""
    precision_masque = _valeur_metrique_yolo(resultats, ("precision", "(m)"))
    rappel_masque = _valeur_metrique_yolo(resultats, ("recall", "(m)"))
    precision_boite = _valeur_metrique_yolo(resultats, ("precision", "(b)"))
    rappel_boite = _valeur_metrique_yolo(resultats, ("recall", "(b)"))

    valeurs = [
        ("mAP@0.5 masque", _valeur_metrique_yolo(resultats, ("map50", "(m)"))),
        ("mAP@0.5:0.95 masque", _valeur_metrique_yolo(resultats, ("map50-95", "(m)"))),
        ("Précision masque", precision_masque),
        ("Rappel masque", rappel_masque),
        ("F1 score masque", _calculer_f1(precision_masque, rappel_masque)),
        ("mAP@0.5 boîte", _valeur_metrique_yolo(resultats, ("map50", "(b)"))),
        ("mAP@0.5:0.95 boîte", _valeur_metrique_yolo(resultats, ("map50-95", "(b)"))),
        ("Précision boîte", precision_boite),
        ("Rappel boîte", rappel_boite),
        ("F1 score boîte", _calculer_f1(precision_boite, rappel_boite)),
    ]
    valeurs = [(label, valeur) for label, valeur in valeurs if valeur is not None]

    if not valeurs:
        print(f"[YOLO] {titre} : métriques détaillées non exposées par cette version.")
        return

    print("\n" + "═" * 55)
    print(f"  {titre}")
    print("═" * 55)
    for label, valeur in valeurs:
        barre = "█" * int(max(0.0, min(1.0, valeur)) * 20)
        print(f"  {label:<24} : {valeur:.4f}  {barre}")
    print("═" * 55 + "\n")


def installer_journaux_yolo(modele: Any, actif: bool = True) -> None:
    """Ajoute un callback léger pour éviter les entraînements YOLO muets."""
    if not actif or not hasattr(modele, "add_callback"):
        return

    def journaliser_epoque(trainer: Any) -> None:
        epoque = int(getattr(trainer, "epoch", -1)) + 1
        args = getattr(trainer, "args", None)
        total = getattr(args, "epochs", "?")
        metriques = getattr(trainer, "metrics", None)
        elements = [f"[YOLO] Époque {epoque}/{total}"]

        if isinstance(metriques, dict) and metriques:
            for cle, valeur in metriques.items():
                cle_min = str(cle).lower()
                if "map50" in cle_min or "precision" in cle_min or "recall" in cle_min:
                    try:
                        elements.append(f"{cle}={float(valeur):.4f}")
                    except (TypeError, ValueError):
                        pass

        loss_items = getattr(trainer, "loss_items", None)
        if loss_items is not None:
            try:
                valeurs = [float(valeur) for valeur in loss_items]
                elements.append("loss=" + ",".join(f"{valeur:.4f}" for valeur in valeurs))
            except (TypeError, ValueError):
                pass

        print(" | ".join(elements), flush=True)

    try:
        modele.add_callback("on_fit_epoch_end", journaliser_epoque)
    except Exception as exc:
        print(
            f"[YOLO] Callback de journalisation non installé "
            f"({type(exc).__name__}: {exc}). Ultralytics gardera ses logs natifs."
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
    afficher_resume_dataset_yolo(racine_dataset_yolo)

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

    print("\n" + "═" * 55)
    print("  CONFIGURATION YOLOV11-SEG")
    print("═" * 55)
    print(f"  Modèle      : {args.modele}")
    print(f"  Époques     : {args.epoques}")
    print(f"  Batch       : {args.lot}")
    print(f"  Image       : {args.taille_image}")
    print(f"  LR initial  : {args.lr}")
    print(f"  Patience    : {args.patience}")
    print(f"  Workers     : {args.workers}")
    print(f"  Dispositif  : {args.dispositif}")
    print("═" * 55 + "\n")

    modele = YOLO(args.modele)
    installer_journaux_yolo(modele, actif=not args.silencieux)
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
        verbose=not args.silencieux,
        plots=True,
        val=True,
        save_period=args.save_period,
    )

    print(f"[YOLO] Entraînement terminé : {resultats.save_dir}")
    afficher_metriques_yolo(resultats, "MÉTRIQUES YOLO — VALIDATION")

    metriques_test = modele.val(
        data=str(chemin_yaml),
        split="test",
        imgsz=args.taille_image,
        batch=1,
        device=_device_ultralytics(args.dispositif),
        project=str(racine_sorties / "evaluations"),
        name=f"{args.nom}_test",
        exist_ok=True,
        verbose=not args.silencieux,
        plots=True,
    )
    print(f"[YOLO] Évaluation test terminée : {metriques_test.save_dir}")
    afficher_metriques_yolo(metriques_test, "MÉTRIQUES YOLO — TEST")


if __name__ == "__main__":
    main()
