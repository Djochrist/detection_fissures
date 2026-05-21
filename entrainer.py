"""
Script principal d'entraînement du modèle de détection de fissures.

Usage :
    python entrainer.py
    python entrainer.py --epoques 100 --lot 8 --lr 5e-5
    python entrainer.py --donnees /chemin/vers/dataset

Structure attendue du dataset :
    dataset/
        train/
            _annotations.coco.json
            *.jpg
        valid/
            _annotations.coco.json
            *.jpg
        test/
            _annotations.coco.json
            *.jpg

Environnements supportés :
    Local  : python entrainer.py
    Colab  : !python entrainer.py
    Kaggle : python entrainer.py
"""

import argparse
import json
import sys
from pathlib import Path

RACINE_PROJET = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE_PROJET.parent))

from detection_fissures.configuration.parametres import (
    ARCHITECTURES_MODELES_SEGMENTATION,
    NOM_FICHIER_ANNOTATIONS_COCO,
    NOM_HISTORIQUE_ENTRAINEMENT,
    SPLITS_DATASET,
    ConfigurationGlobale,
)


def analyser_arguments() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    configuration = ConfigurationGlobale()
    analyseur = argparse.ArgumentParser(
        description="Entraînement Mask R-CNN — Détection de fissures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    analyseur.add_argument(
        "--donnees",
        type=str,
        default=str(RACINE_PROJET / configuration.chemins.donnees_racine),
        help="Répertoire racine du dataset (contient train/, valid/, test/)",
    )
    analyseur.add_argument(
        "--epoques",
        type=int,
        default=configuration.entrainement.nombre_epoques,
        help="Nombre maximum d'époques",
    )
    analyseur.add_argument(
        "--lot",
        type=int,
        default=configuration.entrainement.taille_lot,
        help="Taille du lot (4 pour GPU 8Go, 8 pour GPU 16Go+)",
    )
    analyseur.add_argument(
        "--lr",
        type=float,
        default=configuration.entrainement.taux_apprentissage,
        help="Taux d'apprentissage initial",
    )
    analyseur.add_argument(
        "--patience",
        type=int,
        default=configuration.entrainement.patience_arret_precoce,
        help="Patience de l'arrêt anticipé (early stopping)",
    )
    analyseur.add_argument(
        "--taille-image",
        type=int,
        default=configuration.modele.taille_image_min,
        help="Résolution des images (carré, en pixels)",
    )
    analyseur.add_argument(
        "--architecture",
        type=str,
        default=configuration.modele.architecture,
        choices=ARCHITECTURES_MODELES_SEGMENTATION,
        help="Architecture Mask R-CNN autorisée pour ce projet",
    )
    analyseur.add_argument(
        "--dispositif",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Dispositif de calcul",
    )
    analyseur.add_argument(
        "--graine",
        type=int,
        default=configuration.entrainement.graine_aleatoire,
        help="Graine aléatoire pour la reproductibilité",
    )
    analyseur.add_argument(
        "--sorties",
        type=str,
        default="sorties",
        help="Répertoire de sortie pour les modèles et journaux",
    )
    analyseur.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Chemin vers un checkpoint .pth pour reprendre l'entraînement",
    )
    analyseur.add_argument(
        "--sans-mixte",
        action="store_true",
        default=False,
        help="Désactiver la précision mixte (float16)",
    )

    return analyseur.parse_args()


def verifier_dataset(chemin_donnees: Path) -> None:
    """
    Vérifie la structure COCO attendue avant de démarrer l'entraînement.

    La validation reste volontairement légère : elle confirme les dossiers,
    les fichiers `_annotations.coco.json` et les images référencées par COCO.
    """
    erreurs = []

    if not chemin_donnees.is_dir():
        raise FileNotFoundError(
            f"Dataset introuvable : {chemin_donnees}\n"
            "Indiquez un dossier contenant train/, valid/ et test/ avec --donnees."
        )

    for split in SPLITS_DATASET:
        dossier_split = chemin_donnees / split
        chemin_annotations = dossier_split / NOM_FICHIER_ANNOTATIONS_COCO

        if not dossier_split.is_dir():
            erreurs.append(f"- Dossier manquant : {dossier_split}")
            continue

        if not chemin_annotations.is_file():
            erreurs.append(f"- Annotations manquantes : {chemin_annotations}")
            continue

        try:
            donnees_coco = json.loads(chemin_annotations.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            erreurs.append(f"- JSON COCO invalide : {chemin_annotations} ({exc})")
            continue

        images = donnees_coco.get("images")
        annotations = donnees_coco.get("annotations")
        categories = donnees_coco.get("categories")
        if not all(
            isinstance(element, list)
            for element in (images, annotations, categories)
        ):
            erreurs.append(
                f"- Structure COCO invalide : {chemin_annotations} "
                "(clés attendues : images, annotations, categories)"
            )
            continue

        images_manquantes = [
            image.get("file_name", "<file_name absent>")
            for image in images
            if not (dossier_split / str(image.get("file_name", ""))).is_file()
        ]
        if images_manquantes:
            exemples = ", ".join(images_manquantes[:5])
            suffixe = "..." if len(images_manquantes) > 5 else ""
            erreurs.append(
                f"- {len(images_manquantes)} image(s) référencée(s) introuvable(s) "
                f"dans {dossier_split} : {exemples}{suffixe}"
            )

    if erreurs:
        details = "\n".join(erreurs)
        raise FileNotFoundError(f"Dataset COCO incomplet ou invalide :\n{details}")


def main() -> None:
    """
    Point d'entrée principal de l'entraînement.

    Étapes :
    1. Configuration et reproductibilité
    2. Chargement du dataset
    3. Construction du modèle
    4. Entraînement
    5. Évaluation finale sur le jeu de test
    6. Sauvegarde de l'historique
    """
    args = analyser_arguments()

    chemin_donnees = Path(args.donnees).expanduser().resolve()
    verifier_dataset(chemin_donnees)

    import torch

    from detection_fissures.donnees.chargeur import creer_chargeurs_donnees
    from detection_fissures.entrainement.entraineur import Entraineur
    from detection_fissures.entrainement.metriques import (
        calculer_metriques_segmentation,
        afficher_tableau_metriques,
    )
    from detection_fissures.modeles.masque_rcnn import construire_modele_masque_rcnn
    from detection_fissures.utilitaires.dispositif import (
        detecter_dispositif,
        afficher_info_dispositif,
    )
    from detection_fissures.utilitaires.graine import fixer_graine_aleatoire

    # ── 1. Configuration ──────────────────────────────────────────────────────
    config = ConfigurationGlobale()

    config.entrainement.nombre_epoques = args.epoques
    config.entrainement.taille_lot = args.lot
    config.entrainement.taux_apprentissage = args.lr
    config.entrainement.patience_arret_precoce = args.patience
    config.entrainement.precision_mixte = not args.sans_mixte
    config.modele.taille_image_min = args.taille_image
    config.modele.taille_image_max = args.taille_image
    config.modele.architecture = args.architecture

    config.chemins.definir_racine_donnees(chemin_donnees)
    config.chemins.definir_racine_sorties(args.sorties)
    config.chemins.creer_dossiers()

    # ── 2. Reproductibilité ────────────────────────────────────────────────────
    fixer_graine_aleatoire(args.graine)

    # ── 3. Dispositif ─────────────────────────────────────────────────────────
    dispositif = detecter_dispositif(
        forcer=None if args.dispositif == "auto" else args.dispositif
    )
    afficher_info_dispositif(dispositif)

    # ── 4. Chargement des données ─────────────────────────────────────────────
    chargeurs = creer_chargeurs_donnees(
        chemin_train=config.chemins.chemin_entrainement,
        chemin_valid=config.chemins.chemin_validation,
        chemin_test=config.chemins.chemin_test,
        annotations_train=config.chemins.annotations_entrainement,
        annotations_valid=config.chemins.annotations_validation,
        annotations_test=config.chemins.annotations_test,
        taille_lot=config.entrainement.taille_lot,
        taille_image=config.modele.taille_image_min,
        nombre_workers=config.entrainement.nombre_workers,
        epingler_memoire=config.entrainement.epingler_memoire and dispositif.type == "cuda",
    )

    # ── 5. Construction du modèle ─────────────────────────────────────────────
    modele = construire_modele_masque_rcnn(
        nombre_classes=config.modele.nombre_classes,
        architecture=config.modele.architecture,
        seuil_score_detection=config.modele.seuil_score_detection,
        seuil_iou_nms=config.modele.seuil_iou_nms,
        detections_max_par_image=config.modele.detections_max_par_image,
        taille_image_min=config.modele.taille_image_min,
        taille_image_max=config.modele.taille_image_max,
    )

    # ── 6. Entraînement ───────────────────────────────────────────────────────
    entraineur = Entraineur(
        modele=modele,
        chargeur_train=chargeurs["entrainement"],
        chargeur_valid=chargeurs["validation"],
        dispositif=dispositif,
        taux_apprentissage=config.entrainement.taux_apprentissage,
        decroissance_poids=config.entrainement.decroissance_poids,
        nombre_epoques=config.entrainement.nombre_epoques,
        epoque_degelage_backbone=config.entrainement.epoque_degelage_backbone,
        epoque_degelage_complet=config.entrainement.epoque_degelage_complet,
        patience_arret_precoce=config.entrainement.patience_arret_precoce,
        valeur_clip_gradient=config.entrainement.valeur_clip_gradient,
        dossier_sorties=config.chemins.dossier_modeles,
        precision_mixte=config.entrainement.precision_mixte,
        frequence_affichage=config.entrainement.frequence_affichage,
    )

    if args.resume:
        checkpoint_resume = Path(args.resume).expanduser().resolve()
        if not checkpoint_resume.is_file():
            raise FileNotFoundError(f"Checkpoint de reprise introuvable : {checkpoint_resume}")
        print(f"[Reprise] Chargement du checkpoint : {checkpoint_resume}")
        checkpoint = torch.load(checkpoint_resume, map_location=dispositif)
        entraineur.reprendre_checkpoint(checkpoint)

    try:
        historique = entraineur.entrainer()
    except KeyboardInterrupt:
        print("\n[Interruption] Entraînement interrompu. Retour à la case départ.")
        entraineur.nettoyer_sorties_interrompues()
        print("Relancez le script pour recommencer l'entraînement depuis zéro.")
        return

    # ── 7. Évaluation finale sur le jeu de test ───────────────────────────────
    print("\n[Évaluation] Chargement du meilleur modèle...")
    entraineur.charger_meilleur_modele()

    modele.eval()
    toutes_predictions = []
    toutes_cibles = []

    with torch.no_grad():
        for images, cibles in chargeurs["test"]:
            images_gpu = [img.to(dispositif) for img in images]
            predictions = modele(images_gpu)
            toutes_predictions.extend([{k: v.cpu() for k, v in p.items()} for p in predictions])
            toutes_cibles.extend(cibles)

    metriques_test = calculer_metriques_segmentation(toutes_predictions, toutes_cibles)
    print("\n[RÉSULTATS FINAUX — JEU DE TEST]")
    afficher_tableau_metriques(metriques_test)

    # ── 8. Sauvegarde de l'historique ─────────────────────────────────────────
    chemin_historique = config.chemins.dossier_journaux / NOM_HISTORIQUE_ENTRAINEMENT
    with open(chemin_historique, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hyperparametres": {
                    "epoques": args.epoques,
                    "taille_lot": args.lot,
                    "taux_apprentissage": args.lr,
                    "patience": args.patience,
                    "taille_image": args.taille_image,
                    "architecture": args.architecture,
                    "graine": args.graine,
                },
                "metriques_test": metriques_test,
                "historique": historique,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[Historique] Sauvegardé : {chemin_historique}")
    print("\nEntraînement terminé avec succès.")


if __name__ == "__main__":
    main()
