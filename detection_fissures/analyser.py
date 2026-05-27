"""
Script d'analyse et classification des fissures après entraînement.

Usage :
    python -m detection_fissures.analyser --modele modele.pth --images chemin/images/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        description="Classification des fissures après inférence Mask R-CNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyseur.add_argument(
        "--modele",
        type=str,
        required=True,
        help="Chemin vers le fichier .pth du modèle entraîné",
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
        help="Architecture Mask R-CNN utilisée pour reconstruire le modèle",
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
    """Charge le modèle Mask R-CNN depuis un checkpoint .pth."""
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


def preparer_image(
    chemin_image: str | Path,
    taille_image: int,
    dispositif: object,
) -> tuple[object, int, int]:
    """Charge et prépare une image pour l'inférence."""
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
    """Point d'entrée principal du script d'analyse."""
    args = analyser_arguments()

    import torch

    from detection_fissures.utilitaires.dispositif import detecter_dispositif

    dispositif = detecter_dispositif(
        forcer=None if args.dispositif == "auto" else args.dispositif
    )
    print(f"[Dispositif] {dispositif}")

    modele = charger_modele(
        chemin_pth=args.modele,
        nombre_classes=args.nombre_classes,
        taille_image=args.taille_image,
        architecture=args.architecture,
        dispositif=dispositif,
    )

    dossier_images = Path(args.images)
    if not dossier_images.exists():
        print(f"[Erreur] Dossier introuvable : {dossier_images}")
        return

    chemins_images = sorted(
        [
            p for p in dossier_images.iterdir()
            if p.suffix.lower() in EXTENSIONS_IMAGES
        ]
    )

    if not chemins_images:
        print(f"[Erreur] Aucune image trouvée dans : {dossier_images}")
        return

    print(f"\n[Analyse] {len(chemins_images)} images à traiter...")

    rapport_global = []

    with torch.no_grad():
        for chemin in chemins_images:
            try:
                tenseur, largeur_orig, hauteur_orig = preparer_image(
                    chemin_image=chemin,
                    taille_image=args.taille_image,
                    dispositif=dispositif,
                )

                predictions = modele(tenseur)

                predictions_filtrees = []
                for pred in predictions:
                    masque_score = pred["scores"] >= args.seuil
                    predictions_filtrees.append({
                        "masks": pred["masks"][masque_score],
                        "scores": pred["scores"][masque_score],
                        "labels": pred["labels"][masque_score],
                        "boxes": pred["boxes"][masque_score],
                    })

                resultats = classifier_predictions(
                    predictions=predictions_filtrees,
                    largeur_image=args.taille_image,
                    hauteur_image=args.taille_image,
                    seuil_score_min=0.0,
                )

                afficher_resultats_classification(resultats, nom_image=chemin.name)

                if args.sortie:
                    rapport = generer_rapport_classification(resultats, nom_image=chemin.name)
                    rapport_global.append(rapport)

            except Exception as e:
                print(f"[Erreur] {chemin.name} : {e}")
                continue

    if args.sortie:
        chemin_sortie = Path(args.sortie)
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

        with open(chemin_sortie, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "parametres": {
                        "modele": args.modele,
                        "seuil": args.seuil,
                        "taille_image": args.taille_image,
                    },
                    "resultats": rapport_global,
                    "resume": {
                        "images_analysees": len(rapport_global),
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

        print(f"\n[Rapport] Sauvegardé : {chemin_sortie}")

    print("\nAnalyse terminée.")


if __name__ == "__main__":
    main()
