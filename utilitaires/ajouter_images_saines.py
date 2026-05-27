"""
Ajout d'images non annotées (murs sains) à un dataset COCO existant.

CONTEXTE
────────
Vous avez téléchargé votre dataset depuis Roboflow (déjà splitté en
train/valid/test avec _annotations.coco.json).
Vous souhaitez y ajouter des images de murs sans fissures pour que
le modèle apprenne à ne rien détecter sur les murs sains.

FONCTIONNEMENT
──────────────
Ce script :
1. Copie vos images saines dans le dossier du split choisi (train/ par défaut)
2. Met à jour le fichier _annotations.coco.json pour y référencer ces images
3. N'ajoute AUCUNE annotation pour ces images → le modèle les traite comme fond

Il est non-destructif : les images et annotations existantes ne sont
jamais modifiées ou supprimées.

USAGE
─────
  # Ajouter vos images saines dans le split train (recommandé)
  python utilitaires/ajouter_images_saines.py \\
      --images-saines /chemin/vers/murs_sains/ \\
      --dataset dataset/

  # Ajouter dans un split spécifique
  python utilitaires/ajouter_images_saines.py \\
      --images-saines /chemin/vers/murs_sains/ \\
      --dataset dataset/ \\
      --split train

  # Répartir automatiquement entre train/valid/test
  python utilitaires/ajouter_images_saines.py \\
      --images-saines /chemin/vers/murs_sains/ \\
      --dataset dataset/ \\
      --repartir

  # Voir ce qui serait fait sans modifier le dataset (mode prévisualisation)
  python utilitaires/ajouter_images_saines.py \\
      --images-saines /chemin/vers/murs_sains/ \\
      --dataset dataset/ \\
      --apercu
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path


EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
NOM_ANNOTATIONS   = "_annotations.coco.json"


def analyser_arguments() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    analyseur = argparse.ArgumentParser(
        description="Ajout d'images de murs sains dans un dataset COCO existant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    analyseur.add_argument(
        "--images-saines",
        type=str,
        required=True,
        help="Dossier contenant vos images de murs SANS fissures",
    )
    analyseur.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Racine du dataset Roboflow (contient train/, valid/, test/)",
    )
    analyseur.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Split cible (défaut: train). Ignoré si --repartir est utilisé.",
    )
    analyseur.add_argument(
        "--repartir",
        action="store_true",
        help=(
            "Répartit les images entre train/valid/test automatiquement. "
            "Proportions : 80%% train / 10%% valid / 10%% test"
        ),
    )
    analyseur.add_argument(
        "--ratio-train",
        type=float,
        default=0.80,
        help="Fraction d'images pour train avec --repartir (défaut: 0.80)",
    )
    analyseur.add_argument(
        "--ratio-valid",
        type=float,
        default=0.10,
        help="Fraction d'images pour valid avec --repartir (défaut: 0.10)",
    )
    analyseur.add_argument(
        "--apercu",
        action="store_true",
        help="Affiche ce qui serait fait sans modifier le dataset",
    )
    analyseur.add_argument(
        "--graine",
        type=int,
        default=42,
        help="Graine aléatoire pour la répartition reproductible",
    )
    return analyseur.parse_args()


def lister_images(dossier: Path) -> list[Path]:
    """Liste toutes les images du dossier (non-récursif)."""
    images = sorted([
        p for p in dossier.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS_IMAGES
    ])
    return images


def charger_coco(chemin_json: Path) -> dict:
    """Charge un fichier COCO JSON."""
    if not chemin_json.is_file():
        raise FileNotFoundError(
            f"Annotations COCO introuvables : {chemin_json}\n"
            "Vérifiez que le dataset Roboflow est bien extrait dans le bon dossier."
        )
    return json.loads(chemin_json.read_text(encoding="utf-8"))


def sauvegarder_coco(donnees: dict, chemin_json: Path) -> None:
    """Sauvegarde un fichier COCO JSON (indentation lisible)."""
    chemin_json.write_text(
        json.dumps(donnees, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def id_max_images(donnees_coco: dict) -> int:
    """Retourne le plus grand ID image dans le fichier COCO."""
    images = donnees_coco.get("images", [])
    if not images:
        return 0
    return max(int(img.get("id", 0)) for img in images)


def noms_images_existants(donnees_coco: dict) -> set[str]:
    """Retourne les noms de fichiers déjà dans le JSON."""
    return {img.get("file_name", "") for img in donnees_coco.get("images", [])}


def ajouter_images_dans_split(
    images_a_ajouter: list[Path],
    dossier_split: Path,
    donnees_coco: dict,
    apercu: bool,
) -> tuple[int, int]:
    """
    Copie les images dans le dossier split et met à jour le JSON COCO.

    Les images déjà présentes (même nom) sont ignorées (idempotent).

    Returns:
        (nb_ajoutees, nb_ignorees)
    """
    noms_existants  = noms_images_existants(donnees_coco)
    prochain_id     = id_max_images(donnees_coco) + 1
    nb_ajoutees     = 0
    nb_ignorees     = 0

    nouvelles_entrees = []

    for chemin_image in images_a_ajouter:
        nom_fichier = chemin_image.name
        destination = dossier_split / nom_fichier

        # Éviter le doublon de nom de fichier
        nom_final = nom_fichier
        if nom_fichier in noms_existants or destination.exists():
            # Renommer avec un suffixe pour éviter les collisions
            stem  = chemin_image.stem
            suffixe = chemin_image.suffix
            compteur = 1
            while (
                nom_final in noms_existants
                or (dossier_split / nom_final).exists()
            ):
                nom_final = f"{stem}_sain{compteur:03d}{suffixe}"
                compteur += 1
            if nom_final == nom_fichier:
                # Nom déjà résolu, pas de collision
                pass

        if not apercu:
            # Déterminer les dimensions réelles de l'image
            import cv2
            img_cv = cv2.imread(str(chemin_image))
            if img_cv is None:
                print(f"  ⚠ Image illisible, ignorée : {chemin_image.name}")
                nb_ignorees += 1
                continue
            hauteur, largeur = img_cv.shape[:2]

            # Copier l'image dans le dossier split
            shutil.copy2(chemin_image, dossier_split / nom_final)

            nouvelles_entrees.append({
                "id":        prochain_id,
                "file_name": nom_final,
                "width":     largeur,
                "height":    hauteur,
            })
        else:
            # Mode aperçu : on simule sans lire l'image
            nouvelles_entrees.append({
                "id":        prochain_id,
                "file_name": nom_final,
                "width":     "?",
                "height":    "?",
            })

        noms_existants.add(nom_final)
        prochain_id  += 1
        nb_ajoutees  += 1

    if not apercu and nouvelles_entrees:
        donnees_coco["images"].extend(nouvelles_entrees)

    return nb_ajoutees, nb_ignorees


def repartir_images(
    images: list[Path],
    ratio_train: float,
    ratio_valid: float,
    graine: int,
) -> dict[str, list[Path]]:
    """
    Répartit une liste d'images entre train / valid / test.

    Args:
        images      : Liste de chemins d'images.
        ratio_train : Fraction pour train (ex: 0.80).
        ratio_valid : Fraction pour valid (ex: 0.10).
        graine      : Graine aléatoire.

    Returns:
        Dictionnaire {"train": [...], "valid": [...], "test": [...]}
    """
    random.seed(graine)
    melange = images.copy()
    random.shuffle(melange)

    n        = len(melange)
    n_train  = math.floor(n * ratio_train)
    n_valid  = math.floor(n * ratio_valid)

    return {
        "train": melange[:n_train],
        "valid": melange[n_train : n_train + n_valid],
        "test":  melange[n_train + n_valid :],
    }


def afficher_etat_avant(
    donnees_coco: dict,
    split: str,
    nb_images_a_ajouter: int,
) -> None:
    """Affiche l'état actuel du split avant modification."""
    images       = donnees_coco.get("images", [])
    annotations  = donnees_coco.get("annotations", [])
    ids_annotes  = {int(ann.get("image_id")) for ann in annotations}
    nb_avec      = sum(1 for img in images if int(img.get("id", -1)) in ids_annotes)
    nb_sans      = len(images) - nb_avec

    print(f"\n  État actuel du split '{split}' :")
    print(f"    Images totales      : {len(images)}")
    print(f"    Avec fissures       : {nb_avec}")
    print(f"    Sans fissures       : {nb_sans}")
    print(f"    → Après ajout       : {len(images) + nb_images_a_ajouter} images")
    print(f"    → Sans fissures     : {nb_sans + nb_images_a_ajouter}")


def main() -> None:
    """Point d'entrée principal."""
    args = analyser_arguments()

    dossier_saines = Path(args.images_saines).expanduser().resolve()
    racine_dataset = Path(args.dataset).expanduser().resolve()

    # Validation des dossiers
    if not dossier_saines.is_dir():
        raise FileNotFoundError(
            f"Dossier d'images saines introuvable : {dossier_saines}\n"
            "Vérifiez le chemin avec --images-saines."
        )
    if not racine_dataset.is_dir():
        raise FileNotFoundError(
            f"Dataset introuvable : {racine_dataset}\n"
            "Vérifiez le chemin avec --dataset."
        )

    # Lister les images saines
    images_saines = lister_images(dossier_saines)
    if not images_saines:
        print(f"⚠ Aucune image trouvée dans : {dossier_saines}")
        print(f"  Extensions acceptées : {', '.join(sorted(EXTENSIONS_IMAGES))}")
        return

    print("\n" + "═" * 62)
    print("  AJOUT D'IMAGES SAINES AU DATASET COCO")
    if args.apercu:
        print("  MODE APERÇU — aucune modification ne sera effectuée")
    print("═" * 62)
    print(f"  Dataset         : {racine_dataset}")
    print(f"  Images saines   : {dossier_saines}")
    print(f"  Images trouvées : {len(images_saines)}")

    # Construire le plan de répartition
    if args.repartir:
        plan = repartir_images(
            images_saines,
            ratio_train=args.ratio_train,
            ratio_valid=args.ratio_valid,
            graine=args.graine,
        )
        print(f"\n  Répartition automatique :")
        print(f"    train : {len(plan['train'])} images ({args.ratio_train*100:.0f}%)")
        print(f"    valid : {len(plan['valid'])} images ({args.ratio_valid*100:.0f}%)")
        ratio_test = 1.0 - args.ratio_train - args.ratio_valid
        print(f"    test  : {len(plan['test'])} images ({ratio_test*100:.0f}%)")
    else:
        plan = {args.split: images_saines}
        print(f"\n  Toutes les images → split '{args.split}'")

    print()

    # Traiter chaque split
    total_ajoutees = 0
    total_ignorees = 0

    for split, images_du_split in plan.items():
        if not images_du_split:
            print(f"  [{split}] Aucune image à ajouter, split ignoré.")
            continue

        dossier_split   = racine_dataset / split
        chemin_json     = dossier_split / NOM_ANNOTATIONS

        if not dossier_split.is_dir():
            print(f"  [{split}] ⚠ Dossier introuvable : {dossier_split}, split ignoré.")
            continue
        if not chemin_json.is_file():
            print(f"  [{split}] ⚠ Annotations introuvables : {chemin_json}, split ignoré.")
            continue

        donnees_coco = charger_coco(chemin_json)
        afficher_etat_avant(donnees_coco, split, len(images_du_split))

        print(f"\n  [{split}] Traitement de {len(images_du_split)} image(s)...")

        nb_ajoutees, nb_ignorees = ajouter_images_dans_split(
            images_a_ajouter=images_du_split,
            dossier_split=dossier_split,
            donnees_coco=donnees_coco,
            apercu=args.apercu,
        )

        if not args.apercu and nb_ajoutees > 0:
            sauvegarder_coco(donnees_coco, chemin_json)
            print(f"  [{split}] ✓ {chemin_json} mis à jour")

        if args.apercu:
            print(f"  [{split}] (aperçu) {nb_ajoutees} image(s) seraient ajoutées")
        else:
            print(f"  [{split}] ✓ {nb_ajoutees} image(s) ajoutées")
        if nb_ignorees:
            print(f"  [{split}] ⚠ {nb_ignorees} image(s) ignorées (illisibles)")

        total_ajoutees += nb_ajoutees
        total_ignorees += nb_ignorees

    # Résumé final
    print("\n" + "═" * 62)
    if args.apercu:
        print(f"  APERÇU : {total_ajoutees} image(s) seraient ajoutées au total")
        print("  Relancez SANS --apercu pour effectuer les modifications.")
    else:
        print(f"  ✓ TERMINÉ : {total_ajoutees} image(s) ajoutées au total")
        if total_ignorees:
            print(f"  ⚠ {total_ignorees} image(s) ignorées (illisibles)")
        print()
        print("  Prochaine étape — entraîner le modèle :")
        print()
        print("    Mask R-CNN :")
        print("      python entrainer.py --donnees dataset/")
        print()
        print("    YOLO11-seg :")
        print("      python entrainer_yolo.py --donnees dataset/")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
