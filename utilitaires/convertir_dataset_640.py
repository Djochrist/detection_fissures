"""
Redimensionne un dataset COCO en 640×640 pixels tout en conservant
les annotations de segmentation.

Usage :
    python utilitaires/convertir_dataset_640.py --donnees dataset/ --sorties dataset_640/

Ce script crée une copie redimensionnée de chaque image et met à jour
les fichiers `_annotations.coco.json` pour chaque split train/valid/test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO


SPLITS_DATASET = ("train", "valid", "test")
NOM_FICHIER_ANNOTATIONS_COCO = "_annotations.coco.json"


def analyser_arguments() -> argparse.Namespace:
    analyseur = argparse.ArgumentParser(
        description="Redimensionne un dataset COCO en un dataset 640×640.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyseur.add_argument(
        "--donnees",
        type=str,
        default="dataset",
        help="Répertoire source du dataset COCO (train/, valid/, test/)",
    )
    analyseur.add_argument(
        "--sorties",
        type=str,
        default="dataset_640",
        help="Répertoire de destination du dataset converti",
    )
    analyseur.add_argument(
        "--taille-image",
        type=int,
        default=640,
        help="Taille carré cible en pixels pour les images et les annotations",
    )
    analyseur.add_argument(
        "--force",
        action="store_true",
        help="Écrase le dossier de sortie s'il existe déjà",
    )
    analyseur.add_argument(
        "--split",
        nargs="+",
        choices=list(SPLITS_DATASET),
        default=list(SPLITS_DATASET),
        help="Split(s) à convertir",
    )
    return analyseur.parse_args()


def _encoder_masque_rle(masque: np.ndarray) -> dict[str, Any]:
    masque_fortran = np.asfortranarray(masque.astype(np.uint8))
    rle = mask_utils.encode(masque_fortran)
    if isinstance(rle["counts"], bytes):
        rle = {
            "size": list(rle["size"]),
            "counts": rle["counts"].decode("ascii"),
        }
    return rle


def _redimensionner_masque(masque: np.ndarray, taille_image: int) -> np.ndarray:
    return cv2.resize(
        masque.astype(np.uint8),
        (taille_image, taille_image),
        interpolation=cv2.INTER_NEAREST,
    )


def _mettre_a_jour_annotation(
    annotation: dict[str, Any],
    mask_redim: np.ndarray,
    taille_image: int,
) -> dict[str, Any]:
    indices_y, indices_x = np.where(mask_redim > 0)
    if len(indices_x) == 0:
        annotation = annotation.copy()
        annotation["segmentation"] = []
        annotation["bbox"] = [0.0, 0.0, 0.0, 0.0]
        annotation["area"] = 0.0
        return annotation

    x1 = float(indices_x.min())
    y1 = float(indices_y.min())
    x2 = float(indices_x.max())
    y2 = float(indices_y.max())
    bbox = [x1, y1, float(x2 - x1 + 1.0), float(y2 - y1 + 1.0)]
    area = float(int(mask_redim.sum()))
    segmentation = _encoder_masque_rle(mask_redim)

    annotation = annotation.copy()
    annotation["segmentation"] = segmentation
    annotation["bbox"] = bbox
    annotation["area"] = area
    return annotation


def _convertir_split(
    chemin_split_src: Path,
    chemin_dst: Path,
    taille_image: int,
) -> None:
    split = chemin_split_src.name
    chemin_annotations_src = chemin_split_src / NOM_FICHIER_ANNOTATIONS_COCO
    if not chemin_annotations_src.is_file():
        raise FileNotFoundError(
            f"Fichier d'annotations manquant pour {split} : {chemin_annotations_src}"
        )

    coco = COCO(str(chemin_annotations_src))
    donnees = json.loads(chemin_annotations_src.read_text(encoding="utf-8"))
    images = donnees["images"]
    annotations = donnees["annotations"]
    categories = donnees["categories"]

    chemin_images_dst = chemin_dst / split
    chemin_images_dst.mkdir(parents=True, exist_ok=True)

    images_redim = []
    annotations_redim = []

    for image in images:
        nom_image = str(image["file_name"])
        chemin_image_src = chemin_src / nom_image
        if not chemin_image_src.is_file():
            raise FileNotFoundError(f"Image introuvable : {chemin_image_src}")

        image_bgr = cv2.imread(str(chemin_image_src))
        if image_bgr is None:
            raise FileNotFoundError(f"Impossible de lire l'image : {chemin_image_src}")

        image_redim = cv2.resize(
            image_bgr,
            (taille_image, taille_image),
            interpolation=cv2.INTER_LINEAR,
        )
        chemin_image_dst = chemin_images_dst / nom_image
        cv2.imwrite(str(chemin_image_dst), image_redim)

        images_redim.append(
            {
                **image,
                "width": taille_image,
                "height": taille_image,
            }
        )

    annotations_par_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        annotations_par_image.setdefault(image_id, []).append(annotation)

    for image in images:
        image_id = int(image["id"])
        image_annotations = annotations_par_image.get(image_id, [])
        if not image_annotations:
            continue

        for annotation in image_annotations:
            if not annotation.get("segmentation"):
                annotations_redim.append(annotation.copy())
                continue

            mask_orig = coco.annToMask(annotation)
            mask_redim = _redimensionner_masque(mask_orig, taille_image)
            annotation_redim = _mettre_a_jour_annotation(annotation, mask_redim, taille_image)
            annotations_redim.append(annotation_redim)

    donnees_redim = {
        "images": images_redim,
        "annotations": annotations_redim,
        "categories": categories,
        "info": donnees.get("info", {}),
        "licenses": donnees.get("licenses", []),
    }

    chemin_annotations_dst = chemin_dst / split / NOM_FICHIER_ANNOTATIONS_COCO
    chemin_annotations_dst.write_text(
        json.dumps(donnees_redim, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Split '{split}' converti : {len(images_redim)} images, {len(annotations_redim)} annotations")


def main() -> None:
    args = analyser_arguments()
    chemin_donnees_src = Path(args.donnees).expanduser().resolve()
    chemin_sorties = Path(args.sorties).expanduser().resolve()

    if chemin_sorties.exists():
        if not args.force:
            raise FileExistsError(
                f"Le dossier de sortie existe déjà : {chemin_sorties}."
                " Utilisez --force pour écraser."
            )
    else:
        chemin_sorties.mkdir(parents=True, exist_ok=True)

    for split in args.split:
        chemin_split_src = chemin_donnees_src / split
        if not chemin_split_src.is_dir():
            raise FileNotFoundError(f"Split introuvable : {chemin_split_src}")
        _convertir_split(chemin_split_src, chemin_sorties, args.taille_image)

    print(f"\nConversion terminée : {chemin_sorties}")


if __name__ == "__main__":
    main()
