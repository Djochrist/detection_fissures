"""
Conversion du dataset COCO Roboflow vers le format YOLO segmentation.

Ultralytics YOLO11-seg attend une arborescence images/labels avec un fichier
texte par image. Chaque ligne de label contient :
    classe x1 y1 x2 y2 ... xn yn
où les coordonnées du polygone sont normalisées entre 0 et 1.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from ..configuration.parametres import (
    NOM_FICHIER_ANNOTATIONS_COCO,
    SPLITS_DATASET,
)


def _creer_lien_ou_copie(source: Path, destination: Path, copier: bool) -> None:
    """Crée un lien symbolique vers l'image, ou copie l'image si demandé."""
    if destination.exists() or destination.is_symlink():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if copier:
        shutil.copy2(source, destination)
        return

    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def _normaliser_polygone(
    points: list[float],
    largeur: int,
    hauteur: int,
) -> list[float]:
    """Normalise et borne une liste plate de coordonnées COCO."""
    coordonnees = []
    for index, valeur in enumerate(points):
        limite = largeur if index % 2 == 0 else hauteur
        normalisee = float(valeur) / float(limite)
        coordonnees.append(min(1.0, max(0.0, normalisee)))
    return coordonnees


def _segments_depuis_rle(
    segmentation: dict[str, Any],
    largeur: int,
    hauteur: int,
) -> list[list[float]]:
    """Convertit un masque RLE COCO en contours polygonaux YOLO."""
    masque = mask_utils.decode(segmentation)
    if masque.ndim == 3:
        masque = np.any(masque, axis=2).astype(np.uint8)

    contours, _ = cv2.findContours(
        masque.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    segments = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        plat = points.astype(float).reshape(-1).tolist()
        segments.append(_normaliser_polygone(plat, largeur, hauteur))
    return segments


def _segments_yolo_annotation(
    annotation: dict[str, Any],
    largeur: int,
    hauteur: int,
) -> list[list[float]]:
    """Retourne les segments YOLO normalisés pour une annotation COCO."""
    segmentation = annotation.get("segmentation")
    if not segmentation:
        return []

    if isinstance(segmentation, list):
        segments = []
        for polygone in segmentation:
            if not isinstance(polygone, list) or len(polygone) < 6:
                continue
            segments.append(_normaliser_polygone(polygone, largeur, hauteur))
        return segments

    if isinstance(segmentation, dict):
        return _segments_depuis_rle(segmentation, largeur, hauteur)

    return []


def _charger_coco(chemin_annotations: Path) -> dict[str, Any]:
    """Charge un fichier COCO et vérifie les clés minimales."""
    donnees = json.loads(chemin_annotations.read_text(encoding="utf-8"))
    for cle in ("images", "annotations", "categories"):
        if not isinstance(donnees.get(cle), list):
            raise ValueError(f"COCO invalide dans {chemin_annotations} : clé '{cle}' absente.")
    return donnees


def convertir_dataset_coco_vers_yolo(
    racine_coco: str | Path,
    racine_yolo: str | Path,
    copier_images: bool = False,
    nom_classe_defaut: str = "fissure",
) -> Path:
    """
    Convertit train/valid/test COCO vers un dataset YOLO11-seg.

    Args:
        racine_coco: dossier contenant train/, valid/ et test/.
        racine_yolo: dossier de sortie YOLO.
        copier_images: si True, copie les images. Sinon crée des symlinks.
        nom_classe_defaut: nom de classe si COCO ne fournit pas de catégorie.

    Returns:
        Chemin du fichier data.yaml généré.
    """
    racine_coco = Path(racine_coco).expanduser().resolve()
    racine_yolo = Path(racine_yolo).expanduser().resolve()
    racine_yolo.mkdir(parents=True, exist_ok=True)

    noms_classes: dict[int, str] = {}
    categories_globales: dict[int, int] = {}

    for split in SPLITS_DATASET:
        dossier_split = racine_coco / split
        chemin_annotations = dossier_split / NOM_FICHIER_ANNOTATIONS_COCO
        if not dossier_split.is_dir() or not chemin_annotations.is_file():
            raise FileNotFoundError(
                f"Split COCO incomplet : {dossier_split} "
                f"avec {NOM_FICHIER_ANNOTATIONS_COCO}"
            )

        donnees = _charger_coco(chemin_annotations)
        categories = sorted(donnees["categories"], key=lambda cat: cat.get("id", 0))
        if not categories:
            categories = [{"id": 1, "name": nom_classe_defaut}]

        categories_split = {
            int(categorie["id"]): index
            for index, categorie in enumerate(categories)
        }
        for categorie in categories:
            index = categories_split[int(categorie["id"])]
            categories_globales.setdefault(int(categorie["id"]), index)
            noms_classes.setdefault(index, str(categorie.get("name") or nom_classe_defaut))

        images = {int(image["id"]): image for image in donnees["images"]}
        annotations_par_image: dict[int, list[dict[str, Any]]] = {
            id_image: [] for id_image in images
        }
        for annotation in donnees["annotations"]:
            id_image = int(annotation.get("image_id", -1))
            if id_image in annotations_par_image:
                annotations_par_image[id_image].append(annotation)

        dossier_images_yolo = racine_yolo / "images" / split
        dossier_labels_yolo = racine_yolo / "labels" / split
        dossier_images_yolo.mkdir(parents=True, exist_ok=True)
        dossier_labels_yolo.mkdir(parents=True, exist_ok=True)

        for id_image, image in images.items():
            nom_fichier = str(image["file_name"])
            chemin_image_source = dossier_split / nom_fichier
            if not chemin_image_source.is_file():
                raise FileNotFoundError(f"Image COCO introuvable : {chemin_image_source}")

            chemin_image_yolo = dossier_images_yolo / nom_fichier
            _creer_lien_ou_copie(chemin_image_source, chemin_image_yolo, copier_images)

            largeur = int(image.get("width") or 0)
            hauteur = int(image.get("height") or 0)
            if largeur <= 0 or hauteur <= 0:
                image_cv = cv2.imread(str(chemin_image_source))
                if image_cv is None:
                    raise FileNotFoundError(f"Image illisible : {chemin_image_source}")
                hauteur, largeur = image_cv.shape[:2]

            lignes = []
            for annotation in annotations_par_image.get(id_image, []):
                if int(annotation.get("iscrowd", 0)) == 1:
                    continue
                id_categorie = int(annotation.get("category_id", 1))
                classe_yolo = categories_globales.get(id_categorie, 0)
                for segment in _segments_yolo_annotation(annotation, largeur, hauteur):
                    if len(segment) < 6:
                        continue
                    valeurs = " ".join(f"{valeur:.6f}" for valeur in segment)
                    lignes.append(f"{classe_yolo} {valeurs}")

            chemin_label = dossier_labels_yolo / f"{Path(nom_fichier).stem}.txt"
            chemin_label.write_text("\n".join(lignes), encoding="utf-8")

    noms_ordonnes = [noms_classes[index] for index in sorted(noms_classes)]
    if not noms_ordonnes:
        noms_ordonnes = [nom_classe_defaut]

    chemin_yaml = racine_yolo / "data.yaml"
    lignes_yaml = [
        f"path: {racine_yolo}",
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        "names:",
    ]
    lignes_yaml.extend(
        f"  {index}: {nom}" for index, nom in enumerate(noms_ordonnes)
    )
    chemin_yaml.write_text("\n".join(lignes_yaml) + "\n", encoding="utf-8")

    return chemin_yaml
