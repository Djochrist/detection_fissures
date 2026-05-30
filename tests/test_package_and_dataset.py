import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from detection_fissures.configuration.parametres import (
    NOM_FICHIER_ANNOTATIONS_COCO,
    SPLITS_DATASET,
)
from detection_fissures.donnees.conversion_yolo import convertir_dataset_coco_vers_yolo
from detection_fissures.entrainer import verifier_dataset


def test_package_imports() -> None:
    import detection_fissures  # noqa: F401
    import detection_fissures.entrainer  # noqa: F401
    import detection_fissures.entrainer_yolo  # noqa: F401
    import detection_fissures.analyser  # noqa: F401


def test_verifier_dataset_valide(tmp_path: Path) -> None:
    racine = tmp_path / "dataset"
    for split in SPLITS_DATASET:
        dossier = racine / split
        dossier.mkdir(parents=True)
        image_path = dossier / "image_001.jpg"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        annotations = {
            "images": [{"id": 1, "file_name": "image_001.jpg"}],
            "annotations": [],
            "categories": [{"id": 1, "name": "fissure"}],
        }
        (dossier / NOM_FICHIER_ANNOTATIONS_COCO).write_text(
            json.dumps(annotations), encoding="utf-8"
        )

    verifier_dataset(racine)


def test_verifier_dataset_images_manquantes(tmp_path: Path) -> None:
    racine = tmp_path / "dataset"
    for split in SPLITS_DATASET:
        dossier = racine / split
        dossier.mkdir(parents=True)
        if split == "train":
            annotations = {
                "images": [{"id": 1, "file_name": "missing.jpg"}],
                "annotations": [],
                "categories": [{"id": 1, "name": "fissure"}],
            }
            (dossier / NOM_FICHIER_ANNOTATIONS_COCO).write_text(
                json.dumps(annotations), encoding="utf-8"
            )
        else:
            annotations = {
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "fissure"}],
            }
            (dossier / NOM_FICHIER_ANNOTATIONS_COCO).write_text(
                json.dumps(annotations), encoding="utf-8"
            )

    with pytest.raises(FileNotFoundError, match=r"image\(s\)"):
        verifier_dataset(racine)


def test_conversion_yolo_redimensionne_images_selon_taille_image(tmp_path: Path) -> None:
    racine = tmp_path / "dataset"
    taille_image = 96

    for split in SPLITS_DATASET:
        dossier = racine / split
        dossier.mkdir(parents=True)
        image_path = dossier / "image_001.jpg"
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        annotations = {
            "images": [
                {
                    "id": 1,
                    "file_name": "image_001.jpg",
                    "width": 80,
                    "height": 40,
                }
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "fissure"}],
        }
        (dossier / NOM_FICHIER_ANNOTATIONS_COCO).write_text(
            json.dumps(annotations), encoding="utf-8"
        )

    convertir_dataset_coco_vers_yolo(
        racine_coco=racine,
        racine_yolo=tmp_path / "dataset_yolo",
        taille_image=taille_image,
    )

    image_convertie = cv2.imread(str(tmp_path / "dataset_yolo/images/train/image_001.jpg"))
    assert image_convertie is not None
    assert image_convertie.shape[:2] == (taille_image, taille_image)
