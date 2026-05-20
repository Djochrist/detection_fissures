"""
Module du jeu de données COCO pour la détection de fissures.

Charge les images et annotations au format COCO (Roboflow export).
Aucune augmentation n'est appliquée — le dataset est déjà prétraité.

Format de sortie attendu par Mask R-CNN (torchvision) :
    image   : FloatTensor[3, H, W] dans [0, 1]
    cibles  : {
        "boxes"    : FloatTensor[N, 4]  (x1, y1, x2, y2)
        "labels"   : Int64Tensor[N]
        "masks"    : UInt8Tensor[N, H, W]
        "image_id" : Int64Tensor[1]
        "area"     : FloatTensor[N]
        "iscrowd"  : UInt8Tensor[N]
    }
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from ..utilitaires.images import (
    charger_image_rgb,
    image_rgb_vers_tenseur,
    redimensionner_image_carree,
)


class JeuDonneesFissuresCOCO(Dataset):
    """
    Jeu de données pour la segmentation d'instance de fissures au format COCO.

    Args:
        chemin_images      : Dossier contenant les images (train/, valid/ ou test/).
        chemin_annotations : Fichier _annotations.coco.json du split.
        taille_image       : Taille de redimensionnement (carré, en pixels).
    """

    def __init__(
        self,
        chemin_images: str | Path,
        chemin_annotations: str | Path,
        taille_image: int = 384,
    ) -> None:
        super().__init__()

        self.chemin_images = Path(chemin_images)
        self.chemin_annotations = Path(chemin_annotations)
        self.taille_image = taille_image

        if not self.chemin_images.is_dir():
            raise FileNotFoundError(
                f"Dossier d'images introuvable : {self.chemin_images}"
            )
        if not self.chemin_annotations.is_file():
            raise FileNotFoundError(
                f"Fichier d'annotations introuvable : {self.chemin_annotations}"
            )

        self.api_coco = COCO(str(self.chemin_annotations))

        self.ids_images = sorted(self.api_coco.imgs.keys())
        self.ids_images = self._filtrer_images_valides()

        print(
            f"[JeuDonnees] {len(self.ids_images)} images valides "
            f"— split '{self.chemin_annotations.parent.name}'"
        )

    def _filtrer_images_valides(self) -> List[int]:
        """Garde uniquement les images ayant au moins une segmentation non vide."""
        ids_valides = []
        for id_image in self.ids_images:
            ids_ann = self.api_coco.getAnnIds(imgIds=id_image)
            annotations = self.api_coco.loadAnns(ids_ann)
            if any(len(ann.get("segmentation", [])) > 0 for ann in annotations):
                ids_valides.append(id_image)
        return ids_valides

    def __len__(self) -> int:
        return len(self.ids_images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Charge une image et ses annotations.

        Étapes :
        1. Charger l'image → RGB numpy
        2. Redimensionner à taille_image × taille_image
        3. Décoder les masques polygonaux COCO
        4. Convertir en tenseurs PyTorch
        5. Convertir en tenseur dans [0, 1]
        """
        id_image = self.ids_images[index]

        # ── 1. Charger l'image ────────────────────────────────────────────────
        metadonnees = self.api_coco.imgs[id_image]
        chemin_complet = self.chemin_images / metadonnees["file_name"]

        image_rgb, largeur_orig, hauteur_orig = charger_image_rgb(chemin_complet)

        # ── 2. Redimensionner l'image ─────────────────────────────────────────
        image_rgb = redimensionner_image_carree(image_rgb, self.taille_image)

        # Facteurs d'échelle pour adapter les boites et masques
        echelle_x = self.taille_image / largeur_orig
        echelle_y = self.taille_image / hauteur_orig

        # ── 3. Charger les annotations ────────────────────────────────────────
        ids_ann = self.api_coco.getAnnIds(imgIds=id_image)
        annotations_brutes = self.api_coco.loadAnns(ids_ann)

        boites, masques, etiquettes, aires, est_foule = [], [], [], [], []

        for ann in annotations_brutes:
            if not ann.get("segmentation"):
                continue

            # Décoder le masque polygonal → matrice binaire (taille originale)
            masque_orig = self.api_coco.annToMask(ann)

            if masque_orig.sum() < 50:
                continue

            # Redimensionner le masque à taille_image
            masque_redim = cv2.resize(
                masque_orig.astype(np.uint8),
                (self.taille_image, self.taille_image),
                interpolation=cv2.INTER_NEAREST,
            )

            indices_y, indices_x = np.where(masque_redim > 0)
            if len(indices_x) == 0:
                continue

            x1 = float(indices_x.min())
            y1 = float(indices_y.min())
            x2 = float(indices_x.max())
            y2 = float(indices_y.max())

            if x2 <= x1 or y2 <= y1:
                continue

            boites.append([x1, y1, x2, y2])
            masques.append(masque_redim)
            etiquettes.append(1)
            aires.append(float(masque_redim.sum()))
            est_foule.append(0)

        # ── 4. Construire les cibles ──────────────────────────────────────────
        if len(boites) == 0:
            cibles = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,), dtype=torch.int64),
                "masks":    torch.zeros((0, self.taille_image, self.taille_image), dtype=torch.uint8),
                "image_id": torch.tensor([id_image], dtype=torch.int64),
                "area":     torch.zeros((0,), dtype=torch.float32),
                "iscrowd":  torch.zeros((0,), dtype=torch.uint8),
            }
        else:
            cibles = {
                "boxes":    torch.as_tensor(boites, dtype=torch.float32),
                "labels":   torch.as_tensor(etiquettes, dtype=torch.int64),
                "masks":    torch.as_tensor(np.stack(masques), dtype=torch.uint8),
                "image_id": torch.tensor([id_image], dtype=torch.int64),
                "area":     torch.as_tensor(aires, dtype=torch.float32),
                "iscrowd":  torch.as_tensor(est_foule, dtype=torch.uint8),
            }

        # ── 5. Convertir l'image en tenseur ───────────────────────────────────
        tenseur_image = image_rgb_vers_tenseur(image_rgb)

        return tenseur_image, cibles
