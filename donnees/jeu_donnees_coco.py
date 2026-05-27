"""
Jeu de données COCO pour la segmentation d'instance de fissures.

Charge les images et annotations au format COCO (Roboflow export).

IMAGES SANS ANNOTATIONS (murs sans fissures)
─────────────────────────────────────────────
Le dataset peut contenir deux types d'images :
  - Annotées   : images avec fissures → segmentation d'instance
  - Non annotées : images de murs SANS fissures → exemples négatifs

Les deux sont incluses dans l'entraînement. Mask R-CNN bénéficie
des images négatives : elles apprennent au RPN à ne rien proposer
sur des murs sains, réduisant les faux positifs.

Pour les images non annotées, les cibles retournées sont :
    boxes = Tensor(0, 4), labels = Tensor(0), masks = Tensor(0, H, W)

Format de sortie attendu par Mask R-CNN (torchvision) :
    image  : FloatTensor[3, H, W] dans [0, 1]
    cibles : {
        "boxes"    : FloatTensor[N, 4]  (x1, y1, x2, y2)
        "labels"   : Int64Tensor[N]
        "masks"    : UInt8Tensor[N, H, W]
        "image_id" : Int64Tensor[1]
        "area"     : FloatTensor[N]
        "iscrowd"  : UInt8Tensor[N]
    }
    où N = 0 pour les images sans fissures.
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
    Jeu de données pour la segmentation d'instance de fissures (format COCO).

    Inclut TOUTES les images du dataset :
      - Images annotées (avec fissures) → cibles non vides
      - Images non annotées (sans fissures) → cibles vides, exemples négatifs

    Args:
        chemin_images               : Dossier contenant les images (train/, valid/ ou test/).
        chemin_annotations          : Fichier _annotations.coco.json du split.
        taille_image                : Taille de redimensionnement (carré, en pixels).
        aire_min_masque             : Surface minimale (px²) d'un masque conservé après
                                      redimensionnement. Élimine les artefacts d'annotation.
        inclure_images_sans_fissures: Si False, exclut les images sans fissure annotée.
    """

    def __init__(
        self,
        chemin_images: str | Path,
        chemin_annotations: str | Path,
        taille_image: int = 384,
        aire_min_masque: int = 8,
        inclure_images_sans_fissures: bool = True,
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
        self.aire_min_masque = aire_min_masque
        self.inclure_images_sans_fissures = inclure_images_sans_fissures

        # Toutes les images du split (annotées ET non annotées par défaut)
        tous_ids = sorted(self.api_coco.imgs.keys())

        if inclure_images_sans_fissures:
            self.ids_images = tous_ids
        else:
            # Exclure les images sans aucune annotation de segmentation
            self.ids_images = [
                id_img for id_img in tous_ids
                if any(
                    len(ann.get("segmentation", [])) > 0
                    for ann in self.api_coco.loadAnns(self.api_coco.getAnnIds(imgIds=id_img))
                )
            ]

        # Stats pour l'affichage
        self._nb_avec_fissures, self._nb_sans_fissures = self._compter_types()

        split = self.chemin_annotations.parent.name
        print(
            f"[JeuDonnees] split='{split}' | "
            f"{len(self.ids_images)} images total | "
            f"{self._nb_avec_fissures} avec fissures | "
            f"{self._nb_sans_fissures} sans fissures (fond négatif)"
        )

    def _compter_types(self) -> Tuple[int, int]:
        """Compte les images avec et sans annotations de segmentation."""
        avec = 0
        for id_image in self.ids_images:
            ids_ann = self.api_coco.getAnnIds(imgIds=id_image)
            annotations = self.api_coco.loadAnns(ids_ann)
            if any(len(ann.get("segmentation", [])) > 0 for ann in annotations):
                avec += 1
        return avec, len(self.ids_images) - avec

    def __len__(self) -> int:
        return len(self.ids_images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Charge une image et ses annotations.

        Pour les images sans fissures, les cibles retournées ont N=0
        (tenseurs vides). Mask R-CNN gère nativement ce cas et traite
        l'image comme un exemple purement négatif (fond).

        Étapes :
        1. Charger l'image → RGB numpy
        2. Redimensionner à taille_image × taille_image
        3. Décoder les masques polygonaux COCO (vide si pas d'annotation)
        4. Construire les tenseurs PyTorch
        """
        id_image = self.ids_images[index]

        # ── 1. Charger l'image ────────────────────────────────────────────────
        metadonnees = self.api_coco.imgs[id_image]
        chemin_complet = self.chemin_images / metadonnees["file_name"]

        image_rgb, largeur_orig, hauteur_orig = charger_image_rgb(chemin_complet)

        # ── 2. Redimensionner l'image ─────────────────────────────────────────
        image_rgb = redimensionner_image_carree(image_rgb, self.taille_image)

        # ── 3. Charger les annotations ────────────────────────────────────────
        ids_ann = self.api_coco.getAnnIds(imgIds=id_image)
        annotations_brutes = self.api_coco.loadAnns(ids_ann)

        boites, masques, etiquettes, aires, est_foule = [], [], [], [], []

        for ann in annotations_brutes:
            if not ann.get("segmentation"):
                continue

            # Décoder le masque polygonal → matrice binaire (taille originale)
            masque_orig = self.api_coco.annToMask(ann)

            # Redimensionner le masque dans l'espace cible
            masque_redim = cv2.resize(
                masque_orig.astype(np.uint8),
                (self.taille_image, self.taille_image),
                interpolation=cv2.INTER_NEAREST,
            )

            # Ignorer les masques trop petits après redimensionnement
            # (artéfacts d'annotation, masques dégénérés ou quasi-vides)
            aire_redim = int(masque_redim.sum())
            if aire_redim < self.aire_min_masque:
                continue

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
            etiquettes.append(1)  # 1 = fissure (0 = fond, réservé à Mask R-CNN)
            aires.append(float(masque_redim.sum()))
            est_foule.append(0)

        # ── 4. Construire les cibles ──────────────────────────────────────────
        # N=0 pour les images sans fissures → exemple négatif (fond uniquement)
        if len(boites) == 0:
            cibles = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,), dtype=torch.int64),
                "masks":    torch.zeros(
                    (0, self.taille_image, self.taille_image), dtype=torch.uint8
                ),
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
