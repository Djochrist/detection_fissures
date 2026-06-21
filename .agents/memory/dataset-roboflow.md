---
name: Dataset Roboflow YOLOv11 natif
description: Le dataset est en format YOLOv11 natif (pas COCO). Paramètres calibrés pour 640×640px, 3794 images, 1 classe crack.
---

## Règle

Le dataset est au format **YOLOv11 natif Roboflow** — ne pas convertir depuis COCO.

**How to apply:** Toujours passer `--yaml dataset/data.yaml` à `entrainer_yolo.py`. Le script détecte aussi automatiquement le format si `data.yaml` est présent dans `--donnees`.

## Paramètres calibrés 640×640px

- `mask_ratio = 1` (pleine résolution — fissures de 1-3px)
- `SEUIL_LARGEUR_SUPERFICIELLE = 6.0 px` (classificateur — était 3.5 pour 384px)
- `SEUIL_LARGEUR_PROFONDE = 12.0 px` (classificateur — était 7.0 pour 384px)
- `taille_image_min/max = 640` dans ParametresModele

## Hyperparamètres optimaux (3794 images, déjà augmenté 5×)

- modele: yolo11m-seg.pt, imgsz=640, epochs=150, batch=8
- lr0=0.01, lrf=0.01, patience=50, warmup_epochs=5, close_mosaic=20
- mosaic=0.4 (modéré — déjà augmenté), copy_paste=0.3, degrees=10, flipud=0.1

**Why:** Dataset déjà augmenté 5× par Roboflow (flip, ±5° rot, ±10% lum, ±5% expo) → augmentation complémentaire modérée seulement. 3794 images justifie yolo11m (Medium).
