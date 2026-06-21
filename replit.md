# Détection de Fissures Structurelles

A deep learning project for structural fissure (crack) detection and segmentation. Supports two model architectures:

- **Mask R-CNN** (ResNet50-FPN-V2) — high precision, slower training
- **YOLO11-seg / YOLO26-seg** (Ultralytics) — fast, real-time inference

## Project Structure

- `detection_fissures/` — core Python package
- `modeles/` — model architecture definitions (Mask R-CNN)
- `entrainement/` — training loop, loss functions, metrics
- `donnees/` — data loading, COCO/YOLO format utilities
- `analyse/` — post-detection logic (danger index, orientation, localization)
- `configuration/` — global hyperparameters
- `utilitaires/` — helpers (image processing, device detection)
- `documentation/` — guides for Colab, Kaggle, Local

## Entry Points

- `main.py` — displays all available commands and environment info
- `entrainer.py` — train Mask R-CNN
- `entrainer_yolo.py` — train YOLO11-seg or YOLO26-seg
- `analyser.py` — run inference and danger classification

## Dataset

- Format: **YOLOv11 natif** (Roboflow) — NOT COCO
- Resolution: 640×640 px
- Images: 3794 (≈632 source × 5 augmentations with flip, rotation ±5°, brightness ±10%, exposure ±5%)
- Classes: 1 — `crack`
- Structure: `dataset/data.yaml` + `images/{train,valid,test}/` + `labels/{train,valid,test}/`

## Setup

Python 3.10 is required. Dependencies installed:
- `opencv-python`, `numpy`, `scipy`, `rich`
- `torch`, `torchvision` (CPU build)
- `torchmetrics[detection]`, `pycocotools`
- `ultralytics`

## Usage

See `main.py` output or `GUIDE_ENTRAINEMENT.md` for full command reference.

For YOLO training with the Roboflow dataset (native format, no conversion needed):
```bash
python entrainer_yolo.py --yaml dataset/data.yaml --modele yolo11m-seg.pt --taille-image 640 --epoques 150 --mask-ratio 1 --patience 50
```

## User preferences

(none yet)
