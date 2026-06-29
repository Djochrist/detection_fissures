# Détection de Fissures Structurelles

Segmentation d'instance de fissures par deep learning.  
Deux architectures supportées : **YOLO11-seg / YOLO26-seg** (Ultralytics) et **Mask R-CNN** (ResNet50-FPN-V2).

---

## Dataset

| Propriété | Valeur |
|-----------|--------|
| Format | **YOLOv11 natif** (Roboflow) |
| Résolution | 640 × 640 px |
| Images totales | 3 794 (≈ 632 sources × 5 augmentations) |
| Classes | 1 — `crack` |
| Augmentations Roboflow | Flip horizontal, rotation ±5°, luminosité ±10%, exposition ±5% |

Structure du dataset :

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

> Le `data.yaml` de Roboflow contient des chemins absolus Colab. Adapter le champ `path:` à votre environnement (voir `GUIDE_ENTRAINEMENT.md`).

---

## Structure du projet

```
detection_fissures/       — package Python principal
  configuration/          — hyperparamètres globaux
  modeles/                — architecture Mask R-CNN
  entrainement/           — boucle d'entraînement Mask R-CNN
  donnees/                — chargeurs COCO et conversion COCO→YOLO
  analyse/                — classification danger/orientation/profondeur
  utilitaires/            — détection device, helpers image

entrainer_yolo.py         — entraîner YOLO11-seg ou YOLO26-seg
entrainer.py              — entraîner Mask R-CNN
analyser.py               — inférence + classification des fissures
main.py                   — affiche les commandes et l'état de l'environnement

documentation/            — guides Colab, Kaggle, local
GUIDE_ENTRAINEMENT.md     — référence rapide des commandes
```

---

## Installation

Python 3.10 requis.

```bash
pip install -e .
pip install opencv-python numpy scipy rich
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU
pip install "torchmetrics[detection]" pycocotools ultralytics
```

Sur GPU (Colab / Kaggle) : PyTorch est déjà fourni avec CUDA, ajouter seulement :

```bash
pip install -q opencv-python pycocotools "torchmetrics[detection]" rich ultralytics
```

---

## Entraînement YOLO (recommandé)

La commande optimale pour ce dataset (YOLO11m, GPU, 3794 images 640px) :

```bash
python entrainer_yolo.py \
  --yaml          dataset/data.yaml \
  --murs-sains    murs_sains \
  --modele        yolo11m-seg.pt \
  --taille-image  1024 \
  --epoques       300 \
  --lot           4 \
  --lr            0.01 \
  --lrf           0.01 \
  --patience      100 \
  --warmup-epochs 5.0 \
  --mask-ratio    1 \
  --nom           yolo11m_fissures \
  --sorties       sorties_yolo
```

> `--yaml` charge directement le dataset Roboflow natif sans conversion.  
> `--murs-sains` (optionnel) intègre des images sans fissure comme exemples
> négatifs (label vide, 1 classe `crack`) ; retire-le si tu n'en as pas.  
> `--augmentation moderee` (défaut) ajoute une augmentation dynamique douce pour
> viser un meilleur mAP ; `forte` pousse plus, `desactivee` revient à zéro.  
> `--mask-ratio 1` = masques pleine résolution — critique pour les fissures de 1-3 px.

Voir `GUIDE_ENTRAINEMENT.md` pour toutes les options et les commandes Colab/Kaggle.

---

## Inférence

```bash
python analyser.py \
  --modele  sorties_yolo/entrainements/yolo11m_fissures/weights/best.pt \
  --backend yolo \
  --images  dataset/images/test/ \
  --seuil   0.25
```

---

## Documentation

| Fichier | Contenu |
|---------|---------|
| `GUIDE_ENTRAINEMENT.md` | Référence rapide — toutes les commandes |
| `documentation/guide_colab.md` | Entraîner sur Google Colab (GPU gratuit) |
| `documentation/guide_kaggle.md` | Entraîner sur Kaggle (GPU T4/P100) |
| `documentation/guide_local_uv.md` | Entraîner en local avec uv |
| `documentation/guide_orientation.md` | Théorie — classification d'orientation |
| `documentation/guide_geometrie.md` | Théorie — analyse géométrique des fissures |
| `documentation/guide_masks.md` | Théorie — masques de segmentation |
