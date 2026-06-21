# Entraîner sur Google Colab avec dataset sur Drive

Dataset : format YOLOv11 natif Roboflow — 640×640px — 3794 images — 1 classe (crack).

## 1. Préparer Colab

1. Ouvre Google Colab.
2. `Exécution` → `Modifier le type d'exécution` → choisir `GPU`.
3. Lance les cellules ci-dessous dans l'ordre.

## 2. Monter Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Place ton dataset dans Drive avec cette structure :

```text
/content/drive/MyDrive/dataset/
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

Définis les chemins :

```python
DATASET_YAML = "/content/drive/MyDrive/dataset/data.yaml"
SORTIES_YOLO = "/content/drive/MyDrive/detection_fissures_sorties_yolo"
```

## 3. Cloner le projet

```bash
%cd /content
!rm -rf detection_fissures
!git clone https://github.com/Djochrist/detection_fissures.git
%cd /content/detection_fissures
```

## 4. Installer les dépendances

Colab fournit déjà PyTorch GPU. Installer les dépendances projet :

```bash
!pip install -q -U opencv-python numpy scipy pycocotools "torchmetrics[detection]" rich ultralytics
!pip install -q -e .
```

Vérification :

```bash
!python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
!python entrainer_yolo.py --help
```

## 5. Adapter data.yaml

Le `data.yaml` Roboflow contient un chemin absolu qui doit pointer vers votre Drive :

```python
import yaml, pathlib

yaml_path = pathlib.Path(DATASET_YAML)
config = yaml.safe_load(yaml_path.read_text())
config['path'] = str(yaml_path.parent)
yaml_path.write_text(yaml.dump(config))
print("data.yaml mis à jour :", config['path'])
```

## 6. Entraîner YOLO11m (recommandé)

```bash
!python entrainer_yolo.py \
  --yaml          "$DATASET_YAML" \
  --modele        yolo11m-seg.pt \
  --taille-image  640 \
  --epoques       150 \
  --lot           8 \
  --lr            0.01 \
  --lrf           0.01 \
  --patience      50 \
  --warmup-epochs 5.0 \
  --close-mosaic  20 \
  --mask-ratio    1 \
  --mosaic        0.4 \
  --copy-paste    0.3 \
  --degrees       10.0 \
  --flipud        0.1 \
  --dispositif    auto \
  --nom           yolo11m_fissures \
  --sorties       "$SORTIES_YOLO"
```

**GPU < 6 Go** — utiliser YOLO11s avec lot=16 :

```bash
!python entrainer_yolo.py \
  --yaml          "$DATASET_YAML" \
  --modele        yolo11s-seg.pt \
  --taille-image  640 \
  --epoques       150 \
  --lot           16 \
  --lr            0.01 \
  --patience      50 \
  --mask-ratio    1 \
  --mosaic        0.4 \
  --copy-paste    0.3 \
  --nom           yolo11s_fissures \
  --sorties       "$SORTIES_YOLO"
```

## 7. Reprendre après déconnexion

```bash
LAST_PT = "$SORTIES_YOLO/entrainements/yolo11m_fissures/weights/last.pt"

!python entrainer_yolo.py --resume "$LAST_PT"
```

## 8. Analyser des images après entraînement

```bash
BEST_PT = "$SORTIES_YOLO/entrainements/yolo11m_fissures/weights/best.pt"

!python analyser.py \
  --modele   "$BEST_PT" \
  --backend  yolo \
  --images   /content/drive/MyDrive/dataset/images/test/ \
  --seuil    0.25
```

## 9. Récupérer les résultats

```python
from google.colab import files

# Télécharger les poids
files.download(f"{SORTIES_YOLO}/entrainements/yolo11m_fissures/weights/best.pt")
```

## Notes

- Colab déconnecte après ~12h d'inactivité. Utiliser `--resume` pour reprendre.
- Sauvegarder les poids sur Drive (en définissant `--sorties` sur un dossier Drive).
- `data.yaml` doit avoir `path:` pointant vers le dossier racine du dataset.
