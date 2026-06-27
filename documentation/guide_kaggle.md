# Entraîner sur Kaggle

Dataset : format YOLOv11 natif Roboflow — 640×640px — 3794 images — 1 classe (crack).

## 1. Préparer le Notebook Kaggle

1. Crée un nouveau Notebook.
2. Active un GPU : `Settings` → `Accelerator` → `GPU T4 x2` ou `GPU P100/T4`.
3. Active Internet pour cloner le dépôt GitHub et télécharger les poids préentraînés.
4. Ajoute ton dataset Roboflow dans `Input`.

Le dataset doit contenir cette structure :

```text
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

## 2. Installer le projet

```bash
%cd /kaggle/working
!git clone https://github.com/Djochrist/detection_fissures.git
%cd /kaggle/working/detection_fissures

!pip install -q opencv-python pycocotools "torchmetrics[detection]" rich ultralytics
!pip install -q -e .
```

Vérification :

```bash
!python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
!python entrainer_yolo.py --help
```

## 3. Trouver le chemin du dataset

```bash
!find /kaggle/input -maxdepth 4 -name data.yaml -print
```

Exemple :

```python
DATASET_YAML = "/kaggle/input/detection-fissures/dataset/data.yaml"
SORTIES_YOLO = "/kaggle/working/sorties_yolo"
```

## 4. Adapter data.yaml (important)

Le `data.yaml` Roboflow contient un chemin absolu Colab qui doit être corrigé :

```python
import yaml, pathlib

yaml_path = pathlib.Path(DATASET_YAML)
config = yaml.safe_load(yaml_path.read_text())
config['path'] = str(yaml_path.parent)
yaml_path.write_text(yaml.dump(config))
print("data.yaml mis à jour :", config['path'])
```

## 5. Test rapide (1 époque)

```bash
!python entrainer_yolo.py \
  --yaml         "$DATASET_YAML" \
  --modele       yolo11s-seg.pt \
  --epoques      1 \
  --lot          8 \
  --taille-image 640 \
  --mask-ratio   1 \
  --dispositif   cuda \
  --nom          test_rapide \
  --sorties      /kaggle/working/test_sorties
```

Si ce test passe, lance l'entraînement complet.

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
  --weight-decay  0.0005 \
  --patience      50 \
  --warmup-epochs 5.0 \
  --mask-ratio    1 \
  --dispositif    cuda \
  --nom           yolo11m_fissures \
  --sorties       "$SORTIES_YOLO"
```

**GPU T4 (6 Go VRAM)** — si CUDA out of memory, passer à `--lot 4` ou utiliser YOLO11s :

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
  --dispositif    cuda \
  --nom           yolo11s_fissures \
  --sorties       "$SORTIES_YOLO"
```

## 7. Reprendre depuis un ancien output Kaggle

Ajouter l'ancien output comme Input, puis :

```bash
LAST_PT = "/kaggle/input/ancien-output/sorties_yolo/entrainements/yolo11m_fissures/weights/last.pt"

!python entrainer_yolo.py --resume "$LAST_PT"
```

## 8. Analyser des images après entraînement

```bash
BEST_PT = "$SORTIES_YOLO/entrainements/yolo11m_fissures/weights/best.pt"

!python analyser.py \
  --modele   "$BEST_PT" \
  --backend  yolo \
  --images   /kaggle/input/detection-fissures/dataset/images/test/ \
  --seuil    0.25
```

## 9. Vérifier les fichiers de sortie

```bash
!find /kaggle/working -maxdepth 6 -type f \( -name "*.pt" -o -name "results.csv" -o -name "*.png" \) -print
```

Les poids finaux sont dans :
```
/kaggle/working/sorties_yolo/entrainements/yolo11m_fissures/weights/best.pt
```

Ils apparaissent dans l'onglet `Output` du Notebook après exécution.
