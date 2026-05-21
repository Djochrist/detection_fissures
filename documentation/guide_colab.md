# Entraîner sur Google Colab avec dataset sur Drive

Ce guide part d'un Notebook Colab vide et d'un dataset stocké dans Google Drive.

## 1. Préparer Colab

1. Ouvre Google Colab.
2. `Exécution` -> `Modifier le type d'exécution`.
3. Choisis `GPU`.
4. Lance les cellules ci-dessous dans l'ordre.

## 2. Monter Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Place ton dataset dans Drive avec cette structure :

```text
/content/drive/MyDrive/dataset/
├── train/
│   ├── _annotations.coco.json
│   └── images...
├── valid/
│   ├── _annotations.coco.json
│   └── images...
└── test/
    ├── _annotations.coco.json
    └── images...
```

Définis le chemin :

```python
DATASET = "/content/drive/MyDrive/dataset"
SORTIES_MASK = "/content/drive/MyDrive/detection_fissures_sorties_maskrcnn"
SORTIES_YOLO = "/content/drive/MyDrive/detection_fissures_sorties_yolo11"
```

## 3. Cloner le projet

```bash
%cd /content
!rm -rf detection_fissures
!git clone https://github.com/Djochrist/detection_fissures.git
%cd /content/detection_fissures
```

## 4. Installer les dépendances

Colab fournit déjà PyTorch GPU dans la plupart des runtimes. On installe les
dépendances projet autour de cette version.

```bash
!pip install -q -U opencv-python numpy scipy pycocotools "torchmetrics[detection]" rich ultralytics
```

Vérification :

```bash
!python -c "import torch, torchvision, torchmetrics, cv2, pycocotools; print('cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 5. Vérifier le dataset

```bash
!ls "$DATASET"
!ls "$DATASET/train" | head
!test -f "$DATASET/train/_annotations.coco.json" && echo "train OK"
!test -f "$DATASET/valid/_annotations.coco.json" && echo "valid OK"
!test -f "$DATASET/test/_annotations.coco.json" && echo "test OK"
```

## 6. Test court Mask R-CNN

```bash
!python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees "$DATASET" \
  --sorties /content/drive/MyDrive/detection_fissures_test_maskrcnn \
  --epoques 1 \
  --lot 1 \
  --lr 5e-5 \
  --taille-image 384 \
  --dispositif cuda
```

## 7. Entraîner Mask R-CNN

```bash
!python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees "$DATASET" \
  --sorties "$SORTIES_MASK" \
  --epoques 50 \
  --lot 2 \
  --lr 5e-5 \
  --taille-image 384 \
  --dispositif cuda
```

Si Colab donne `CUDA out of memory`, relance avec `--lot 1`.

## 8. Reprendre Mask R-CNN après coupure

```bash
!python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees "$DATASET" \
  --sorties "$SORTIES_MASK" \
  --epoques 100 \
  --lot 2 \
  --lr 5e-5 \
  --taille-image 384 \
  --seuil-score 0.05 \
  --dispositif cuda \
  --resume "$SORTIES_MASK/modeles/dernier_modele.pth"
```

## 9. Entraîner YOLOv11-seg

```bash
!python entrainer_yolo.py \
  --donnees "$DATASET" \
  --sorties "$SORTIES_YOLO" \
  --modele yolo11n-seg.pt \
  --epoques 50 \
  --lot 8 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --patience 15 \
  --taille-image 384 \
  --save-period 5 \
  --dispositif cuda
```

Si la mémoire manque, utilise `--lot 4` ou `--lot 2`.
Le script affiche un résumé du dataset YOLO converti, la configuration
d'entraînement, les métriques par époque quand Ultralytics les fournit, puis les
métriques validation/test finales avec précision, rappel, F1 score et mAP.

Pour reprendre après une coupure Colab :

```bash
!python entrainer_yolo.py \
  --donnees "$DATASET" \
  --sorties "$SORTIES_YOLO" \
  --resume "$SORTIES_YOLO/entrainements/yolo11_seg_fissures/weights/last.pt" \
  --dispositif cuda
```

## 10. Fichiers à récupérer

Mask R-CNN :

```text
$SORTIES_MASK/modeles/meilleur_modele.pth
$SORTIES_MASK/modeles/dernier_modele.pth
$SORTIES_MASK/journaux/historique_entrainement.json
```

YOLOv11 :

```text
$SORTIES_YOLO/entrainements/
$SORTIES_YOLO/evaluations/
```
