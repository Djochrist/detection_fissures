# Entraîner en local avec uv

Ce guide valide le chemin local recommandé : environnement isolé avec `uv`,
dataset COCO local, test court, puis entraînement réel.

## 1. Prérequis

- Python 3.10 ou plus récent
- `git`
- un GPU NVIDIA conseillé pour l'entraînement réel
- un dataset COCO avec `train/`, `valid/`, `test/`

Structure attendue :

```text
dataset/
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

## 2. Installer uv et le projet

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/Djochrist/detection_fissures.git
cd detection_fissures
uv sync
```

Toutes les commandes Python suivantes doivent passer par `uv run`.

## 3. Vérifier l'installation

```bash
uv run python -c "import torch, torchvision, torchmetrics, cv2, pycocotools; print('OK')"
uv run python entrainer.py --help
uv run python entrainer_yolo.py --help
```

Vérifie aussi le GPU :

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 4. Définir le chemin du dataset

Si le dataset est dans le projet :

```bash
DATASET=dataset
```

Sinon :

```bash
DATASET=/chemin/absolu/vers/dataset
```

Vérification :

```bash
ls "$DATASET"
ls "$DATASET/train" | head
```

## 5. Test court Mask R-CNN

Ce test lance une époque avec un petit batch. Il vérifie les imports, le dataset,
TorchMetrics segmentation et l'écriture des checkpoints.

```bash
uv run python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees "$DATASET" \
  --sorties sorties_test_maskrcnn \
  --epoques 1 \
  --lot 1 \
  --lr 3e-4 \
  --taille-image 384 \
  --dispositif auto
```

## 6. Entraînement Mask R-CNN

GPU 8 Go :

```bash
uv run python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees "$DATASET" \
  --sorties sorties_maskrcnn \
  --epoques 50 \
  --lot 2 \
  --lr 3e-4 \
  --taille-image 384 \
  --dispositif cuda
```

Si tu as plus de mémoire GPU, essaie `--lot 4`. Si tu as `CUDA out of memory`,
relance avec `--lot 1`.

## 7. Entraînement YOLO26-seg

```bash
uv run python entrainer_yolo.py \
  --donnees "$DATASET" \
  --sorties sorties_yolo26 \
  --modele yolo26s-seg.pt \
  --epoques 50 \
  --lot 8 \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --patience 25 \
  --taille-image 384 \
  --mask-ratio 2 \
  --dispositif cuda
```

Pour un GPU plus petit, utilise `--lot 4` ou `--lot 2`.

## 8. Reprendre YOLO-seg

```bash
uv run python entrainer_yolo.py \
  --donnees "$DATASET" \
  --sorties sorties_yolo26 \
  --resume sorties_yolo26/entrainements/yolo_seg_fissures/weights/last.pt \
  --dispositif cuda
```

## 9. Reprendre Mask R-CNN

```bash
uv run python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees "$DATASET" \
  --sorties sorties_maskrcnn \
  --epoques 100 \
  --lot 2 \
  --lr 3e-4 \
  --taille-image 384 \
  --dispositif cuda \
  --resume sorties_maskrcnn/modeles/dernier_modele.pth
```

## 10. Analyser des images et retrouver les sorties

Les images source restent dans le dossier passé à `--images`. Les images
annotées sont écrites dans `analyses/<backend>/images_annotees/`.

```bash
uv run python analyser.py \
  --modele sorties_yolo26/entrainements/yolo_seg_fissures/weights/best.pt \
  --images photos_test/ \
  --dossier-sortie analyses
```

Sorties principales :

```text
sorties_maskrcnn/
├── modeles/meilleur_modele.pth
├── modeles/dernier_modele.pth
└── journaux/historique_entrainement.json

sorties_yolo26/
├── dataset_yolo/images/{train,valid,test}/
├── dataset_yolo/labels/{train,valid,test}/
├── entrainements/yolo_seg_fissures/weights/best.pt
├── entrainements/yolo_seg_fissures/weights/last.pt
└── evaluations/yolo_seg_fissures_test/

analyses/
└── yolo/
    ├── rapport_analyse.json
    └── images_annotees/*_analyse.jpg
```
