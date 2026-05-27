# Entraîner sur Kaggle

Ce projet entraîne uniquement deux modèles :

- Mask R-CNN ResNet50-FPN-V2 avec `entrainer.py`
- YOLO26-seg avec `entrainer_yolo.py` (YOLO11 reste compatible)

## 1. Préparer le Notebook Kaggle

Dans Kaggle :

1. Crée un nouveau Notebook.
2. Active un GPU : `Settings` -> `Accelerator` -> `GPU T4 x2` ou `GPU P100/T4`.
3. Active Internet si tu veux cloner le dépôt GitHub et télécharger les poids préentraînés.
4. Ajoute ton dataset dans `Input`.

Le dataset doit contenir cette structure COCO :

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

## 2. Installer le projet

Cellule Kaggle :

```bash
%cd /kaggle/working
!git clone https://github.com/Djochrist/detection_fissures.git
%cd /kaggle/working/detection_fissures

!pip install -q opencv-python pycocotools "torchmetrics[detection]" rich ultralytics
```

Vérification :

```bash
!python -c "import torch, torchvision, torchmetrics, cv2, pycocotools; print('cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
!python entrainer.py --help
!python entrainer_yolo.py --help
```

Si tu as envoyé le projet comme dataset Kaggle au lieu de le cloner, copie-le dans
`/kaggle/working` puis entre dans son dossier avant de lancer les commandes.

## 3. Trouver le chemin du dataset

Cellule Kaggle :

```bash
!find /kaggle/input -maxdepth 4 -name _annotations.coco.json -print
```

Choisis le dossier qui contient directement `train/`, `valid/` et `test/`.
Exemple :

```python
DATASET = "/kaggle/input/detection-fissures/dataset"
```

Vérification rapide :

```bash
!ls "$DATASET"
!ls "$DATASET/train" | head
```

## 4. Test rapide avant long entraînement

Ce test vérifie le dataset, les imports, le GPU, la sauvegarde et une époque complète.

```bash
!python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees "$DATASET" \
  --sorties /kaggle/working/sorties_maskrcnn_test \
  --epoques 1 \
  --lot 1 \
  --lr 3e-4 \
  --taille-image 384 \
  --dispositif cuda
```

Si ce test passe, lance ensuite l'entraînement réel.

## 5. Entraîner Mask R-CNN ResNet50-FPN-V2

Réglage prudent pour Kaggle :

```bash
!python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees "$DATASET" \
  --sorties /kaggle/working/sorties_maskrcnn \
  --epoques 50 \
  --lot 2 \
  --lr 3e-4 \
  --taille-image 384 \
  --dispositif cuda
```

Si la mémoire GPU tient, tu peux essayer `--lot 4`. Si Kaggle affiche une erreur
`CUDA out of memory`, relance avec `--lot 1`.

Les fichiers importants seront :

```text
/kaggle/working/sorties_maskrcnn/modeles/meilleur_modele.pth
/kaggle/working/sorties_maskrcnn/modeles/dernier_modele.pth
/kaggle/working/sorties_maskrcnn/journaux/historique_entrainement.json
```

## 6. Entraîner YOLO26-seg

YOLO convertit automatiquement le COCO vers YOLO-seg dans le dossier de sortie.

```bash
!python entrainer_yolo.py \
  --donnees "$DATASET" \
  --sorties /kaggle/working/sorties_yolo26 \
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

Pour un modèle plus fort si le GPU tient :

```bash
!python entrainer_yolo.py \
  --donnees "$DATASET" \
  --sorties /kaggle/working/sorties_yolo26_m \
  --modele yolo26m-seg.pt \
  --epoques 50 \
  --lot 4 \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --patience 25 \
  --taille-image 384 \
  --mask-ratio 2 \
  --dispositif cuda
```

## 7. Sauvegarder les résultats Kaggle

Pour analyser des images après entraînement YOLO-seg :

```bash
!python analyser.py \
  --modele /kaggle/working/sorties_yolo26/entrainements/yolo_seg_fissures/weights/best.pt \
  --images /kaggle/input/images-a-tester \
  --dossier-sortie /kaggle/working/analyses \
  --dispositif cuda
```

Le rapport sera dans `/kaggle/working/analyses/yolo/rapport_analyse.json`.
Les copies annotées seront dans `/kaggle/working/analyses/yolo/images_annotees/`.
Les images originales dans `/kaggle/input/...` ne sont jamais modifiées.

Tout ce qui est dans `/kaggle/working` apparaît dans l'onglet `Output` après
l'exécution du Notebook. Avant de fermer Kaggle, vérifie :

```bash
!find /kaggle/working -maxdepth 5 -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.json" -o -name "results.csv" -o -name "*_analyse.jpg" -o -name "*_pred.jpg" \) -print
```

Pour reprendre YOLO-seg depuis un ancien output Kaggle :

```bash
!python entrainer_yolo.py \
  --donnees "$DATASET" \
  --sorties /kaggle/working/sorties_yolo26 \
  --resume /kaggle/input/ancien-output/sorties_yolo26/entrainements/yolo_seg_fissures/weights/last.pt \
  --dispositif cuda
```

Pour reprendre Mask R-CNN dans une nouvelle session, ajoute l'ancien output comme
Input Kaggle puis lance :

```bash
!python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees "$DATASET" \
  --sorties /kaggle/working/sorties_maskrcnn \
  --epoques 100 \
  --lot 2 \
  --lr 3e-4 \
  --taille-image 384 \
  --dispositif cuda \
  --resume /kaggle/input/ancien-output/sorties_maskrcnn/modeles/dernier_modele.pth
```
