# Détection de Fissures Structurelles — Mask R-CNN et YOLOv11

Segmentation d'instance de fissures sur structures (béton, maçonnerie)
avec deux modèles entraînables :

- **Mask R-CNN** (`maskrcnn_resnet50_fpn_v2`) : modèle principal précis,
  entraîné directement depuis le dataset COCO.
- **YOLOv11-seg** (`yolo11n-seg.pt`, `yolo11s-seg.pt`, etc.) : modèle rapide
  entraîné via Ultralytics après conversion automatique COCO → YOLO-seg.

---

## Structure du projet

```
detection_fissures/
│
├── pyproject.toml              # Dépendances (UV)
├── entrainer.py                # Entraînement Mask R-CNN
├── entrainer_yolo.py           # Entraînement YOLOv11-seg
├── analyser.py                 # Script de classification post-entraînement
│
├── configuration/
│   └── parametres.py           # Hyperparamètres et chemins
│
├── donnees/
│   ├── jeu_donnees_coco.py     # Dataset PyTorch au format COCO
│   ├── chargeur.py             # DataLoaders avec collate_fn
│   └── conversion_yolo.py      # Conversion COCO vers YOLO-seg
│
├── modeles/
│   └── masque_rcnn.py          # Factory Mask R-CNN
│
├── entrainement/
│   ├── entraineur.py           # Boucle d'entraînement + early stopping
│   ├── pertes.py               # PerteCombineeMaskRCNN + UFC
│   └── metriques.py            # mAP COCO, IoU pixel, F1
│
├── analyse/
│   └── classificateur_fissures.py  # Classification par orientation et localisation
│
├── utilitaires/
│   ├── dispositif.py           # Détection CUDA/MPS/CPU
│   └── graine.py               # Reproductibilité
│
└── sorties/                    # Créé automatiquement
    ├── modeles/                # Checkpoints .pth
    └── journaux/               # Historique d'entraînement JSON
```

---

## Structure du dataset attendue

Le chemin par défaut du dataset est `dataset/` à la racine du projet.
Vous pouvez aussi fournir un chemin absolu ou relatif avec `--donnees`.

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image_001.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   └── ...
└── test/
    ├── _annotations.coco.json
    └── ...
```

Le fichier d'annotations doit être au format **COCO** (export Roboflow).
Pour Mask R-CNN, le dataset est utilisé tel quel.
Pour YOLOv11-seg, `entrainer_yolo.py` génère automatiquement une copie
YOLO-seg dans le dossier de sortie.
Les images sont fournies aux modèles sous forme de tenseurs float `[0, 1]`.
La normalisation attendue par Mask R-CNN est gérée par le transform interne
de `torchvision`.
Au démarrage, `entrainer.py` vérifie que `train/`, `valid/`, `test/`,
leurs fichiers `_annotations.coco.json` et les images référencées existent.

---

## Installation

```bash
# Avec UV (recommandé)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd detection_fissures
uv sync
```

Vérification locale :

```bash
uv run python -c "import torch, torchvision, torchmetrics, cv2, pycocotools; print('OK')"
uv run python entrainer.py --help
uv run python entrainer_yolo.py --help
```

Guides complets :

- Local avec uv : [documentation/guide_local_uv.md](documentation/guide_local_uv.md)
- Google Colab avec dataset sur Drive : [documentation/guide_colab.md](documentation/guide_colab.md)
- Kaggle : [documentation/guide_kaggle.md](documentation/guide_kaggle.md)
- Analyse de l'orientation : [documentation/guide_orientation.md](documentation/guide_orientation.md)

---

## Dataset, GitHub, Colab et Kaggle

- Ne pousse pas le dossier `dataset/` sur GitHub. Le dataset est ignoré dans `.gitignore`.
- Stocke ton dataset sur Google Drive, un stockage externe ou un lien Roboflow.
- Sur Colab, clone d'abord le projet GitHub dans `/content`, puis utilise Google Drive uniquement pour le dataset et les sorties.
- Indique le dossier du dataset avec `--donnees /content/drive/MyDrive/dataset`.
- Indique le dossier de sauvegarde avec `--sorties /content/drive/MyDrive/detection_fissures_sorties_*`.
- Si Colab coupe l'exécution, relance avec `--resume /content/drive/MyDrive/.../modeles/dernier_modele.pth`.
- Sur Kaggle, utilise `/kaggle/input/...` pour le dataset et `/kaggle/working/...`
  pour les sorties. Voir [documentation/guide_kaggle.md](documentation/guide_kaggle.md).

### Commandes complètes pour Google Colab

Le guide détaillé de bout en bout est ici :
[documentation/guide_colab.md](documentation/guide_colab.md).

Monter Google Drive pour accéder au dataset et sauvegarder les checkpoints :

```python
from google.colab import drive
drive.mount("/content/drive")
```

Cloner le projet GitHub dans l'environnement Colab :

```bash
cd /content
git clone https://github.com/Djochrist/detection_fissures.git
cd detection_fissures
```

Si le dossier existe déjà dans la session Colab, mets-le à jour :

```bash
cd /content/detection_fissures
git pull origin main
```

Installer les dépendances :

```bash
pip install -U torch torchvision opencv-python numpy scipy pycocotools "torchmetrics[detection]" rich ultralytics
```

Structure du dataset sur Drive, à passer au script avec `--donnees /content/drive/MyDrive/dataset` :

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

### Modèle 1 — Mask R-CNN

```bash
python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees /content/drive/MyDrive/dataset \
  --sorties /content/drive/MyDrive/detection_fissures_sorties_mask_v2 \
  --epoques 100 \
  --lot 4 \
  --lr 5e-5 \
  --taille-image 384 \
  --seuil-score 0.05 \
  --dispositif cuda
```

### Modèle 2 — YOLOv11-seg

```bash
python entrainer_yolo.py \
  --donnees /content/drive/MyDrive/dataset \
  --sorties /content/drive/MyDrive/detection_fissures_sorties_yolo11 \
  --modele yolo11n-seg.pt \
  --epoques 100 \
  --lot 8 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --patience 15 \
  --taille-image 384 \
  --save-period 5 \
  --dispositif cuda
```

Pour un GPU plus puissant, vous pouvez remplacer `yolo11n-seg.pt` par
`yolo11s-seg.pt`, `yolo11m-seg.pt`, `yolo11l-seg.pt` ou `yolo11x-seg.pt`.
Le script YOLO affiche maintenant un résumé du dataset converti, la configuration,
les métriques par époque quand Ultralytics les expose, puis un tableau validation/test
avec précision, rappel, F1 score et mAP.
Ajoutez `--silencieux` seulement si vous voulez réduire ces journaux.

### Reprendre un entraînement interrompu

Exemple pour reprendre YOLOv11-seg depuis le dernier checkpoint :

```bash
python entrainer_yolo.py \
  --donnees /content/drive/MyDrive/dataset \
  --sorties /content/drive/MyDrive/detection_fissures_sorties_yolo11 \
  --resume /content/drive/MyDrive/detection_fissures_sorties_yolo11/entrainements/yolo11_seg_fissures/weights/last.pt \
  --dispositif cuda
```

Exemple pour reprendre le modèle Mask R-CNN recommandé :

```bash
python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees /content/drive/MyDrive/dataset \
  --sorties /content/drive/MyDrive/detection_fissures_sorties_mask_v2 \
  --epoques 100 \
  --lot 4 \
  --lr 5e-5 \
  --taille-image 384 \
  --dispositif cuda \
  --resume /content/drive/MyDrive/detection_fissures_sorties_mask_v2/modeles/dernier_modele.pth
```

### Analyse après entraînement Mask R-CNN

Après l'entraînement, lancez `analyser.py` avec le checkpoint `.pth` et le dossier
d'images à analyser. Le script détecte les fissures, calcule leur orientation
(`horizontale`, `verticale`, `inclinée`) et peut exporter un rapport JSON.

Commande locale :

```bash
python analyser.py \
  --modele sorties/modeles/meilleur_modele.pth \
  --images photos_test/ \
  --seuil 0.4 \
  --sortie resultats_analyse.json
```

Commande Google Colab / Drive :

```bash
python analyser.py \
  --modele /content/drive/MyDrive/detection_fissures_sorties_mask_v2/modeles/meilleur_modele.pth \
  --images /content/drive/MyDrive/images_a_tester \
  --sortie /content/drive/MyDrive/resultats_fissures.json \
  --seuil 0.5
```

Les résultats YOLOv11 sont sauvegardés par Ultralytics dans
`detection_fissures_sorties_yolo11/entrainements/`.

---

## Utilisation

### 1. Entraînement
```bash
python entrainer.py --architecture maskrcnn_resnet50_fpn_v2 --donnees /chemin/vers/dataset --dispositif cuda
python entrainer_yolo.py --donnees /chemin/vers/dataset --modele yolo11n-seg.pt --dispositif cuda
```

Les checkpoints conservent l'architecture utilisée, ce qui permet à `analyser.py`
de reconstruire automatiquement le bon modèle pendant l'inférence.

### 2. Classification des fissures (après entraînement)
```bash
# Analyser un dossier d'images et afficher les résultats dans le terminal
python analyser.py --modele sorties/modeles/meilleur_modele.pth --images photos/

# Exporter un rapport JSON complet
python analyser.py --modele sorties/modeles/meilleur_modele.pth --images photos/ --sortie resultats.json

# Ajuster le seuil de confiance (0.4 = plus de détections, 0.7 = plus strict)
python analyser.py --modele modele.pth --images photos/ --seuil 0.4
```

Pour comprendre comment l'orientation est calculée après la détection, voir
[documentation/guide_orientation.md](documentation/guide_orientation.md).

---

## Stratégie d'entraînement — 3 phases

```
Époque 1–5   : Backbone GELÉ → entraînement têtes uniquement
Époque 5–15  : Dégelage layer3/layer4 → fine-tuning features haut niveau
Époque 15+   : Dégelage complet → fine-tuning global
```

Anti-overfitting : AdamW + weight decay, gradient clipping, early stopping, précision mixte float16.

---

## Classification des fissures

Après entraînement, chaque fissure détectée est classifiée selon deux axes :

### Orientation (analyse PCA sur les pixels du masque)
| Type | Angle de l'axe principal | Signification |
|---|---|---|
| **Horizontale** | < 20° | Parallèle au sol — souvent tassement différentiel |
| **Verticale** | > 70° | Perpendiculaire au sol — souvent flexion ou retrait |
| **Inclinée** | 20° – 70° | Cisaillement — souvent le plus dangereux |

### Localisation / Profondeur (analyse morphologique du masque)
| Type | Critère | Danger structurel |
|---|---|---|
| **Superficielle** | Largeur < 3.5 px | Faible — enduit/peinture uniquement |
| **Profonde** | Largeur > 7 px | Élevé — pénètre dans le matériau porteur |
| **Transversale** | Bbox > 65% de l'image | Critique — traverse tout l'élément |

La largeur est estimée via la **distance transform** (rayon médian × 2 = largeur locale).
Un **indice de danger composite** [0, 1] est calculé pour chaque fissure.

---

## Métriques calculées

- mAP [0.5:0.95] (standard COCO)
- mAP@IoU=0.50 et mAP@IoU=0.75
- Précision, Rappel, F1-Score pixel

---

## Compatibilité

- Linux local, Google Colab (T4/A100), Kaggle GPU
- CUDA 11.8+, Apple Silicon (MPS), CPU (fallback)
