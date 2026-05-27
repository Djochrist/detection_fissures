# Détection de Fissures Structurelles

Segmentation d'instance automatique de fissures dans les matériaux de construction (béton, maçonnerie) par deep learning. Deux architectures sont fournies : **Mask R-CNN** (précision maximale) et **YOLO11-seg** (rapidité).

---

## Table des matières

1. [Architecture du projet](#architecture-du-projet)
2. [Prérequis et installation](#prérequis-et-installation)
3. [Préparation du dataset](#préparation-du-dataset)
4. [Entraînement Mask R-CNN](#entraînement-mask-r-cnn)
5. [Entraînement YOLO11-seg](#entraînement-yolo11-seg)
6. [Analyse et classification](#analyse-et-classification)
7. [Paramètres expliqués](#paramètres-expliqués)
8. [Reprendre un entraînement](#reprendre-un-entraînement)
9. [Résultats attendus](#résultats-attendus)
10. [Structure des sorties](#structure-des-sorties)
11. [Dépannage](#dépannage)

---

## Architecture du projet

```
detection-fissures/
│
├── entrainer.py                   # Entraînement Mask R-CNN (commande principale)
├── entrainer_yolo.py              # Entraînement YOLO11-seg (commande principale)
├── analyser.py                    # Analyse + classification des fissures détectées
├── main.py                        # Point d'entrée — affiche toutes les commandes
│
├── configuration/
│   └── parametres.py              # Hyperparamètres centralisés et justifiés
│
├── donnees/
│   ├── jeu_donnees_coco.py        # Dataset PyTorch (COCO → Mask R-CNN)
│   ├── chargeur.py                # DataLoader avec collate_fn adapté
│   └── conversion_yolo.py         # Conversion COCO → YOLO-seg (labels + data.yaml)
│
├── modeles/
│   └── masque_rcnn.py             # Mask R-CNN construction + stratégie 3 phases
│
├── entrainement/
│   ├── entraineur.py              # Boucle d'entraînement Mask R-CNN
│   ├── metriques.py               # mAP, précision, rappel, F1 (torchmetrics + COCO)
│   └── pertes.py                  # Pertes personnalisées (Dice, Focal)
│
├── analyse/
│   └── classificateur_fissures.py # Classification PCA + distance transform + danger
│
└── utilitaires/
    ├── ajouter_images_saines.py   # Ajout d'images non annotées au dataset COCO
    ├── dispositif.py              # Détection GPU/MPS/CPU
    ├── graine.py                  # Fixation de la graine aléatoire
    └── images.py                  # Chargement, redimensionnement, conversion tenseur
```

### Comparaison des architectures

| Critère              | Mask R-CNN ResNet50-FPN-v2     | YOLO11-seg (small)          |
|----------------------|--------------------------------|-----------------------------|
| Paramètres           | ~44 M                          | ~11 M                       |
| Vitesse inférence    | ~0.5 s/image (CPU)             | ~0.05 s/image (CPU)         |
| Précision masque     | Très haute                     | Haute                       |
| Fissures fines       | Excellente (FPN multi-échelle) | Bonne (mask_ratio=1)        |
| Déploiement          | PyTorch natif                  | ONNX, TensorRT, CoreML      |
| Recommandé pour      | Analyse hors-ligne             | Inspection temps réel       |

---

## Prérequis et installation

### Python et dépendances

```bash
# CPU uniquement
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU NVIDIA (recommandé pour YOLO)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dépendances communes
pip install ultralytics torchmetrics pycocotools
pip install opencv-python-headless numpy scipy rich
```

### Installation optionnelle par extras
Pour limiter l'installation aux dépendances nécessaires, utilisez les extras du package :

```bash
pip install -e .          # installation minimale
pip install -e .[yolo]     # installe YOLO et ultralytics
pip install -e .[maskrcnn] # installe Mask R-CNN + dépendances PyTorch
pip install -e .[full]     # installe l'ensemble complet
```

### Vérifier l'installation et afficher les commandes

```bash
python main.py
```

---

## Préparation du dataset

### Format attendu (export Roboflow COCO)

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image001.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   └── ...
└── test/
    ├── _annotations.coco.json
    └── ...
```

Ce projet utilise des images **384×384 pixels** (format standard Roboflow).  
La structure COCO requiert 2 classes : `0 = fond`, `1 = fissure`.

### Ajouter des images de murs sains (exemples négatifs)

Les images non annotées (murs sans fissures) sont prises en charge nativement.  
Mask R-CNN les reçoit avec des cibles vides (`N=0`). YOLO les reçoit avec un fichier `.txt` vide.

**Aperçu sans modifier :**
```bash
python utilitaires/ajouter_images_saines.py \
  --images-saines murs_sains/ \
  --dataset       dataset/ \
  --apercu
```

**Ajouter au split train :**
```bash
python utilitaires/ajouter_images_saines.py \
  --images-saines murs_sains/ \
  --dataset       dataset/
```

**Répartir automatiquement (80 % train / 10 % valid / 10 % test) :**
```bash
python utilitaires/ajouter_images_saines.py \
  --images-saines murs_sains/ \
  --dataset       dataset/ \
  --repartir
```

---

## Entraînement Mask R-CNN

### Commande complète pour ce projet

```bash
python entrainer.py \
  --donnees            dataset/ \
  --epoques            60 \
  --lot                2 \
  --lr                 0.0001 \
  --patience           15 \
  --taille-image       384 \
  --seuil-score        0.05 \
  --architecture       maskrcnn_resnet50_fpn_v2 \
  --dispositif         auto \
  --graine             42 \
  --sorties            sorties \
  --decroissance-poids 0.0005
```

> **Sur GPU 8 Go :** `--lot 4 --dispositif cuda`  
> **Sur GPU 16 Go :** `--lot 8 --dispositif cuda`

La commande complète est également affichée au démarrage et à la fin de l'entraînement.

### Stratégie d'entraînement en 3 phases

| Phase | Époques | Couches actives             | Objectif                              |
|-------|---------|-----------------------------|---------------------------------------|
| 1     | 1–5     | Têtes seulement             | Convergence rapide des prédicteurs    |
| 2     | 5–15    | Backbone layer3/layer4      | Fine-tuning features fissures         |
| 3     | 15–60   | Tout le réseau              | Fine-tuning complet + early stopping  |

### Structure des sorties

```
sorties/
├── modeles/
│   ├── meilleur_modele.pth         # Modèle avec le meilleur mAP@0.5 validation
│   └── dernier_modele.pth          # Checkpoint de la dernière époque
└── journaux/
    └── historique_entrainement.json
```

---

## Entraînement YOLO11-seg

### Commande complète pour ce projet

```bash
python entrainer_yolo.py \
  --donnees          dataset/ \
  --modele           yolo11s-seg.pt \
  --taille-image     384 \
  --epoques          150 \
  --lot              8 \
  --lr               0.001 \
  --lrf              0.01 \
  --weight-decay     0.0005 \
  --patience         50 \
  --warmup-epoques   5.0 \
  --close-mosaic     30 \
  --mask-ratio       1 \
  --overlap-mask \
  --copy-paste       0.4 \
  --mosaic           1.0 \
  --mixup            0.0 \
  --degrees          45.0 \
  --translate        0.1 \
  --scale            0.5 \
  --fliplr           0.5 \
  --flipud           0.0 \
  --hsv-h            0.02 \
  --hsv-s            0.5 \
  --hsv-v            0.4 \
  --cos-lr \
  --freeze           0 \
  --amp \
  --workers          2 \
  --save-period      10 \
  --dispositif       auto \
  --nom              yolo11_seg_fissures \
  --sorties          sorties_yolo
```

La commande complète est affichée au démarrage et à la fin de l'entraînement.

### Conversion COCO → YOLO automatique

Le script convertit automatiquement votre dataset COCO avant l'entraînement :

```
sorties_yolo/dataset_yolo/
├── images/{train,valid,test}/   → symlinks vers dataset/
├── labels/{train,valid,test}/   → fichiers .txt YOLO-seg (vide = mur sain)
└── data.yaml                    → configuration Ultralytics
```

### Structure des sorties

```
sorties_yolo/
└── entrainements/
    └── yolo11_seg_fissures/
        ├── weights/
        │   ├── best.pt            # Meilleur mAP validation
        │   └── last.pt            # Dernier checkpoint
        ├── results.csv            # Courbes numériques
        └── *.png                  # Courbes, matrice de confusion
```

---

## Analyse et classification

### Analyser avec Mask R-CNN

```bash
python analyser.py \
  --modele   sorties/modeles/meilleur_modele.pth \
  --backend  maskrcnn \
  --images   dataset/test/ \
  --seuil    0.40
```

### Analyser avec YOLO11-seg

```bash
python analyser.py \
  --modele   sorties_yolo/entrainements/yolo11_seg_fissures/weights/best.pt \
  --backend  yolo \
  --images   dataset/test/ \
  --seuil    0.25
```

### Sauvegarder les résultats (JSON)

```bash
python analyser.py \
  --modele   sorties/modeles/meilleur_modele.pth \
  --backend  maskrcnn \
  --images   dataset/test/ \
  --seuil    0.40 \
  --sortie   resultats.json
```

### Classification automatique de chaque fissure

| Champ        | Valeurs possibles                              | Méthode                          |
|--------------|------------------------------------------------|----------------------------------|
| Orientation  | horizontale / verticale / inclinée / inconnue  | PCA sur les pixels du masque     |
| Localisation | superficielle / profonde / transversale        | Distance transform (largeur max) |
| Danger       | 0.0 → 1.0                                      | Indice composite                 |

**Formule de l'indice de danger :**
```
danger = 0.5 × (largeur normalisée)
       + 0.3 × (longueur normalisée)
       + 0.2 × (excentricité PCA)
```

**Interprétation :**

| Indice      | Niveau        | Action recommandée              |
|-------------|---------------|---------------------------------|
| 0.0 – 0.30  | Faible        | Surveillance annuelle           |
| 0.30 – 0.60 | Modéré        | Inspection à 6 mois             |
| 0.60 – 1.0  | Critique      | Intervention structurelle       |

### Conseil sur le seuil de détection

> **Inspection de sécurité** (ne rien manquer) → `--seuil 0.10`  
> **Rapport standard** (moins de faux positifs) → `--seuil 0.40` (Mask R-CNN) / `--seuil 0.25` (YOLO)

---

## Paramètres expliqués

### Mask R-CNN — justification par paramètre

| Paramètre            | Valeur  | Pourquoi cette valeur pour les fissures                             |
|----------------------|---------|---------------------------------------------------------------------|
| `--epoques`          | 60      | Minimum pour les 3 phases (5+10+45) + early stopping               |
| `--lot`              | 2       | Mask R-CNN charge des masques [N,H,W] → 3–4 Go RAM par lot sur CPU |
| `--lr`               | 0.0001  | Fine-tuning COCO ; backbone entraîné à lr/10 = 1e-5                |
| `--patience`         | 15      | Transitions de phases = creux temporaires de mAP → 15 de marge     |
| `--taille-image`     | 384     | Identique au dataset Roboflow (ne pas modifier)                     |
| `--seuil-score`      | 0.05    | Standard COCO mAP (courbe PR complète). Inférence : utiliser 0.30–0.50 |
| `--decroissance-poids` | 0.0005 | L2 AdamW essentiel pour datasets < 5000 images (anti-overfitting) |

### YOLO11-seg — justification par paramètre

| Paramètre            | Valeur  | Pourquoi cette valeur pour les fissures                             |
|----------------------|---------|---------------------------------------------------------------------|
| `--modele`           | yolo11s | Small > nano pour objets fins ; medium overkill sans GPU dédié      |
| `--taille-image`     | 384     | Multiple de 32 ✓, identique au dataset, pas 640 (upscale inutile)  |
| `--mask-ratio`       | 1       | **CRITIQUE** : 384/4=96px avec le défaut → fissures de 1–2px perdues |
| `--copy-paste`       | 0.4     | Augmentation objet rare : fissures < 5 % des pixels                |
| `--degrees`          | 45.0    | Fissures à 0°, 45°, 90° et tous angles structurels                 |
| `--flipud`           | 0.0     | Flip vertical = inversion du contexte gravitationnel : irréaliste   |
| `--mixup`            | 0.0     | Mixup brouille les contours fins des masques de fissures            |
| `--close-mosaic`     | 30      | 30 dernières époques sans mosaic → adaptation aux images réelles    |
| `--patience`         | 50      | Convergence longue sur objets rares et petits                       |
| `--epoques`          | 150     | Plus long que YOLO standard (100) pour objets difficiles            |
| `--hsv-v`            | 0.4     | Béton sec vs humide vs ombre → forte variation de luminosité        |
| `--hsv-s`            | 0.5     | Surface sèche (désaturée) vs humide (saturée)                       |
| `--cos-lr`           | true    | Cosine annealing > linéaire pour petits datasets                    |
| `--warmup-epoques`   | 5.0     | Démarrage progressif du LR pour stabiliser les premières mises à jour |

---

## Reprendre un entraînement

### Mask R-CNN

```bash
python entrainer.py \
  --donnees            dataset/ \
  --epoques            60 \
  --lot                2 \
  --lr                 0.0001 \
  --patience           15 \
  --taille-image       384 \
  --seuil-score        0.05 \
  --architecture       maskrcnn_resnet50_fpn_v2 \
  --dispositif         auto \
  --graine             42 \
  --sorties            sorties \
  --decroissance-poids 0.0005 \
  --resume             sorties/modeles/dernier_modele.pth
```

### YOLO11-seg

```bash
python entrainer_yolo.py \
  --resume sorties_yolo/entrainements/yolo11_seg_fissures/weights/last.pt
```

---

## Résultats attendus

Les métriques dépendent de la taille et de la qualité du dataset.

| Taille dataset | mAP@0.5 masque | Rappel masque |
|----------------|----------------|---------------|
| < 500 images   | 0.35 – 0.55    | 0.50 – 0.65   |
| 500 – 2000     | 0.55 – 0.70    | 0.65 – 0.80   |
| > 2000 images  | 0.70 – 0.85    | 0.75 – 0.90   |

> **Priorité absolue : le RAPPEL.** Une fissure non détectée est plus dangereuse qu'un faux positif. Si le rappel est inférieur à 0.60, réduire le seuil d'inférence à 0.10.

---

## Structure des sorties

```
sorties/                                  Mask R-CNN
├── modeles/
│   ├── meilleur_modele.pth
│   └── dernier_modele.pth
└── journaux/
    └── historique_entrainement.json

sorties_yolo/                             YOLO11-seg
├── dataset_yolo/
│   ├── images/{train,valid,test}/
│   ├── labels/{train,valid,test}/
│   └── data.yaml
├── entrainements/
│   └── yolo11_seg_fissures/
│       ├── weights/{best.pt,last.pt}
│       ├── results.csv
│       └── *.png
└── evaluations/
    └── yolo11_seg_fissures_test/
```

---

## Dépannage

| Problème                          | Solution                                                            |
|-----------------------------------|---------------------------------------------------------------------|
| `CUDA out of memory`              | Réduire `--lot` : 8 → 4 → 2 → 1                                   |
| `FileNotFoundError annotations`   | Vérifier que `dataset/train/_annotations.coco.json` existe         |
| `mAP = 0.00`                      | Vérifier annotations COCO (champ `segmentation` requis)            |
| Rappel < 0.50                     | Réduire le seuil : `python analyser.py --seuil 0.10`               |
| Fissures fines mal segmentées     | Vérifier `--mask-ratio 1` (défaut YOLO=4 est insuffisant)          |
| Convergence très lente sur CPU    | Normal : ~2–5 min/époque Mask R-CNN, ~1–3 min/époque YOLO          |
| `ModuleNotFoundError ultralytics` | `pip install -U ultralytics`                                       |

---

## Dépendances

| Paquet                    | Rôle                                       |
|---------------------------|--------------------------------------------|
| `torch`                   | Deep learning (CPU ou CUDA)                |
| `torchvision`             | Mask R-CNN, transformations images         |
| `ultralytics`             | YOLO11-seg                                 |
| `torchmetrics`            | mAP, précision, rappel, F1                 |
| `pycocotools`             | Décodage RLE, évaluation COCO              |
| `opencv-python-headless`  | Lecture images, distance transform         |
| `numpy`                   | Calculs numériques                         |
| `scipy`                   | Analyse composantes connexes               |
| `rich`                    | Affichage terminal formaté                 |
