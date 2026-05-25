# Détection de Fissures Structurelles par Segmentation d'Instance

Détection et classification automatique de fissures dans les matériaux de construction (béton, maçonnerie) par apprentissage profond. Deux architectures sont implémentées et comparables :

| Architecture | Fichier d'entraînement | Cadre | Vitesse | Précision |
|---|---|---|---|---|
| **Mask R-CNN** ResNet-50-FPN-v2 | `entrainer.py` | torchvision | Lente | Haute |
| **YOLO11-seg** (n/s/m/l/x) | `entrainer_yolo.py` | ultralytics | Rapide | Haute |

---

## Sommaire

1. [Structure du projet](#structure-du-projet)
2. [Installation](#installation)
3. [Format du dataset](#format-du-dataset)
4. [Entraînement Mask R-CNN](#entraînement-mask-r-cnn)
5. [Entraînement YOLO11-seg](#entraînement-yolo11-seg)
6. [Analyse post-entraînement](#analyse-post-entraînement)
7. [Métriques — référence complète](#métriques--référence-complète)
8. [Bonnes pratiques et réglages](#bonnes-pratiques-et-réglages)
9. [Dépannage](#dépannage)

---

## Structure du projet

```
detection-fissures/
├── entrainer.py              # Script principal Mask R-CNN
├── entrainer_yolo.py         # Script principal YOLO11-seg
├── analyser.py               # Analyse et classification post-inférence
│
├── configuration/
│   └── parametres.py         # Hyperparamètres et chemins centralisés
│
├── donnees/
│   ├── jeu_donnees_coco.py   # Dataset PyTorch (format COCO → Mask R-CNN)
│   └── conversion_yolo.py    # Conversion COCO → YOLO-seg (labels + data.yaml)
│
├── modeles/
│   └── masque_rcnn.py        # Construction Mask R-CNN + stratégie de gel backbone
│
├── entrainement/
│   ├── entraineur.py         # Boucle d'entraînement Mask R-CNN
│   ├── metriques.py          # Calcul mAP, précision, rappel, F1 (torchmetrics)
│   └── pertes.py             # Losses personnalisées (Dice, Focal)
│
├── analyse/
│   └── classificateur_fissures.py  # Classification orientation/sévérité par PCA
│
└── utilitaires/
    ├── materiel.py           # Détection GPU/MPS/CPU
    ├── reproductibilite.py   # Graine aléatoire
    └── images.py             # Chargement, redimensionnement, tenseur
```

---

## Installation

### Prérequis

- Python 3.11 (géré via Nix dans l'environnement Replit)
- `uv` (gestionnaire de paquets rapide)

### Création de l'environnement

```bash
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
```

### Vérification

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -c "from ultralytics import YOLO; print('YOLO OK')"
```

---

## Format du dataset

Le projet attend un dataset au **format COCO** avec segmentation d'instance (export Roboflow recommandé) :

```
mon_dataset/
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

### Recommandations dataset

| Paramètre | Minimum | Recommandé |
|---|---|---|
| Images d'entraînement | 500 | 2 000+ |
| Images de validation | 100 | 400+ |
| Images de test | 50 | 200+ |
| Résolution | 320×320 | 640×640 |
| Instances par image | 1 | 3–10 |

**Sources de datasets annotés :**
- [Roboflow Universe — crack detection](https://universe.roboflow.com/search?q=crack+segmentation)
- [COCO-format crack datasets (Zenodo)](https://zenodo.org)

**Conseils d'annotation :**
- Annoter toutes les fissures visibles, même les plus fines (réduit les faux négatifs)
- Utiliser des polygones précis plutôt que des bounding boxes
- Inclure des fissures de différentes largeurs, orientations et éclairages
- Exporter en COCO instance segmentation (pas semantic segmentation)

---

## Entraînement Mask R-CNN

### Commande de base

```bash
.venv/bin/python entrainer.py \
    --donnees /chemin/vers/dataset \
    --epoques 50 \
    --architecture maskrcnn_resnet50_fpn_v2
```

### Tous les arguments

```
Arguments obligatoires :
  --donnees PATH              Dossier racine du dataset COCO
                              (doit contenir train/, valid/, test/)

Arguments d'entraînement :
  --epoques N                 Nombre d'époques (défaut: 30)
  --lot N                     Taille de lot (défaut: 4 ; réduire si OOM GPU)
  --taille-image N            Résolution (défaut: 384)
  --lr FLOAT                  Taux d'apprentissage initial AdamW (défaut: 1e-4)
  --decroissance-poids FLOAT  Régularisation L2 AdamW (défaut: 1e-4)
  --patience N                Patience early stopping sur mAP val (défaut: 10)
  --graine N                  Graine pour reproductibilité (défaut: 42)
  --seuil-score FLOAT         Seuil de confiance minimum à l'inférence (défaut: 0.5)
  --sans-mixte                Désactiver la précision mixte float16

Modèle :
  --architecture STR          Backbone (défaut: maskrcnn_resnet50_fpn_v2)
  --poids STR                 Poids préentraînés : DEFAULT | NONE (défaut: DEFAULT)

Chemins :
  --sorties PATH              Dossier de sortie (défaut: sorties/)
  --resume PATH               Reprendre depuis un checkpoint .pth

Exemples :
  # Entraînement standard, GPU, 50 époques
  .venv/bin/python entrainer.py --donnees ./dataset --epoques 50

  # Fine-tuning avec lr réduit
  .venv/bin/python entrainer.py --donnees ./dataset --lr 5e-5 --epoques 30

  # Reprise depuis checkpoint
  .venv/bin/python entrainer.py --donnees ./dataset --resume sorties/modeles/meilleur.pth
```

### Stratégie d'entraînement (3 phases)

Le modèle utilise un **dégelage progressif du backbone** :

| Phase | Époques | Couches entraînées | Objectif |
|---|---|---|---|
| 1 — Têtes seulement | 1 – 5 | Têtes détection + masque | Convergence rapide |
| 2 — Dégelage partiel | 6 – 15 | + layer3, layer4 ResNet | Fine-tuning haut niveau |
| 3 — Dégelage complet | 16+ | Toutes les couches | Fine-tuning fin |

---

## Entraînement YOLO11-seg

### Commande de base

```bash
.venv/bin/python entrainer_yolo.py \
    --donnees /chemin/vers/dataset \
    --modele yolo11s-seg.pt \
    --epoques 100
```

La conversion COCO → YOLO est effectuée automatiquement avant l'entraînement.

### Modèles disponibles

| Modèle | Paramètres | Vitesse | mAP (COCO) | Usage recommandé |
|---|---|---|---|---|
| `yolo11n-seg.pt` | 2.9 M | Très rapide | Faible | Test rapide, edge device |
| `yolo11s-seg.pt` | 9.9 M | Rapide | Moyen | Équilibre vitesse/précision |
| `yolo11m-seg.pt` | 20.1 M | Moyen | Bon | Recommandé pour fissures |
| `yolo11l-seg.pt` | 26.2 M | Lent | Très bon | Dataset 2 000+ images |
| `yolo11x-seg.pt` | 56.9 M | Très lent | Excellent | Dataset 5 000+ images |

### Tous les arguments

```
Arguments obligatoires :
  --donnees PATH              Dossier racine du dataset COCO

Modèle :
  --modele STR                Poids YOLO11 (défaut: yolo11s-seg.pt)

Entraînement général :
  --epoques N                 Époques totales (défaut: 100)
  --lot N                     Batch size (-1 = autobatch GPU, défaut: 8)
  --taille-image N            Résolution multiple de 32 (défaut: 640)
  --patience N                Patience early stopping (défaut: 20)
  --workers N                 Workers DataLoader (défaut: 2)

Taux d'apprentissage :
  --lr FLOAT                  lr0 : LR initial (défaut: 1e-3)
  --lrf FLOAT                 Ratio LR final = lr0 × lrf (défaut: 0.01)
                              Le LR décroît de lr0 → lr0×lrf par cosine annealing
  --cos-lr                    Scheduler cosine (activé par défaut)
  --warmup-epoques FLOAT      Époques de montée linéaire du LR (défaut: 3)
  --weight-decay FLOAT        Régularisation L2 (défaut: 5e-4)

Masques :
  --mask-ratio N              Sous-échantillonnage masques : 1=full, 4=défaut
                              Réduire à 1 pour améliorer la précision des contours
  --overlap-mask              Autoriser masques superposés (défaut: activé)

Augmentation :
  --close-mosaic N            Désactiver mosaic N époques avant fin (défaut: 10)
                              Stabilise l'entraînement sur les dernières époques
  --copy-paste FLOAT          Probabilité copy-paste [0–1] (défaut: 0.2)
                              Colle des instances de fissures dans d'autres images
  --mosaic FLOAT              Probabilité mosaic [0–1] (défaut: 1.0)
  --mixup FLOAT               Probabilité mixup [0–1] (défaut: 0.0)
  --degrees FLOAT             Rotation aléatoire ±N degrés (défaut: 10.0)
  --translate FLOAT           Translation aléatoire (fraction image, défaut: 0.1)
  --scale FLOAT               Mise à l'échelle aléatoire ±N (défaut: 0.5)
  --fliplr FLOAT              Flip horizontal (défaut: 0.5)
  --flipud FLOAT              Flip vertical (défaut: 0.0)
  --hsv-h FLOAT               Variation teinte HSV (défaut: 0.015)
  --hsv-s FLOAT               Variation saturation HSV (défaut: 0.7)
  --hsv-v FLOAT               Variation luminosité HSV (défaut: 0.4)

Backbone :
  --freeze N                  Geler N premières couches backbone (défaut: 0)
                              Utile si dataset < 500 images

Performances :
  --amp                       Précision mixte AMP (défaut: activé)

Contrôle :
  --dispositif STR            auto | cuda | mps | cpu (défaut: auto)
  --nom STR                   Nom de l'expérience dans le dossier sorties/
  --save-period N             Sauvegarder checkpoint toutes les N époques
  --exist-ok                  Réutiliser un dossier d'expérience existant
  --resume PATH               Reprendre depuis un checkpoint last.pt
  --convertir-seulement       Convertir COCO→YOLO sans entraîner
  --silencieux                Réduire les logs Ultralytics

Exemples :
  # Dataset standard ~1000 images
  .venv/bin/python entrainer_yolo.py --donnees ./dataset --modele yolo11s-seg.pt

  # Dataset large, GPU puissant
  .venv/bin/python entrainer_yolo.py \
      --donnees ./dataset --modele yolo11m-seg.pt \
      --epoques 150 --lot 16 --taille-image 640

  # Fine-tuning avec backbone partiellement gelé (petit dataset)
  .venv/bin/python entrainer_yolo.py \
      --donnees ./dataset --modele yolo11s-seg.pt \
      --freeze 10 --lr 5e-4 --epoques 80 --copy-paste 0.3

  # Reprendre un entraînement interrompu
  .venv/bin/python entrainer_yolo.py \
      --donnees ./dataset \
      --resume sorties_yolo/entrainements/yolo11_seg_fissures/weights/last.pt

  # Conversion dataset seulement (sans entraînement)
  .venv/bin/python entrainer_yolo.py \
      --donnees ./dataset --convertir-seulement
```

---

## Analyse post-entraînement

Le script `analyser.py` applique un modèle entraîné sur des images et classifie chaque fissure détectée selon :
- **Orientation** : horizontale / verticale / inclinée (PCA sur les pixels du masque)
- **Sévérité** : superficielle / profonde / transversale (distance transform)
- **Indice de danger** : score composite [0, 1] pondéré par l'orientation et la sévérité

```bash
.venv/bin/python analyser.py \
    --modele sorties_yolo/entrainements/yolo11_seg_fissures/weights/best.pt \
    --images /chemin/vers/photos \
    --sortie resultats.json
```

### Interprétation de l'indice de danger

| Indice | Icône | Signification |
|---|---|---|
| 0.00 – 0.35 | `▓░░` | Fissure superficielle — surveillance périodique |
| 0.35 – 0.65 | `▓▓░` | Fissure profonde — inspection approfondie recommandée |
| 0.65 – 1.00 | `▓▓▓` | Fissure critique — intervention urgente |

---

## Métriques — référence complète

Ces métriques sont affichées dans le terminal à la fin de chaque évaluation (validation et test). Une légende explicative est automatiquement imprimée après chaque tableau.

---

### Métriques Mask R-CNN

#### `mAP [0.5:0.95]` — mAP standard COCO

**Définition :** Moyenne des mAP calculées sur 10 seuils IoU : 0.50, 0.55, 0.60, …, 0.95.

**IoU (Intersection over Union) :** Pour un masque prédit M_pred et un masque annoté M_gt :

```
IoU = |M_pred ∩ M_gt| / |M_pred ∪ M_gt|
```

Une prédiction est un **vrai positif (VP)** si son IoU dépasse le seuil ; sinon c'est un **faux positif (FP)**.

**Pourquoi c'est exigeant :** Un masque qui recouvre correctement la fissure mais dont les bords débordent de 30 % passera le seuil IoU=0.50 mais échouera à IoU=0.75. La moyenne sur dix seuils pénalise donc les masques imprécis au niveau du pixel.

**Cibles :**
- < 0.25 : Insuffisant
- 0.25 – 0.40 : Acceptable pour fissures complexes
- 0.40 – 0.55 : Bon
- > 0.55 : Excellent

---

#### `mAP@0.5` — Métrique de détection principale

**Définition :** mAP avec un seuil IoU unique à 0.50.

**Calcul :**
1. Trier toutes les prédictions du dataset par score de confiance décroissant
2. Calculer précision et rappel cumulatifs → courbe PR
3. AP = aire sous la courbe PR interpolée en 101 points
4. mAP = moyenne des AP sur toutes les classes (ici : 1 classe = fissure)

**Usage :** Métrique de référence pour comparer ce modèle à d'autres publications.

**Cibles :**
- < 0.40 : Insuffisant
- 0.40 – 0.55 : Acceptable
- 0.55 – 0.70 : Bon
- > 0.70 : Excellent

---

#### `mAP@0.75` — Précision fine des contours

**Définition :** Même calcul que mAP@0.50 mais seuil IoU porté à 0.75.

**Interprétation de l'écart mAP50 − mAP75 :**
- Écart < 0.10 : Excellente qualité des contours de masques
- Écart > 0.20 : Contours imprécis — augmenter `--taille-image`, réduire `--mask-ratio`
- Écart > 0.30 : Masques grossiers même si les boîtes sont correctes

---

#### `mAR@100` — Rappel maximal COCO

**Définition :** Rappel moyen en autorisant jusqu'à 100 détections par image, calculé sur IoU 0.50→0.95.

**Diagnostic :** mAR élevé + mAP faible → le modèle *trouve* les fissures mais les score avec une mauvaise confiance. Solution : réduire `--seuil-score`.

---

#### `Précision` — Taux de vraies détections (pixel)

```
Précision = VP_pixels / (VP_pixels + FP_pixels)
```

Où VP_pixels = pixels prédits fissure qui sont bien annotés fissure,
et FP_pixels = pixels prédits fissure qui sont en réalité fond.

**Précision faible (< 0.50) :** Trop de fausses alarmes. Actions :
- Augmenter le seuil de score (`--seuil-score 0.6`)
- Vérifier si le dataset contient des textures similaires aux fissures (joints, ombres, fils)

---

#### `Rappel` — Taux de fissures détectées (pixel)

```
Rappel = VP_pixels / (VP_pixels + FN_pixels)
```

Où FN_pixels = pixels annotés fissure mais prédits fond.

**Rappel faible (< 0.50) :** Des fissures sont manquées — dangereux pour l'inspection structurelle. Actions :
- Réduire le seuil de score (`--seuil-score 0.3`)
- Vérifier les annotations (fissures manquées dans les labels)
- Augmenter le dataset avec des fissures fines

**Priorité :** Dans le contexte de la sécurité structurelle, **le rappel est prioritaire sur la précision**. Mieux vaut une fausse alarme qu'une fissure critique manquée.

---

#### `F1 Score` — Bilan global

```
F1 = 2 × Précision × Rappel / (Précision + Rappel)
```

**Cibles :**
- < 0.50 : Insuffisant
- 0.50 – 0.65 : Acceptable
- 0.65 – 0.75 : Bon
- > 0.75 : Excellent

---

### Métriques YOLO11-seg

YOLO calcule des métriques séparées pour les **masques de segmentation** et les **boîtes englobantes**. Les métriques masque sont les plus importantes.

---

#### `mAP@0.5 masque` — Métrique principale YOLO

Identique à `mAP@0.50` Mask R-CNN mais calculée sur les masques polygonaux YOLO après reconstruction. C'est la **métrique de comparaison inter-architectures**.

---

#### `mAP@0.5:0.95 masque`

Équivalent YOLO du `mAP[0.5:0.95]` COCO. Pénalise les masques dont les contours s'écartent de l'annotation.

**Diagnostic :** si `mAP@0.5 masque` est bon mais `mAP@0.5:0.95 masque` est faible (écart > 0.20) :
- Augmenter `--mask-ratio 1` (masques pleine résolution)
- Augmenter `--taille-image 640` ou supérieur
- Augmenter `--epoques` (les masques précis nécessitent plus d'entraînement)

---

#### `mAP@0.5 boîte` vs `mAP@0.5 masque`

| Situation | Interprétation |
|---|---|
| Boîte ≈ Masque (écart < 0.10) | Architecture équilibrée |
| Boîte >> Masque (écart > 0.15) | Localisation bonne, masques imprécis — backbone OK, tête segmentation à améliorer |
| Masque >> Boîte | Cas rare ; vérifier les annotations de boîtes |

---

#### `Précision masque` et `Rappel masque`

Identiques aux définitions Mask R-CNN ci-dessus, calculées sur les masques YOLO après seuillage à 0.5.

---

#### `F1 score masque`

Calculé depuis précision masque et rappel masque. Référence unique pour comparer rapidement deux runs.

---

### Récapitulatif des cibles

| Métrique | Acceptable | Bon | Excellent |
|---|---|---|---|
| mAP [0.5:0.95] | > 0.25 | > 0.40 | > 0.55 |
| mAP@0.5 | > 0.40 | > 0.55 | > 0.70 |
| mAP@0.75 | > 0.25 | > 0.38 | > 0.55 |
| Rappel | > 0.50 | > 0.65 | > 0.75 |
| Précision | > 0.50 | > 0.65 | > 0.75 |
| F1 Score | > 0.50 | > 0.65 | > 0.75 |

> Ces cibles sont indicatives pour un dataset de fissures structurelles de taille moyenne (1 000–5 000 images). Les fissures fines sur fond texturé sont difficiles : un mAP@0.5 > 0.55 est déjà un excellent résultat pour ce type de donnée.

---

## Bonnes pratiques et réglages

### Hyperparamètres YOLO11-seg recommandés

```bash
# Dataset moyen (1 000–3 000 images) — configuration de départ
.venv/bin/python entrainer_yolo.py \
    --donnees ./dataset \
    --modele yolo11s-seg.pt \
    --epoques 120 \
    --lot 16 \
    --taille-image 640 \
    --lr 1e-3 \
    --lrf 0.01 \
    --warmup-epoques 3 \
    --close-mosaic 10 \
    --copy-paste 0.2 \
    --mask-ratio 4 \
    --degrees 15 \
    --scale 0.5
```

### Paramètres clés expliqués

**`--close-mosaic 10`** (recommandé par Ultralytics)
L'augmentation mosaïque assemble 4 images en une, très utile pour apprendre des fissures à différentes échelles. Mais elle est désactivée lors des 10 dernières époques pour stabiliser l'entraînement et affiner le modèle sur des images normales.

**`--warmup-epoques 3`**
Les 3 premières époques, le taux d'apprentissage monte linéairement de 0 vers `lr0`. Évite les mises à jour instables au début quand les poids sont loin de l'optimum.

**`--copy-paste 0.2`** (particulièrement efficace pour la segmentation d'instance)
20 % des images d'entraînement auront des instances de fissures collées depuis d'autres images avec leurs masques. Augmente la diversité des configurations de fissures sans collecter de nouvelles données.

**`--lrf 0.01`**
Le LR final vaudra `lr0 × lrf = 1e-3 × 0.01 = 1e-5`. Le scheduler cosinus descend progressivement, permettant un fine-tuning stable en fin d'entraînement.

**`--mask-ratio 1`** (qualité des masques)
Par défaut (4), les masques sont calculés à ¼ de la résolution puis interpolés. Passer à 1 donne des masques pleine résolution — meilleur pour les fissures fines — au prix d'un entraînement ~20 % plus lent.

### Réglage selon la taille du dataset

| Taille dataset | Modèle conseillé | Freeze | Copy-paste | Époques |
|---|---|---|---|---|
| < 300 images | yolo11n-seg.pt | 10 | 0.3 | 100 |
| 300 – 1 000 | yolo11s-seg.pt | 5 | 0.2 | 120 |
| 1 000 – 5 000 | yolo11m-seg.pt | 0 | 0.2 | 150 |
| > 5 000 | yolo11l-seg.pt | 0 | 0.1 | 200 |

---

## Dépannage

### GPU non détecté

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Forcer le CPU
.venv/bin/python entrainer_yolo.py --donnees ./dataset --dispositif cpu
```

### Out of Memory (OOM)

```bash
# Réduire le batch size
.venv/bin/python entrainer_yolo.py --donnees ./dataset --lot 4

# Réduire la résolution
.venv/bin/python entrainer_yolo.py --donnees ./dataset --taille-image 416

# Autobatch : YOLO calcule automatiquement le batch optimal selon la VRAM
.venv/bin/python entrainer_yolo.py --donnees ./dataset --lot -1
```

### Loss NaN en début d'entraînement

Causes possibles :
1. **LR trop élevé** — essayer `--lr 1e-4` au lieu de `1e-3`
2. **AMP instable** — ajouter `--sans-mixte` (Mask R-CNN)
3. **Annotations corrompues** — vérifier le JSON COCO

```bash
.venv/bin/python -c "
from pycocotools.coco import COCO
coco = COCO('mon_dataset/train/_annotations.coco.json')
print(f'{len(coco.imgs)} images, {len(coco.anns)} annotations')
"
```

### mAP stagne ou régresse

1. **Overfitting** — réduire `--epoques`, augmenter `--copy-paste 0.3`
2. **LR trop bas** — essayer `--lr 3e-3` pour YOLO
3. **Dataset déséquilibré** — vérifier la distribution images/annotations par split
4. **Résolution insuffisante** — augmenter `--taille-image 640` ou `768`

### Rappel faible malgré bonne précision

Le modèle trouve bien ce qu'il détecte, mais manque des fissures :

```bash
.venv/bin/python analyser.py \
    --modele best.pt \
    --images ./photos \
    --seuil-score 0.25
```

### Erreur `FileNotFoundError: _annotations.coco.json`

Le dataset doit contenir exactement les dossiers `train/`, `valid/`, `test/` avec le fichier `_annotations.coco.json` dans chacun :

```bash
ls mon_dataset/train/
# Doit afficher : _annotations.coco.json  image_001.jpg  image_002.jpg  ...
```

### Modèle YOLO non reconnu

Seuls les modèles `yolo11n/s/m/l/x-seg.pt` sont acceptés. Vérifier l'orthographe : YOLO**11**, pas YOLO**v11** ni YOLO**8**.

```bash
.venv/bin/python -c "
from configuration.parametres import MODELES_YOLOV11_SEG_AUTORISES
print(MODELES_YOLOV11_SEG_AUTORISES)
"
```

### Erreur OpenCV lors de l'import

```bash
# Réinstaller les dépendances système X11 (environnement Nix/Replit)
nix-env -iA nixpkgs.xorg.libxcb nixpkgs.xorg.libX11

# Ou utiliser la version headless
uv pip install opencv-python-headless
```
