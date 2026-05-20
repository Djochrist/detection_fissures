# Détection de Fissures Structurelles — Mask R-CNN

Segmentation d'instance de fissures sur structures (béton, maçonnerie)
basée sur **Mask R-CNN** (torchvision), avec deux architectures
fine-tunables sur les fissures :

- `maskrcnn_resnet50_fpn_v2` : modèle principal recommandé.
- `maskrcnn_resnet50_fpn` : modèle alternatif pour benchmark comparatif.

---

## Structure du projet

```
detection_fissures/
│
├── pyproject.toml              # Dépendances (UV)
├── entrainer.py                # Script d'entraînement principal
├── analyser.py                 # Script de classification post-entraînement
│
├── configuration/
│   └── parametres.py           # Hyperparamètres et chemins
│
├── donnees/
│   ├── jeu_donnees_coco.py     # Dataset PyTorch au format COCO
│   └── chargeur.py             # DataLoaders avec collate_fn
│
├── modeles/
│   └── masque_rcnn.py          # Factories Mask R-CNN V2 et FPN classique
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
Aucun prétraitement supplémentaire n'est appliqué — le dataset est utilisé tel quel.
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

---

## Dataset et GitHub

- Ne pousse pas le dossier `dataset/` sur GitHub. Le dataset est ignoré dans `.gitignore`.
- Stocke ton dataset sur Google Drive, un stockage externe ou un lien Roboflow.
- Sur Colab, utilise `--donnees /content/drive/MyDrive/dataset` et `--sorties /content/drive/MyDrive/detection_fissures_sorties`.
- Si Colab coupe l'exécution, relance le même notebook avec `--resume sorties/modeles/dernier_modele.pth`.

Exemple de commande Colab :

```bash
python entrainer.py \
  --architecture maskrcnn_resnet50_fpn_v2 \
  --donnees /content/drive/MyDrive/dataset \
  --sorties /content/drive/MyDrive/detection_fissures_sorties \
  --epoques 100 \
  --lot 8 \
  --lr 5e-5 \
  --dispositif cuda \
  --resume sorties/modeles/dernier_modele.pth
```

---

## Utilisation

### 1. Entraînement
```bash
python entrainer.py
python entrainer.py --architecture maskrcnn_resnet50_fpn_v2 --donnees /chemin/vers/dataset --epoques 100 --lot 8 --lr 5e-5
python entrainer.py --architecture maskrcnn_resnet50_fpn --donnees /chemin/vers/dataset --sorties sorties_fpn_v1
python entrainer.py --dispositif cuda
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
