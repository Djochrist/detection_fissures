# Entraîner sur Google Colab — Guide complet

Dataset : format **YOLOv11 natif Roboflow** — 640×640px — 3794 images — 1 classe (`crack`).

Ce guide couvre **tout** : entraînement YOLO11 (s/m), YOLO26 (s/m), Mask R-CNN,
reprise après déconnexion, analyse + **orientation des fissures**, et une section
finale de **conseils de réglage** selon les performances obtenues.

---

## 1. Préparer Colab

1. Ouvre [Google Colab](https://colab.research.google.com/).
2. `Exécution` → `Modifier le type d'exécution` → choisir **GPU** (T4 gratuit suffit).
3. Lance les cellules ci-dessous **dans l'ordre**.

Vérifier le GPU attribué :

```bash
!nvidia-smi
```

---

## 2. Monter Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Place ton dataset YOLO dans Drive avec cette structure :

```text
/content/drive/MyDrive/dataset/
├── data.yaml
├── images/
│   ├── train/   (≈3160 images)
│   ├── valid/   (≈476 images)
│   └── test/    (≈158 images)
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

Définis les chemins (variables réutilisées partout ensuite) :

```python
DATASET_YAML = "/content/drive/MyDrive/dataset/data.yaml"
SORTIES_YOLO = "/content/drive/MyDrive/sorties_yolo"
SORTIES_MASK = "/content/drive/MyDrive/sorties_maskrcnn"
```

> 💡 En définissant `--sorties` sur un dossier Drive, les poids survivent à une
> déconnexion de Colab.

---

## 3. Cloner le projet

```bash
%cd /content
!rm -rf detection_fissures
!git clone https://github.com/Djochrist/detection_fissures.git
%cd /content/detection_fissures
```

---

## 4. Installer les dépendances

Colab fournit déjà PyTorch GPU. Installer les dépendances du projet :

```bash
!pip install -q -U opencv-python numpy scipy pycocotools "torchmetrics[detection]" rich ultralytics
!pip install -q -e .
```

Vérification :

```bash
!python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
!python entrainer_yolo.py --help
!python analyser.py --help
```

---

## 5. Adapter data.yaml

Le `data.yaml` de Roboflow contient un chemin absolu qui doit pointer vers ton Drive :

```python
import yaml, pathlib

yaml_path = pathlib.Path(DATASET_YAML)
config = yaml.safe_load(yaml_path.read_text())
config['path'] = str(yaml_path.parent)   # racine du dataset sur Drive
yaml_path.write_text(yaml.dump(config))
print("data.yaml mis à jour :", config)
```

---

## 6. Entraînement YOLO-seg

Choisis **une** des 4 variantes selon ton GPU et tes besoins. Les commandes sont
complètes : tous les hyperparamètres sont explicités pour pouvoir les régler
ensuite (voir section 11).

### 6.1 — YOLO11 Medium (recommandé, GPU ≥ 6 Go)

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
  --dispositif    auto \
  --nom           yolo11m_fissures \
  --sorties       "$SORTIES_YOLO"
```

### 6.2 — YOLO11 Small (GPU < 6 Go ou test rapide)

```bash
!python entrainer_yolo.py \
  --yaml          "$DATASET_YAML" \
  --modele        yolo11s-seg.pt \
  --taille-image  640 \
  --epoques       150 \
  --lot           16 \
  --lr            0.01 \
  --lrf           0.01 \
  --weight-decay  0.0005 \
  --patience      50 \
  --warmup-epochs 5.0 \
  --mask-ratio    1 \
  --dispositif    auto \
  --nom           yolo11s_fissures \
  --sorties       "$SORTIES_YOLO"
```

### 6.3 — YOLO26 Medium (génération 2026, GPU ≥ 6 Go)

```bash
!python entrainer_yolo.py \
  --yaml          "$DATASET_YAML" \
  --modele        yolo26m-seg.pt \
  --taille-image  640 \
  --epoques       150 \
  --lot           8 \
  --lr            0.01 \
  --lrf           0.01 \
  --weight-decay  0.0005 \
  --patience      50 \
  --warmup-epochs 5.0 \
  --mask-ratio    1 \
  --dispositif    auto \
  --nom           yolo26m_fissures \
  --sorties       "$SORTIES_YOLO"
```

### 6.4 — YOLO26 Small (génération 2026, GPU < 6 Go)

```bash
!python entrainer_yolo.py \
  --yaml          "$DATASET_YAML" \
  --modele        yolo26s-seg.pt \
  --taille-image  640 \
  --epoques       150 \
  --lot           16 \
  --lr            0.01 \
  --lrf           0.01 \
  --weight-decay  0.0005 \
  --patience      50 \
  --warmup-epochs 5.0 \
  --mask-ratio    1 \
  --dispositif    auto \
  --nom           yolo26s_fissures \
  --sorties       "$SORTIES_YOLO"
```

Meilleur modèle → `$SORTIES_YOLO/entrainements/<nom>/weights/best.pt`

---

## 7. Entraînement Mask R-CNN (précision maximale)

> ⚠️ **Mask R-CNN exige le format COCO** (`_annotations.coco.json`), pas le format
> YOLOv11. Sur Roboflow, ré-exporte le même dataset au format **COCO Segmentation**
> et place-le sur Drive sous `dataset_coco/` avec `train/`, `valid/`, `test/`.

```python
DATASET_COCO = "/content/drive/MyDrive/dataset_coco"
```

### 7.1 — Entraînement GPU (recommandé, ~2-6h)

```bash
!python entrainer.py \
  --donnees       "$DATASET_COCO" \
  --epoques       100 \
  --lot           4 \
  --lr            0.0001 \
  --weight-decay  0.0005 \
  --patience      20 \
  --taille-image  640 \
  --seuil-score   0.05 \
  --architecture  maskrcnn_resnet50_fpn_v2 \
  --dispositif    auto \
  --graine        42 \
  --sorties       "$SORTIES_MASK"
```

> Les variables `$DATASET_COCO`, `$SORTIES_MASK`, etc. définies en Python sont
> automatiquement substituées par Colab dans les cellules `!`.

Meilleur modèle → `$SORTIES_MASK/modeles/meilleur_modele.pth`

---

## 8. Reprendre un entraînement interrompu

Colab déconnecte après ~12h. **Relance les cellules 1→5** (notamment celle qui
définit `DATASET_YAML`, `SORTIES_YOLO`, etc.), puis :

```bash
# YOLO11m (adapter le nom selon la variante entraînée)
!python entrainer_yolo.py \
  --yaml    "$DATASET_YAML" \
  --sorties "$SORTIES_YOLO" \
  --resume  "$SORTIES_YOLO/entrainements/yolo11m_fissures/weights/last.pt"
```

```bash
# Mask R-CNN
!python entrainer.py \
  --donnees "$DATASET_COCO" \
  --sorties "$SORTIES_MASK" \
  --resume  "$SORTIES_MASK/modeles/dernier_modele.pth"
```

---

## 9. Analyse + orientation des fissures

Le script `analyser.py` détecte les fissures, **classe leur orientation**
(horizontale / verticale / inclinée par PCA), estime la **largeur** et la
**profondeur** (superficielle / profonde / transversale), calcule un **indice de
danger**, génère des **images annotées** et un **rapport JSON**.

### 9.1 — Analyse avec un modèle YOLO

```bash
!python analyser.py \
  --backend        yolo \
  --modele         "$SORTIES_YOLO/entrainements/yolo11m_fissures/weights/best.pt" \
  --images         /content/drive/MyDrive/dataset/images/test/ \
  --taille-image   640 \
  --seuil          0.25 \
  --dossier-sortie /content/drive/MyDrive/analyses \
  --dispositif     auto
```

Résultats produits :

```text
/content/drive/MyDrive/analyses/yolo/
├── rapport_analyse.json
└── images_annotees/
    ├── <image>_analyse.jpg
    └── ...
```

### 9.2 — Analyse avec un modèle Mask R-CNN

```bash
!python analyser.py \
  --backend        maskrcnn \
  --modele         "$SORTIES_MASK/modeles/meilleur_modele.pth" \
  --images         /content/drive/MyDrive/dataset/images/test/ \
  --taille-image   640 \
  --seuil          0.40 \
  --dossier-sortie /content/drive/MyDrive/analyses \
  --dispositif     auto
```

### 9.3 — Options utiles d'analyse

| Option | Effet |
|--------|-------|
| `--seuil 0.25` | Score de confiance minimal (baisser = plus de détections, plus de faux positifs) |
| `--sans-images` | Produit **uniquement** le JSON (plus rapide, pas de copies annotées) |
| `--sortie chemin.json` | Nomme explicitement le fichier JSON de sortie |
| `--taille-image 640` | Doit correspondre à la résolution d'entraînement |

### 9.4 — Lire le rapport JSON

```python
import json
rapport = json.load(open("/content/drive/MyDrive/analyses/yolo/rapport_analyse.json"))
print("Images analysées :", rapport["resume"]["images_analysees"])
print("Fissures totales :", rapport["resume"]["total_fissures"])
print("Danger moyen     :", rapport["resume"]["danger_moyen_global"])
for img in rapport["resultats"][:3]:
    for f in img["fissures"]:
        print(img["image"], "→", f["orientation"], f["localisation"],
              f"angle={f['angle_degres']}°", f"danger={f['indice_danger']}")
```

---

## 10. Récupérer les résultats

Si `--sorties` pointe déjà sur Drive, tout y est déjà sauvegardé. Pour télécharger
un fichier précis sur ton ordinateur :

```python
from google.colab import files
files.download(f"{SORTIES_YOLO}/entrainements/yolo11m_fissures/weights/best.pt")
```

Visualiser une image annotée directement dans le notebook :

```python
from IPython.display import Image
Image("/content/drive/MyDrive/analyses/yolo/images_annotees/<image>_analyse.jpg")
```

---

## 11. Conseils de réglage selon les performances

Après l'entraînement, regarde les métriques affichées (mAP@0.5 masque, précision,
rappel, F1) et les courbes dans `$SORTIES_YOLO/entrainements/<nom>/results.png`.
Voici quoi faire selon ce que tu observes.

### 11.1 — Le mAP est trop bas (< 0.5)

Le modèle ne détecte pas assez bien. À essayer, dans l'ordre :

1. **Entraîner plus longtemps** — passe `--epoques` de 150 à `250` ou `300`, et
   augmente `--patience` à `80` pour ne pas arrêter trop tôt.
2. **Modèle plus gros** — passe de `yolo11s-seg.pt` à `yolo11m-seg.pt`
   (ou `yolo11l-seg.pt` si le GPU le permet). Plus de capacité = meilleure précision.
3. **Garder la pleine résolution des masques** — vérifie `--mask-ratio 1`
   (essentiel pour les fissures fines de 1-3 px).
4. **Augmenter la résolution** — `--taille-image 768` ou `960` (multiple de 32)
   si le GPU a assez de mémoire ; baisse alors `--lot` (ex. 4).
5. **Vérifier les annotations** — un mAP très bas vient souvent d'un dataset mal
   étiqueté, pas du modèle. Inspecte quelques images annotées.

> ⚠️ **Augmentation désactivée.** Le dataset Roboflow est déjà pré-augmenté (5×)
> et l'entraînement n'applique plus aucune augmentation à la volée. Les leviers
> ci-dessous portent donc sur le modèle, les époques, le seuil et les **données**
> (ajouter des images réelles ou des **murs sains**), pas sur des réglages d'augmentation.

### 11.2 — Le rappel est bas (le modèle rate des fissures)

Beaucoup de fissures non détectées (faux négatifs) :

- **Réduis le seuil à l'analyse** : `--seuil 0.15` au lieu de 0.25.
- **Entraîne plus longtemps** : `--epoques 250` et augmente la taille d'image (voir 11.1).
- **Modèle plus gros** : `yolo11m-seg.pt` → `yolo11l-seg.pt`.
- **Ajoute des images de fissures** réelles et variées au dataset.

### 11.3 — La précision est basse (trop de faux positifs)

Le modèle voit des fissures là où il n'y en a pas :

- **Ajoute plus de murs sains** (images sans fissure, label vide) au dataset : c'est
  le meilleur moyen d'apprendre au modèle à ne PAS halluciner de fissures.
- **Augmente le seuil à l'analyse** : `--seuil 0.35` ou `0.4`.
- **Entraîne un peu plus longtemps** pour mieux discriminer.

### 11.4 — Surapprentissage (train bon, validation mauvaise)

La courbe de perte d'entraînement descend mais la validation stagne/remonte :

- **Augmente la régularisation** : `--weight-decay 0.001`.
- **Ajoute plus de données** variées (images réelles + murs sains).
- **Réduis `--epoques`** ou laisse l'early stopping agir (`--patience 30`).
- **Modèle plus petit** : repasse de `m` à `s`.

### 11.5 — Sous-apprentissage (train ET validation mauvais)

- **Modèle plus gros** (`m` → `l`).
- **Plus d'époques** (`--epoques 300`).
- **Augmente le LR** : `--lr 0.02` (puis surveille que la perte ne diverge pas).

### 11.6 — La perte diverge / devient NaN

- **Baisse le LR** : `--lr 0.005` puis `0.001`.
- **Augmente le warmup** : `--warmup-epochs 8`.
- **Garde la précision mixte par défaut** (déjà activée).

### 11.7 — Mémoire GPU insuffisante (CUDA out of memory)

- **Réduis `--lot`** : 8 → 4 → 2.
- **Réduis `--taille-image`** : 640 → 512.
- **Modèle plus petit** : `m` → `s`.
- **Cache disque** : `--cache disk` (au lieu de `ram`).

### 11.8 — Entraînement trop lent

- **Modèle plus petit** : `yolo11s-seg.pt`.
- **Cache RAM** si ≥ 32 Go : `--cache ram`.
- **Augmente `--lot`** si la mémoire le permet (meilleure utilisation GPU).
- **Réduis `--save-period`** ou mets `-1` pour moins d'écritures disque.

### 11.9 — Les seuils de classification ne collent pas à tes photos

L'orientation et la profondeur sont calibrées pour des images **640×640 px**. Si tes
photos ont une autre résolution ou une distance caméra différente, ajuste les seuils
dans `analyse/classificateur_fissures.py` :

- `SEUIL_LARGEUR_SUPERFICIELLE` / `SEUIL_LARGEUR_PROFONDE` (en pixels) : largeurs
  qui séparent superficielle / profonde.
- `SEUIL_ANGLE_HORIZONTAL` / `SEUIL_ANGLE_VERTICAL` (en degrés) : bornes d'orientation.
- `SEUIL_TRAVERSEE` (ratio 0-1) : seuil pour qu'une fissure soit jugée transversale.

---

## Notes

- Colab déconnecte après ~12h d'inactivité → utiliser `--resume`.
- Toujours sauvegarder les poids sur Drive via `--sorties`.
- `data.yaml` doit avoir `path:` pointant vers la racine du dataset.
- Pour comparer plusieurs modèles, lance-les avec des `--nom` différents : ils sont
  rangés dans des sous-dossiers distincts de `$SORTIES_YOLO/entrainements/`.
