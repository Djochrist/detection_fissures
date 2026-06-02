# Guide des masques prédits et des sorties des modèles Mask R-CNN / YOLO-seg

Ce document explique en détail la chaîne complète de prédiction des masques dans le projet.
Il décrit les sorties produites par les deux architectures supportées : `Mask R-CNN` et `YOLO-seg`.
Chaque terme technique est expliqué, et les commandes pour lancer l'inférence et visualiser les masques sont fournies.

---

## 1. Objectif

Ce guide a trois objectifs :

1. Expliquer comment le projet prédit les masques de fissure.
2. Décrire les sorties de `Mask R-CNN` et de `YOLO-seg`.
3. Fournir les commandes et exemples pratiques pour voir et sauvegarder les masques.

---

## 2. Concepts fondamentaux

### 2.1 Qu'est-ce qu'un "mask" ?

Un mask de segmentation est une image binaire de même taille que l'image d'entrée.
Chaque pixel est soit :

- `1` (ou blanc) : ce pixel appartient à la fissure.
- `0` (ou noir) : ce pixel appartient au fond (mur, béton, etc.).

Dans ce projet, chaque fissure détectée reçoit un mask distinct.
C'est ce qu'on appelle la **segmentation d'instance**.

### 2.2 Segmentation d'instance vs détection classique

- Détection classique : le modèle retourne une boîte autour de l'objet.
- Segmentation d'instance : le modèle retourne un masque, donc le contour exact de l'objet.

Pour les fissures, le mask est essentiel car il permet de mesurer :
- l'orientation,
- la largeur,
- la longueur,
- le périmètre,
- la surface en pixels.

### 2.3 Formats de sortie communs

Les deux architectures du projet renvoient les mêmes types de sorties :

- `boxes` : boîtes englobantes.
- `scores` : score de confiance.
- `labels` : étiquette de classe.
- `masks` : masques de segmentation.

Pour ce projet, il n'y a qu'une seule classe utile : la fissure.
Donc `labels` est ici surtout un indicateur interne.

---

## 3. Mask R-CNN dans ce projet

### 3.1 Architecture utilisée

Le projet utilise `Mask R-CNN ResNet50-FPN-V2` via `torchvision`.

Fichier principal : `modeles/masque_rcnn.py`

### 3.2 Objectif de chaque composant

- `Backbone` : ResNet-50.
  - Il lit l'image et extrait des caractéristiques complexes.
  - Il transforme les pixels en vecteurs de caractéristiques.
- `FPN` : Feature Pyramid Network.
  - Il regarde l'image sur plusieurs échelles simultanément.
  - Il permet de détecter les fissures très fines et les fissures plus larges.
- `RPN` : Region Proposal Network.
  - Il propose des régions candidates où une fissure peut exister.
- `Tête boîtes` : prédit les boîtes englobantes.
- `Tête masques` : prédit un mask de segmentation pour chaque fissure.

### 3.3 Sortie brute de Mask R-CNN

Après inférence, le modèle retourne un objet de type `list[dict]`.
Chaque dict correspond à une image et contient :

- `boxes` : Tensor[N, 4], coordonnées des boîtes.
- `scores` : Tensor[N], score de confiance.
- `labels` : Tensor[N], classe prédite.
- `masks` : Tensor[N, 1, H, W], valeurs continues.

### 3.4 Format des masks Mask R-CNN

Le format `Tensor[N, 1, H, W]` signifie :

- `N` : nombre de détections (nombre de fissures reconnues).
- `1` : canal unique, puisque c'est un mask binaire.
- `H, W` : hauteur et largeur de l'image d'entrée.

Les valeurs du mask sont des probabilités entre 0 et 1.
Pour obtenir un mask binaire on utilise le seuil standard :

```python
masque_binaire = (masque > 0.5).astype(np.uint8)
```

Une fois binarisé, le mask est exploitable comme image de segmentation.

### 3.5 Seuils et filtrage

Dans `detection_fissures/analyser.py`, les prédictions sont filtrées selon le score :

```python
masque_score = pred["scores"] >= args.seuil
```

- Pour `Mask R-CNN`, le guide recommande un seuil autour de `0.40`.
- Cela élimine les détections peu fiables et conserve les masques les plus solides.

### 3.6 Sorties d'analyse du pipeline Mask R-CNN

Quand l'inférence est lancée avec `Mask R-CNN`, les sorties attendues sont :

- `analyses/maskrcnn/rapport_analyse.json`
- `analyses/maskrcnn/images_annotees/`

Le rapport contient :
- les images traitées,
- le nombre de fissures détectées,
- les classifications produites.

Le dossier `images_annotees` contient des images dupliquées avec :
- les boîtes tracées,
- les labels de fissure,
- éventuellement les contours ou le dessin d'annotation.

### 3.7 Termes techniques principaux pour Mask R-CNN

- `box` : boîte englobante autour d'une fissure.
- `score` : probabilité que la détection soit correcte.
- `label` : classe détectée (ici fissure).
- `mask` : contour pixel par pixel de la fissure.
- `IoU` : Intersection over Union, utilisé par la NMS.
- `NMS` : Non-Maximum Suppression, supprime les doublons entre boîtes.
- `mAP` : mean Average Precision, métrique d'évaluation.

### 3.8 Exemple de commande Mask R-CNN

```bash
python analyser.py \
  --modele sorties/modeles/meilleur_modele.pth \
  --backend maskrcnn \
  --images dataset/test/ \
  --seuil 0.40 \
  --sortie analyses/maskrcnn/rapport_analyse.json
```

---

## 4. YOLO-seg dans ce projet

### 4.1 Architecture disponible

Le projet supporte `YOLO-seg` via les scripts dans :
- `detection_fissures/entrainer_yolo.py`
- `donnees/conversion_yolo.py`

YOLO-seg est un modèle de segmentation plus rapide que Mask R-CNN.
Il reste compatible avec la segmentation d'instance.

### 4.2 Format du dataset YOLO-seg

YOLO-seg attend un dataset converti en :
- `images/` : photos normales,
- `labels/*.txt` : annotations de segmentation en format YOLO.

Le fichier `.txt` contient une ligne par instance :

```
class_id x_center y_center width height segmentation_points...
```

Pour les images sans fissure, le fichier `.txt` est vide.

### 4.3 Sortie brute de YOLO-seg

Le résultat d'inférence YOLO-seg comprend également :

- `boxes` : boîte englobante de chaque instance.
- `scores` : score de confiance pour chaque détection.
- `labels` : étiquette de classe.
- `masks` : masque de segmentation par instance.

Le format exact dépend de la version Ultralytics, mais le concept est identique :
une liste de détections avec masques.

### 4.4 Différences pratiques entre Mask R-CNN et YOLO-seg

- `Mask R-CNN` : meilleure précision sur les contours de fissure.
- `YOLO-seg` : plus rapide, utile pour des besoins d'analyse en production.
- Les deux produisent des masques d'instance.
- Pour des fissures très fines ou des contours très détaillés, Mask R-CNN est préféré.

### 4.5 Sorties d'analyse du pipeline YOLO-seg

Quand l'inférence est lancée avec `YOLO`, les sorties attendues sont :

- `analyses/yolo/rapport_analyse.json`
- `analyses/yolo/images_annotees/`

Ces dossiers contiennent le même type de livrables que pour Mask R-CNN.

### 4.6 Seuil recommandé pour YOLO-seg

Le guide du projet recommande :

- `--seuil 0.25` pour `YOLO-seg`
- ce seuil est inférieur à Mask R-CNN car YOLO utilise souvent des scores plus prudents
  pour couvrir un maximum d'instances pertinentes.

### 4.7 Exemple de commande YOLO-seg

```bash
python analyser.py \
  --modele sorties_yolo/entrainements/yolo11_seg_fissures/weights/best.pt \
  --backend yolo \
  --images dataset/test/ \
  --seuil 0.25 \
  --sortie analyses/yolo/rapport_analyse.json
```

---

## 5. Structure des fichiers de modèle

### 5.1 Mask R-CNN

Le checkpoint Mask R-CNN se trouve dans :
- `sorties/modeles/meilleur_modele.pth`
- `sorties/modeles/dernier_modele.pth`

C'est un fichier PyTorch contenant l'état des poids.

### 5.2 YOLO-seg

Le checkpoint YOLO-seg se trouve généralement dans :
- `sorties_yolo/entrainements/yolo11_seg_fissures/weights/best.pt`
- `sorties_yolo/entrainements/yolo11_seg_fissures/weights/last.pt`

Ce sont des fichiers Ultralytics/YOLO.

---

## 6. Comment voir les masques directement

### 6.1 Ce que le code actuel fait

Le script d'analyse (`detection_fissures/analyser.py`) :

- charge l'image,
- exécute le modèle,
- filtre les detections par score,
- convertit les masques en format utilisable,
- calcule les mesures géométriques.

Il n'écrit pas automatiquement des fichiers `mask_*.png` dans sa version d'origine,
mais il dispose de tous les éléments pour le faire.

### 6.2 Exemple de script pour sauvegarder les masks en PNG

```python
import cv2
import numpy as np
from pathlib import Path
from detection_fissures.utilitaires.images import charger_image_rgb, redimensionner_image_carree, image_rgb_vers_tenseur
from detection_fissures.detection_fissures.analyser import charger_modele

# Exemple rapide : charger un modèle et sauver les masks de la première image.
modele = charger_modele(
    chemin_pth="sorties/modeles/meilleur_modele.pth",
    nombre_classes=2,
    taille_image=384,
    architecture="maskrcnn_resnet50_fpn_v2",
    dispositif="cpu",
)

image_path = Path("dataset/test/image_exemple.jpg")
image_rgb, _, _ = charger_image_rgb(image_path)
image_redim = redimensionner_image_carree(image_rgb, 384)
tenseur = image_rgb_vers_tenseur(image_redim).unsqueeze(0)

with torch.no_grad():
    predictions = modele(tenseur)

for i, masque in enumerate(predictions[0]["masks"].squeeze(1).cpu().numpy()):
    masque_bin = (masque > 0.5).astype(np.uint8) * 255
    cv2.imwrite(f"mask_{image_path.stem}_{i}.png", masque_bin)
```

### 6.3 Visualisation simple en terminal

Pour afficher les images annotées déjà produites par le projet :

```bash
find analyses/maskrcnn/images_annotees -type f | sort | head
find analyses/yolo/images_annotees -type f | sort | head
```

---

## 7. Explication détaillée des termes techniques

### Mask
C'est la sortie la plus importante pour la segmentation d'instance.
Il représente la zone précise de la fissure pixel par pixel.

### Boîte englobante (`box`)
Une boîte rectangulaire autour de la fissure.
Elle donne une localisation rapide mais pas le contour exact.

### Score de confiance (`score`)
Une valeur entre 0 et 1.
Plus elle est haute, plus le modèle est confiant dans la détection.

### Label
Dans ce projet, il s'agit de la classe détectée.
Il existe essentiellement une classe utile : la fissure.

### IoU (Intersection over Union)
C'est une mesure de chevauchement entre deux boîtes.
Il est utilisé pour supprimer les doubles détections.

### NMS (Non-Maximum Suppression)
Un algorithme qui garde la meilleure boîte parmi plusieurs boîtes similaires.
Il évite de retourner plusieurs fois la même fissure.

### PCA (Analyse en Composantes Principales)
Une méthode mathématique qui trouve la direction principale des pixels d'un mask.
C'est la technique utilisée pour mesurer l'orientation de la fissure.

### Distance transform
Une méthode qui calcule, pour chaque pixel du mask, la distance au bord le plus proche.
Elle permet d'estimer la largeur d'une fissure de manière robuste.

### mAP (mean Average Precision)
Une métrique d'évaluation standard en détection et segmentation.
Plus elle est haute, meilleur est le modèle.

---

## 8. Résumé des sorties pour le client

### Masques disponibles
- `Mask R-CNN` : prédictions ultra-précises pour contours de fissure.
- `YOLO-seg` : prédictions rapides et toujours correctes pour des cas plus légers.

### Fichiers produits par l'inférence
- `analyses/maskrcnn/rapport_analyse.json`
- `analyses/maskrcnn/images_annotees/`
- `analyses/yolo/rapport_analyse.json`
- `analyses/yolo/images_annotees/`

### Formats de fichiers de modèle
- `sorties/modeles/*.pth` pour Mask R-CNN.
- `sorties_yolo/.../weights/*.pt` pour YOLO-seg.

---

## 9. Commandes recommandées

### Analyse Mask R-CNN
```bash
python analyser.py \
  --modele sorties/modeles/meilleur_modele.pth \
  --backend maskrcnn \
  --images dataset/test/ \
  --seuil 0.40 \
  --sortie analyses/maskrcnn/rapport_analyse.json
```

### Analyse YOLO-seg
```bash
python analyser.py \
  --modele sorties_yolo/entrainements/yolo11_seg_fissures/weights/best.pt \
  --backend yolo \
  --images dataset/test/ \
  --seuil 0.25 \
  --sortie analyses/yolo/rapport_analyse.json
```

### Voir les masques PNG générés

Le projet n'écrit pas encore de `mask_*.png` par défaut, mais le code contient déjà
les étapes nécessaires pour le faire.

---

## 10. Compléments importants

- Si le client veut un export de masques PNG instantané, le plus simple est d'ajouter
  un petit script qui lit `pred["masks"]`, binarise avec `> 0.5`, puis écrit en PNG.
- Si le client veut des rapports JSON, il suffit d'utiliser `--sortie` avec `analyser.py`.
- Pour un usage industriel, on recommande de garder les deux backends :
  - Mask R-CNN pour la précision maximale,
  - YOLO-seg pour la rapidité et les tests de production.
