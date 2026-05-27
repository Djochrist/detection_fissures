# Guide d'Analyse Géométrique — Classification des Fissures

> **À qui s'adresse ce guide ?**
> À toute personne qui veut comprendre comment le programme classifie les fissures
> après les avoir détectées. On explique ici chaque étape mathématique avec des
> analogies simples et des exemples tirés directement du projet.

---

## Sommaire

1. [Vue d'ensemble : que fait l'analyse géométrique ?](#1-vue-densemble--que-fait-lanalyse-géométrique)
2. [Le masque de segmentation : notre matière première](#2-le-masque-de-segmentation--notre-matière-première)
3. [Classification par orientation : la méthode PCA](#3-classification-par-orientation--la-méthode-pca)
4. [Classification par localisation : la distance transform](#4-classification-par-localisation--la-distance-transform)
5. [La détection transversale : analyse de l'étendue](#5-la-détection-transversale--analyse-de-létendue)
6. [L'indice de danger composite](#6-lindice-de-danger-composite)
7. [Comment utiliser le script d'analyse](#7-comment-utiliser-le-script-danalyse)
8. [Lire les résultats](#8-lire-les-résultats)
9. [Ajuster les seuils de classification](#9-ajuster-les-seuils-de-classification)

---

## 1. Vue d'ensemble : que fait l'analyse géométrique ?

Après l'entraînement, on a un modèle capable de trouver les fissures dans une photo.
Mais "il y a une fissure" n'est pas suffisant en génie civil. On veut savoir :

- **Quelle est son orientation ?** Une fissure verticale n'a pas les mêmes causes ni
  les mêmes conséquences qu'une fissure horizontale.
- **Est-elle superficielle ou profonde ?** Une fissure capillaire dans l'enduit est
  moins préoccupante qu'une fissure qui traverse le matériau porteur.
- **Traverse-t-elle toute la structure ?** Une fissure transversale peut indiquer une
  séparation structurelle critique.

L'analyse géométrique répond à ces trois questions en étudiant la **forme du masque**
prédit par le modèle. Pas besoin de refaire une inférence : on travaille directement sur
les pixels du masque.

### Le flux complet

```
Photo de mur
    ↓
[ Modèle Mask R-CNN ]
    ↓
Masques de segmentation (un par fissure)
    ↓
[ Analyse géométrique ]
    ├── PCA → orientation (horizontale / verticale / inclinée)
    ├── Distance Transform → largeur → localisation (superficielle / profonde)
    ├── Ratio bbox/image → transversale ?
    └── Calcul de l'indice de danger
    ↓
Classification finale + indice de danger [0, 1]
```

---

## 2. Le masque de segmentation : notre matière première

### C'est quoi un masque ?

Quand le modèle détecte une fissure, il produit un **masque** : une image en noir et blanc
de la même taille que la photo originale (384×384 pixels), où :
- Les pixels **blancs (valeur 1)** appartiennent à la fissure
- Les pixels **noirs (valeur 0)** appartiennent au fond (mur, béton, etc.)

**Exemple visuel :**
```
Photo originale (384×384)          Masque correspondant (384×384)
┌────────────────────────┐         ┌────────────────────────┐
│  mur en béton gris     │         │  0 0 0 0 0 0 0 0 0 0  │
│                        │         │  0 0 0 1 1 1 0 0 0 0  │  ← fissure = 1
│  ╱ fissure en diagonale│  →      │  0 0 0 0 1 1 1 0 0 0  │
│     ╱                  │         │  0 0 0 0 0 1 1 1 0 0  │
│                        │         │  0 0 0 0 0 0 0 0 0 0  │
└────────────────────────┘         └────────────────────────┘
```

### Ce qu'on extrait du masque

Toute l'analyse géométrique se base sur la **position des pixels blancs** dans ce masque.
On ne regarde plus la photo originale — seulement le masque.

On note la liste de tous les pixels blancs sous forme de coordonnées (x, y) :
```
Pixel 1 : (120, 45)
Pixel 2 : (121, 46)
Pixel 3 : (122, 46)
Pixel 4 : (123, 47)
...
```

C'est avec ces coordonnées qu'on calcule l'orientation et la largeur.

---

## 3. Classification par orientation : la méthode PCA

### Pourquoi PCA et pas simplement l'angle de la boîte ?

On pourrait penser à utiliser les coins de la boîte englobante pour calculer l'angle.
Mais ce n'est pas fiable pour les fissures qui ne sont pas rectilignes (courbées,
ramifiées, en zigzag). La PCA, elle, trouve toujours la **direction principale** même
pour des formes complexes.

### Qu'est-ce que la PCA ?

**PCA** = Analyse en Composantes Principales.

**L'analogie de la foule :**
Imaginez une foule de gens qui marchent dans une rue. Si vous regardez la foule depuis
le dessus, les gens sont dispersés mais globalement orientés dans la même direction (la
rue). La PCA trouverait cet axe principal de la rue, même si certaines personnes font
des zigzags.

Pour nos fissures : les pixels blancs du masque forment une "foule". La PCA trouve l'axe
principal de cette foule = la direction principale de la fissure.

### Comment ça fonctionne concrètement ?

**Étape 1 — Lister les coordonnées de tous les pixels blancs**
```
Pixels de la fissure :
(120, 45), (121, 46), (122, 46), (123, 47), (124, 48), (125, 48), ...
```

**Étape 2 — Centrer les coordonnées (soustraire la moyenne)**
```
Moyenne x = 122.5,  Moyenne y = 46.8
Pixel centré 1 : (120-122.5, 45-46.8) = (-2.5, -1.8)
Pixel centré 2 : (121-122.5, 46-46.8) = (-1.5, -0.8)
...
```

**Étape 3 — Calculer la "matrice de covariance"**

C'est une façon de mesurer si les variations en x et en y sont liées.
- Si quand x augmente, y augmente aussi → la fissure est diagonale (inclinée)
- Si y ne change presque pas quand x change → la fissure est horizontale
- Si x ne change presque pas quand y change → la fissure est verticale

**Étape 4 — Trouver l'axe principal (vecteur propre dominant)**

La matrice de covariance nous donne deux vecteurs :
- Le **vecteur dominant** : la direction dans laquelle la fissure s'étend le plus
- Le vecteur secondaire : perpendiculaire au premier (la largeur)

**Étape 5 — Calculer l'angle**

L'angle du vecteur dominant par rapport à l'horizontal nous donne l'orientation :
```python
angle = arctan(composante_y / composante_x)  # en degrés
```

### Les 3 types d'orientation et leurs seuils

```
         90°
          │
Verticale │ (angle > 70°)
          │
          │    Inclinée
 ─────────┼─────────── 0° (Horizontale)
          │    (20°–70°)
          │
          │ (angle < 20°)
```

| Type | Plage d'angle | Causes structurelles possibles |
|---|---|---|
| **Horizontale** | 0° – 20° | Tassement différentiel, surcharge verticale, retrait du béton |
| **Inclinée** | 20° – 70° | Cisaillement (force appliquée en biais), contraintes combinées |
| **Verticale** | 70° – 90° | Flexion d'une poutre ou d'un mur porteur, retrait thermique |

**Exemple avec des angles réels :**

```
Fissure A : angle = 8°   → Horizontale  (parallèle au sol)
Fissure B : angle = 45°  → Inclinée     (à 45° = cisaillement pur)
Fissure C : angle = 83°  → Verticale    (presque perpendiculaire au sol)
```

### Pourquoi l'angle est dans [0°, 90°] et pas [0°, 180°] ?

Une fissure à 100° est exactement la même chose qu'une fissure à 80° — c'est juste qu'on
la regarde de l'autre côté. La symétrie fait qu'on ramène tout dans [0°, 90°].

---

## 4. Classification par localisation : la distance transform

### Le problème : comment mesurer l'épaisseur d'une fissure ?

Pour savoir si une fissure est superficielle (fine) ou profonde (large), on a besoin de
mesurer son épaisseur. Sur un masque en noir et blanc, comment fait-on ça ?

### La distance transform : l'analogie du feu de forêt

Imaginez la forêt vue du dessus. Les zones noires (fond) sont des zones inondées. Les
zones blanches (fissure) sont des zones sèches.

Si on allume un incendie depuis le bord de la fissure (depuis les pixels noirs adjacents),
le feu progresse vers le centre de la fissure. Plus un pixel blanc est au centre de la
fissure (loin du bord), plus il met longtemps à brûler.

La **distance transform** calcule, pour chaque pixel blanc, sa distance au bord le plus
proche (au pixel noir le plus proche). Ce nombre = le rayon local de la fissure.

```
Masque (fissure = 1, fond = 0) :

0 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 0 0    ← fissure épaisse
0 0 1 1 1 1 1 0 0
0 0 1 1 1 1 1 0 0
0 0 0 0 0 0 0 0 0

Distance transform (distance au bord pour chaque pixel blanc) :

0 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 0 0
0 0 1 2 2 2 1 0 0    ← les pixels du centre sont à distance 2 du bord
0 0 1 1 1 1 1 0 0
0 0 0 0 0 0 0 0 0
```

**Largeur = valeur médiane × 2** (rayon → diamètre)

Dans cet exemple :
- Valeurs de distance : 1, 1, 1, 2, 2, 2, 1, 1, 1, ...
- Médiane = 1.0 (si beaucoup de bords) ou 2.0 (si fissure épaisse)
- Largeur estimée = 2 × 1.5 = 3 pixels (exemple)

### Pourquoi la médiane et pas la moyenne ?

La médiane est **robuste aux valeurs extrêmes**. Les extrémités d'une fissure
peuvent être plus larges (par exemple, une fissure qui s'ouvre à son extrémité).
Si on prenait la moyenne, ces extrémités élargies biaiseraient le résultat.
La médiane représente mieux la largeur "typique" de la fissure.

### Les 3 types de localisation et leurs seuils

| Type | Largeur estimée | Interprétation structurelle |
|---|---|---|
| **Superficielle** | < 3.5 pixels | Fissure fine, capillaire. Affecte seulement l'enduit ou la peinture. Surveillance recommandée mais pas d'urgence. |
| **Profonde** | > 7 pixels | Fissure large qui pénètre dans le matériau porteur. Nécessite une inspection approfondie. |
| Zone intermédiaire | 3.5 – 7 pixels | Classifiée selon d'autres critères (étendue dans l'image) |

**Conversion en millimètres réels :**
Ces seuils sont en pixels pour des images 384×384. Si vous connaissez la résolution
physique de votre caméra (par exemple, 0.5 mm par pixel à 2 mètres de distance) :
```
Largeur réelle = Largeur en pixels × résolution mm/pixel
Exemple : 3.5 pixels × 0.5 mm/pixel = 1.75 mm
```

Pour les fissures structurelles, les seuils normatifs courants sont :
- < 0.3 mm → surveillance simple
- 0.3 – 1 mm → attention
- > 1 mm → intervention recommandée

---

## 5. La détection transversale : analyse de l'étendue

### Qu'est-ce qu'une fissure transversale ?

Une fissure **transversale** est une fissure qui traverse l'élément structurel d'un côté
à l'autre — elle parcourt toute la largeur ou toute la hauteur du mur/poteau/dalle.

C'est la plus préoccupante car elle peut indiquer une **séparation complète** du matériau.

### Comment la détecter sur une image ?

On ne peut pas savoir avec certitude si une fissure est transversale depuis une photo 2D
(il faudrait voir les deux faces de la structure). Mais on peut en avoir un bon indice :
si la fissure occupe une grande partie de l'image, elle traverse probablement toute la
structure visible.

**Le ratio de couverture :**
```
ratio = max(largeur_bbox / largeur_image, hauteur_bbox / hauteur_image)
```

Où `bbox` est la boîte englobante de la fissure (le plus petit rectangle qui l'entoure).

**Exemple :**
```
Image 384×384  :  ┌────────────────────────────────────┐
                  │                                    │
                  │  ────────────────────────────────  │  ← fissure horizontale
                  │  ←────────── 320 pixels ──────────→│
                  └────────────────────────────────────┘

ratio = 320 / 384 = 0.83  →  83% de l'image → transversale (> seuil de 65%)
```

```
Image 384×384  :  ┌────────────────────────────────────┐
                  │         ╱                           │
                  │        ╱  ← petite fissure          │
                  │       ╱   (80 pixels de largeur)   │
                  └────────────────────────────────────┘

ratio = 80 / 384 = 0.21  →  21% de l'image → pas transversale
```

**Seuil actuel : 65% (0.65)**
Si la boîte englobante couvre plus de 65% de la dimension de l'image dans au moins une
direction, la fissure est classifiée comme **transversale**.

---

## 6. L'indice de danger composite

### Pourquoi un indice unique ?

On a maintenant 3 informations :
- L'orientation (horizontale / verticale / inclinée)
- La localisation (superficielle / profonde / transversale)
- La largeur et l'étendue (chiffres précis)

Pour aider à la prise de décision, on calcule un **indice de danger** entre 0 et 1 qui
synthétise tout ça.

**0.0** = aucun danger apparent
**0.5** = attention requise
**1.0** = danger critique, inspection immédiate recommandée

### La formule

```
Indice = 0.45 × score_localisation
       + 0.20 × score_orientation
       + 0.20 × (largeur / 20 pixels)
       + 0.15 × ratio_couverture
```

Les poids (0.45, 0.20, 0.20, 0.15) reflètent l'importance relative de chaque facteur
en génie civil.

**Scores par type :**

| Localisation | Score |
|---|---|
| Superficielle | 0.20 |
| Profonde | 0.75 |
| Transversale | 1.00 |

| Orientation | Score |
|---|---|
| Horizontale | 0.30 |
| Inclinée | 0.60 |
| Verticale | 0.85 |

### Exemples de calcul

**Fissure 1 — Petite fissure superficielle horizontale**
```
Localisation  = superficielle → 0.20 × 0.45 = 0.090
Orientation   = horizontale   → 0.30 × 0.20 = 0.060
Largeur       = 2px / 20     → 0.10 × 0.20 = 0.020
Couverture    = 15%           → 0.15 × 0.15 = 0.023
─────────────────────────────────────────────────────
Indice de danger = 0.193  ≈  0.19   ← Faible risque
```

**Fissure 2 — Fissure verticale profonde**
```
Localisation  = profonde    → 0.75 × 0.45 = 0.338
Orientation   = verticale   → 0.85 × 0.20 = 0.170
Largeur       = 10px / 20  → 0.50 × 0.20 = 0.100
Couverture    = 40%         → 0.40 × 0.15 = 0.060
─────────────────────────────────────────────────────
Indice de danger = 0.668  ≈  0.67   ← Risque élevé
```

**Fissure 3 — Fissure transversale inclinée**
```
Localisation  = transversale → 1.00 × 0.45 = 0.450
Orientation   = inclinée     → 0.60 × 0.20 = 0.120
Largeur       = 8px / 20    → 0.40 × 0.20 = 0.080
Couverture    = 80%          → 0.80 × 0.15 = 0.120
─────────────────────────────────────────────────────
Indice de danger = 0.770  ≈  0.77   ← Risque critique
```

### Interprétation pratique de l'indice

| Plage | Interprétation | Action recommandée |
|---|---|---|
| 0.00 – 0.35 | Risque faible | Surveillance photographique |
| 0.35 – 0.65 | Risque modéré | Inspection par un technicien |
| 0.65 – 0.85 | Risque élevé | Expertise d'un ingénieur civil |
| 0.85 – 1.00 | Risque critique | Intervention urgente |

**Important :** l'indice est un **outil d'aide à la décision**, pas un diagnostic médical.
Toute décision structurelle doit être validée par un ingénieur civil qualifié.

---

## 7. Comment utiliser le script d'analyse

### Commande de base

```bash
python analyser.py \
    --modele sorties_yolo26/entrainements/yolo_seg_fissures/weights/best.pt \
    --images chemin/vers/mes/photos/
```

### Avec dossier de sortie personnalisé

```bash
python analyser.py \
    --modele sorties_yolo26/entrainements/yolo_seg_fissures/weights/best.pt \
    --images chemin/vers/mes/photos/ \
    --dossier-sortie resultats/
```

### Paramètres disponibles

| Paramètre | Valeur par défaut | Explication |
|---|---|---|
| `--modele` | *obligatoire* | Chemin vers `best.pt` YOLO-seg ou un `.pth` Mask R-CNN |
| `--backend` | `yolo` | `yolo` par défaut, `maskrcnn` pour un checkpoint `.pth` |
| `--images` | *obligatoire* | Dossier contenant les photos à analyser |
| `--seuil` | 0.25 YOLO / 0.4 Mask | Score de confiance minimal pour analyser une fissure. |
| `--taille-image` | 384 YOLO / config Mask | Résolution d'inférence. YOLO exige un multiple de 32. |
| `--dispositif` | auto | `cuda` pour GPU NVIDIA, `cpu` pour processeur |
| `--dossier-sortie` | `analyses` | Dossier racine du rapport et des images annotées |
| `--sortie` | `analyses/<backend>/rapport_analyse.json` | Chemin du fichier JSON si vous voulez le nommer vous-même |
| `--sans-images` | désactivé | Ne produit que le JSON, sans copies annotées |

### Formats d'images acceptés

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

---

## 8. Lire les résultats

### Affichage dans le terminal

```
════════════════════════════════════════════════════════════════════
  CLASSIFICATION DES FISSURES — mur_batiment_A.jpg
════════════════════════════════════════════════════════════════════
  #   | Orientation   | Localisation  |  Angle  |  Largeur | Danger | Score
  ---+-+---------------+-+-------------+-+-------+-+--------+-+------+-+------
    1 | verticale     | profonde      |  81.3°  |    9.2px |  0.71 ▓▓▓ |  0.87
    2 | horizontale   | superficielle |   7.8°  |    2.1px |  0.18 ▓░░ |  0.72
    3 | inclinée      | transversale  |  43.2°  |    5.8px |  0.77 ▓▓▓ |  0.65
────────────────────────────────────────────────────────────────────

  Répartition par orientation :
    horizontale     :   1  █
    inclinée        :   1  █
    verticale       :   1  █

  Répartition par localisation :
    profonde        :   1  █
    superficielle   :   1  █
    transversale    :   1  █

  Indice de danger  → Moyen : 0.55  |  Max : 0.77
════════════════════════════════════════════════════════════════════
```

**Comment lire chaque colonne :**

- **#** : numéro de la fissure sur cette image
- **Orientation** : le type déterminé par la PCA
- **Localisation** : superficielle, profonde ou transversale
- **Angle** : l'angle précis en degrés (0°=horizontal, 90°=vertical)
- **Largeur** : largeur estimée en pixels (médiane de la distance transform × 2)
- **Danger** : l'indice composite [0, 1] + indicateur visuel (▓ = danger, ░ = sûr)
- **Score** : le score de confiance du modèle pour cette détection

### Fichiers produits

Par défaut :

```text
analyses/
└── yolo/
    ├── rapport_analyse.json
    └── images_annotees/
        ├── mur_batiment_A_analyse.jpg
        └── ...
```

Les images d'origine restent dans le dossier fourni avec `--images`. Le dossier
`images_annotees/` contient seulement des copies avec contours, boîtes et labels.

### Le fichier JSON de résultats

Un fichier JSON est créé. Son contenu :

```json
{
  "parametres": {
    "modele": "sorties_yolo26/entrainements/yolo_seg_fissures/weights/best.pt",
    "backend": "yolo",
    "seuil": 0.4,
    "taille_image": 384,
    "dossier_images_annotees": "analyses/yolo/images_annotees"
  },
  "resultats": [
    {
      "image": "mur_batiment_A.jpg",
      "chemin_image_source": "photos_test/mur_batiment_A.jpg",
      "chemin_image_annotee": "analyses/yolo/images_annotees/mur_batiment_A_analyse.jpg",
      "fissures": [
        {
          "id": 1,
          "orientation": "verticale",
          "localisation": "profonde",
          "angle_degres": 81.3,
          "largeur_moy_pixels": 9.2,
          "ratio_couverture": 0.38,
          "aire_pixels": 1240,
          "indice_danger": 0.71,
          "score_detection": 0.87
        },
        {
          "id": 2,
          "orientation": "horizontale",
          "localisation": "superficielle",
          "angle_degres": 7.8,
          "largeur_moy_pixels": 2.1,
          "ratio_couverture": 0.12,
          "aire_pixels": 340,
          "indice_danger": 0.18,
          "score_detection": 0.72
        }
      ],
      "statistiques": {
        "nombre_fissures": 2,
        "danger_moyen": 0.445,
        "danger_maximum": 0.71,
        "orientation": {
          "verticale": 1,
          "horizontale": 1
        },
        "localisation": {
          "profonde": 1,
          "superficielle": 1
        }
      }
    }
  ],
  "resume": {
    "images_analysees": 1,
    "total_fissures": 2
  }
}
```

**Explication de chaque champ :**

| Champ | Signification |
|---|---|
| `orientation` | Type d'orientation : `"horizontale"`, `"verticale"` ou `"inclinée"` |
| `localisation` | Type de localisation : `"superficielle"`, `"profonde"` ou `"transversale"` |
| `angle_degres` | Angle précis en degrés [0, 90] |
| `largeur_moy_pixels` | Largeur médiane estimée en pixels |
| `ratio_couverture` | Fraction de l'image couverte par la boîte englobante [0, 1] |
| `aire_pixels` | Nombre total de pixels blancs dans le masque |
| `indice_danger` | Score de danger composite [0, 1] |
| `score_detection` | Confiance du modèle pour cette détection [0, 1] |

---

## 9. Ajuster les seuils de classification

Si vous trouvez que les classifications ne correspondent pas à vos observations terrain,
vous pouvez ajuster les seuils dans le fichier :

```
detection_fissures/analyse/classificateur_fissures.py
```

Lignes à modifier (tout en haut du fichier) :

```python
SEUIL_ANGLE_HORIZONTAL: float = 20.0   # Degrés : en dessous → horizontale
SEUIL_ANGLE_VERTICAL: float   = 70.0   # Degrés : au dessus → verticale

SEUIL_LARGEUR_SUPERFICIELLE: float = 3.5   # Pixels : en dessous → superficielle
SEUIL_LARGEUR_PROFONDE: float      = 7.0   # Pixels : au dessus → profonde

SEUIL_TRAVERSEE: float = 0.65   # Ratio : au dessus → transversale
```

### Exemple d'ajustement selon la résolution caméra

Si vos images représentent une surface de 60 cm × 60 cm sur 384 pixels :
- Résolution = 600 mm / 384 px ≈ 1.56 mm/px

Pour classer "superficielle" les fissures < 0.5 mm :
```python
SEUIL_LARGEUR_SUPERFICIELLE = 0.5 / 1.56 ≈ 0.32 pixels  # très fin
```

Pour classer "profonde" les fissures > 2 mm :
```python
SEUIL_LARGEUR_PROFONDE = 2.0 / 1.56 ≈ 1.28 pixels
```

### Quand augmenter le seuil de confiance `--seuil` ?

| Problème observé | Solution |
|---|---|
| Trop de fissures fantômes (fausses alarmes) | Augmenter `--seuil 0.6` ou `0.7` |
| Des vraies fissures ne sont pas détectées | Baisser `--seuil 0.3` |
| Résultats instables (varie beaucoup) | Essayer `--seuil 0.5` (compromis) |

---

*Ce guide est spécifique au projet de détection de fissures. Pour comprendre
l'entraînement du modèle, se référer au guide d'entraînement.*
