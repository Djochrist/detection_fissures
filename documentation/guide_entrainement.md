# Guide d'Entraînement — Détection de Fissures par IA

> **À qui s'adresse ce guide ?**
> À toute personne qui veut comprendre comment fonctionne l'entraînement du modèle,
> même sans formation en informatique ou en intelligence artificielle.
> Chaque terme technique est expliqué simplement, avec des exemples tirés
> directement de ce projet.

---

## Sommaire

1. [Qu'est-ce qu'on fait concrètement ?](#1-quest-ce-quon-fait-concrètement)
2. [Le dataset : nos images annotées](#2-le-dataset--nos-images-annotées)
3. [Mask R-CNN : comment le modèle "voit" une fissure](#3-mask-r-cnn--comment-le-modèle-voit-une-fissure)
4. [Le transfer learning : ne pas repartir de zéro](#4-le-transfer-learning--ne-pas-repartir-de-zéro)
5. [Les 3 phases d'entraînement](#5-les-3-phases-dentraînement)
6. [Les paramètres clés expliqués](#6-les-paramètres-clés-expliqués)
7. [Les métriques : comment mesurer si le modèle est bon](#7-les-métriques--comment-mesurer-si-le-modèle-est-bon)
8. [Comment lire les sorties du terminal](#8-comment-lire-les-sorties-du-terminal)
9. [Lancer l'entraînement](#9-lancer-lentraînement)
10. [Problèmes courants et solutions](#10-problèmes-courants-et-solutions)

---

## 1. Qu'est-ce qu'on fait concrètement ?

L'objectif est d'apprendre à un programme informatique à **repérer et délimiter les fissures**
dans des photos de murs, de béton ou de maçonnerie, **comme le ferait un expert en génie civil**
en regardant une photo.

### L'analogie de l'inspecteur

Imaginez un jeune inspecteur qui n'a jamais vu de fissures. Pour le former :
- On lui montre **34 349 photos** de murs (notre dataset augmenté)
- Sur chaque photo, on lui dit exactement **où se trouve la fissure** (les annotations)
- Il regarde, se trompe, se corrige, recommence — c'est ça l'**entraînement**
- Après 50 séances d'apprentissage (les **époques**), il est capable de trouver les fissures tout seul

Notre programme apprend exactement de la même façon.

### Ce que le modèle produit

Pour chaque photo analysée, le modèle retourne :
- **Des boîtes englobantes** : des rectangles autour de chaque fissure
- **Des masques de segmentation** : le contour exact pixel par pixel de chaque fissure
- **Des scores de confiance** : entre 0 et 1, "je suis sûr à 87% que c'est une fissure"

---

## 2. Le dataset : nos images annotées

### Structure du dataset

Notre dataset vient de **4 907 photos sources** prises de murs et structures.
Roboflow a créé **10 versions augmentées** de chaque image d'entraînement.
Il est divisé en 3 groupes selon un ratio **70 % / 15 % / 15 %** :

```

> Les images de validation et de test ne sont **pas augmentées** : elles restent dans
> leur état original pour mesurer les vraies performances du modèle face à des images
> qu'il n'a jamais vues sous une forme modifiée.

**Pourquoi 3 groupes séparés ?**

Imaginez que vous préparez un examen :
- `train/` : c'est votre **cahier de cours**, vous l'étudiez en boucle
- `valid/` : ce sont les **contrôles blancs** pendant la préparation — vous vérifiez où vous en êtes sans que ça compte
- `test/` : c'est l'**examen final** — une seule fois, à la fin, pour avoir la vraie note

Si on utilisait les mêmes images pour apprendre et pour évaluer, on mesurerait si le modèle a
mémorisé les images, pas s'il sait vraiment détecter les fissures.

### Le format COCO

COCO est un format de fichier standard pour les annotations d'images.
Chaque dossier contient un fichier `_annotations.coco.json` qui ressemble à ceci (simplifié) :

```json
{
  "images": [
    { "id": 1, "file_name": "mur_001.jpg", "width": 384, "height": 384 }
  ],
  "annotations": [
    {
      "image_id": 1,
      "segmentation": [[120, 45, 135, 50, 140, 80, ...]],
      "bbox": [120, 45, 20, 35]
    }
  ]
}
```

- `segmentation` : la liste des coordonnées des pixels qui forment le contour de la fissure
- `bbox` : la boîte englobante [x, y, largeur, hauteur]

### La résolution des images

Toutes nos images sont en **384 × 384 pixels** (carré).
C'est la résolution choisie par Roboflow lors de l'export du dataset.
Le modèle recevra toujours des images de cette taille exacte.

### Ce qui a déjà été fait (prétraitement Roboflow)

Le dataset a déjà été préparé et augmenté par Roboflow. Chaque image originale a été
transformée en plusieurs variantes pour enrichir l'apprentissage :

| Transformation appliquée | Paramètre exact | Pourquoi |
|---|---|---|
| Auto-orientation EXIF | — | Corrige l'orientation si la caméra était penchée lors de la prise de vue |
| Redimensionnement | 384 × 384 px (étirement) | Standardise toutes les images à la même taille |
| Flip horizontal (miroir) | 50 % de probabilité | La même fissure peut être à gauche ou à droite |
| Rotations 90° | Aucune / horaire / anti-horaire (équiprobable) | La photo peut être prise sous différents angles |
| Recadrage aléatoire | 0 – 20 % de la boîte | L'inspecteur ne voit jamais tout le mur |
| Rotation aléatoire | –15° à +15° | La caméra n'est jamais parfaitement droite |
| Cisaillement (shear) | ±10° horizontal et vertical | Simule une perspective légèrement oblique |
| Variation de luminosité | ±15 % | Éclairage différent selon l'heure et la saison |
| Variation d'exposition | ±10 % | Simule une caméra sur/sous-exposée |
| Flou gaussien | 0 – 2.5 pixels | Simulation d'une caméra pas parfaitement mise au point |
| Bruit sel-et-poivre | 0.1 % des pixels | Simulation d'un capteur de mauvaise qualité |

**Conséquence pratique :** notre code ne fait aucune augmentation supplémentaire.
On charge les images telles quelles et on applique uniquement la **normalisation ImageNet**
(explication ci-dessous).

---

## 3. Mask R-CNN : comment le modèle "voit" une fissure

### L'architecture en langage simple

Mask R-CNN est un programme organisé en plusieurs étapes qui se passent l'une après l'autre.
Voici comment une image passe à travers le modèle :

```
Photo de mur (384×384 pixels)
        ↓
┌─────────────────────────────────┐
│  BACKBONE (les "yeux")          │
│  ResNet-50                      │
│  → Lit l'image et extrait des   │
│    "caractéristiques" invisibles│
│    (textures, contours, formes) │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  FPN (vue multi-échelle)        │
│  Feature Pyramid Network        │
│  → Cherche les fissures à       │
│    TOUTES les tailles en même   │
│    temps (fine et large)        │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  RPN (le "chercheur")           │
│  Region Proposal Network        │
│  → Propose des zones suspectes  │
│    "peut-être une fissure ici"  │
└─────────────────────────────────┘
        ↓
┌──────────────┬──────────────────┐
│ TÊTE BOÎTES  │  TÊTE MASQUES    │
│ → Affine les │  → Dessine le    │
│   rectangles │  contour exact   │
│   autour des │  de chaque       │
│   fissures   │  fissure         │
└──────────────┴──────────────────┘
        ↓
Résultat : fissures localisées et segmentées
```

### Qu'est-ce que le backbone ?

Le **backbone** (littéralement "colonne vertébrale") est la partie du modèle qui lit l'image
et la transforme en informations numériques que le reste du programme peut utiliser.

Ici on utilise **ResNet-50** : un réseau qui a été entraîné sur 1,2 million d'images de la vie
courante (chiens, voitures, maisons, etc.). Il sait donc déjà reconnaître des contours, des
textures, des formes. On réutilise ce savoir pour détecter les fissures.

### Qu'est-ce que le FPN ?

Le **FPN** (Feature Pyramid Network) permet au modèle de voir l'image à plusieurs niveaux
de détail en même temps :

- Vue "de loin" : repère les grandes fissures larges
- Vue "de près" : repère les microfissures capillaires

Sans le FPN, le modèle manquerait soit les grandes fissures soit les petites.

### Pourquoi Mask R-CNN et pas YOLO ou autre chose ?

| Question | Mask R-CNN | YOLO |
|---|---|---|
| Segmentation d'instance ? | Oui — chaque fissure a son propre masque | Oui mais moins précis |
| Précision des masques ? | Très haute | Correcte |
| Adapté à petit dataset ? | Oui (transfer learning COCO) | Oui mais moins |
| Analyse géométrique possible ? | Oui — masques précis | Difficile |

Pour analyser la **largeur, l'orientation et la longueur** de chaque fissure, on a besoin de
masques très précis pixel par pixel. C'est pourquoi Mask R-CNN est le meilleur choix ici.

---

## 4. Le transfer learning : ne pas repartir de zéro

### L'analogie de la langue

Imaginez que vous parlez parfaitement l'espagnol et qu'on vous demande d'apprendre le
portugais. Vous ne repartez pas de zéro : vous utilisez déjà les accents, la grammaire
romane, les racines latines communes. Vous apprenez 10 fois plus vite.

Le **transfer learning** fonctionne de la même façon pour les réseaux de neurones :
- Notre modèle a déjà appris à reconnaître des formes sur 1,2 million d'images (ImageNet)
- On "réutilise" ce savoir pour apprendre à détecter les fissures
- Résultat : on atteint de bonnes performances avec seulement 4 907 images sources
  (soit ~35 822 images après augmentation)
  (sans transfer learning, il en faudrait des centaines de milliers)

### La normalisation ImageNet — pourquoi est-elle obligatoire ?

Quand ResNet-50 a appris sur ImageNet, les images étaient normalisées avec des valeurs
spécifiques. Si on lui donne des images non-normalisées, c'est comme lui parler dans une
langue qu'il ne reconnaît pas.

Concrètement : au lieu de valeurs de pixels entre 0 et 255, on les transforme pour avoir
une moyenne autour de 0. Ces valeurs exactes sont utilisées dans notre code :

```
Moyenne par canal RGB : [0.485, 0.456, 0.406]
Écart-type par canal : [0.229, 0.224, 0.225]
```

**Ce que ça change en pratique :** un pixel blanc (255, 255, 255) devient environ
(2.24, 2.43, 2.64) après normalisation. Le modèle "comprend" mieux ces valeurs.

---

## 5. Les 3 phases d'entraînement

L'entraînement de notre modèle est divisé en 3 phases progressives.
C'est une stratégie pour **éviter que le modèle oublie ce qu'il sait déjà** tout en
l'adaptant aux fissures.

### Phase 1 — Époques 1 à 5 : Backbone gelé

**Ce qui se passe :**
Le backbone (ResNet-50) est **gelé** — ses paramètres ne changent pas.
Seules les **têtes** (la partie qui prédit les boîtes et les masques) apprennent.

**L'analogie :**
Un expert en peinture change de spécialité pour étudier les fissures. Au début, il garde
son œil formé (backbone gelé) et apprend juste les nouveaux concepts spécifiques aux
fissures (têtes entraînées). S'il changeait tout en même temps, il risquerait de "désapprendre"
ses compétences de base.

**Résultat attendu :**
Les pertes diminuent rapidement car les têtes convergent vite.

```
Époque 1 | Perte = 3.21  ↓
Époque 2 | Perte = 2.87  ↓
Époque 3 | Perte = 2.54  ↓
Époque 4 | Perte = 2.31  ↓
Époque 5 | Perte = 2.15  ↓  ← transition phase 2
```

### Phase 2 — Époques 5 à 15 : Dégelage progressif

**Ce qui se passe :**
Les couches **layer3 et layer4** du backbone sont dégelées et commencent à apprendre,
mais avec un taux d'apprentissage 10 fois plus faible que les têtes.

**Pourquoi seulement layer3 et layer4 ?**
ResNet-50 a 4 blocs de couches (layer1, layer2, layer3, layer4).
- Layer1 et layer2 apprennent des choses très génériques (contours, textures basiques)
  — pas besoin de les modifier
- Layer3 et layer4 apprennent des formes complexes — utile pour les fissures

**L'analogie :**
Notre expert commence à "réentraîner son œil" pour voir spécifiquement les fissures,
mais doucement pour ne pas perdre ses compétences générales.

### Phase 3 — Époque 15 et au-delà : Fine-tuning complet

**Ce qui se passe :**
Tout le modèle est dégelé et apprend ensemble.
Le taux d'apprentissage continue de diminuer progressivement (scheduler cosinus).

**L'analogie :**
L'expert est maintenant suffisamment stable pour affiner toute sa vision en même temps,
jusqu'à atteindre sa meilleure performance possible.

---

## 6. Les paramètres clés expliqués

Voici chaque paramètre de notre fichier `configuration/parametres.py` expliqué simplement.

### La taille du lot (`taille_lot = 4`)

**C'est quoi ?** Le nombre d'images traitées en même temps avant de corriger le modèle.

**L'analogie :** Imaginez corriger des copies d'examen. Vous pouvez :
- Corriger une copie, puis ajuster vos critères, puis corriger la suivante (lot=1, lent)
- Corriger 4 copies en même temps, puis ajuster une fois (lot=4, plus rapide)
- Corriger 16 copies ensemble (lot=16, très rapide mais besoin de beaucoup de mémoire)

**Valeur choisie : 4**
- Fonctionne sur un GPU de 8 Go (comme Colab T4)
- Augmenter à 8 si vous avez un GPU de 16 Go+

### Le nombre d'époques (`nombre_epoques = 50`)

**C'est quoi ?** Le nombre de fois que le modèle voit l'intégralité du dataset.

**L'analogie :** Réviser un cours 50 fois. À chaque révision, on comprend un peu mieux.

**Valeur choisie : 50**
L'early stopping (voir ci-dessous) arrêtera automatiquement si le modèle ne progresse plus.

### Le taux d'apprentissage (`taux_apprentissage = 0.0001`)

**C'est quoi ?** La taille du "pas" que le modèle fait à chaque correction.

**L'analogie :** Chercher le fond d'une vallée dans le brouillard.
- Pas trop grand (0.1) → vous risquez de dépasser le fond et d'aller de l'autre côté
- Pas trop petit (0.000001) → vous mettez une éternité à descendre
- Valeur équilibrée (0.0001) → vous descendez régulièrement vers le fond

**Valeur choisie : 0.0001** — standard pour le fine-tuning de modèles préentraînés.

### La décroissance des poids (`decroissance_poids = 0.0005`)

**C'est quoi ?** Une pénalité appliquée à chaque correction pour éviter que les paramètres
deviennent trop grands (ce qui causerait du "surajustement").

**L'analogie :** Un étudiant qui révise se rappelle mieux les grandes idées que les détails
trop précis. La décroissance des poids empêche le modèle de mémoriser des détails
anecdotiques qui ne sont vrais que pour les images d'entraînement.

### L'early stopping (`patience_arret_precoce = 10`)

**C'est quoi ?** L'entraînement s'arrête automatiquement si le modèle ne s'améliore plus
pendant 10 époques consécutives.

**L'analogie :** Si votre score à un examen blanc ne progresse plus depuis 10 séances de
révision, c'est probablement que vous avez atteint votre maximum et que continuer ne sert
à rien (voire nuit en mémorisant trop).

**Valeur choisie : 10 époques**

### Le scheduler cosinus (`CosineAnnealingLR`)

**C'est quoi ?** Le taux d'apprentissage diminue progressivement au cours de l'entraînement
selon une courbe en cosinus.

**Pourquoi cosinus ?** Au début de l'entraînement, on fait de grands ajustements. Vers la
fin, on fait de petits ajustements fins pour ne pas rater le meilleur point.

```
Époque 1  → LR = 0.0001  (grand pas)
Époque 25 → LR = 0.00005 (pas moyen)
Époque 50 → LR = 0.000001 (tout petit pas)
```

### La précision mixte (`precision_mixte = True`)

**C'est quoi ?** Le GPU calcule en "demi-précision" (float16) au lieu de "pleine précision"
(float32), ce qui va 2 à 3 fois plus vite et utilise 2 fois moins de mémoire.

**Est-ce que ça change les résultats ?** Très légèrement, mais la différence est négligeable
sur les performances du modèle. C'est activé par défaut car le gain de vitesse est significatif.

**Note :** Uniquement disponible sur GPU NVIDIA (CUDA). Sur CPU, ce paramètre est ignoré.

### Le gradient clipping (`valeur_clip_gradient = 1.0`)

**C'est quoi ?** Si les corrections deviennent anormalement grandes (ce qui peut arriver
en début d'entraînement), on les limite artificiellement à une valeur maximale de 1.0.

**L'analogie :** Un étudiant qui révise et soudain "réalise" qu'il avait tout faux ne va pas
tout réécrire d'un coup — il fait des corrections progressives et raisonnées.

### La graine aléatoire (`graine_aleatoire = 42`)

**C'est quoi ?** Un nombre qui fixe tous les hasards du programme (l'ordre de mélange
des images, l'initialisation des poids, etc.).

**Pourquoi 42 ?** C'est une convention humoristique dans le domaine (référence à
"La Grande Question sur la Vie, l'Univers et le Reste").

**Utilité pratique :** En relançant l'entraînement avec la même graine, vous obtenez
exactement les mêmes résultats — parfait pour vérifier qu'une modification a vraiment
amélioré les performances.

---

## 7. Les métriques : comment mesurer si le modèle est bon

### La perte (loss) — pendant l'entraînement

**C'est quoi ?** Un nombre qui mesure à quel point le modèle se trompe.
Plus la perte est basse, mieux c'est.

Notre modèle calcule 5 pertes internes en même temps :

| Nom interne | Ce qu'il mesure | Explication simple |
|---|---|---|
| `loss_classifier` | Erreur de classification | "Est-ce une fissure ou pas ?" |
| `loss_box_reg` | Erreur de localisation | "Le rectangle est-il bien positionné ?" |
| `loss_mask` | Erreur de segmentation | "Le contour de la fissure est-il précis ?" × 2 |
| `loss_objectness` | Erreur de détection | "Le RPN a-t-il bien repéré les zones suspectes ?" |
| `loss_rpn_box_reg` | Erreur RPN | "Les propositions de zones sont-elles bonnes ?" |

La **perte totale = somme de ces 5 pertes**, avec `loss_mask` multipliée par 2
car la précision des masques est notre priorité principale.

**Valeurs typiques au cours de l'entraînement :**
```
Époque 1  → Perte totale ≈ 3.5–5.0  (le modèle débute)
Époque 10 → Perte totale ≈ 1.5–2.5  (amélioration notable)
Époque 30 → Perte totale ≈ 0.8–1.5  (modèle qui converge)
Époque 50 → Perte totale ≈ 0.5–1.0  (modèle stable)
```

### Le mAP — la métrique principale d'évaluation

**mAP** = Mean Average Precision = Précision Moyenne sur l'ensemble des classes.

C'est la métrique internationale standard pour évaluer les détecteurs d'objets (utilisée
dans les compétitions COCO et Pascal VOC).

**mAP@0.5** (notre métrique principale) :
- On dit qu'une détection est "bonne" si la fissure prédite recouvre au moins 50%
  de la vraie fissure (IoU ≥ 0.5)
- On calcule la précision sur toutes les images de validation
- On fait la moyenne

**Exemple concret :**

```
Image 1 : 3 fissures réelles
  → Le modèle en trouve 3, toutes bien délimitées  → Bonne détection ✓ ✓ ✓

Image 2 : 2 fissures réelles
  → Le modèle en trouve 3 (1 de trop = fausse alarme)  → 2 bonnes, 1 fausse ✓ ✓ ✗

Image 3 : 4 fissures réelles
  → Le modèle en trouve 3 (en manque 1)  → 3 bonnes, 1 manquée ✓ ✓ ✓ ✗

mAP@0.5 ≈ 0.78  →  le modèle est bon à 78% selon ce critère
```

### L'IoU — comment mesure-t-on si une détection est "bonne" ?

**IoU** = Intersection over Union = chevauchement entre la prédiction et la vérité.

```
     Vraie fissure
    ┌──────────┐
    │  ██████  │  ← Zone commune (intersection)
    │  ██████  │
    └──────────┘
  ┌──────────────┐
  │ Prédiction   │
  └──────────────┘

IoU = Surface commune / (Surface union totale)
    = 6 / 10 = 0.60  →  60% de chevauchement
```

- IoU = 1.0 → prédiction parfaite (identique à la réalité)
- IoU = 0.5 → chevauchement de 50% → acceptable
- IoU < 0.5 → mauvaise détection

### La Précision et le Rappel

**Précision** = Parmi toutes les fissures détectées par le modèle, combien sont réellement
des fissures ?

**Rappel** = Parmi toutes les vraies fissures dans les images, combien le modèle en a-t-il
trouvé ?

**Exemple concret sur nos images :**
```
Réalité : 100 fissures dans le jeu de test
Le modèle trouve : 90 vraies fissures + 15 fausses alarmes

Précision = 90 / (90 + 15) = 85.7%   ("85% de mes détections sont correctes")
Rappel    = 90 / 100       = 90.0%   ("j'ai trouvé 90% des vraies fissures")
```

**Quelle métrique privilégier pour la sécurité structurelle ?**

Le **rappel** est plus important. Manquer une fissure réelle (faux négatif) est plus
dangereux qu'une fausse alarme (faux positif) dans un contexte de génie civil.

**Objectifs raisonnables pour notre projet :**
```
mAP@0.5   > 0.60  → acceptable  |  > 0.80  → excellent
Rappel     > 0.85  → acceptable  |  > 0.92  → excellent
Précision  > 0.80  → acceptable  |  > 0.90  → excellent
```

### Le F1-Score

**F1** = 2 × (Précision × Rappel) / (Précision + Rappel)

C'est la moyenne harmonique entre précision et rappel. Il résume les deux en un seul
chiffre. F1 = 0.88 signifie que le modèle est globalement bon dans les deux dimensions.

---

## 8. Comment lire les sorties du terminal

Pendant l'entraînement, le programme affiche des informations régulièrement.
Voici comment les interpréter.

### Affichage en cours d'époque

```
  Époque   3/50 | Lot  10/235 | Perte = 2.4531
  Époque   3/50 | Lot  20/235 | Perte = 2.2874
  Époque   3/50 | Lot  30/235 | Perte = 2.1203
```

- `Époque 3/50` : on est à la 3ème époque sur 50 prévues
- `Lot 10/235` : on a traité 10 lots sur 235 dans cette époque
  (235 lots = 3 760 images ÷ 4 images par lot ≈ 940, ou selon la taille du dataset/lot)
- `Perte = 2.45` : la perte moyenne des 10 derniers lots — doit diminuer au fil du temps

### Affichage de fin d'époque

```
────────────────────────────────────────────────────────────
  Époque   3/ 50 | Durée : 42.3s
  Perte train   : 2.1847
  mAP@0.5 valid : 0.3412 ↑  (meilleur : 0.3412 @ ép.3)
  Patience      : 0/10
────────────────────────────────────────────────────────────
```

- `Durée : 42.3s` : cette époque a pris 42 secondes
- `Perte train : 2.18` : perte moyenne sur toutes les images d'entraînement
- `mAP@0.5 valid : 0.34 ↑` : performance sur la validation — la flèche ↑ indique une amélioration
- `Patience : 0/10` : on repart à 0 car il y a eu amélioration

### Affichage d'un changement de phase

```
═══ PHASE 2 : Dégelage couches supérieures ═══
```
Cela indique que l'époque 5 est atteinte et que layer3/layer4 du backbone sont maintenant
actifs dans l'apprentissage.

### Affichage d'early stopping

```
⚠ Arrêt anticipé à l'époque 28 (patience=10 atteinte)
  Meilleur modèle : époque 18 avec mAP@0.5 = 0.7234
```

Cela signifie que depuis l'époque 18, le modèle n'a pas fait mieux pendant 10 époques.
Le meilleur modèle sauvegardé est celui de l'époque 18 avec mAP@0.5 = 0.72.

### Résultats finaux sur le jeu de test

```
═══════════════════════════════════════════════════════
  MÉTRIQUES — MASK R-CNN (SEGMENTATION D'INSTANCE)
═══════════════════════════════════════════════════════
  mAP [0.5:0.95]  (Standard COCO) : 0.5234  ██████████░░░░░░░░░░
  mAP @ IoU=0.50  (Principal)     : 0.7812  ████████████████░░░░
  mAP @ IoU=0.75  (Précis)        : 0.4521  █████████░░░░░░░░░░░
  Précision       (VP / VP+FP)    : 0.8345  █████████████████░░░
  Rappel          (VP / VP+FN)    : 0.8923  ██████████████████░░
  F1 Score        (2PR / P+R)     : 0.8624  █████████████████░░░
═══════════════════════════════════════════════════════
```

Les barres `█` sont proportionnelles à la valeur (1.0 = 20 blocs pleins).

---

## 9. Lancer l'entraînement

### Commande de base

```bash
python entrainer.py
```

Utilise les paramètres par défaut :
- Dataset dans `dataset/`
- 50 époques
- Lot de 4 images
- LR = 0.0001
- Taille image : 384×384

### Commandes avancées

```bash
# Changer le dossier du dataset
python entrainer.py --donnees /chemin/vers/mon/dataset

# Entraînement plus long avec plus de mémoire
python entrainer.py --epoques 100 --lot 8

# Entraînement plus précis avec LR plus faible
python entrainer.py --lr 5e-5

# Forcer l'utilisation du CPU (si pas de GPU)
python entrainer.py --dispositif cpu

# Désactiver la précision mixte (si problèmes de compatibilité GPU)
python entrainer.py --sans-mixte
```

### Fichiers produits après l'entraînement

```
sorties/
├── modeles/
│   ├── meilleur_modele.pth   ← LE PLUS IMPORTANT : le modèle à la meilleure époque
│   └── dernier_modele.pth    ← Le modèle à la dernière époque (pas forcément le meilleur)
└── journaux/
    └── historique_entrainement.json  ← Toutes les métriques époque par époque
```

### Sur Google Colab

```python
# 1. Monter votre Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Placer votre dataset dans Drive puis lancer :
!python entrainer.py --donnees /content/drive/MyDrive/dataset --lot 8
```

---

## 10. Problèmes courants et solutions

### "CUDA out of memory" (mémoire GPU insuffisante)

**Symptôme :** `RuntimeError: CUDA out of memory. Tried to allocate X MiB`

**Solutions :**
```bash
# Réduire la taille du lot
python entrainer.py --lot 2

# Désactiver la précision mixte (moins efficace mais plus stable)
python entrainer.py --sans-mixte
```

### La perte ne diminue pas (plateau dès le début)

**Symptôme :** La perte reste à 4.0 ou plus après 5 époques.

**Causes possibles et solutions :**

1. **Taux d'apprentissage trop élevé** → essayer `--lr 1e-5`
2. **Problème dans les annotations** → vérifier que `_annotations.coco.json` est correct
3. **Dataset mal structuré** → vérifier que les images sont bien dans `train/`, `valid/`, `test/`

### La validation ne s'améliore plus mais la perte train continue de baisser

**Symptôme :** `Perte train` descend mais `mAP@0.5 valid` stagne ou monte.

**Diagnostic :** C'est du **surajustement** (overfitting) — le modèle mémorise les images
d'entraînement au lieu de vraiment comprendre les fissures.

**Solutions :**
- Réduire le nombre d'époques (`--epoques 30`)
- Augmenter la décroissance des poids dans `parametres.py` : `decroissance_poids = 1e-3`
- L'early stopping gère ça automatiquement — vérifier que `patience = 10` est bien actif

### "FileNotFoundError: Image introuvable"

**Symptôme :** `FileNotFoundError: Image introuvable : dataset/train/image_001.jpg`

**Solution :** Vérifier que les chemins dans `_annotations.coco.json` correspondent
exactement aux noms des fichiers présents dans le dossier.
```bash
# Vérifier que le fichier existe
ls dataset/train/ | head -5
```

---

