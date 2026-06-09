# Guide d'entraînement — Détection de Fissures Structurelles

Ce guide explique comment préparer ton dataset, entraîner le modèle, et utiliser le résultat pour analyser des images.

---

## Prérequis

### Matériel recommandé

| Configuration | RAM | Durée estimée (60 époques) |
|---|---|---|
| CPU seul | 8 Go minimum | 12-48h |
| GPU NVIDIA 8 Go | 8 Go RAM | 2-4h |
| GPU NVIDIA 16 Go | 16 Go RAM | 1-2h |

### Python

Version 3.10 ou 3.11 recommandée.

### Installation des dépendances

**Sur CPU (sans GPU) :**
```bash
pip install opencv-python numpy scipy rich pycocotools torchmetrics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps
```

**Sur GPU NVIDIA (CUDA 12.1) :**
```bash
pip install opencv-python numpy scipy rich pycocotools torchmetrics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e . --no-deps
```

---

## Étape 1 — Préparer le dataset

### Format attendu (COCO — compatible Roboflow)

```
dataset/
  train/
    photo1.jpg
    photo2.jpg
    _annotations.coco.json    ← généré par Roboflow
  valid/
    photo3.jpg
    _annotations.coco.json
  test/                       ← optionnel
    photo4.jpg
    _annotations.coco.json
```

### Exporter depuis Roboflow

1. Ouvrir le projet dans Roboflow
2. Cliquer sur **Generate** → **Export**
3. Choisir le format **COCO** (pas YOLOv5 ni autres)
4. **Important** : dans les options de génération, choisir **"Split BEFORE augmentation"**
   - Sinon des photos augmentées du même mur peuvent se retrouver dans train ET valid → le modèle "triche"

### Ajouter des images de murs sans fissures (recommandé)

Ces images apprennent au modèle à ne pas inventer de fissures là où il n'y en a pas.

```bash
python utilitaires/ajouter_images_saines.py \
  --images-saines dossier_murs_sains/ \
  --dataset       dataset/
```

---

## Étape 2 — Entraîner Mask R-CNN

### Commande de base

```bash
python entrainer.py \
  --donnees  dataset/ \
  --epoques  60 \
  --lot      2 \
  --sorties  sorties
```

### Tous les paramètres disponibles

```bash
python entrainer.py \
  --donnees            dataset/          # Dossier contenant train/ valid/ test/
  --epoques            60                # Nombre maximum d'époques
  --lot                2                 # Images par lot (2 sur CPU, 4-8 sur GPU)
  --lr                 0.0001            # Taux d'apprentissage initial
  --patience           15                # Époques sans amélioration avant arrêt
  --taille-image       640               # Résolution (ne pas changer)
  --seuil-score        0.05              # Seuil pour l'évaluation mAP (ne pas changer)
  --architecture       maskrcnn_resnet50_fpn_v2
  --dispositif         auto              # auto = GPU si disponible, sinon CPU
  --graine             42                # Graine aléatoire (reproductibilité)
  --sorties            sorties           # Dossier de sortie des modèles
  --decroissance-poids 0.0005            # Régularisation L2
```

### Ce que tu vas voir pendant l'entraînement

```
══════════════════════════════════════════════════════════
  DÉBUT DE L'ENTRAÎNEMENT
  Dispositif : cpu
  Époques : 60
══════════════════════════════════════════════════════════

═══ PHASE 1 : entraînement des têtes ═══
  Époque   1/60 | Lot   10/45 | Perte = 2.3451
  Époque   1/60 | Lot   20/45 | Perte = 1.8923
  ...
────────────────────────────────────────────────────────
  Époque   1/60 | Durée : 47.3s
  Perte train   : 1.9234
  mAP@0.5 valid : 0.1823 ↑  (meilleur : 0.1823 @ ép.1)
  Patience      : 0/15
────────────────────────────────────────────────────────
  ✓ Meilleur modèle sauvegardé (mAP@0.5 = 0.1823)
```

**3 phases se succèdent automatiquement :**
- **Phase 1** (époques 1-5) : seules les têtes de détection s'entraînent
- **Phase 2** (époques 5-15) : les couches supérieures du backbone se débloquent
- **Phase 3** (époque 15+) : tout le réseau s'entraîne

### Reprendre un entraînement interrompu

```bash
python entrainer.py \
  --donnees   dataset/ \
  --checkpoint sorties/modeles/dernier_modele.pth \
  --epoques   60
```

### Fichiers générés

```
sorties/
  modeles/
    meilleur_modele.pth    ← à utiliser pour l'inférence
    dernier_modele.pth     ← dernière époque (reprendre si interrompu)
```

---

## Étape 3 — Entraîner YOLO-seg (YOLO11 ou YOLO26)

YOLO est plus rapide à entraîner et à l'inférence. Le même script `entrainer_yolo.py` gère YOLO11 et YOLO26 — il suffit de changer `--modele`.

### Choix du modèle

| Modèle | Paramètres | Quand l'utiliser |
|---|---|---|
| `yolo11s-seg.pt` | ~9M | Dataset < 1000 images, GPU < 6 Go |
| `yolo11m-seg.pt` | ~20M | Dataset ≥ 1000 images, GPU ≥ 6 Go (**défaut**) |
| `yolo26s-seg.pt` | ~10M | Génération 2026, rapide |
| `yolo26m-seg.pt` | ~21M | Génération 2026, précis |

### YOLO11 Medium (défaut recommandé)

```bash
python entrainer_yolo.py \
  --donnees      dataset/ \
  --modele       yolo11m-seg.pt \
  --taille-image 640 \
  --epoques      100 \
  --lot          8 \
  --patience     50 \
  --mask-ratio   2 \
  --dispositif   auto \
  --nom          yolo11m_fissures \
  --sorties      sorties_yolo
```

### YOLO11 Small (CPU ou petit GPU)

```bash
python entrainer_yolo.py \
  --donnees      dataset/ \
  --modele       yolo11s-seg.pt \
  --taille-image 640 \
  --epoques      100 \
  --lot          16 \
  --patience     50 \
  --mask-ratio   2 \
  --dispositif   auto \
  --nom          yolo11s_fissures \
  --sorties      sorties_yolo
```

### YOLO26 Medium

```bash
python entrainer_yolo.py \
  --donnees      dataset/ \
  --modele       yolo26m-seg.pt \
  --taille-image 640 \
  --epoques      100 \
  --lot          8 \
  --patience     50 \
  --mask-ratio   2 \
  --dispositif   auto \
  --nom          yolo26m_fissures \
  --sorties      sorties_yolo
```

### YOLO26 Small

```bash
python entrainer_yolo.py \
  --donnees      dataset/ \
  --modele       yolo26s-seg.pt \
  --taille-image 640 \
  --epoques      100 \
  --lot          16 \
  --patience     50 \
  --mask-ratio   2 \
  --dispositif   auto \
  --nom          yolo26s_fissures \
  --sorties      sorties_yolo
```

Le modèle de base sera téléchargé automatiquement au premier lancement si non présent localement.

### Reprendre un entraînement YOLO interrompu

```bash
python entrainer_yolo.py \
  --resume sorties_yolo/entrainements/<nom>/weights/last.pt
```

---

## Étape 4 — Analyser des images

### Avec Mask R-CNN

```bash
python analyser.py \
  --modele   sorties/modeles/meilleur_modele.pth \
  --backend  maskrcnn \
  --images   dossier_photos/ \
  --seuil    0.40
```

### Avec YOLO11 Medium

```bash
python analyser.py \
  --modele   sorties_yolo/entrainements/yolo11m_fissures/weights/best.pt \
  --backend  yolo \
  --images   dossier_photos/ \
  --seuil    0.25
```

### Avec YOLO11 Small

```bash
python analyser.py \
  --modele   sorties_yolo/entrainements/yolo11s_fissures/weights/best.pt \
  --backend  yolo \
  --images   dossier_photos/ \
  --seuil    0.25
```

### Avec YOLO26 Medium

```bash
python analyser.py \
  --modele   sorties_yolo/entrainements/yolo26m_fissures/weights/best.pt \
  --backend  yolo \
  --images   dossier_photos/ \
  --seuil    0.25
```

### Avec YOLO26 Small

```bash
python analyser.py \
  --modele   sorties_yolo/entrainements/yolo26s_fissures/weights/best.pt \
  --backend  yolo \
  --images   dossier_photos/ \
  --seuil    0.25
```

### Paramètres d'analyse

| Paramètre | Description | Valeur recommandée |
|---|---|---|
| `--seuil` | Confiance minimale pour afficher une fissure | 0.30 à 0.50 |
| `--images` | Dossier ou fichier image | chemin du dossier |
| `--backend` | Modèle à utiliser | `maskrcnn` ou `yolo` |

---

## Comprendre les métriques

| Métrique | Ce qu'elle mesure | Cible |
|---|---|---|
| **mAP@0.5** | Chaque fissure détectée individuellement (≥50% chevauchement) | > 0.55 |
| **Précision** | Parmi les fissures détectées, combien sont vraies | > 0.80 |
| **Rappel** | Parmi les vraies fissures, combien sont trouvées | > 0.85 |
| **F1** | Équilibre précision/rappel | > 0.82 |

**Priorité sécurité** : le rappel est plus important que la précision. Manquer une fissure est plus dangereux qu'une fausse alarme.

---

## Conseils pratiques

### Si le mAP reste bas après 30 époques

1. Vérifier la qualité des annotations Roboflow (polygones précis ?)
2. Augmenter le dataset (minimum 200-500 images annotées)
3. Vérifier l'absence de data leakage (split avant augmentation)

### Si la perte explose (NaN, Inf)

```bash
# Réduire le taux d'apprentissage
python entrainer.py --donnees dataset/ --lr 0.00005 --lot 2
```

### Si le GPU manque de mémoire

```bash
# Réduire la taille du lot
python entrainer.py --donnees dataset/ --lot 1
```

### Temps d'entraînement estimés (CPU)

| Taille dataset | Époques | Temps estimé |
|---|---|---|
| 200 images | 60 | 8-12h |
| 500 images | 60 | 20-30h |
| 1000 images | 60 | 40-60h |

Avec un GPU NVIDIA, diviser ces temps par 8-15.
