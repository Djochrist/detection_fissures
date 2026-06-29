# Guide d'entraînement — Détection de Fissures Structurelles

Toutes les commandes, prêtes à copier-coller de A à Z.

---

## 0. Dataset — structure attendue

```
dataset/
├── data.yaml
├── images/
│   ├── train/    (≈3160 images)
│   ├── valid/    (≈476 images)
│   └── test/     (≈158 images)
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

Adapter `path:` dans `data.yaml` à votre environnement :

```yaml
path: /chemin/absolu/vers/dataset   # ← modifier ici
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['crack']
```

---

## 1. YOLO11 Medium — GPU ≥ 6 Go (recommandé)

```bash
python entrainer_yolo.py \
  --yaml          dataset/data.yaml \
  --murs-sains    murs_sains \
  --modele        yolo11m-seg.pt \
  --taille-image  1024 \
  --epoques       300 \
  --lot           4 \
  --lr            0.01 \
  --lrf           0.01 \
  --weight-decay  0.0001 \
  --patience      100 \
  --warmup-epochs 5.0 \
  --mask-ratio    1 \
  --dispositif    auto \
  --nom           yolo11m_fissures \
  --sorties       sorties_yolo
```

> `--murs-sains` (optionnel) : dossier d'images **sans fissure** placé à côté de
> `dataset/`. Le code les copie dans le dataset avec un label **vide** (exemples
> négatifs), reste à **1 classe** (`crack`), et c'est **idempotent**. Les
> sous-dossiers `train/valid/test` sont respectés ; sinon répartition automatique.
> Retire ce flag si tu n'as pas de murs sains.

> `--augmentation` (défaut `moderee`) pilote l'augmentation **dynamique** à
> l'entraînement : `moderee` = mosaic partiel + flip + légères variations
> couleur/échelle (prudent pour fissures fines), `forte` = plus agressif
> (rotation, shear, mixup, copy_paste, erasing), `desactivee` = aucune. C'est
> souvent le levier le plus efficace pour faire monter le mAP.

Meilleur modèle → `sorties_yolo/entrainements/yolo11m_fissures/weights/best.pt`

---

## 2. YOLO11 Small — GPU < 6 Go ou test rapide

```bash
python entrainer_yolo.py \
  --yaml          dataset/data.yaml \
  --modele        yolo11s-seg.pt \
  --taille-image  640 \
  --epoques       150 \
  --lot           16 \
  --lr            0.01 \
  --lrf           0.01 \
  --weight-decay  0.0001 \
  --patience      50 \
  --warmup-epochs 5.0 \
  --mask-ratio    1 \
  --dispositif    auto \
  --nom           yolo11s_fissures \
  --sorties       sorties_yolo
```

Meilleur modèle → `sorties_yolo/entrainements/yolo11s_fissures/weights/best.pt`

---

## 3. YOLO26 Medium — GPU ≥ 6 Go (génération 2026)

```bash
python entrainer_yolo.py \
  --yaml          dataset/data.yaml \
  --modele        yolo26m-seg.pt \
  --taille-image  640 \
  --epoques       150 \
  --lot           8 \
  --lr            0.01 \
  --lrf           0.01 \
  --weight-decay  0.0001 \
  --patience      50 \
  --warmup-epochs 5.0 \
  --mask-ratio    1 \
  --dispositif    auto \
  --nom           yolo26m_fissures \
  --sorties       sorties_yolo
```

Meilleur modèle → `sorties_yolo/entrainements/yolo26m_fissures/weights/best.pt`

---

## 4. YOLO26 Small — GPU < 6 Go (génération 2026)

```bash
python entrainer_yolo.py \
  --yaml          dataset/data.yaml \
  --modele        yolo26s-seg.pt \
  --taille-image  640 \
  --epoques       150 \
  --lot           16 \
  --lr            0.01 \
  --lrf           0.01 \
  --weight-decay  0.0001 \
  --patience      50 \
  --warmup-epochs 5.0 \
  --mask-ratio    1 \
  --dispositif    auto \
  --nom           yolo26s_fissures \
  --sorties       sorties_yolo
```

Meilleur modèle → `sorties_yolo/entrainements/yolo26s_fissures/weights/best.pt`

---

## 5. Reprendre un entraînement interrompu

```bash
# YOLO11m
python entrainer_yolo.py \
  --resume sorties_yolo/entrainements/yolo11m_fissures/weights/last.pt

# YOLO11s
python entrainer_yolo.py \
  --resume sorties_yolo/entrainements/yolo11s_fissures/weights/last.pt

# YOLO26m
python entrainer_yolo.py \
  --resume sorties_yolo/entrainements/yolo26m_fissures/weights/last.pt

# YOLO26s
python entrainer_yolo.py \
  --resume sorties_yolo/entrainements/yolo26s_fissures/weights/last.pt
```

---

## 6. Mask R-CNN — précision maximale

> ⚠️ Mask R-CNN nécessite le format **COCO** (`_annotations.coco.json`), pas le format YOLOv11.
> Si vous avez uniquement le dataset Roboflow natif, utiliser YOLO-seg (sections 1-4).

### CPU (lent, ~12-48h)

```bash
python entrainer.py \
  --donnees            dataset_coco/ \
  --epoques            100 \
  --lot                2 \
  --lr                 0.0001 \
  --patience           20 \
  --taille-image       640 \
  --seuil-score        0.05 \
  --architecture       maskrcnn_resnet50_fpn_v2 \
  --dispositif         auto \
  --graine             42 \
  --sorties            sorties_maskrcnn \
  --decroissance-poids 0.0005
```

### GPU (recommandé, ~2-6h)

```bash
python entrainer.py \
  --donnees            dataset_coco/ \
  --epoques            100 \
  --lot                4 \
  --lr                 0.0001 \
  --patience           20 \
  --taille-image       640 \
  --seuil-score        0.05 \
  --architecture       maskrcnn_resnet50_fpn_v2 \
  --dispositif         auto \
  --graine             42 \
  --sorties            sorties_maskrcnn \
  --decroissance-poids 0.0005
```

### Reprendre Mask R-CNN

```bash
python entrainer.py \
  --donnees  dataset_coco/ \
  --sorties  sorties_maskrcnn \
  --resume   sorties_maskrcnn/modeles/dernier_modele.pth
```

Meilleur modèle → `sorties_maskrcnn/modeles/meilleur_modele.pth`

---

## 7. Analyser des images après entraînement

```bash
# Avec YOLO11m
python analyser.py \
  --modele   sorties_yolo/entrainements/yolo11m_fissures/weights/best.pt \
  --backend  yolo \
  --images   dataset/images/test/ \
  --seuil    0.25

# Avec YOLO11s
python analyser.py \
  --modele   sorties_yolo/entrainements/yolo11s_fissures/weights/best.pt \
  --backend  yolo \
  --images   dataset/images/test/ \
  --seuil    0.25

# Avec YOLO26m
python analyser.py \
  --modele   sorties_yolo/entrainements/yolo26m_fissures/weights/best.pt \
  --backend  yolo \
  --images   dataset/images/test/ \
  --seuil    0.25

# Avec YOLO26s
python analyser.py \
  --modele   sorties_yolo/entrainements/yolo26s_fissures/weights/best.pt \
  --backend  yolo \
  --images   dataset/images/test/ \
  --seuil    0.25

# Avec Mask R-CNN
python analyser.py \
  --modele   sorties_maskrcnn/modeles/meilleur_modele.pth \
  --backend  maskrcnn \
  --images   dataset/images/test/ \
  --seuil    0.40
```

---

## Résumé — quel modèle choisir ?

| Situation | Commande à utiliser |
|-----------|---------------------|
| GPU ≥ 6 Go, précision max | **Section 1 — YOLO11m** |
| GPU < 6 Go ou test rapide | **Section 2 — YOLO11s** |
| GPU ≥ 6 Go, génération 2026 | **Section 3 — YOLO26m** |
| GPU < 6 Go, génération 2026 | **Section 4 — YOLO26s** |
| Précision absolue (format COCO requis) | **Section 6 — Mask R-CNN** |

---

## Guides environnement

- **Google Colab** → `documentation/guide_colab.md`
- **Kaggle** → `documentation/guide_kaggle.md`
- **Local** → `documentation/guide_local_uv.md`
