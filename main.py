"""
Point d'entrée du projet Détection de Fissures Structurelles.

Affiche les informations sur l'environnement et les commandes d'utilisation.
"""

import time

from detection_fissures.configuration.parametres import ConfigurationGlobale
from detection_fissures.utilitaires.dispositif import detecter_dispositif, afficher_info_dispositif


def afficher_commandes() -> None:
    """Affiche toutes les commandes du projet avec leurs paramètres complets."""

    print("""
════════════════════════════════════════════════════════════════════════
  COMMANDES DU PROJET
════════════════════════════════════════════════════════════════════════

  DATASET : Format YOLOv11 natif Roboflow — 640×640px — 3794 images — 1 classe (crack)
  Structure attendue :
    dataset/
    ├── data.yaml
    ├── images/  train/  valid/  test/
    └── labels/  train/  valid/  test/

  ── 1. ENTRAÎNER YOLO11-SEG (recommandé) ─────────────────────────────

  ┌─ YOLO11 Medium (précis — GPU ≥ 6 Go, défaut pour 3794 images) ─────┐

    python entrainer_yolo.py \\
      --yaml             dataset/data.yaml \\
      --modele           yolo11m-seg.pt \\
      --taille-image     640 \\
      --epoques          150 \\
      --lot              8 \\
      --lr               0.01 \\
      --lrf              0.01 \\
      --weight-decay     0.0005 \\
      --patience         50 \\
      --warmup-epochs    5.0 \\
      --mask-ratio       1 \\
      --dispositif       auto \\
      --nom              yolo11m_fissures \\
      --sorties          sorties_yolo

  └────────────────────────────────────────────────────────────────────┘

  ┌─ YOLO11 Small (rapide — GPU < 6 Go ou test rapide) ────────────────┐

    python entrainer_yolo.py \\
      --yaml             dataset/data.yaml \\
      --modele           yolo11s-seg.pt \\
      --taille-image     640 \\
      --epoques          150 \\
      --lot              16 \\
      --lr               0.01 \\
      --lrf              0.01 \\
      --patience         50 \\
      --warmup-epochs    5.0 \\
      --mask-ratio       1 \\
      --dispositif       auto \\
      --nom              yolo11s_fissures \\
      --sorties          sorties_yolo

  └────────────────────────────────────────────────────────────────────┘

  ── 2. ENTRAÎNER YOLO26-SEG ─────────────────────────────────────────

  ┌─ YOLO26 Medium (génération 2026 — GPU ≥ 6 Go) ─────────────────────┐

    python entrainer_yolo.py \\
      --yaml             dataset/data.yaml \\
      --modele           yolo26m-seg.pt \\
      --taille-image     640 \\
      --epoques          150 \\
      --lot              8 \\
      --lr               0.01 \\
      --lrf              0.01 \\
      --patience         50 \\
      --warmup-epochs    5.0 \\
      --mask-ratio       1 \\
      --dispositif       auto \\
      --nom              yolo26m_fissures \\
      --sorties          sorties_yolo

  └────────────────────────────────────────────────────────────────────┘

  Reprendre un entraînement YOLO interrompu :
    python entrainer_yolo.py \\
      --resume sorties_yolo/entrainements/<nom>/weights/last.pt

  ── 3. ENTRAÎNER MASK R-CNN ─────────────────────────────────────────

  Note : Mask R-CNN attend un dataset au format COCO (_annotations.coco.json).
  Pour un dataset YOLO natif, utiliser YOLO-seg (section 1-2 ci-dessus).

  CPU (lot=2, ~12-48h selon dataset) :
    python entrainer.py \\
      --donnees            dataset/ \\
      --epoques            100 \\
      --lot                2 \\
      --lr                 0.0001 \\
      --patience           20 \\
      --taille-image       640 \\
      --seuil-score        0.05 \\
      --architecture       maskrcnn_resnet50_fpn_v2 \\
      --dispositif         auto \\
      --graine             42 \\
      --sorties            sorties \\
      --decroissance-poids 0.0005

  GPU (lot=4-8, ~2-6h) :
    python entrainer.py \\
      --donnees            dataset/ \\
      --epoques            100 \\
      --lot                4 \\
      --lr                 0.0001 \\
      --patience           20 \\
      --dispositif         auto \\
      --sorties            sorties

  Reprendre un entraînement interrompu :
    python entrainer.py [mêmes paramètres] \\
      --resume sorties/modeles/dernier_modele.pth

  ── 4. ANALYSER LES IMAGES ──────────────────────────────────────────

  Avec YOLO11 Medium (après entraînement) :
    python analyser.py \\
      --modele   sorties_yolo/entrainements/yolo11m_fissures/weights/best.pt \\
      --backend  yolo \\
      --images   dataset/images/test/ \\
      --seuil    0.25

  Avec YOLO11 Small :
    python analyser.py \\
      --modele   sorties_yolo/entrainements/yolo11s_fissures/weights/best.pt \\
      --backend  yolo \\
      --images   dataset/images/test/ \\
      --seuil    0.25

  Avec Mask R-CNN :
    python analyser.py \\
      --modele   sorties/modeles/meilleur_modele.pth \\
      --backend  maskrcnn \\
      --images   dataset/images/test/ \\
      --seuil    0.40

════════════════════════════════════════════════════════════════════════
  GUIDE DE CHOIX DU MODÈLE
════════════════════════════════════════════════════════════════════════

  Dataset ≥ 1000 images  → YOLO11m  ou  YOLO26m  (Medium, défaut)
  GPU < 6 Go / CPU       → YOLO11s  ou  YOLO26s  (Small, lot=16)
  Précision max          → Mask R-CNN (nécessite format COCO)
  YOLO11 vs YOLO26       → YOLO26 si ultralytics ≥ 8.x supporte yolo26
  Dataset natif Roboflow → utiliser --yaml dataset/data.yaml (pas de conversion)

════════════════════════════════════════════════════════════════════════
""")


def main():
    print("=" * 68)
    print("  Détection de Fissures Structurelles")
    print("  Segmentation d'instance — Mask R-CNN · YOLO11-seg · YOLO26-seg")
    print("=" * 68)

    dispositif = detecter_dispositif()
    afficher_info_dispositif(dispositif)
    afficher_commandes()

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
