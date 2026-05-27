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

  ── 0. PRÉPARER LE DATASET ──────────────────────────────────────────

  Ajouter des images de murs sains (sans fissures) au dataset Roboflow :

    python utilitaires/ajouter_images_saines.py \\
      --images-saines murs_sains/ \\
      --dataset       dataset/

  ── 1. ENTRAÎNER MASK R-CNN ─────────────────────────────────────────

    python entrainer.py \\
      --donnees            dataset/ \\
      --epoques            60 \\
      --lot                2 \\
      --lr                 0.0001 \\
      --patience           15 \\
      --taille-image       384 \\
      --seuil-score        0.05 \\
      --architecture       maskrcnn_resnet50_fpn_v2 \\
      --dispositif         auto \\
      --graine             42 \\
      --sorties            sorties \\
      --decroissance-poids 0.0005

  ── 2. ENTRAÎNER YOLO11-SEG ─────────────────────────────────────────

    python entrainer_yolo.py \\
      --donnees          dataset/ \\
      --modele           yolo11s-seg.pt \\
      --taille-image     384 \\
      --epoques          150 \\
      --lot              8 \\
      --lr               0.001 \\
      --lrf              0.01 \\
      --weight-decay     0.0005 \\
      --patience         50 \\
      --warmup-epoques   5.0 \\
      --close-mosaic     30 \\
      --mask-ratio       1 \\
      --overlap-mask \\
      --copy-paste       0.4 \\
      --mosaic           1.0 \\
      --mixup            0.0 \\
      --degrees          45.0 \\
      --translate        0.1 \\
      --scale            0.5 \\
      --fliplr           0.5 \\
      --flipud           0.0 \\
      --hsv-h            0.02 \\
      --hsv-s            0.5 \\
      --hsv-v            0.4 \\
      --cos-lr \\
      --freeze           0 \\
      --amp \\
      --workers          2 \\
      --save-period      10 \\
      --dispositif       auto \\
      --nom              yolo11_seg_fissures \\
      --sorties          sorties_yolo

  ── 3. ANALYSER LES IMAGES ──────────────────────────────────────────

  Avec Mask R-CNN :
    python analyser.py \\
      --modele   sorties/modeles/meilleur_modele.pth \\
      --backend  maskrcnn \\
      --images   dataset/test/ \\
      --seuil    0.40

  Avec YOLO11-seg :
    python analyser.py \\
      --modele   sorties_yolo/entrainements/yolo11_seg_fissures/weights/best.pt \\
      --backend  yolo \\
      --images   dataset/test/ \\
      --seuil    0.25

  ── 4. REPRENDRE UN ENTRAÎNEMENT INTERROMPU ─────────────────────────

  Mask R-CNN :
    python entrainer.py [mêmes paramètres] \\
      --resume sorties/modeles/dernier_modele.pth

  YOLO :
    python entrainer_yolo.py \\
      --resume sorties_yolo/entrainements/yolo11_seg_fissures/weights/last.pt

════════════════════════════════════════════════════════════════════════
""")


def main():
    print("=" * 68)
    print("  Détection de Fissures Structurelles")
    print("  Segmentation d'instance — Mask R-CNN & YOLO11-seg")
    print("=" * 68)

    dispositif = detecter_dispositif()
    afficher_info_dispositif(dispositif)
    afficher_commandes()

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
