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

  GPU (lot=4-8, ~2-4h) — même commande + ajuster --lot :
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

  ── 2. ENTRAÎNER YOLO11-SEG ─────────────────────────────────────────

  ┌─ YOLO11 Medium (défaut — précis, dataset ≥ 1000 images, GPU ≥ 6 Go) ─┐

    python entrainer_yolo.py \\
      --donnees          dataset/ \\
      --modele           yolo11m-seg.pt \\
      --taille-image     640 \\
      --epoques          100 \\
      --lot              8 \\
      --lr               0.001 \\
      --lrf              0.01 \\
      --weight-decay     0.0005 \\
      --patience         50 \\
      --warmup-epochs    3.0 \\
      --close-mosaic     15 \\
      --mask-ratio       2 \\
      --mosaic           0.3 \\
      --dispositif       auto \\
      --nom              yolo11m_fissures \\
      --sorties          sorties_yolo

  └────────────────────────────────────────────────────────────────────┘

  ┌─ YOLO11 Small (rapide — dataset < 1000 images ou GPU < 6 Go) ──────┐

    python entrainer_yolo.py \\
      --donnees          dataset/ \\
      --modele           yolo11s-seg.pt \\
      --taille-image     640 \\
      --epoques          100 \\
      --lot              16 \\
      --lr               0.001 \\
      --lrf              0.01 \\
      --weight-decay     0.0005 \\
      --patience         50 \\
      --warmup-epochs    3.0 \\
      --close-mosaic     15 \\
      --mask-ratio       2 \\
      --mosaic           0.3 \\
      --dispositif       auto \\
      --nom              yolo11s_fissures \\
      --sorties          sorties_yolo

  └────────────────────────────────────────────────────────────────────┘

  ── 3. ENTRAÎNER YOLO26-SEG ─────────────────────────────────────────

  ┌─ YOLO26 Medium (précis — génération 2026, GPU ≥ 6 Go) ─────────────┐

    python entrainer_yolo.py \\
      --donnees          dataset/ \\
      --modele           yolo26m-seg.pt \\
      --taille-image     640 \\
      --epoques          100 \\
      --lot              8 \\
      --lr               0.001 \\
      --lrf              0.01 \\
      --weight-decay     0.0005 \\
      --patience         50 \\
      --warmup-epochs    3.0 \\
      --close-mosaic     15 \\
      --mask-ratio       2 \\
      --mosaic           0.3 \\
      --dispositif       auto \\
      --nom              yolo26m_fissures \\
      --sorties          sorties_yolo

  └────────────────────────────────────────────────────────────────────┘

  ┌─ YOLO26 Small (rapide — génération 2026, GPU < 6 Go) ──────────────┐

    python entrainer_yolo.py \\
      --donnees          dataset/ \\
      --modele           yolo26s-seg.pt \\
      --taille-image     640 \\
      --epoques          100 \\
      --lot              16 \\
      --lr               0.001 \\
      --lrf              0.01 \\
      --weight-decay     0.0005 \\
      --patience         50 \\
      --warmup-epochs    3.0 \\
      --close-mosaic     15 \\
      --mask-ratio       2 \\
      --mosaic           0.3 \\
      --dispositif       auto \\
      --nom              yolo26s_fissures \\
      --sorties          sorties_yolo

  └────────────────────────────────────────────────────────────────────┘

  Reprendre un entraînement YOLO interrompu :
    python entrainer_yolo.py \\
      --resume sorties_yolo/entrainements/<nom>/weights/last.pt

  ── 4. ANALYSER LES IMAGES ──────────────────────────────────────────

  Avec Mask R-CNN :
    python analyser.py \\
      --modele   sorties/modeles/meilleur_modele.pth \\
      --backend  maskrcnn \\
      --images   dataset/test/ \\
      --seuil    0.40

  Avec YOLO11 Medium :
    python analyser.py \\
      --modele   sorties_yolo/entrainements/yolo11m_fissures/weights/best.pt \\
      --backend  yolo \\
      --images   dataset/test/ \\
      --seuil    0.25

  Avec YOLO11 Small :
    python analyser.py \\
      --modele   sorties_yolo/entrainements/yolo11s_fissures/weights/best.pt \\
      --backend  yolo \\
      --images   dataset/test/ \\
      --seuil    0.25

  Avec YOLO26 Medium :
    python analyser.py \\
      --modele   sorties_yolo/entrainements/yolo26m_fissures/weights/best.pt \\
      --backend  yolo \\
      --images   dataset/test/ \\
      --seuil    0.25

  Avec YOLO26 Small :
    python analyser.py \\
      --modele   sorties_yolo/entrainements/yolo26s_fissures/weights/best.pt \\
      --backend  yolo \\
      --images   dataset/test/ \\
      --seuil    0.25

════════════════════════════════════════════════════════════════════════
  GUIDE DE CHOIX DU MODÈLE
════════════════════════════════════════════════════════════════════════

  Dataset < 1000 images  → YOLO11s  ou  YOLO26s  (Small)
  Dataset ≥ 1000 images  → YOLO11m  ou  YOLO26m  (Medium, défaut)
  Précision max          → Mask R-CNN (plus lent à entraîner)
  GPU < 6 Go / CPU       → Small obligatoire (lot réduit à 8 ou moins)
  YOLO11 vs YOLO26       → YOLO26 si ultralytics ≥ 8.x supporte yolo26

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
