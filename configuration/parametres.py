"""
Configuration centrale — Détection de fissures structurelles.

JUSTIFICATION DES PARAMÈTRES CHOISIS
═══════════════════════════════════════════════════════════════════════

MASK R-CNN (maskrcnn_resnet50_fpn_v2)
──────────────────────────────────────

  taille_image = 384×384
    Images Roboflow exportées en 384×384. Changer cette valeur
    sans redimensionner le dataset dégrade les performances.

  taille_lot = 2  (CPU) / 4-8  (GPU 8-16 Go)
    Mask R-CNN stocke des tenseurs de masques [N, H, W] par image.
    À 384×384, lot=2 sur CPU consomme ~3-4 Go de RAM.
    Sur GPU 8 Go : lot=4. Sur GPU 16 Go : lot=8.

  nombre_epoques = 60
    La stratégie 3 phases nécessite au moins :
      - Phase 1 (backbone gelé)   : 5 époques  → têtes stables
      - Phase 2 (dégelage partiel) : 10 époques → features fissures
      - Phase 3 (dégelage complet) : 45 époques → fine-tuning
    60 époques = durée minimale raisonnable avec early stopping.

  taux_apprentissage = 1e-4
    Taux standard pour fine-tuning d'un modèle COCO préentraîné.
    Trop élevé (>1e-3) : destruction des features pré-apprises.
    Trop bas (<1e-5) : convergence trop lente pour petit dataset.
    Le backbone est entraîné à lr/10 = 1e-5 (moins agressif).

  patience_arret_precoce = 15
    15 époques sans amélioration mAP@0.5 avant arrêt.
    Valeur élevée car les transitions de phases peuvent provoquer
    une baisse temporaire du mAP (normal, pas un vrai plateau).

  epoque_degelage_backbone = 5
    Les têtes convergent rapidement (5 époques suffisent).
    Dégeler trop tôt → gradient explosif sur les features ImageNet.

  epoque_degelage_complet = 15
    layer3/layer4 dégelés de l'époque 5 à 15, puis backbone entier.

  seuil_score_detection = 0.05
    CRITIQUE pour l'évaluation mAP. La métrique COCO construit une
    courbe précision-rappel en variant le seuil de 0 à 1.
    Un seuil élevé (0.5) tronque la courbe → mAP sous-estimé.
    0.05 est la valeur standard COCO (toutes les détections passent).
    Pour l'INFÉRENCE (analyser.py), utiliser 0.30-0.50.

  seuil_iou_nms = 0.3
    NMS supprime les boîtes qui se chevauchent à > 30% IoU.
    Valeur basse nécessaire : deux fissures distinctes peuvent être
    très proches. 0.5 (standard) fusionnerait des fissures séparées.

  detections_max_par_image = 50
    Une image de 384×384 peut contenir 5-20 fissures visibles.
    50 est largement suffisant et évite les faux positifs multiples.

  decroissance_poids = 5e-4
    Régularisation L2 (AdamW). Critique pour les petits datasets
    (< 1000 images) pour éviter l'overfitting.
    Valeur empiriquement validée sur datasets de segmentation COCO.

YOLO11-SEG
───────────────────────────────────────────────────────────────────────
  Voir entrainer_yolo.py pour la justification des paramètres YOLO.
  Les paramètres critiques pour les fissures :
    mask_ratio = 1    → masques pleine résolution (fissures de 1-2px)
    degrees    = 45   → fissures à tous les angles possibles
    copy_paste = 0.4  → augmentation rare objet (fissures < 5% pixels)
    patience   = 50   → les fissures nécessitent plus de convergence
"""

from dataclasses import dataclass, field
from pathlib import Path


SPLITS_DATASET: tuple[str, str, str] = ("train", "valid", "test")
SPLIT_ENTRAINEMENT = "train"
SPLIT_VALIDATION   = "valid"
SPLIT_TEST         = "test"
NOM_FICHIER_ANNOTATIONS_COCO = "_annotations.coco.json"

NOM_MEILLEUR_MODELE        = "meilleur_modele.pth"
NOM_DERNIER_MODELE         = "dernier_modele.pth"
NOM_HISTORIQUE_ENTRAINEMENT = "historique_entrainement.json"

ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2 = "maskrcnn_resnet50_fpn_v2"
ARCHITECTURES_MODELES_SEGMENTATION: tuple[str, ...] = (
    ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2,
)

# YOLO11-seg : yolo11s recommandé pour fissures (nano trop simple pour objet fin)
MODELE_YOLOV11_SEG_DEFAUT = "yolo11s-seg.pt"
MODELES_YOLOV11_SEG_OFFICIELS: tuple[str, ...] = (
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
)


@dataclass
class CheminsProjets:
    """Centralise tous les chemins de fichiers et dossiers du projet."""

    racine: Path = Path(".")

    donnees_racine:           Path = Path("dataset")
    chemin_entrainement:      Path = Path("dataset") / SPLIT_ENTRAINEMENT
    chemin_validation:        Path = Path("dataset") / SPLIT_VALIDATION
    chemin_test:              Path = Path("dataset") / SPLIT_TEST

    annotations_entrainement: Path = (
        Path("dataset") / SPLIT_ENTRAINEMENT / NOM_FICHIER_ANNOTATIONS_COCO
    )
    annotations_validation:   Path = (
        Path("dataset") / SPLIT_VALIDATION / NOM_FICHIER_ANNOTATIONS_COCO
    )
    annotations_test:         Path = Path("dataset") / SPLIT_TEST / NOM_FICHIER_ANNOTATIONS_COCO

    sorties_racine:   Path = Path("sorties")
    dossier_modeles:  Path = Path("sorties/modeles")
    dossier_journaux: Path = Path("sorties/journaux")

    meilleur_modele: Path = Path("sorties/modeles") / NOM_MEILLEUR_MODELE
    dernier_modele:  Path = Path("sorties/modeles") / NOM_DERNIER_MODELE

    def definir_racine_donnees(self, racine: str | Path) -> None:
        """Met à jour tous les chemins dataset COCO depuis une racine unique."""
        self.donnees_racine           = Path(racine)
        self.chemin_entrainement      = self.donnees_racine / SPLIT_ENTRAINEMENT
        self.chemin_validation        = self.donnees_racine / SPLIT_VALIDATION
        self.chemin_test              = self.donnees_racine / SPLIT_TEST
        self.annotations_entrainement = self.chemin_entrainement / NOM_FICHIER_ANNOTATIONS_COCO
        self.annotations_validation   = self.chemin_validation   / NOM_FICHIER_ANNOTATIONS_COCO
        self.annotations_test         = self.chemin_test         / NOM_FICHIER_ANNOTATIONS_COCO

    def definir_racine_sorties(self, racine: str | Path) -> None:
        """Met à jour tous les chemins de sortie depuis une racine unique."""
        self.sorties_racine  = Path(racine)
        self.dossier_modeles = self.sorties_racine / "modeles"
        self.dossier_journaux = self.sorties_racine / "journaux"
        self.meilleur_modele = self.dossier_modeles / NOM_MEILLEUR_MODELE
        self.dernier_modele  = self.dossier_modeles / NOM_DERNIER_MODELE

    def creer_dossiers(self) -> None:
        """Crée tous les dossiers de sortie s'ils n'existent pas."""
        for dossier in [self.sorties_racine, self.dossier_modeles, self.dossier_journaux]:
            Path(dossier).mkdir(parents=True, exist_ok=True)


@dataclass
class ParametresModele:
    """
    Hyperparamètres de l'architecture Mask R-CNN pour la détection de fissures.

    Ces valeurs sont calibrées pour :
      - Images 384×384 (export Roboflow standard)
      - Fissures fines (< 5% des pixels)
      - Transfer learning depuis COCO (91 classes → 2 classes)
    """

    architecture: str = ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2

    # 2 classes : 0 = fond (background), 1 = fissure
    nombre_classes: int = 2

    # Résolution identique à l'export Roboflow
    taille_image_min: int = 384
    taille_image_max: int = 384

    # 0.05 = standard COCO pour l'évaluation mAP (pas pour l'inférence)
    # Pour l'inférence (analyser.py) : utiliser 0.30 à 0.50
    seuil_score_detection: float = 0.05

    # Bas (0.3) car deux fissures distinctes peuvent être très proches
    seuil_iou_nms: float = 0.3

    # Masque : seuil de binarisation (0.5 = standard)
    seuil_masque: float = 0.5

    # 50 détections max par image (plus que suffisant pour fissures)
    detections_max_par_image: int = 50


@dataclass
class ParametresEntrainement:
    """
    Hyperparamètres de la boucle d'entraînement pour la détection de fissures.

    Stratégie 3 phases (voir masque_rcnn.py pour le détail) :
      Phase 1 : époques 1-5   → backbone gelé
      Phase 2 : époques 5-15  → couches supérieures dégelées
      Phase 3 : époques 15-60 → fine-tuning complet
    """

    # Lot de 2 sur CPU (RAM ~3-4 Go) ; 4-8 sur GPU selon VRAM
    taille_lot: int = 2

    # 60 = durée minimale pour les 3 phases + early stopping
    nombre_epoques: int = 60

    # Standard pour fine-tuning COCO préentraîné
    taux_apprentissage: float = 1e-4

    # L2 essentiel pour dataset < 5000 images (anti-overfitting)
    decroissance_poids: float = 5e-4

    # SGD momentum (non utilisé avec AdamW, conservé pour référence)
    momentum: float = 0.9

    # 15 époques de marge : les transitions de phases créent des creux temporaires
    patience_arret_precoce: int = 15

    # Phases de dégelage progressif (voir masque_rcnn.py)
    epoque_degelage_backbone: int = 5
    epoque_degelage_complet:  int = 15

    # Gradient clipping : stabilise l'entraînement si gradient explosif
    valeur_clip_gradient: float = 1.0

    nombre_workers: int = 2
    epingler_memoire: bool = True
    graine_aleatoire: int  = 42

    # Précision mixte float16 : activée sur GPU (ignorée sur CPU)
    precision_mixte: bool = True

    frequence_sauvegarde: int = 5
    frequence_affichage:  int = 10


@dataclass
class ConfigurationGlobale:
    """
    Configuration racine du projet détection de fissures structurelles.

    Usage :
        from configuration.parametres import ConfigurationGlobale
        config = ConfigurationGlobale()
        config.chemins.creer_dossiers()
    """
    chemins:      CheminsProjets        = field(default_factory=CheminsProjets)
    modele:       ParametresModele      = field(default_factory=ParametresModele)
    entrainement: ParametresEntrainement = field(default_factory=ParametresEntrainement)

    dispositif:     str  = "auto"
    mode_verbose:   bool = True
    nom_experience: str  = "detection_fissures_v1"


configuration_defaut = ConfigurationGlobale()
