"""
Module de configuration centrale du système de détection de fissures.

Regroupe tous les hyperparamètres, chemins et réglages du projet.
"""

from dataclasses import dataclass, field
from pathlib import Path


SPLITS_DATASET: tuple[str, str, str] = ("train", "valid", "test")
SPLIT_ENTRAINEMENT = "train"
SPLIT_VALIDATION = "valid"
SPLIT_TEST = "test"
NOM_FICHIER_ANNOTATIONS_COCO = "_annotations.coco.json"

NOM_MEILLEUR_MODELE = "meilleur_modele.pth"
NOM_DERNIER_MODELE = "dernier_modele.pth"
NOM_HISTORIQUE_ENTRAINEMENT = "historique_entrainement.json"

ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2 = "maskrcnn_resnet50_fpn_v2"
ARCHITECTURE_MASKRCNN_RESNET50_FPN = "maskrcnn_resnet50_fpn"
ARCHITECTURES_MODELES_SEGMENTATION: tuple[str, str] = (
    ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2,
    ARCHITECTURE_MASKRCNN_RESNET50_FPN,
)


@dataclass
class CheminsProjets:
    """
    Centralise tous les chemins de fichiers et dossiers du projet.

    Structure attendue du dataset :
        dataset/
            train/
                _annotations.coco.json
                image1.jpg
                ...
            valid/
                _annotations.coco.json
                ...
            test/
                _annotations.coco.json
                ...
    """
    racine: Path = Path(".")

    donnees_racine: Path = Path("dataset")
    chemin_entrainement: Path = Path("dataset") / SPLIT_ENTRAINEMENT
    chemin_validation: Path = Path("dataset") / SPLIT_VALIDATION
    chemin_test: Path = Path("dataset") / SPLIT_TEST

    annotations_entrainement: Path = (
        Path("dataset") / SPLIT_ENTRAINEMENT / NOM_FICHIER_ANNOTATIONS_COCO
    )
    annotations_validation: Path = (
        Path("dataset") / SPLIT_VALIDATION / NOM_FICHIER_ANNOTATIONS_COCO
    )
    annotations_test: Path = Path("dataset") / SPLIT_TEST / NOM_FICHIER_ANNOTATIONS_COCO

    sorties_racine: Path = Path("sorties")
    dossier_modeles: Path = Path("sorties/modeles")
    dossier_journaux: Path = Path("sorties/journaux")

    meilleur_modele: Path = Path("sorties/modeles") / NOM_MEILLEUR_MODELE
    dernier_modele: Path = Path("sorties/modeles") / NOM_DERNIER_MODELE

    def definir_racine_donnees(self, racine: str | Path) -> None:
        """Met à jour tous les chemins dataset COCO depuis une racine unique."""
        self.donnees_racine = Path(racine)
        self.chemin_entrainement = self.donnees_racine / SPLIT_ENTRAINEMENT
        self.chemin_validation = self.donnees_racine / SPLIT_VALIDATION
        self.chemin_test = self.donnees_racine / SPLIT_TEST
        self.annotations_entrainement = (
            self.chemin_entrainement / NOM_FICHIER_ANNOTATIONS_COCO
        )
        self.annotations_validation = (
            self.chemin_validation / NOM_FICHIER_ANNOTATIONS_COCO
        )
        self.annotations_test = self.chemin_test / NOM_FICHIER_ANNOTATIONS_COCO

    def definir_racine_sorties(self, racine: str | Path) -> None:
        """Met à jour tous les chemins de sortie depuis une racine unique."""
        self.sorties_racine = Path(racine)
        self.dossier_modeles = self.sorties_racine / "modeles"
        self.dossier_journaux = self.sorties_racine / "journaux"
        self.meilleur_modele = self.dossier_modeles / NOM_MEILLEUR_MODELE
        self.dernier_modele = self.dossier_modeles / NOM_DERNIER_MODELE

    def creer_dossiers(self) -> None:
        """Crée tous les dossiers de sortie s'ils n'existent pas."""
        for dossier in [self.sorties_racine, self.dossier_modeles, self.dossier_journaux]:
            Path(dossier).mkdir(parents=True, exist_ok=True)


@dataclass
class ParametresModele:
    """Hyperparamètres définissant l'architecture du modèle Mask R-CNN."""

    architecture: str = ARCHITECTURE_MASKRCNN_RESNET50_FPN_V2
    nombre_classes: int = 2

    taille_image_min: int = 384
    taille_image_max: int = 384

    seuil_score_detection: float = 0.5
    seuil_iou_nms: float = 0.3
    seuil_masque: float = 0.5
    detections_max_par_image: int = 100


@dataclass
class ParametresEntrainement:
    """Hyperparamètres contrôlant la boucle d'entraînement."""

    taille_lot: int = 4
    nombre_epoques: int = 50
    taux_apprentissage: float = 1e-4
    decroissance_poids: float = 5e-4
    momentum: float = 0.9

    patience_arret_precoce: int = 10

    epoque_degelage_backbone: int = 5
    epoque_degelage_complet: int = 15

    valeur_clip_gradient: float = 1.0
    nombre_workers: int = 2
    epingler_memoire: bool = True
    graine_aleatoire: int = 42
    precision_mixte: bool = True

    frequence_sauvegarde: int = 5
    frequence_affichage: int = 10


@dataclass
class ConfigurationGlobale:
    """
    Classe racine agrégeant toute la configuration du projet.

    Usage :
        from configuration.parametres import ConfigurationGlobale
        config = ConfigurationGlobale()
        config.chemins.creer_dossiers()
    """
    chemins: CheminsProjets = field(default_factory=CheminsProjets)
    modele: ParametresModele = field(default_factory=ParametresModele)
    entrainement: ParametresEntrainement = field(default_factory=ParametresEntrainement)

    dispositif: str = "auto"
    mode_verbose: bool = True
    nom_experience: str = "detection_fissures_v1"


configuration_defaut = ConfigurationGlobale()
