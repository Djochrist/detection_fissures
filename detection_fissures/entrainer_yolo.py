"""
Entraînement YOLO-seg pour la détection de fissures.

Supporte deux modes d'entrée :
  - Dataset YOLO natif (Roboflow YOLOv11) : utiliser --yaml data.yaml
  - Dataset COCO (conversion automatique)  : utiliser --donnees dataset/
    (détection automatique si data.yaml présent dans le dossier)

YOLO11-seg est recommandé. YOLO26-seg est aussi supporté.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

from detection_fissures.configuration.parametres import (
    MODELE_YOLO_SEG_DEFAUT,
    MODELES_YOLO_SEG_OFFICIELS,
    ConfigurationGlobale,
)


def analyser_arguments() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    configuration = ConfigurationGlobale()
    analyseur = argparse.ArgumentParser(
        description="Entraînement YOLO-seg — Détection de fissures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyseur.add_argument(
        "--yaml",
        type=str,
        default=None,
        help=(
            "Chemin direct vers le fichier data.yaml du dataset YOLO natif. "
            "Quand fourni, aucune conversion COCO n'est effectuée. "
            "Exemple : --yaml dataset/data.yaml"
        ),
    )
    analyseur.add_argument(
        "--donnees",
        type=str,
        default=str(configuration.chemins.donnees_racine),
        help=(
            "Répertoire racine du dataset. Si data.yaml y est présent, le format "
            "YOLO natif est détecté automatiquement. Sinon, conversion COCO."
        ),
    )
    analyseur.add_argument(
        "--murs-sains",
        type=str,
        default=None,
        help=(
            "Dossier d'images de murs sains (sans fissure) à intégrer comme "
            "exemples négatifs. Le code les copie dans le dataset avec un label "
            "VIDE (reste 1 classe 'crack'). Les sous-dossiers train/valid/test "
            "sont respectés ; sinon répartition proportionnelle aux splits "
            "existants. Idempotent. Exemple : --murs-sains murs_sains"
        ),
    )
    analyseur.add_argument(
        "--sorties",
        type=str,
        default="sorties_yolo",
        help="Répertoire de sortie YOLO",
    )
    analyseur.add_argument(
        "--modele",
        type=str,
        default=MODELE_YOLO_SEG_DEFAUT,
        help=(
            "Poids YOLO segmentation. "
            "YOLO11 : yolo11s-seg.pt (rapide) | yolo11m-seg.pt (précis, défaut). "
            "YOLO26 : yolo26s-seg.pt (rapide) | yolo26m-seg.pt (précis). "
            "Un checkpoint .pt local est aussi accepté."
        ),
    )
    analyseur.add_argument("--epoques", type=int, default=150)
    analyseur.add_argument("--lot", type=int, default=8, help="Batch size")
    analyseur.add_argument(
        "--taille-image",
        type=int,
        default=640,
        help="Taille cible des images (imgsz YOLO). Doit correspondre au dataset.",
    )
    analyseur.add_argument("--lr", type=float, default=1e-2, help="lr0 Ultralytics")
    analyseur.add_argument("--lrf", type=float, default=0.01, help="Fraction finale du LR")
    analyseur.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Décroissance des poids Ultralytics (frein léger : 1e-4)",
    )
    analyseur.add_argument("--patience", type=int, default=50, help="Patience early stopping")
    analyseur.add_argument("--workers", type=int, default=2)
    analyseur.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        help="Optimiseur Ultralytics : auto, SGD, Adam, AdamW, MuSGD selon version",
    )
    analyseur.add_argument(
        "--warmup-epochs",
        type=float,
        default=5.0,
        help="Nombre d'époques de warmup",
    )
    analyseur.add_argument(
        "--no-cos-lr",
        dest="cos_lr",
        action="store_false",
        default=True,
        help="Désactive le scheduler cosinus",
    )
    analyseur.add_argument(
        "--mask-ratio",
        type=int,
        default=1,
        help=(
            "Sous-échantillonnage des masques. "
            "1 = pleine résolution (recommandé pour fissures fines de 1-3px)."
        ),
    )
    analyseur.add_argument(
        "--sans-overlap-mask",
        dest="overlap_mask",
        action="store_false",
        default=True,
        help="Désactive la fusion des masques qui se chevauchent",
    )
    analyseur.add_argument(
        "--multi-classes",
        dest="classe_unique",
        action="store_false",
        default=True,
        help="Désactive single_cls (utile si dataset multi-classes)",
    )
    analyseur.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction du dataset utilisée pour entraînement rapide/diagnostic",
    )
    analyseur.add_argument(
        "--cache",
        choices=["none", "ram", "disk"],
        default="none",
        help="Cache Ultralytics. 'ram' accélère si RAM ≥ 32 Go.",
    )
    analyseur.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Nombre maximal de détections par image pendant la validation",
    )
    analyseur.add_argument(
        "--save-period",
        type=int,
        default=5,
        help="Sauvegarde un checkpoint YOLO toutes les N époques (-1 désactive)",
    )
    analyseur.add_argument(
        "--dispositif",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Dispositif de calcul",
    )
    analyseur.add_argument(
        "--nom",
        type=str,
        default="yolo_seg_fissures",
        help="Nom de l'expérience Ultralytics",
    )
    analyseur.add_argument(
        "--copier-images",
        action="store_true",
        help="(Mode COCO→YOLO) Copie les images au lieu de créer des liens symboliques",
    )
    analyseur.add_argument(
        "--convertir-seulement",
        action="store_true",
        help="(Mode COCO→YOLO) Convertit le dataset sans entraîner",
    )
    analyseur.add_argument(
        "--exist-ok",
        action="store_true",
        help="Autorise Ultralytics à réutiliser un dossier d'expérience existant",
    )
    analyseur.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Reprend un entraînement YOLO depuis un checkpoint last.pt",
    )
    analyseur.add_argument(
        "--silencieux",
        action="store_true",
        help="Réduit les journaux YOLO au minimum",
    )
    return analyseur.parse_args()


def _device_ultralytics(dispositif: str) -> str | None:
    """Traduit le nom de dispositif du projet vers Ultralytics."""
    if dispositif == "auto":
        return None
    if dispositif == "cuda":
        return "0"
    return dispositif


def valider_arguments_yolo(args: argparse.Namespace) -> None:
    """Valide les réglages qui causent sinon des erreurs tardives Ultralytics."""
    if args.taille_image <= 0:
        raise ValueError("--taille-image doit être un entier positif.")
    if args.taille_image % 32 != 0:
        print(
            "[YOLO] Attention : --taille-image n'est pas un multiple de 32. "
            "Ultralytics peut ajuster imgsz pendant l'entraînement."
        )
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction doit être dans l'intervalle ]0, 1].")
    if args.mask_ratio < 1:
        raise ValueError("--mask-ratio doit être >= 1.")


def _cache_ultralytics(cache: str) -> bool | str:
    """Traduit l'option cache CLI vers Ultralytics."""
    if cache == "ram":
        return True
    if cache == "disk":
        return "disk"
    return False


def verifier_modele_yolo_seg(chemin_modele: str) -> None:
    """Bloque les poids YOLO qui ne sont pas des modèles de segmentation.

    Accepte :
    - Les noms officiels YOLO11-seg et YOLO26-seg (liste dans parametres.py)
    - Tout fichier .pt dont le nom contient 'yolo' ET '-seg' (checkpoints custom)
    """
    nom_modele = Path(chemin_modele).name

    if nom_modele in MODELES_YOLO_SEG_OFFICIELS:
        return

    stem = Path(nom_modele).stem.lower()
    if stem.startswith("yolo") and "-seg" in stem:
        return

    from detection_fissures.configuration.parametres import (
        MODELES_YOLOV11_SEG_OFFICIELS,
        MODELES_YOLO26_SEG_OFFICIELS,
    )
    choix_yolo11 = ", ".join(MODELES_YOLOV11_SEG_OFFICIELS)
    choix_yolo26 = ", ".join(MODELES_YOLO26_SEG_OFFICIELS)
    raise ValueError(
        f"Modèle YOLO non autorisé : '{chemin_modele}'.\n"
        f"  YOLO11 acceptés : {choix_yolo11}\n"
        f"  YOLO26 acceptés : {choix_yolo26}\n"
        "  Ou un checkpoint local nommé 'yolo*-seg*.pt'."
    )


def _compter_lignes_labels(dossier_labels: Path) -> tuple[int, int]:
    """Retourne (images annotees, instances) pour un dossier labels YOLO."""
    fichiers = sorted(dossier_labels.glob("*.txt"))
    images_annotees = 0
    instances = 0
    for fichier in fichiers:
        lignes = [
            ligne
            for ligne in fichier.read_text(encoding="utf-8").splitlines()
            if ligne.strip()
        ]
        if lignes:
            images_annotees += 1
            instances += len(lignes)
    return images_annotees, instances


def afficher_resume_dataset_yolo(racine_dataset_yolo: Path) -> None:
    """Affiche un résumé compact du dataset YOLO."""
    print("\n" + "═" * 55)
    print("  DATASET YOLO-SEG")
    print("═" * 55)
    for split in ("train", "valid", "test"):
        dossier_images = racine_dataset_yolo / "images" / split
        dossier_labels = racine_dataset_yolo / "labels" / split
        if not dossier_images.exists():
            continue
        nb_images = sum(
            1
            for chemin in dossier_images.iterdir()
            if chemin.is_file() or chemin.is_symlink()
        )
        images_annotees, instances = _compter_lignes_labels(dossier_labels)
        murs_sains = max(0, nb_images - images_annotees)
        print(
            f"  {split:<5} : {nb_images:5d} image(s) | "
            f"{images_annotees:5d} avec fissure(s) | "
            f"{murs_sains:5d} mur(s) sain(s) | {instances:6d} instance(s)"
        )
    print("═" * 55 + "\n")


_EXTENSIONS_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collecter_images(dossier: Path) -> list[Path]:
    """Liste (récursivement) toutes les images d'un dossier."""
    return sorted(
        chemin
        for chemin in dossier.rglob("*")
        if chemin.is_file() and chemin.suffix.lower() in _EXTENSIONS_IMAGE
    )


def _split_depuis_chemin(chemin: Path, racine: Path) -> str | None:
    """Déduit le split (train/valid/test) depuis un sous-dossier nommé, sinon None."""
    for element in chemin.relative_to(racine).parts:
        token = element.lower()
        if token in ("train", "entrainement", "entrainements"):
            return "train"
        if token in ("valid", "val", "validation"):
            return "valid"
        if token == "test":
            return "test"
    return None


def _compter_images_split(racine_dataset: Path, split: str) -> int:
    """Compte les images d'un split du dataset YOLO."""
    dossier = racine_dataset / "images" / split
    if not dossier.exists():
        return 0
    return sum(
        1 for chemin in dossier.iterdir() if chemin.is_file() or chemin.is_symlink()
    )


def _repartir_murs_sains(
    images: list[Path], racine_dataset: Path
) -> list[tuple[Path, str]]:
    """Répartit les murs sains proportionnellement aux splits existants."""
    comptes = {
        split: _compter_images_split(racine_dataset, split)
        for split in ("train", "valid", "test")
    }
    total = sum(comptes.values())
    if total == 0:
        return [(chemin, "train") for chemin in images]

    nombre = len(images)
    n_valid = round(nombre * comptes["valid"] / total)
    n_test = round(nombre * comptes["test"] / total)
    n_train = nombre - n_valid - n_test

    assignations: list[tuple[Path, str]] = []
    for index, chemin in enumerate(images):
        if index < n_train:
            assignations.append((chemin, "train"))
        elif index < n_train + n_valid:
            assignations.append((chemin, "valid"))
        else:
            assignations.append((chemin, "test"))
    return assignations


def integrer_murs_sains(
    racine_dataset: Path, dossier_murs_sains: Path, prefixe: str = "mur_sain_"
) -> dict[str, int]:
    """Intègre des murs sains dans le dataset YOLO comme exemples négatifs.

    Chaque image est copiée dans ``images/<split>/`` avec un fichier label
    VIDE dans ``labels/<split>/`` : c'est ce qui signale « aucune fissure »
    à YOLO tout en gardant une seule classe (``crack``).

    - Si ``dossier_murs_sains`` contient des sous-dossiers train/valid/test,
      ils sont respectés ; sinon les images sont réparties proportionnellement
      aux splits déjà présents dans le dataset.
    - Idempotent : une image déjà copiée est ignorée (le nom de destination
      inclut un hachage du chemin source), donc relancer ne duplique rien et
      deux images de même nom dans des sous-dossiers différents ne s'écrasent pas.
    """
    if not dossier_murs_sains.exists():
        raise FileNotFoundError(
            f"Dossier de murs sains introuvable : {dossier_murs_sains}"
        )

    # Garde-fou : le dossier source ne doit pas être à l'intérieur du dataset
    # (ni l'inverse), sinon rglob ré-ingérerait les copies à chaque relance.
    racine_resolue = racine_dataset.resolve()
    source_resolue = dossier_murs_sains.resolve()
    if source_resolue.is_relative_to(racine_resolue) or racine_resolue.is_relative_to(
        source_resolue
    ):
        raise ValueError(
            "--murs-sains ne doit pas être situé à l'intérieur du dataset "
            f"({racine_resolue}). Place le dossier des murs sains à côté du "
            "dataset (ex. murs_sains/ frère de dataset/)."
        )

    images = _collecter_images(dossier_murs_sains)
    if not images:
        raise ValueError(
            f"Aucune image trouvée dans le dossier de murs sains : {dossier_murs_sains}"
        )

    avec_split = [(chemin, _split_depuis_chemin(chemin, dossier_murs_sains)) for chemin in images]
    if any(split for _, split in avec_split):
        assignations = [(chemin, split or "train") for chemin, split in avec_split]
    else:
        assignations = _repartir_murs_sains(images, racine_dataset)

    ajoutes = {"train": 0, "valid": 0, "test": 0}
    ignores = 0
    for chemin_source, split in assignations:
        dossier_images = racine_dataset / "images" / split
        dossier_labels = racine_dataset / "labels" / split
        dossier_images.mkdir(parents=True, exist_ok=True)
        dossier_labels.mkdir(parents=True, exist_ok=True)

        # Nom de destination unique et déterministe : hachage du chemin relatif
        # source → pas de collision entre fichiers homonymes, et idempotent.
        relatif = chemin_source.relative_to(source_resolue)
        empreinte = hashlib.sha1(str(relatif).encode("utf-8")).hexdigest()[:8]
        dest_image = dossier_images / f"{prefixe}{empreinte}_{chemin_source.name}"
        dest_label = dossier_labels / f"{prefixe}{empreinte}_{chemin_source.stem}.txt"

        if dest_image.exists():
            ignores += 1
        else:
            shutil.copy2(chemin_source, dest_image)
            ajoutes[split] += 1
        if not dest_label.exists():
            dest_label.write_text("", encoding="utf-8")  # label vide = exemple négatif

    print("\n" + "═" * 55)
    print("  INTÉGRATION DES MURS SAINS (exemples négatifs)")
    print("═" * 55)
    print(f"  Source            : {dossier_murs_sains}")
    print(f"  Images trouvées   : {len(images)}")
    for split in ("train", "valid", "test"):
        if ajoutes[split]:
            print(f"  Ajoutés ({split:<5})  : {ajoutes[split]}")
    if ignores:
        print(f"  Déjà présents     : {ignores} (ignorés)")
    print("  Label par image   : VIDE  →  reste 1 classe (crack)")
    print("═" * 55 + "\n")
    return ajoutes


def _valeur_metrique_yolo(resultats: Any, fragments_cles: tuple[str, ...]) -> float | None:
    """Récupère une métrique Ultralytics depuis results_dict si disponible."""
    dictionnaire = getattr(resultats, "results_dict", None)
    if not isinstance(dictionnaire, dict):
        return None

    for cle, valeur in dictionnaire.items():
        cle_min = str(cle).lower()
        if all(fragment.lower() in cle_min for fragment in fragments_cles):
            try:
                return float(valeur)
            except (TypeError, ValueError):
                return None
    return None


def _calculer_f1(precision: float | None, rappel: float | None) -> float | None:
    """Calcule le F1 score depuis précision et rappel si les deux existent."""
    if precision is None or rappel is None:
        return None
    denominateur = precision + rappel
    if denominateur <= 0:
        return 0.0
    return 2.0 * precision * rappel / denominateur


def afficher_metriques_yolo(resultats: Any, titre: str) -> None:
    """Affiche les métriques YOLO principales quand Ultralytics les expose."""
    precision_masque = _valeur_metrique_yolo(resultats, ("precision", "(m)"))
    rappel_masque = _valeur_metrique_yolo(resultats, ("recall", "(m)"))
    precision_boite = _valeur_metrique_yolo(resultats, ("precision", "(b)"))
    rappel_boite = _valeur_metrique_yolo(resultats, ("recall", "(b)"))

    valeurs = [
        ("mAP@0.5 masque", _valeur_metrique_yolo(resultats, ("map50", "(m)"))),
        ("mAP@0.5:0.95 masque", _valeur_metrique_yolo(resultats, ("map50-95", "(m)"))),
        ("Précision masque", precision_masque),
        ("Rappel masque", rappel_masque),
        ("F1 score masque", _calculer_f1(precision_masque, rappel_masque)),
        ("mAP@0.5 boîte", _valeur_metrique_yolo(resultats, ("map50", "(b)"))),
        ("mAP@0.5:0.95 boîte", _valeur_metrique_yolo(resultats, ("map50-95", "(b)"))),
        ("Précision boîte", precision_boite),
        ("Rappel boîte", rappel_boite),
        ("F1 score boîte", _calculer_f1(precision_boite, rappel_boite)),
    ]
    valeurs = [(label, valeur) for label, valeur in valeurs if valeur is not None]

    if not valeurs:
        print(f"[YOLO] {titre} : métriques détaillées non exposées par cette version.")
        return

    print("\n" + "═" * 55)
    print(f"  {titre}")
    print("═" * 55)
    for label, valeur in valeurs:
        barre = "█" * int(max(0.0, min(1.0, valeur)) * 20)
        print(f"  {label:<24} : {valeur:.4f}  {barre}")
    print("═" * 55 + "\n")


def installer_journaux_yolo(modele: Any, actif: bool = True) -> None:
    """Ajoute un callback léger pour éviter les entraînements YOLO muets."""
    if not actif or not hasattr(modele, "add_callback"):
        return

    def journaliser_epoque(trainer: Any) -> None:
        epoque = int(getattr(trainer, "epoch", -1)) + 1
        args = getattr(trainer, "args", None)
        total = getattr(args, "epochs", "?")
        metriques = getattr(trainer, "metrics", None)
        elements = [f"[YOLO] Époque {epoque}/{total}"]

        if isinstance(metriques, dict) and metriques:
            for cle, valeur in metriques.items():
                cle_min = str(cle).lower()
                if "map50" in cle_min or "precision" in cle_min or "recall" in cle_min:
                    try:
                        elements.append(f"{cle}={float(valeur):.4f}")
                    except (TypeError, ValueError):
                        pass

        loss_items = getattr(trainer, "loss_items", None)
        if loss_items is not None:
            try:
                valeurs = [float(valeur) for valeur in loss_items]
                elements.append("loss=" + ",".join(f"{valeur:.4f}" for valeur in valeurs))
            except (TypeError, ValueError):
                pass

        print(" | ".join(elements), flush=True)

    try:
        modele.add_callback("on_fit_epoch_end", journaliser_epoque)
    except Exception as exc:
        print(
            f"[YOLO] Callback de journalisation non installé "
            f"({type(exc).__name__}: {exc}). Ultralytics gardera ses logs natifs."
        )


def construire_arguments_train_yolo(
    args: argparse.Namespace,
    chemin_yaml: Path,
    racine_sorties: Path,
) -> dict[str, Any]:
    """Centralise les arguments Ultralytics pour éviter les divergences."""
    return {
        "data": str(chemin_yaml),
        "task": "segment",
        "epochs": args.epoques,
        "imgsz": args.taille_image,
        "batch": args.lot,
        "optimizer": args.optimizer,
        "lr0": args.lr,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "device": _device_ultralytics(args.dispositif),
        "workers": args.workers,
        "project": str(racine_sorties / "entrainements"),
        "name": args.nom,
        "exist_ok": args.exist_ok,
        "resume": args.resume is not None,
        "verbose": not args.silencieux,
        "plots": True,
        "val": True,
        "save_period": args.save_period,
        "cos_lr": args.cos_lr,
        # Augmentations entièrement désactivées : le dataset Roboflow est déjà
        # pré-augmenté (5×) et les murs sains servent d'exemples négatifs bruts.
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "bgr": 0.0,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "auto_augment": None,
        "multi_scale": False,
        "mask_ratio": args.mask_ratio,
        "overlap_mask": args.overlap_mask,
        "single_cls": args.classe_unique,
        "fraction": args.fraction,
        "cache": _cache_ultralytics(args.cache),
        "max_det": args.max_det,
        "amp": True,
        "deterministic": True,
    }


def afficher_configuration_yolo(args: argparse.Namespace, checkpoint_resume: Path | None) -> None:
    """Affiche la configuration YOLO utile en un seul endroit."""
    lignes = [
        ("Modèle", args.modele),
        ("Époques", args.epoques),
        ("Batch", args.lot),
        ("Image", args.taille_image),
        ("Optimiseur", args.optimizer),
        ("LR initial", args.lr),
        ("LR final frac", args.lrf),
        ("Weight decay", args.weight_decay),
        ("Warmup", args.warmup_epochs),
        ("Patience", args.patience),
        ("cos_lr", args.cos_lr),
        ("Augmentation", "désactivée"),
        ("Mask ratio", args.mask_ratio),
        ("Classe unique", args.classe_unique),
        ("Cache", args.cache),
        ("Workers", args.workers),
        ("Dispositif", args.dispositif),
    ]
    print("\n" + "═" * 55)
    print("  CONFIGURATION YOLO-SEG")
    print("═" * 55)
    for label, valeur in lignes:
        print(f"  {label:<13}: {valeur}")
    if checkpoint_resume is not None:
        print(f"  Reprise      : {checkpoint_resume}")
    print("═" * 55 + "\n")


def main() -> None:
    """Entraîne YOLO-seg — supporte les datasets YOLO natifs et COCO."""
    args = analyser_arguments()
    valider_arguments_yolo(args)
    verifier_modele_yolo_seg(args.modele)
    racine_sorties = Path(args.sorties).expanduser().resolve()
    checkpoint_resume = Path(args.resume).expanduser().resolve() if args.resume else None
    if checkpoint_resume is not None and not checkpoint_resume.is_file():
        raise FileNotFoundError(f"Checkpoint YOLO de reprise introuvable : {checkpoint_resume}")

    # ── Déterminer le chemin YAML ──────────────────────────────────────────────
    if args.yaml:
        # Mode explicite : dataset YOLO natif fourni via --yaml
        chemin_yaml = Path(args.yaml).expanduser().resolve()
        if not chemin_yaml.is_file():
            raise FileNotFoundError(f"Fichier data.yaml introuvable : {chemin_yaml}")
        print(f"[YOLO] Dataset YOLO natif (--yaml) : {chemin_yaml}")
        racine_dataset_yolo = chemin_yaml.parent
    else:
        # Détection automatique : data.yaml présent dans --donnees ?
        chemin_yaml_auto = Path(args.donnees).expanduser().resolve() / "data.yaml"
        if chemin_yaml_auto.is_file():
            chemin_yaml = chemin_yaml_auto
            print(f"[YOLO] Format YOLO natif détecté automatiquement : {chemin_yaml}")
            racine_dataset_yolo = chemin_yaml.parent
        else:
            # Conversion depuis COCO
            racine_dataset_yolo = racine_sorties / "dataset_yolo"
            from detection_fissures.donnees.conversion_yolo import convertir_dataset_coco_vers_yolo
            chemin_yaml = convertir_dataset_coco_vers_yolo(
                racine_coco=args.donnees,
                racine_yolo=racine_dataset_yolo,
                copier_images=args.copier_images,
                taille_image=args.taille_image,
            )
            print(f"[YOLO] Dataset converti depuis COCO : {chemin_yaml}")

    # Intégration optionnelle des murs sains comme exemples négatifs (label vide)
    if args.murs_sains:
        dossier_murs_sains = Path(args.murs_sains).expanduser().resolve()
        integrer_murs_sains(racine_dataset_yolo, dossier_murs_sains)

    afficher_resume_dataset_yolo(racine_dataset_yolo)

    if args.convertir_seulement:
        print("[YOLO] Conversion terminée, entraînement ignoré.")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Le paquet 'ultralytics>=8.4.0' est requis pour YOLO11/YOLO26-seg. "
            "Installez-le avec : pip install -U ultralytics"
        ) from exc

    afficher_configuration_yolo(args, checkpoint_resume)

    modele = YOLO(str(checkpoint_resume) if checkpoint_resume is not None else args.modele)
    installer_journaux_yolo(modele, actif=not args.silencieux)
    resultats = modele.train(**construire_arguments_train_yolo(args, chemin_yaml, racine_sorties))

    print(f"[YOLO] Entraînement terminé : {resultats.save_dir}")
    afficher_metriques_yolo(resultats, "MÉTRIQUES YOLO — VALIDATION")

    metriques_test = modele.val(
        data=str(chemin_yaml),
        split="test",
        imgsz=args.taille_image,
        batch=1,
        device=_device_ultralytics(args.dispositif),
        project=str(racine_sorties / "evaluations"),
        name=f"{args.nom}_test",
        exist_ok=True,
        single_cls=args.classe_unique,
        max_det=args.max_det,
        verbose=not args.silencieux,
        plots=True,
    )
    print(f"[YOLO] Évaluation test terminée : {metriques_test.save_dir}")
    afficher_metriques_yolo(metriques_test, "MÉTRIQUES YOLO — TEST")


if __name__ == "__main__":
    main()
