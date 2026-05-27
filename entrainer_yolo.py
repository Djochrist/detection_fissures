"""
Entraînement YOLO11-seg — Détection de fissures structurelles.

COMMANDE COMPLÈTE RECOMMANDÉE POUR CE PROJET
═════════════════════════════════════════════════════════════════════

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

  Pour reprendre un entraînement interrompu :
    python entrainer_yolo.py \\
      --resume sorties_yolo/entrainements/yolo11_seg_fissures/weights/last.pt

JUSTIFICATION DES PARAMÈTRES SPÉCIFIQUES AUX FISSURES
══════════════════════════════════════════════════════════════════════

  --modele yolo11s-seg.pt
      Le modèle SMALL (11.2M paramètres) est recommandé pour les fissures.
      - nano (yolo11n) : trop peu de capacité pour détecter des traits fins
      - small (yolo11s) : bon équilibre profondeur/vitesse pour 384×384
      - medium/large : gains marginaux, risque d'overfitting (petit dataset)

  --taille-image 384
      Identique à la résolution du dataset (export Roboflow 384×384).
      Ne pas mettre 640 : le modèle serait upscalé → artefacts.
      384 est un multiple de 32 (exigé par YOLO) ✓

  --mask-ratio 1  ← PARAMÈTRE CRITIQUE POUR LES FISSURES
      mask_ratio contrôle la résolution des masques de segmentation :
        mask_ratio=4 (défaut YOLO) → masques 384/4 = 96×96 px
        mask_ratio=1              → masques 384×384 px (pleine résolution)
      Les fissures ont souvent 1–3 pixels de large.
      Avec mask_ratio=4, les contours sont flous et la classification
      PCA/distance-transform devient imprécise.
      mask_ratio=1 conserve la qualité des contours au coût de +~30% RAM.

  --copy-paste 0.4
      Probabilité de "coller" des instances de fissures entre images.
      Efficace pour les objets rares (fissures < 5% des pixels).
      0.4 = 40% des lots bénéficient de cet augmentation.

  --degrees 45.0
      Rotation aléatoire ±45°. Les fissures structurelles peuvent être
      horizontales, verticales, inclinées ou en diagonale.

  --flipud 0.0
      PAS de flip vertical. Un flip vertical transformerait une fissure
      verticale descendant avec la gravité en montant : irréaliste.

  --mixup 0.0
      Pas de mixup. Le mixup mélange deux images, brouillant les
      contours fins des fissures et dégradant les masques.

  --mosaic 1.0 + --close-mosaic 30
      Mosaic actif 100% du temps sauf les 30 dernières époques.
      La fin sans mosaic permet au modèle de s'adapter aux images réelles
      (sans collage) et de mieux prédire sur les images de test.

  --patience 50
      50 époques sans amélioration avant arrêt. Élevé car :
      - Les fissures sont des objets difficiles (long à converger)
      - La mosaic et copy-paste introduisent de la variance

  --epoques 150
      Plus long que le défaut YOLO (100). Justifié par :
      - Objets rares et petits (lente convergence)
      - close-mosaic à 30 → 120 époques avec mosaic + 30 sans

  --cos-lr
      Cosine annealing : LR décroît en cosinus de lr0 à lr0×lrf.
      Meilleur que le scheduler linéaire pour les petits datasets.

  --hsv-v 0.4 + --hsv-s 0.5
      Variation de luminosité et saturation importante pour les murs :
      béton sec vs humide, ombre vs plein soleil.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

RACINE_PROJET = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE_PROJET.parent))

from detection_fissures.configuration.parametres import (
    MODELE_YOLOV11_SEG_DEFAUT,
    MODELES_YOLOV11_SEG_OFFICIELS,
    ConfigurationGlobale,
)


def analyser_arguments() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    p = argparse.ArgumentParser(
        description="Entraînement YOLO11-seg — Détection de fissures structurelles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--donnees", type=str, default="dataset",
        help="Répertoire COCO racine (contient train/, valid/, test/)",
    )
    p.add_argument(
        "--sorties", type=str, default="sorties_yolo",
        help="Répertoire de sortie YOLO",
    )
    p.add_argument(
        "--modele", type=str, default=MODELE_YOLOV11_SEG_DEFAUT,
        help="Poids YOLO11-seg : yolo11n/s/m/l/x-seg.pt (s recommandé pour fissures)",
    )
    p.add_argument(
        "--taille-image", type=int, default=384,
        help="Résolution d'entrée (doit correspondre au dataset, multiple de 32)",
    )
    p.add_argument(
        "--epoques", type=int, default=150,
        help="Nombre d'époques (150 pour fissures : objets difficiles à converger)",
    )
    p.add_argument(
        "--lot", type=int, default=8,
        help="Batch size. 8=CPU/GPU, -1=autobatch YOLO (GPU requis)",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Taux d'apprentissage initial (lr0)",
    )
    p.add_argument(
        "--lrf", type=float, default=0.01,
        help="Ratio lr final : lr_final = lr0 × lrf (cosine annealing)",
    )
    p.add_argument(
        "--weight-decay", type=float, default=5e-4,
        help="Décroissance L2 (anti-overfitting datasets < 5000 images)",
    )
    p.add_argument(
        "--patience", type=int, default=50,
        help="Patience early stopping (50 = fissures difficiles, convergence lente)",
    )
    p.add_argument(
        "--workers", type=int, default=2,
        help="Processus de chargement des données",
    )
    p.add_argument(
        "--save-period", type=int, default=10,
        help="Checkpoint toutes les N époques (-1 = seulement meilleur/dernier)",
    )
    p.add_argument(
        "--warmup-epoques", type=float, default=5.0,
        help="Époques d'échauffement LR (linéaire de 0 → lr0)",
    )
    p.add_argument(
        "--close-mosaic", type=int, default=30,
        help=(
            "Désactive mosaic N époques avant la fin. "
            "Les dernières époques sans mosaic améliorent la généralisation."
        ),
    )
    p.add_argument(
        "--mask-ratio", type=int, default=1,
        help=(
            "Résolution des masques (1=pleine résolution CRITIQUE pour fissures fines). "
            "mask_ratio=4 (défaut YOLO) → 96px pour images 384px → fissures floues."
        ),
    )
    p.add_argument(
        "--overlap-mask", action="store_true", default=True,
        help="Autorise les masques qui se chevauchent (recommandé pour fissures denses)",
    )
    p.add_argument(
        "--copy-paste", type=float, default=0.4,
        help="Probabilité copy-paste (0.4 = augmentation pour objets rares comme les fissures)",
    )
    p.add_argument(
        "--mosaic", type=float, default=1.0,
        help="Probabilité mosaic [0,1] (1.0 = actif tout le temps sauf close-mosaic)",
    )
    p.add_argument(
        "--mixup", type=float, default=0.0,
        help="Probabilité mixup (0 = désactivé : brouille les contours fins des fissures)",
    )
    p.add_argument(
        "--degrees", type=float, default=45.0,
        help="Rotation aléatoire ±degrés (45° car fissures à tous les angles structurels)",
    )
    p.add_argument(
        "--translate", type=float, default=0.1,
        help="Translation aléatoire (fraction de l'image)",
    )
    p.add_argument(
        "--scale", type=float, default=0.5,
        help="Mise à l'échelle aléatoire ±scale",
    )
    p.add_argument(
        "--fliplr", type=float, default=0.5,
        help="Probabilité flip horizontal (0.5 = symétrie des murs OK)",
    )
    p.add_argument(
        "--flipud", type=float, default=0.0,
        help="Probabilité flip vertical (0 = non : inverserait la direction des fissures de gravité)",
    )
    p.add_argument(
        "--hsv-h", type=float, default=0.02,
        help="Variation de teinte HSV (béton = palette restreinte)",
    )
    p.add_argument(
        "--hsv-s", type=float, default=0.5,
        help="Variation saturation HSV (mur sec vs humide)",
    )
    p.add_argument(
        "--hsv-v", type=float, default=0.4,
        help="Variation luminosité HSV (ombre vs plein soleil sur murs)",
    )
    p.add_argument(
        "--cos-lr", action="store_true", default=True,
        help="Cosine annealing LR (meilleur que linéaire pour petit dataset)",
    )
    p.add_argument(
        "--freeze", type=int, default=0,
        help=(
            "Couches backbone à geler (0 = tout entraînable). "
            "Augmenter si overfitting sévère (ex: --freeze 10)."
        ),
    )
    p.add_argument(
        "--amp", action="store_true", default=True,
        help="Précision mixte AMP (GPU uniquement, ignoré sur CPU)",
    )
    p.add_argument(
        "--dispositif", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Dispositif de calcul",
    )
    p.add_argument(
        "--nom", type=str, default="yolo11_seg_fissures",
        help="Nom de l'expérience (dossier de sortie Ultralytics)",
    )
    p.add_argument(
        "--copier-images", action="store_true",
        help="Copier les images au lieu de créer des symlinks",
    )
    p.add_argument(
        "--convertir-seulement", action="store_true",
        help="Convertir COCO → YOLO sans entraîner",
    )
    p.add_argument(
        "--exist-ok", action="store_true",
        help="Réutiliser un dossier d'expérience existant",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Reprendre depuis un checkpoint last.pt. Ex : sorties_yolo/.../weights/last.pt",
    )
    p.add_argument(
        "--silencieux", action="store_true",
        help="Réduire les logs YOLO au minimum",
    )
    return p.parse_args()


def generer_commande_yolo(args: argparse.Namespace) -> str:
    """Génère la commande complète avec tous les paramètres utilisés."""
    lignes = [
        "python entrainer_yolo.py \\",
        f"  --donnees          {args.donnees} \\",
        f"  --modele           {args.modele} \\",
        f"  --taille-image     {args.taille_image} \\",
        f"  --epoques          {args.epoques} \\",
        f"  --lot              {args.lot} \\",
        f"  --lr               {args.lr} \\",
        f"  --lrf              {args.lrf} \\",
        f"  --weight-decay     {args.weight_decay} \\",
        f"  --patience         {args.patience} \\",
        f"  --warmup-epoques   {args.warmup_epoques} \\",
        f"  --close-mosaic     {args.close_mosaic} \\",
        f"  --mask-ratio       {args.mask_ratio} \\",
        f"  --overlap-mask \\" if args.overlap_mask else "  # --overlap-mask désactivé \\",
        f"  --copy-paste       {args.copy_paste} \\",
        f"  --mosaic           {args.mosaic} \\",
        f"  --mixup            {args.mixup} \\",
        f"  --degrees          {args.degrees} \\",
        f"  --translate        {args.translate} \\",
        f"  --scale            {args.scale} \\",
        f"  --fliplr           {args.fliplr} \\",
        f"  --flipud           {args.flipud} \\",
        f"  --hsv-h            {args.hsv_h} \\",
        f"  --hsv-s            {args.hsv_s} \\",
        f"  --hsv-v            {args.hsv_v} \\",
        f"  --cos-lr \\" if args.cos_lr else "  # --cos-lr désactivé \\",
        f"  --freeze           {args.freeze} \\",
        f"  --amp \\" if args.amp else "  # --amp désactivé \\",
        f"  --workers          {args.workers} \\",
        f"  --save-period      {args.save_period} \\",
        f"  --dispositif       {args.dispositif} \\",
        f"  --nom              {args.nom} \\",
        f"  --sorties          {args.sorties}",
    ]
    if args.resume:
        lignes[-1] += " \\"
        lignes.append(f"  --resume           {args.resume}")
    return "\n".join(lignes)


def afficher_commande_complete(args: argparse.Namespace, titre: str = "COMMANDE UTILISÉE") -> None:
    """Affiche la commande complète avec tous les paramètres."""
    commande = generer_commande_yolo(args)
    print("\n" + "═" * 68)
    print(f"  {titre}")
    print("═" * 68)
    print()
    for ligne in commande.splitlines():
        print(f"  {ligne}")
    print()
    print("═" * 68 + "\n")


def verifier_modele_yolov11_seg(chemin_modele: str) -> None:
    """Vérifie que le modèle est bien un YOLO11-seg valide."""
    nom = Path(chemin_modele).name
    if nom in MODELES_YOLOV11_SEG_OFFICIELS:
        return
    if nom.startswith("yolo11") and "-seg" in Path(nom).stem:
        return
    choix = ", ".join(MODELES_YOLOV11_SEG_OFFICIELS)
    raise ValueError(
        f"Modèle non autorisé : {chemin_modele}\n"
        f"Acceptés : {choix}\n"
        "Ou un checkpoint .pt entraîné par ce projet (last.pt, best.pt)."
    )


def _device_ultralytics(dispositif: str) -> str | None:
    """Traduit le dispositif vers le format Ultralytics."""
    if dispositif == "auto":
        return None
    if dispositif == "cuda":
        return "0"
    return dispositif


def _compter_lignes_labels(dossier_labels: Path) -> tuple[int, int]:
    """Retourne (nb_avec_fissures, nb_instances) pour un dossier labels."""
    fichiers = sorted(dossier_labels.glob("*.txt"))
    avec, instances = 0, 0
    for f in fichiers:
        lignes = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lignes:
            avec += 1
            instances += len(lignes)
    return avec, instances


def afficher_resume_dataset_yolo(racine_dataset_yolo: Path) -> None:
    """Affiche un résumé du dataset converti avec statistiques."""
    print("\n" + "═" * 68)
    print("  DATASET YOLO-SEG CONVERTI")
    print("═" * 68)
    print(
        f"  {'Split':<6} {'Total':>7} {'Avec fissures':>14} "
        f"{'Murs sains':>11} {'Instances':>10}"
    )
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*14}-+-{'-'*11}-+-{'-'*10}")
    for split in ("train", "valid", "test"):
        dossier_images = racine_dataset_yolo / "images" / split
        dossier_labels = racine_dataset_yolo / "labels" / split
        if not dossier_images.exists():
            continue
        nb_images = sum(1 for p in dossier_images.iterdir() if p.is_file() or p.is_symlink())
        nb_avec, instances = _compter_lignes_labels(dossier_labels)
        print(
            f"  {split:<6} {nb_images:>7} {nb_avec:>14} "
            f"{nb_images - nb_avec:>11} {instances:>10}"
        )
    print("═" * 68)
    print(
        "  Murs sains → label .txt vide → exemples négatifs YOLO.\n"
        "  mask_ratio=1 → masques pleine résolution pour fissures fines."
    )
    print("═" * 68 + "\n")


def _valeur_metrique_yolo(resultats: Any, fragments: tuple[str, ...]) -> float | None:
    """Récupère une métrique YOLO depuis results_dict."""
    d = getattr(resultats, "results_dict", None)
    if not isinstance(d, dict):
        return None
    for cle, val in d.items():
        cle_min = str(cle).lower()
        if all(f.lower() in cle_min for f in fragments):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _f1(p: float | None, r: float | None) -> float | None:
    """Calcule le F1 depuis précision et rappel."""
    if p is None or r is None:
        return None
    d = p + r
    return 0.0 if d <= 0 else 2.0 * p * r / d


def afficher_metriques_yolo(resultats: Any, titre: str) -> None:
    """Affiche et interprète les métriques YOLO pour la détection de fissures."""
    pm = _valeur_metrique_yolo(resultats, ("precision", "(m)"))
    rm = _valeur_metrique_yolo(resultats, ("recall", "(m)"))
    pb = _valeur_metrique_yolo(resultats, ("precision", "(b)"))
    rb = _valeur_metrique_yolo(resultats, ("recall", "(b)"))

    valeurs = [
        ("mAP@0.5 masque",      _valeur_metrique_yolo(resultats, ("map50", "(m)"))),
        ("mAP@0.5:0.95 masque", _valeur_metrique_yolo(resultats, ("map50-95", "(m)"))),
        ("Précision masque",    pm),
        ("Rappel masque",       rm),
        ("F1 masque",           _f1(pm, rm)),
        ("mAP@0.5 boîte",       _valeur_metrique_yolo(resultats, ("map50", "(b)"))),
        ("Précision boîte",     pb),
        ("Rappel boîte",        rb),
        ("F1 boîte",            _f1(pb, rb)),
    ]
    valeurs = [(l, v) for l, v in valeurs if v is not None]
    if not valeurs:
        print(f"[Métriques] {titre} : non exposées par cette version Ultralytics.")
        return

    print("\n" + "═" * 68)
    print(f"  {titre}")
    print("═" * 68)
    for label, valeur in valeurs:
        barre = "█" * int(max(0.0, min(1.0, valeur)) * 25)
        print(f"  {label:<26} : {valeur:.4f}  {barre}")

    print()
    print("  INTERPRÉTATION POUR LA DÉTECTION DE FISSURES")
    print(f"  {'─'*64}")
    print("  PRIORITÉ → Rappel masque : fissure manquée = danger structurel.")

    rappel_m = next((v for l, v in valeurs if "rappel masque" in l.lower()), 0.0)
    map50_m  = next((v for l, v in valeurs if "map@0.5 masque" in l.lower() or "map50 masque" in l.lower()), 0.0)
    f1_m     = next((v for l, v in valeurs if "f1 masque" in l.lower()), 0.0)

    if rappel_m < 0.60 and map50_m > 0.40:
        print()
        print("  ⚠  Rappel faible → fissures manquées probables.")
        print("     → Réduire le seuil : python analyser.py --seuil 0.10")

    print()
    if map50_m >= 0.70 and f1_m >= 0.70:
        print("  ✓✓ EXCELLENT — modèle fiable pour inspection structurelle")
    elif map50_m >= 0.50 and f1_m >= 0.60:
        print("  ✓  BON — résultats exploitables, affinage possible")
    elif map50_m >= 0.35:
        print("  ~  ACCEPTABLE — continuer l'entraînement ou augmenter le dataset")
    else:
        print("  ✗  INSUFFISANT — vérifier données, augmentations, annotations")

    print("═" * 68 + "\n")


def installer_callback_epoque(modele: Any, actif: bool = True) -> None:
    """Affiche la progression époque par époque."""
    if not actif or not hasattr(modele, "add_callback"):
        return

    def journaliser(trainer: Any) -> None:
        epoque  = int(getattr(trainer, "epoch", -1)) + 1
        total   = getattr(getattr(trainer, "args", None), "epochs", "?")
        elems   = [f"[YOLO] Époque {epoque}/{total}"]
        metr    = getattr(trainer, "metrics", None)
        if isinstance(metr, dict):
            for k, v in metr.items():
                km = str(k).lower()
                if "map50" in km or "precision" in km or "recall" in km:
                    try:
                        elems.append(f"{k}={float(v):.4f}")
                    except (TypeError, ValueError):
                        pass
        loss = getattr(trainer, "loss_items", None)
        if loss is not None:
            try:
                elems.append("loss=" + ",".join(f"{float(v):.4f}" for v in loss))
            except (TypeError, ValueError):
                pass
        print(" | ".join(elems), flush=True)

    try:
        modele.add_callback("on_fit_epoch_end", journaliser)
    except Exception as exc:
        print(f"[YOLO] Callback non installé ({type(exc).__name__}). Logs natifs conservés.")


def afficher_commandes_analyse(racine_sorties: Path, nom: str, args: argparse.Namespace) -> None:
    """Affiche les commandes exactes pour analyser les images après l'entraînement."""
    best = str(racine_sorties / "entrainements" / nom / "weights" / "best.pt")
    last = str(racine_sorties / "entrainements" / nom / "weights" / "last.pt")
    test = str(Path(args.donnees) / "test")

    print("\n" + "═" * 68)
    print("  ENTRAÎNEMENT YOLO11-SEG TERMINÉ")
    print("═" * 68)
    print()
    print("  Modèles sauvegardés :")
    print(f"    Meilleur : {best}")
    print(f"    Dernier  : {last}")
    print()
    print("  ┌─── COMMANDES D'ANALYSE ───────────────────────────────────┐")
    print("  │")
    print("  │  Analyser le jeu de test :")
    print(f"  │    python analyser.py \\")
    print(f"  │      --modele   {best} \\")
    print(f"  │      --backend  yolo \\")
    print(f"  │      --images   {test} \\")
    print(f"  │      --seuil    0.25")
    print("  │")
    print("  │  Analyser un dossier quelconque :")
    print(f"  │    python analyser.py \\")
    print(f"  │      --modele   {best} \\")
    print(f"  │      --backend  yolo \\")
    print(f"  │      --images   /chemin/vers/images/ \\")
    print(f"  │      --sortie   resultats_yolo.json")
    print("  │")
    print("  │  Seuil plus bas (éviter les fissures manquées) :")
    print(f"  │    python analyser.py \\")
    print(f"  │      --modele   {best} \\")
    print(f"  │      --backend  yolo \\")
    print(f"  │      --images   {test} \\")
    print(f"  │      --seuil    0.10")
    print("  │")
    print("  │  Reprendre l'entraînement :")
    print(f"  │    python entrainer_yolo.py --resume {last}")
    print("  │")
    print("  └───────────────────────────────────────────────────────────┘")
    print()
    print("  Classification de chaque fissure détectée :")
    print("    Orientation  : horizontale / verticale / inclinée (PCA)")
    print("    Localisation : superficielle / profonde / transversale")
    print("    Danger [0→1] : 0.0 faible | 0.5 modéré | 1.0 critique")
    print("═" * 68 + "\n")


def main() -> None:
    """
    Point d'entrée de l'entraînement YOLO11-seg.

    Étapes :
    1. Conversion automatique COCO → YOLO-seg
    2. Affichage de la commande complète utilisée
    3. Configuration et affichage des paramètres
    4. Entraînement avec transfer learning et augmentations adaptées aux fissures
    5. Évaluation validation (fin d'entraînement) + test
    6. Affichage des métriques interprétées
    7. Affichage des commandes d'analyse
    """
    args = analyser_arguments()

    if args.resume is None:
        verifier_modele_yolov11_seg(args.modele)

    racine_sorties       = Path(args.sorties).expanduser().resolve()
    racine_dataset_yolo  = racine_sorties / "dataset_yolo"

    checkpoint_resume = Path(args.resume).expanduser().resolve() if args.resume else None
    if checkpoint_resume is not None and not checkpoint_resume.is_file():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_resume}\n"
            "Lancez sans --resume pour un entraînement neuf."
        )

    # ── 1. Conversion COCO → YOLO ─────────────────────────────────────────
    print("\n[Conversion] COCO → YOLO-seg en cours...")
    from detection_fissures.donnees.conversion_yolo import convertir_dataset_coco_vers_yolo

    chemin_yaml = convertir_dataset_coco_vers_yolo(
        racine_coco=args.donnees,
        racine_yolo=racine_dataset_yolo,
        copier_images=args.copier_images,
    )
    print(f"[Conversion] Terminée → {chemin_yaml}")
    afficher_resume_dataset_yolo(racine_dataset_yolo)

    if args.convertir_seulement:
        print("[Conversion] Mode --convertir-seulement : entraînement ignoré.")
        return

    # ── 2. Commande complète ───────────────────────────────────────────────
    afficher_commande_complete(args, titre="COMMANDE COMPLÈTE UTILISÉE")

    # ── 3. Import YOLO ────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Le paquet 'ultralytics' est requis.\n"
            "Installez-le avec : pip install -U ultralytics"
        ) from exc

    # ── 4. Configuration affichée ─────────────────────────────────────────
    print("═" * 68)
    print("  CONFIGURATION YOLO11-SEG — FISSURES STRUCTURELLES")
    print("═" * 68)
    print(f"  Modèle de départ    : {args.modele}")
    print(f"  Époques             : {args.epoques}")
    print(f"  Batch size          : {args.lot}")
    print(f"  Résolution          : {args.taille_image}px (identique au dataset)")
    print(f"  LR initial → final  : {args.lr}  →  {args.lr * args.lrf:.2e}")
    print(f"  Weight decay        : {args.weight_decay}")
    print(f"  Warmup              : {args.warmup_epoques} époques")
    print(f"  Close mosaic        : {args.close_mosaic} dernières époques sans mosaic")
    print(f"  Patience ES         : {args.patience}")
    print(f"  mask_ratio          : {args.mask_ratio} (1 = pleine résolution pour fissures fines)")
    print(f"  Overlap mask        : {args.overlap_mask}")
    print(f"  Copy-paste          : {args.copy_paste} (augmentation objet rare)")
    print(f"  Mosaic              : {args.mosaic}")
    print(f"  Mixup               : {args.mixup} (désactivé : brouille contours)")
    print(f"  Rotation            : ±{args.degrees}° (fissures à tous angles)")
    print(f"  Flip horizontal     : {args.fliplr}")
    print(f"  Flip vertical       : {args.flipud} (désactivé : contexte gravitationnel)")
    print(f"  Cosine LR           : {args.cos_lr}")
    print(f"  AMP                 : {args.amp}")
    print(f"  Workers             : {args.workers}")
    if checkpoint_resume is not None:
        print(f"  Reprise checkpoint  : {checkpoint_resume}")
    print("═" * 68 + "\n")

    # ── 5. Entraînement ───────────────────────────────────────────────────
    modele = YOLO(str(checkpoint_resume) if checkpoint_resume is not None else args.modele)
    installer_callback_epoque(modele, actif=not args.silencieux)

    resultats = modele.train(
        data=str(chemin_yaml),
        task="segment",
        epochs=args.epoques,
        imgsz=args.taille_image,
        batch=args.lot,
        lr0=args.lr,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epoques,
        close_mosaic=args.close_mosaic,
        patience=args.patience,
        mask_ratio=args.mask_ratio,
        overlap_mask=args.overlap_mask,
        copy_paste=args.copy_paste,
        mosaic=args.mosaic,
        mixup=args.mixup,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        cos_lr=args.cos_lr,
        freeze=args.freeze if args.freeze > 0 else None,
        amp=args.amp,
        device=_device_ultralytics(args.dispositif),
        workers=args.workers,
        project=str(racine_sorties / "entrainements"),
        name=args.nom,
        exist_ok=args.exist_ok,
        resume=checkpoint_resume is not None,
        verbose=not args.silencieux,
        plots=True,
        val=True,
        save_period=args.save_period,
    )

    print(f"\n[YOLO] Entraînement terminé. Dossier : {resultats.save_dir}")
    afficher_metriques_yolo(resultats, "MÉTRIQUES VALIDATION — FIN D'ENTRAÎNEMENT")

    # ── 6. Évaluation test ────────────────────────────────────────────────
    print("[Évaluation] Évaluation finale sur le jeu de test...")
    metriques_test = modele.val(
        data=str(chemin_yaml),
        split="test",
        imgsz=args.taille_image,
        batch=1,
        device=_device_ultralytics(args.dispositif),
        project=str(racine_sorties / "evaluations"),
        name=f"{args.nom}_test",
        exist_ok=True,
        verbose=not args.silencieux,
        plots=True,
    )
    afficher_metriques_yolo(metriques_test, "MÉTRIQUES TEST — ÉVALUATION FINALE")

    # ── 7. Commande complète + analyse ────────────────────────────────────
    afficher_commande_complete(args, titre="RAPPEL — COMMANDE UTILISÉE POUR CET ENTRAÎNEMENT")
    afficher_commandes_analyse(racine_sorties, args.nom, args)


if __name__ == "__main__":
    main()
