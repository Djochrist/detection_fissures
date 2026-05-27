"""
Entraînement Mask R-CNN — Détection de fissures structurelles.

COMMANDE COMPLÈTE RECOMMANDÉE POUR CE PROJET
═════════════════════════════════════════════════════════════════════

  python entrainer.py \\
    --donnees        dataset/ \\
    --epoques        60 \\
    --lot            2 \\
    --lr             0.0001 \\
    --patience       15 \\
    --taille-image   384 \\
    --seuil-score    0.05 \\
    --architecture   maskrcnn_resnet50_fpn_v2 \\
    --dispositif     auto \\
    --graine         42 \\
    --sorties        sorties \\
    --decroissance-poids 0.0005

  Sur GPU (adapter --lot selon la VRAM disponible) :
    --lot 4    → GPU 8  Go  (ex : RTX 3060)
    --lot 8    → GPU 16 Go  (ex : RTX 3080, A100)
    --dispositif cuda

  Pour reprendre un entraînement interrompu :
    python entrainer.py [mêmes paramètres] \\
      --resume sorties/modeles/dernier_modele.pth

JUSTIFICATION DES PARAMÈTRES CLÉS
════════════════════════════════════
  --epoques 60
      3 phases de transfer learning :
        Phase 1 (ép. 1–5)   : backbone gelé → convergence des têtes
        Phase 2 (ép. 5–15)  : dégelage layer3/layer4 → features fissures
        Phase 3 (ép. 15–60) : fine-tuning complet + early stopping

  --lr 0.0001
      Taux standard pour fine-tuning depuis COCO préentraîné.
      Backbone entraîné à lr/10 = 1e-5 (moins agressif).

  --seuil-score 0.05
      Standard COCO pour l'évaluation mAP (courbe précision-rappel complète).
      Pour l'inférence finale → utiliser 0.30–0.50 dans analyser.py.

  --lot 2
      Mask R-CNN stocke des tenseurs masques [N, H, W].
      À 384×384, lot=2 consomme ~3-4 Go RAM sur CPU.

  --patience 15
      15 époques sans amélioration avant arrêt.
      Valeur élevée : les transitions de phases créent des creux temporaires.

  --decroissance-poids 0.0005
      Régularisation L2 (AdamW) essentielle pour datasets < 5000 images.
"""

import argparse
import json
import sys
from pathlib import Path

RACINE_PROJET = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE_PROJET.parent))

from detection_fissures.configuration.parametres import (
    ARCHITECTURES_MODELES_SEGMENTATION,
    NOM_FICHIER_ANNOTATIONS_COCO,
    NOM_HISTORIQUE_ENTRAINEMENT,
    SPLITS_DATASET,
    ConfigurationGlobale,
)


def analyser_arguments() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    cfg = ConfigurationGlobale()
    p = argparse.ArgumentParser(
        description="Entraînement Mask R-CNN — Détection de fissures structurelles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--donnees", type=str,
        default=str(RACINE_PROJET / cfg.chemins.donnees_racine),
        help="Répertoire racine du dataset (contient train/, valid/, test/)",
    )
    p.add_argument(
        "--epoques", type=int, default=cfg.entrainement.nombre_epoques,
        help="Nombre maximum d'époques (60 = 3 phases complètes pour fissures)",
    )
    p.add_argument(
        "--lot", type=int, default=cfg.entrainement.taille_lot,
        help="Taille du lot. 2=CPU, 4=GPU 8Go, 8=GPU 16Go",
    )
    p.add_argument(
        "--lr", type=float, default=cfg.entrainement.taux_apprentissage,
        help="Taux d'apprentissage initial (1e-4 standard pour fine-tuning COCO)",
    )
    p.add_argument(
        "--patience", type=int, default=cfg.entrainement.patience_arret_precoce,
        help="Patience early stopping (15 = couvre les transitions de phases)",
    )
    p.add_argument(
        "--taille-image", type=int, default=cfg.modele.taille_image_min,
        help="Résolution des images en pixels (doit correspondre à votre dataset)",
    )
    p.add_argument(
        "--seuil-score", type=float, default=cfg.modele.seuil_score_detection,
        help=(
            "Score minimal de détection pour l'évaluation mAP. "
            "0.05 = standard COCO (ne pas modifier sauf expertise)."
        ),
    )
    p.add_argument(
        "--architecture", type=str, default=cfg.modele.architecture,
        choices=ARCHITECTURES_MODELES_SEGMENTATION,
        help="Architecture Mask R-CNN",
    )
    p.add_argument(
        "--dispositif", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Dispositif de calcul (auto = détection automatique GPU/CPU)",
    )
    p.add_argument(
        "--graine", type=int, default=cfg.entrainement.graine_aleatoire,
        help="Graine aléatoire pour la reproductibilité",
    )
    p.add_argument(
        "--sorties", type=str, default="sorties",
        help="Répertoire de sortie pour les modèles et journaux",
    )
    p.add_argument(
        "--decroissance-poids", type=float, default=cfg.entrainement.decroissance_poids,
        help="Décroissance L2 AdamW (5e-4 = anti-overfitting datasets < 5000 images)",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint .pth pour reprendre l'entraînement. Ex : sorties/modeles/dernier_modele.pth",
    )
    p.add_argument(
        "--sans-mixte", action="store_true", default=False,
        help="Désactiver la précision mixte float16 (activée par défaut sur CUDA)",
    )
    return p.parse_args()


def generer_commande_maskrcnn(args: argparse.Namespace) -> str:
    """Génère la commande complète avec tous les paramètres utilisés."""
    lignes = [
        "python entrainer.py \\",
        f"  --donnees           {args.donnees} \\",
        f"  --epoques           {args.epoques} \\",
        f"  --lot               {args.lot} \\",
        f"  --lr                {args.lr} \\",
        f"  --patience          {args.patience} \\",
        f"  --taille-image      {args.taille_image} \\",
        f"  --seuil-score       {args.seuil_score} \\",
        f"  --architecture      {args.architecture} \\",
        f"  --dispositif        {args.dispositif} \\",
        f"  --graine            {args.graine} \\",
        f"  --sorties           {args.sorties} \\",
        f"  --decroissance-poids {args.decroissance_poids}",
    ]
    if args.resume:
        lignes[-1] += " \\"
        lignes.append(f"  --resume            {args.resume}")
    if args.sans_mixte:
        lignes[-1] += " \\"
        lignes.append("  --sans-mixte")
    return "\n".join(lignes)


def afficher_commande_complete(args: argparse.Namespace, titre: str = "COMMANDE UTILISÉE") -> None:
    """Affiche la commande complète avec tous les paramètres."""
    commande = generer_commande_maskrcnn(args)
    print("\n" + "═" * 68)
    print(f"  {titre}")
    print("═" * 68)
    print()
    for ligne in commande.splitlines():
        print(f"  {ligne}")
    print()
    print("═" * 68 + "\n")


def verifier_dataset(chemin_donnees: Path) -> None:
    """
    Vérifie la structure COCO et affiche les statistiques du dataset.

    Accepte les images sans annotations (murs sains = exemples négatifs).
    """
    if not chemin_donnees.is_dir():
        raise FileNotFoundError(
            f"Dataset introuvable : {chemin_donnees}\n"
            "Indiquez un dossier contenant train/, valid/ et test/ avec --donnees."
        )

    erreurs = []
    stats = {}

    for split in SPLITS_DATASET:
        dossier_split = chemin_donnees / split
        chemin_annotations = dossier_split / NOM_FICHIER_ANNOTATIONS_COCO

        if not dossier_split.is_dir():
            erreurs.append(f"  - Dossier manquant : {dossier_split}")
            continue
        if not chemin_annotations.is_file():
            erreurs.append(f"  - Annotations manquantes : {chemin_annotations}")
            continue

        try:
            donnees_coco = json.loads(chemin_annotations.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            erreurs.append(f"  - JSON COCO invalide : {chemin_annotations} ({exc})")
            continue

        images      = donnees_coco.get("images", [])
        annotations = donnees_coco.get("annotations", [])
        categories  = donnees_coco.get("categories", [])

        if not all(isinstance(e, list) for e in (images, annotations, categories)):
            erreurs.append(f"  - Structure COCO invalide dans {chemin_annotations}")
            continue

        images_manquantes = [
            img.get("file_name", "<inconnu>")
            for img in images
            if not (dossier_split / str(img.get("file_name", ""))).is_file()
        ]
        if images_manquantes:
            ex = ", ".join(images_manquantes[:3])
            sfx = f" ... (+{len(images_manquantes) - 3})" if len(images_manquantes) > 3 else ""
            erreurs.append(
                f"  - {len(images_manquantes)} image(s) manquante(s) dans {dossier_split} : {ex}{sfx}"
            )

        ids_annotes = {int(ann.get("image_id")) for ann in annotations}
        nb_avec     = sum(1 for img in images if int(img.get("id", -1)) in ids_annotes)
        stats[split] = {
            "total":        len(images),
            "avec_fissures": nb_avec,
            "sans_fissures": len(images) - nb_avec,
            "instances":    len(annotations),
        }

    if erreurs:
        raise FileNotFoundError("Dataset COCO incomplet :\n" + "\n".join(erreurs))

    print("\n" + "═" * 68)
    print("  STATISTIQUES DU DATASET")
    print("═" * 68)
    print(f"  {'Split':<8} {'Total':>7} {'Fissures':>10} {'Murs sains':>11} {'Instances':>10}")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}")
    for split, s in stats.items():
        print(
            f"  {split:<8} {s['total']:>7} {s['avec_fissures']:>10} "
            f"{s['sans_fissures']:>11} {s['instances']:>10}"
        )
    print("═" * 68)
    print(
        "  Murs sains = images sans annotations → exemples négatifs.\n"
        "  Mask R-CNN les utilise pour réduire les faux positifs."
    )
    print("═" * 68 + "\n")


def afficher_commandes_analyse(dossier_sorties: str, args: argparse.Namespace) -> None:
    """Affiche les commandes exactes pour analyser les images après l'entraînement."""
    chemin_modele = str(Path(dossier_sorties) / "modeles" / "meilleur_modele.pth")
    dossier_test  = str(Path(args.donnees) / "test")

    print("\n" + "═" * 68)
    print("  ENTRAÎNEMENT MASK R-CNN TERMINÉ")
    print("═" * 68)
    print()
    print("  Modèle sauvegardé :")
    print(f"    {chemin_modele}")
    print()
    print("  ┌─── COMMANDES D'ANALYSE ───────────────────────────────────┐")
    print("  │")
    print("  │  Analyser le jeu de test :")
    print(f"  │    python analyser.py \\")
    print(f"  │      --modele   {chemin_modele} \\")
    print(f"  │      --backend  maskrcnn \\")
    print(f"  │      --images   {dossier_test} \\")
    print(f"  │      --seuil    0.40")
    print("  │")
    print("  │  Analyser un dossier quelconque :")
    print(f"  │    python analyser.py \\")
    print(f"  │      --modele   {chemin_modele} \\")
    print(f"  │      --backend  maskrcnn \\")
    print(f"  │      --images   /chemin/vers/images/ \\")
    print(f"  │      --sortie   resultats_maskrcnn.json")
    print("  │")
    print("  │  Reprendre l'entraînement :")
    print(f"  │    python entrainer.py [mêmes paramètres] \\")
    print(f"  │      --resume {str(Path(dossier_sorties) / 'modeles' / 'dernier_modele.pth')}")
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
    Point d'entrée de l'entraînement Mask R-CNN.

    Étapes :
    1. Vérification du dataset (structure COCO + statistiques)
    2. Affichage de la commande complète utilisée
    3. Chargement des données (annotées + non annotées)
    4. Construction du modèle (transfer learning COCO → fissures)
    5. Entraînement 3 phases avec early stopping
    6. Évaluation finale sur le jeu de test
    7. Sauvegarde de l'historique JSON
    8. Affichage des commandes d'analyse
    """
    args = analyser_arguments()

    chemin_donnees = Path(args.donnees).expanduser().resolve()
    verifier_dataset(chemin_donnees)

    # Afficher la commande complète dès le début
    afficher_commande_complete(args, titre="COMMANDE COMPLÈTE UTILISÉE")

    import torch

    from detection_fissures.donnees.chargeur import creer_chargeurs_donnees
    from detection_fissures.entrainement.entraineur import Entraineur
    from detection_fissures.entrainement.metriques import (
        calculer_metriques_segmentation,
        afficher_tableau_metriques,
    )
    from detection_fissures.modeles.masque_rcnn import construire_modele_masque_rcnn
    from detection_fissures.utilitaires.dispositif import (
        detecter_dispositif,
        afficher_info_dispositif,
    )
    from detection_fissures.utilitaires.graine import fixer_graine_aleatoire

    # ── Configuration ───────────────────────────────────────────────────────
    config = ConfigurationGlobale()
    config.entrainement.nombre_epoques          = args.epoques
    config.entrainement.taille_lot              = args.lot
    config.entrainement.taux_apprentissage      = args.lr
    config.entrainement.decroissance_poids      = args.decroissance_poids
    config.entrainement.patience_arret_precoce  = args.patience
    config.entrainement.precision_mixte         = not args.sans_mixte
    config.modele.taille_image_min              = args.taille_image
    config.modele.taille_image_max              = args.taille_image
    config.modele.architecture                  = args.architecture
    config.modele.seuil_score_detection         = args.seuil_score

    config.chemins.definir_racine_donnees(chemin_donnees)
    config.chemins.definir_racine_sorties(args.sorties)
    config.chemins.creer_dossiers()

    # ── Reproductibilité ────────────────────────────────────────────────────
    fixer_graine_aleatoire(args.graine)

    # ── Dispositif ──────────────────────────────────────────────────────────
    dispositif = detecter_dispositif(
        forcer=None if args.dispositif == "auto" else args.dispositif
    )
    afficher_info_dispositif(dispositif)

    # ── Chargement des données ──────────────────────────────────────────────
    chargeurs = creer_chargeurs_donnees(
        chemin_train=config.chemins.chemin_entrainement,
        chemin_valid=config.chemins.chemin_validation,
        chemin_test=config.chemins.chemin_test,
        annotations_train=config.chemins.annotations_entrainement,
        annotations_valid=config.chemins.annotations_validation,
        annotations_test=config.chemins.annotations_test,
        taille_lot=config.entrainement.taille_lot,
        taille_image=config.modele.taille_image_min,
        nombre_workers=config.entrainement.nombre_workers,
        epingler_memoire=(
            config.entrainement.epingler_memoire and dispositif.type == "cuda"
        ),
    )

    # ── Construction du modèle ──────────────────────────────────────────────
    modele = construire_modele_masque_rcnn(
        nombre_classes=config.modele.nombre_classes,
        architecture=config.modele.architecture,
        seuil_score_detection=config.modele.seuil_score_detection,
        seuil_iou_nms=config.modele.seuil_iou_nms,
        detections_max_par_image=config.modele.detections_max_par_image,
        taille_image_min=config.modele.taille_image_min,
        taille_image_max=config.modele.taille_image_max,
    )

    # ── Entraîneur ──────────────────────────────────────────────────────────
    entraineur = Entraineur(
        modele=modele,
        chargeur_train=chargeurs["entrainement"],
        chargeur_valid=chargeurs["validation"],
        dispositif=dispositif,
        taux_apprentissage=config.entrainement.taux_apprentissage,
        decroissance_poids=config.entrainement.decroissance_poids,
        nombre_epoques=config.entrainement.nombre_epoques,
        epoque_degelage_backbone=config.entrainement.epoque_degelage_backbone,
        epoque_degelage_complet=config.entrainement.epoque_degelage_complet,
        patience_arret_precoce=config.entrainement.patience_arret_precoce,
        valeur_clip_gradient=config.entrainement.valeur_clip_gradient,
        dossier_sorties=config.chemins.dossier_modeles,
        precision_mixte=config.entrainement.precision_mixte,
        frequence_affichage=config.entrainement.frequence_affichage,
    )

    # ── Reprise d'un checkpoint ─────────────────────────────────────────────
    if args.resume:
        checkpoint_resume = Path(args.resume).expanduser().resolve()
        if not checkpoint_resume.is_file():
            raise FileNotFoundError(
                f"Checkpoint introuvable : {checkpoint_resume}\n"
                "Relancez sans --resume pour un entraînement neuf."
            )
        print(f"\n[Reprise] Chargement du checkpoint : {checkpoint_resume}")
        checkpoint = torch.load(
            checkpoint_resume, map_location=dispositif, weights_only=False
        )
        entraineur.reprendre_checkpoint(checkpoint)

    # ── Lancement de l'entraînement ─────────────────────────────────────────
    try:
        historique = entraineur.entrainer()
    except KeyboardInterrupt:
        print(
            "\n[Interruption] Entraînement arrêté par l'utilisateur.\n"
            "  Dernier checkpoint sauvegardé.\n"
            "  Pour reprendre :\n"
            f"    python entrainer.py --donnees {args.donnees} "
            f"--resume {args.sorties}/modeles/dernier_modele.pth"
        )
        return

    # ── Évaluation finale sur le jeu de test ───────────────────────────────
    print("\n[Évaluation] Chargement du meilleur modèle pour évaluation finale...")
    entraineur.charger_meilleur_modele()
    modele.eval()

    toutes_predictions = []
    toutes_cibles      = []

    with torch.no_grad():
        for images, cibles in chargeurs["test"]:
            images_gpu  = [img.to(dispositif) for img in images]
            predictions = modele(images_gpu)
            toutes_predictions.extend(
                [{k: v.cpu() for k, v in p.items()} for p in predictions]
            )
            toutes_cibles.extend(cibles)

    metriques_test = calculer_metriques_segmentation(
        toutes_predictions, toutes_cibles, journaliser=False
    )
    print("\n[RÉSULTATS FINAUX — JEU DE TEST]")
    afficher_tableau_metriques(metriques_test)

    # ── Sauvegarde de l'historique ──────────────────────────────────────────
    chemin_historique = config.chemins.dossier_journaux / NOM_HISTORIQUE_ENTRAINEMENT
    with open(chemin_historique, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hyperparametres": {
                    "epoques":           args.epoques,
                    "taille_lot":        args.lot,
                    "taux_apprentissage": args.lr,
                    "decroissance_poids": args.decroissance_poids,
                    "patience":          args.patience,
                    "taille_image":      args.taille_image,
                    "seuil_score":       args.seuil_score,
                    "architecture":      args.architecture,
                    "graine":            args.graine,
                    "commande_complete": generer_commande_maskrcnn(args),
                },
                "metriques_test": metriques_test,
                "historique":     historique,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"[Historique] Sauvegardé : {chemin_historique}")

    # ── Commande complète + commandes d'analyse ─────────────────────────────
    afficher_commande_complete(args, titre="RAPPEL — COMMANDE UTILISÉE POUR CET ENTRAÎNEMENT")
    afficher_commandes_analyse(args.sorties, args)


if __name__ == "__main__":
    main()
