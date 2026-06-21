"""
Script d'analyse et classification des fissures après entraînement.

Deux backends sont supportés :

  - YOLO-seg (--backend yolo)      : charge un fichier .pt (YOLO11/YOLO26-seg).
  - Mask R-CNN (--backend maskrcnn): charge un fichier .pth torchvision.

Pour chaque image, le script :
  1. détecte les fissures (segmentation) ;
  2. classe chaque fissure (orientation PCA + localisation + indice de danger) ;
  3. produit une image annotée (sauf si --sans-images) ;
  4. écrit un rapport JSON.

Usage :
    python analyser.py --backend yolo \
        --modele sorties_yolo/entrainements/yolo11m_fissures/weights/best.pt \
        --images dataset/images/test/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from detection_fissures.analyse.classificateur_fissures import (
    ResultatClassification,
    afficher_resultats_classification,
    classifier_fissure,
)
from detection_fissures.configuration.parametres import (
    ARCHITECTURES_MODELES_SEGMENTATION,
    ConfigurationGlobale,
)

EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

SEUIL_DEFAUT_YOLO = 0.25
SEUIL_DEFAUT_MASKRCNN = 0.40


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENTS
# ──────────────────────────────────────────────────────────────────────────────

def analyser_arguments() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    configuration = ConfigurationGlobale()
    analyseur = argparse.ArgumentParser(
        description="Analyse et classification des fissures (YOLO-seg ou Mask R-CNN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyseur.add_argument(
        "--modele",
        type=str,
        required=True,
        help="Chemin vers le modèle entraîné : .pt (YOLO-seg) ou .pth (Mask R-CNN)",
    )
    analyseur.add_argument(
        "--backend",
        type=str,
        default="yolo",
        choices=["yolo", "maskrcnn"],
        help="Type de modèle : 'yolo' pour un .pt YOLO-seg, 'maskrcnn' pour un .pth",
    )
    analyseur.add_argument(
        "--images",
        type=str,
        required=True,
        help="Dossier contenant les images à analyser",
    )
    analyseur.add_argument(
        "--taille-image",
        type=int,
        default=640,
        help="Résolution d'inférence (doit correspondre à l'entraînement, multiple de 32 pour YOLO)",
    )
    analyseur.add_argument(
        "--seuil",
        type=float,
        default=None,
        help="Score de confiance minimal [0, 1]. Défaut : 0.25 (YOLO) / 0.40 (Mask R-CNN)",
    )
    analyseur.add_argument(
        "--dispositif",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Dispositif de calcul",
    )
    analyseur.add_argument(
        "--dossier-sortie",
        type=str,
        default="analyses",
        help="Dossier racine où écrire le rapport JSON et les images annotées",
    )
    analyseur.add_argument(
        "--sortie",
        type=str,
        default="",
        help="Chemin explicite du JSON (sinon : <dossier-sortie>/<backend>/rapport_analyse.json)",
    )
    analyseur.add_argument(
        "--sans-images",
        action="store_true",
        help="Ne produit que le JSON, sans copies annotées",
    )
    analyseur.add_argument(
        "--architecture",
        type=str,
        default=configuration.modele.architecture,
        choices=ARCHITECTURES_MODELES_SEGMENTATION,
        help="(Mask R-CNN) Architecture utilisée pour reconstruire le modèle",
    )
    analyseur.add_argument(
        "--nombre-classes",
        type=int,
        default=configuration.modele.nombre_classes,
        help="(Mask R-CNN) Nombre de classes du modèle (doit correspondre à l'entraînement)",
    )
    return analyseur.parse_args()


def _device_ultralytics(dispositif: str) -> str | None:
    """Traduit le nom de dispositif du projet vers Ultralytics."""
    if dispositif == "auto":
        return None
    if dispositif == "cuda":
        return "0"
    return dispositif


# ──────────────────────────────────────────────────────────────────────────────
# INFÉRENCE YOLO-SEG
# ──────────────────────────────────────────────────────────────────────────────

def charger_modele_yolo(chemin_pt: str | Path) -> Any:
    """Charge un modèle YOLO-seg depuis un fichier .pt."""
    chemin_pt = Path(chemin_pt)
    if not chemin_pt.exists():
        raise FileNotFoundError(f"Modèle YOLO introuvable : {chemin_pt}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Le paquet 'ultralytics' est requis pour --backend yolo. "
            "Installez-le avec : pip install -U ultralytics"
        ) from exc
    modele = YOLO(str(chemin_pt))
    print(f"[Modele] YOLO-seg chargé : {chemin_pt.name}")
    return modele


def inferer_yolo(
    modele: Any,
    image_bgr: np.ndarray,
    taille_image: int,
    seuil: float,
    dispositif: str,
) -> Tuple[List[np.ndarray], List[float], List[Tuple[int, int, int, int]]]:
    """Exécute l'inférence YOLO-seg sur une image carrée déjà redimensionnée.

    Returns:
        (masques, scores, boites) — masques binaires [taille, taille],
        scores [0,1], boîtes (x1, y1, x2, y2) en coordonnées de l'image carrée.
    """
    import cv2

    resultats = modele.predict(
        source=image_bgr,
        imgsz=taille_image,
        conf=seuil,
        device=_device_ultralytics(dispositif),
        verbose=False,
        retina_masks=True,
    )

    if not resultats:
        return [], [], []

    r = resultats[0]
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return [], [], []

    masques_brut = r.masks.data.cpu().numpy()  # [N, h, w]
    scores = r.boxes.conf.cpu().numpy().tolist()
    boites_xyxy = r.boxes.xyxy.cpu().numpy()

    masques: List[np.ndarray] = []
    boites: List[Tuple[int, int, int, int]] = []
    for i in range(masques_brut.shape[0]):
        masque = masques_brut[i]
        if masque.shape != (taille_image, taille_image):
            masque = cv2.resize(
                masque.astype(np.float32),
                (taille_image, taille_image),
                interpolation=cv2.INTER_NEAREST,
            )
        masques.append((masque > 0.5).astype(np.uint8))
        x1, y1, x2, y2 = boites_xyxy[i]
        boites.append((int(x1), int(y1), int(x2), int(y2)))

    return masques, scores, boites


# ──────────────────────────────────────────────────────────────────────────────
# INFÉRENCE MASK R-CNN
# ──────────────────────────────────────────────────────────────────────────────

def charger_modele_maskrcnn(
    chemin_pth: str | Path,
    nombre_classes: int,
    taille_image: int,
    architecture: str,
    dispositif: object,
) -> object:
    """Charge le modèle Mask R-CNN depuis un checkpoint .pth."""
    chemin_pth = Path(chemin_pth)
    if not chemin_pth.exists():
        raise FileNotFoundError(f"Modèle introuvable : {chemin_pth}")

    import torch

    from detection_fissures.modeles.masque_rcnn import construire_modele_masque_rcnn

    checkpoint = torch.load(chemin_pth, map_location=dispositif, weights_only=False)
    architecture_checkpoint = (
        checkpoint.get("architecture_modele", architecture)
        if isinstance(checkpoint, dict)
        else architecture
    ) or architecture

    modele = construire_modele_masque_rcnn(
        nombre_classes=nombre_classes,
        architecture=architecture_checkpoint,
        taille_image_min=taille_image,
        taille_image_max=taille_image,
    )

    if isinstance(checkpoint, dict) and "etat_modele" in checkpoint:
        modele.load_state_dict(checkpoint["etat_modele"])
        epoque = checkpoint.get("epoque", "?")
        map_50 = checkpoint.get("metriques", {}).get("map_50", 0.0)
        print(f"[Modele] Mask R-CNN chargé — époque {epoque}, mAP@0.5 = {map_50:.4f}")
    else:
        modele.load_state_dict(checkpoint)
        print("[Modele] Poids Mask R-CNN chargés directement.")

    modele.to(dispositif)
    modele.eval()
    return modele


def inferer_maskrcnn(
    modele: object,
    image_rgb: np.ndarray,
    taille_image: int,
    seuil: float,
    dispositif: object,
) -> Tuple[List[np.ndarray], List[float], List[Tuple[int, int, int, int]]]:
    """Exécute l'inférence Mask R-CNN sur une image carrée déjà redimensionnée."""
    import torch

    from detection_fissures.utilitaires.images import image_rgb_vers_tenseur

    tenseur = image_rgb_vers_tenseur(image_rgb).unsqueeze(0).to(dispositif)

    with torch.no_grad():
        predictions = modele(tenseur)

    if not predictions:
        return [], [], []

    pred = predictions[0]
    scores_t = pred["scores"]
    garder = scores_t >= seuil

    masques_t = pred["masks"][garder]      # [N, 1, H, W]
    scores = scores_t[garder].cpu().numpy().tolist()
    boites_t = pred["boxes"][garder].cpu().numpy()

    masques: List[np.ndarray] = []
    boites: List[Tuple[int, int, int, int]] = []
    masques_np = masques_t.squeeze(1).cpu().numpy()  # [N, H, W]
    for i in range(masques_np.shape[0]):
        masques.append((masques_np[i] > 0.5).astype(np.uint8))
        x1, y1, x2, y2 = boites_t[i]
        boites.append((int(x1), int(y1), int(x2), int(y2)))

    return masques, scores, boites


# ──────────────────────────────────────────────────────────────────────────────
# ANNOTATION DES IMAGES
# ──────────────────────────────────────────────────────────────────────────────

def _couleur_danger(indice_danger: float) -> Tuple[int, int, int]:
    """Retourne une couleur BGR selon l'indice de danger."""
    if indice_danger > 0.65:
        return (0, 0, 255)      # rouge
    if indice_danger > 0.35:
        return (0, 165, 255)    # orange
    return (0, 200, 0)          # vert


def dessiner_annotations(
    image_rgb: np.ndarray,
    masques: List[np.ndarray],
    boites: List[Tuple[int, int, int, int]],
    resultats: List[ResultatClassification],
) -> np.ndarray:
    """Dessine contours, boîtes et labels sur une copie BGR de l'image."""
    import cv2

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    for i, (masque, boite, r) in enumerate(zip(masques, boites, resultats), start=1):
        couleur = _couleur_danger(r.indice_danger)

        # Contour du masque
        contours, _ = cv2.findContours(
            masque.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image_bgr, contours, -1, couleur, 2)

        # Boîte englobante
        x1, y1, x2, y2 = boite
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), couleur, 1)

        # Label : #i orientation/localisation danger
        label = (
            f"#{i} {r.orientation.value[:4]}/{r.localisation.value[:5]} "
            f"d={r.indice_danger:.2f}"
        )
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_texte = max(y1 - 5, lh + 2)
        cv2.rectangle(
            image_bgr,
            (x1, y_texte - lh - 3),
            (x1 + lw + 2, y_texte + 2),
            couleur,
            -1,
        )
        cv2.putText(
            image_bgr,
            label,
            (x1 + 1, y_texte - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image_bgr


# ──────────────────────────────────────────────────────────────────────────────
# RAPPORT
# ──────────────────────────────────────────────────────────────────────────────

def construire_entree_rapport(
    nom_image: str,
    chemin_source: str,
    chemin_annotee: str,
    resultats: List[ResultatClassification],
) -> dict:
    """Construit l'entrée JSON pour une image analysée."""
    fissures = [
        {
            "id": i + 1,
            "orientation": r.orientation.value,
            "localisation": r.localisation.value,
            "angle_degres": round(r.angle_degres, 2),
            "largeur_moy_pixels": round(r.largeur_moy_pixels, 2),
            "ratio_couverture": round(r.ratio_couverture, 3),
            "aire_pixels": r.aire_pixels,
            "indice_danger": round(r.indice_danger, 3),
            "score_detection": round(r.score_detection, 3),
        }
        for i, r in enumerate(resultats)
    ]
    return {
        "image": nom_image,
        "chemin_image_source": chemin_source,
        "chemin_image_annotee": chemin_annotee,
        "fissures": fissures,
    }


# ──────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal du script d'analyse."""
    args = analyser_arguments()

    import cv2

    from detection_fissures.utilitaires.dispositif import detecter_dispositif
    from detection_fissures.utilitaires.images import (
        charger_image_rgb,
        redimensionner_image_carree,
    )

    seuil = args.seuil
    if seuil is None:
        seuil = SEUIL_DEFAUT_YOLO if args.backend == "yolo" else SEUIL_DEFAUT_MASKRCNN

    dispositif = detecter_dispositif(
        forcer=None if args.dispositif == "auto" else args.dispositif
    )
    print(f"[Dispositif] {dispositif}")
    print(f"[Backend] {args.backend} | seuil = {seuil} | taille = {args.taille_image}px")

    # ── Charger le modèle selon le backend ────────────────────────────────────
    if args.backend == "yolo":
        modele = charger_modele_yolo(args.modele)
    else:
        modele = charger_modele_maskrcnn(
            chemin_pth=args.modele,
            nombre_classes=args.nombre_classes,
            taille_image=args.taille_image,
            architecture=args.architecture,
            dispositif=dispositif,
        )

    # ── Lister les images ─────────────────────────────────────────────────────
    dossier_images = Path(args.images)
    if not dossier_images.exists():
        print(f"[Erreur] Dossier introuvable : {dossier_images}")
        return

    chemins_images = sorted(
        p for p in dossier_images.iterdir()
        if p.suffix.lower() in EXTENSIONS_IMAGES
    )
    if not chemins_images:
        print(f"[Erreur] Aucune image trouvée dans : {dossier_images}")
        return

    print(f"\n[Analyse] {len(chemins_images)} images à traiter...")

    # ── Préparer les dossiers de sortie ───────────────────────────────────────
    racine_sortie = Path(args.dossier_sortie) / args.backend
    dossier_annotees = racine_sortie / "images_annotees"
    if not args.sans_images:
        dossier_annotees.mkdir(parents=True, exist_ok=True)

    chemin_json = (
        Path(args.sortie) if args.sortie
        else racine_sortie / "rapport_analyse.json"
    )

    rapport_global: List[dict] = []

    for chemin in chemins_images:
        try:
            image_rgb, _, _ = charger_image_rgb(chemin)
            image_carree = redimensionner_image_carree(image_rgb, args.taille_image)

            if args.backend == "yolo":
                image_bgr = cv2.cvtColor(image_carree, cv2.COLOR_RGB2BGR)
                masques, scores, boites = inferer_yolo(
                    modele, image_bgr, args.taille_image, seuil, args.dispositif
                )
            else:
                masques, scores, boites = inferer_maskrcnn(
                    modele, image_carree, args.taille_image, seuil, dispositif
                )

            resultats = [
                classifier_fissure(
                    masque=masque,
                    largeur_image=args.taille_image,
                    hauteur_image=args.taille_image,
                    score_detection=score,
                )
                for masque, score in zip(masques, scores)
            ]

            afficher_resultats_classification(resultats, nom_image=chemin.name)

            chemin_annotee = ""
            if not args.sans_images and masques:
                image_annotee = dessiner_annotations(
                    image_carree, masques, boites, resultats
                )
                chemin_annotee_path = dossier_annotees / f"{chemin.stem}_analyse.jpg"
                cv2.imwrite(str(chemin_annotee_path), image_annotee)
                chemin_annotee = str(chemin_annotee_path)

            rapport_global.append(
                construire_entree_rapport(
                    nom_image=chemin.name,
                    chemin_source=str(chemin),
                    chemin_annotee=chemin_annotee,
                    resultats=resultats,
                )
            )

        except Exception as e:
            print(f"[Erreur] {chemin.name} : {e}")
            continue

    # ── Écrire le rapport JSON ────────────────────────────────────────────────
    chemin_json.parent.mkdir(parents=True, exist_ok=True)
    total_fissures = sum(len(r["fissures"]) for r in rapport_global)
    dangers = [
        f["indice_danger"]
        for r in rapport_global
        for f in r["fissures"]
    ]
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "parametres": {
                    "modele": args.modele,
                    "backend": args.backend,
                    "seuil": seuil,
                    "taille_image": args.taille_image,
                    "dossier_images_annotees": (
                        "" if args.sans_images else str(dossier_annotees)
                    ),
                },
                "resultats": rapport_global,
                "resume": {
                    "images_analysees": len(rapport_global),
                    "total_fissures": total_fissures,
                    "danger_moyen_global": (
                        round(sum(dangers) / len(dangers), 3) if dangers else 0.0
                    ),
                    "danger_maximum_global": (
                        round(max(dangers), 3) if dangers else 0.0
                    ),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[Rapport] JSON sauvegardé : {chemin_json}")
    if not args.sans_images:
        print(f"[Rapport] Images annotées : {dossier_annotees}")
    print("\nAnalyse terminée.")


if __name__ == "__main__":
    main()
