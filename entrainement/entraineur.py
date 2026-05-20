"""
Module de la boucle d'entraînement principale.

STRATÉGIE D'ENTRAÎNEMENT EN 3 PHASES
═══════════════════════════════════════

Phase 1 — Échauffement (époques 1 à epoque_degelage_backbone) :
    → Backbone GELÉ (poids ImageNet conservés)
    → Seules les têtes de détection et masque sont entraînées
    → LR standard (1e-4), convergence rapide
    → Objectif : stabiliser les têtes avant le fine-tuning

Phase 2 — Dégelage partiel (époques degelage_backbone à degelage_complet) :
    → Couches layer3/layer4 dégelées
    → LR backbone = LR_têtes / 10 (évite destruction des features)
    → Adapter les features aux patterns de fissures

Phase 3 — Fine-tuning complet (époques degelage_complet à fin) :
    → Tout le backbone entraînable
    → LR backbone faible conservé
    → CosineAnnealingLR pour une décroissance douce

ANTI-OVERFITTING POUR PETIT DATASET :
═══════════════════════════════════════
1. Transfer learning progressif (Phases 1-2-3)
2. Weight decay L2 (5e-4)
3. Gradient clipping (max_norm=1.0)
4. Early stopping (patience=10 époques)
5. Sauvegarde du meilleur modèle (validation mAP@0.5)
6. Précision mixte (float16) pour entraînement plus rapide + régularisation légère
"""

import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from rich.console import Console

from ..configuration.parametres import NOM_DERNIER_MODELE, NOM_MEILLEUR_MODELE
from ..modeles.masque_rcnn import (
    geler_backbone,
    degeler_couches_superieures,
    degeler_backbone_complet,
    afficher_resume_modele,
)
from .pertes import PerteCombineeMaskRCNN
from .metriques import calculer_metriques_segmentation, afficher_tableau_metriques


console = Console()


class Entraineur:
    """
    Gère l'entraînement complet du modèle Mask R-CNN.

    Responsabilités :
        - Boucle d'entraînement (forward + backward + optimisation)
        - Évaluation sur la validation à chaque époque
        - Gestion des phases de dégelage du backbone
        - Early stopping basé sur mAP@0.5 de validation
        - Sauvegarde des checkpoints et du meilleur modèle
        - Journalisation des métriques

    Args:
        modele : Modèle Mask R-CNN PyTorch.
        chargeur_train : DataLoader d'entraînement.
        chargeur_valid : DataLoader de validation.
        dispositif : 'cuda', 'mps' ou 'cpu'.
        taux_apprentissage : LR initial pour les têtes.
        decroissance_poids : Coefficient L2 (weight decay).
        nombre_epoques : Nombre maximum d'époques.
        epoque_degelage_backbone : Époque de dégelage partiel.
        epoque_degelage_complet : Époque de dégelage total.
        patience_arret_precoce : Époques sans amélioration avant arrêt.
        valeur_clip_gradient : Valeur max pour gradient clipping.
        dossier_sorties : Répertoire de sauvegarde des modèles.
        precision_mixte : Utiliser float16 (True si GPU récent).
        frequence_affichage : Nombre de steps entre affichages.
    """

    def __init__(
        self,
        modele: torch.nn.Module,
        chargeur_train: DataLoader,
        chargeur_valid: DataLoader,
        dispositif: torch.device,
        taux_apprentissage: float = 1e-4,
        decroissance_poids: float = 5e-4,
        nombre_epoques: int = 50,
        epoque_degelage_backbone: int = 5,
        epoque_degelage_complet: int = 15,
        patience_arret_precoce: int = 10,
        valeur_clip_gradient: float = 1.0,
        dossier_sorties: str | Path = "sorties/modeles",
        precision_mixte: bool = True,
        frequence_affichage: int = 10,
    ) -> None:
        self.modele = modele.to(dispositif)
        self.chargeur_train = chargeur_train
        self.chargeur_valid = chargeur_valid
        self.dispositif = dispositif
        self.taux_apprentissage = taux_apprentissage
        self.decroissance_poids = decroissance_poids
        self.nombre_epoques = nombre_epoques
        self.epoque_degelage_backbone = epoque_degelage_backbone
        self.epoque_degelage_complet = epoque_degelage_complet
        self.patience_arret_precoce = patience_arret_precoce
        self.valeur_clip_gradient = valeur_clip_gradient
        self.dossier_sorties = Path(dossier_sorties)
        self.dossier_sorties.mkdir(parents=True, exist_ok=True)
        self.frequence_affichage = frequence_affichage

        # Précision mixte (float16) — uniquement sur CUDA
        self.precision_mixte = precision_mixte and (dispositif.type == "cuda")
        self.gradscaler = GradScaler("cuda") if self.precision_mixte else None

        # Fonction de perte avec poids masque × 2 pour priorité segmentation
        self.perte_combinee = PerteCombineeMaskRCNN(poids_masque=2.0)

        # Historique des métriques pour journalisation
        self.historique: Dict[str, List[float]] = {
            "perte_train": [],
            "map_50_valid": [],
            "map_valid": [],
        }

        # Variables pour early stopping
        self.meilleure_map_50: float = 0.0
        self.epoques_sans_amelioration: int = 0
        self.epoque_meilleur_modele: int = 0
        self.epoque_depart: int = 1

        # Suivi des fichiers de checkpoint créés pendant l'exécution
        self._fichiers_checkpoint_crees: set[Path] = set()

        # Phase 1 : Geler le backbone au démarrage
        geler_backbone(self.modele)
        afficher_resume_modele(self.modele)

        # Optimiseur (configuré après gélification)
        self.optimiseur = self._creer_optimiseur()

        # Scheduler cosinus
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimiseur,
            T_max=nombre_epoques,
            eta_min=1e-6,
        )

    def reprendre_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        """Charge un checkpoint existant pour reprendre l'entraînement."""
        self.modele.load_state_dict(checkpoint["etat_modele"])
        self.optimiseur.load_state_dict(checkpoint["etat_optimiseur"])
        self.scheduler.load_state_dict(checkpoint["etat_scheduler"])
        self.historique = checkpoint.get("historique", self.historique)
        self.meilleure_map_50 = checkpoint.get("meilleure_map_50", self.meilleure_map_50)
        self.epoque_meilleur_modele = int(checkpoint.get("epoque", self.epoque_meilleur_modele))
        self.epoque_depart = int(checkpoint.get("epoque", 0)) + 1
        self.epoques_sans_amelioration = 0
        console.print(
            f"[green]Checkpoint chargé : époque {self.epoque_depart - 1}, "
            f"mAP@0.5 = {checkpoint.get('metriques', {}).get('map_50', 0.0):.4f}[/green]"
        )

    def _creer_optimiseur(self) -> optim.Optimizer:
        """
        Crée l'optimiseur AdamW avec gestion différenciée des LR.

        AdamW vs SGD :
            AdamW (Adam + Weight Decay décorrélé) converge plus vite que SGD
            pour le fine-tuning de modèles préentraînés.
            SGD avec momentum peut atteindre de meilleures performances
            finales mais nécessite un tuning plus minutieux du LR.
            Pour notre use case (petit dataset, convergence rapide souhaitée),
            AdamW est le meilleur choix.

        Returns:
            Optimiseur AdamW configuré.
        """
        # Paramètres des têtes (toujours entraînables)
        groupes_parametres = [
            {
                "params": [
                    p for n, p in self.modele.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": self.taux_apprentissage,
                "weight_decay": self.decroissance_poids,
                "name": "tetes",
            },
        ]

        # Paramètres du backbone (si dégelés)
        params_backbone = [
            p for n, p in self.modele.named_parameters()
            if "backbone" in n and p.requires_grad
        ]
        if params_backbone:
            groupes_parametres.append({
                "params": params_backbone,
                "lr": self.taux_apprentissage / 10.0,  # LR 10× plus bas
                "weight_decay": self.decroissance_poids,
                "name": "backbone",
            })

        return optim.AdamW(groupes_parametres)

    def _gerer_phase_degelage(self, epoque: int) -> None:
        """
        Gère la transition entre les phases de dégelage.

        Reconfigure également l'optimiseur après chaque dégelage
        pour ajouter les nouveaux paramètres entraînables.

        Args:
            epoque : Époque courante (1-indexée).
        """
        if epoque == self.epoque_degelage_backbone:
            console.print(f"\n[bold yellow]═══ PHASE 2 : Dégelage couches supérieures ═══[/bold yellow]")
            degeler_couches_superieures(self.modele)
            self.optimiseur = self._creer_optimiseur()
            # Réinitialiser le scheduler pour les nouvelles phases
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimiseur,
                T_max=self.nombre_epoques - epoque,
                eta_min=1e-6,
            )

        elif epoque == self.epoque_degelage_complet:
            console.print(f"\n[bold green]═══ PHASE 3 : Fine-tuning complet ═══[/bold green]")
            degeler_backbone_complet(self.modele)
            self.optimiseur = self._creer_optimiseur()
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimiseur,
                T_max=self.nombre_epoques - epoque,
                eta_min=1e-6,
            )

    def _etape_entrainement(self, images: List, cibles: List) -> float:
        """
        Effectue une étape d'entraînement (forward + backward + update).

        Args:
            images : Liste de tenseurs image.
            cibles : Liste de dictionnaires de cibles.

        Returns:
            Valeur scalaire de la perte pour cette étape.
        """
        # Transfert vers le dispositif
        images = [img.to(self.dispositif) for img in images]
        cibles = [{k: v.to(self.dispositif) for k, v in c.items()} for c in cibles]

        self.optimiseur.zero_grad()

        if self.precision_mixte:
            # Précision mixte : calcul en float16, accumulation en float32
            with autocast("cuda"):
                dictionnaire_pertes = self.modele(images, cibles)
                perte = self.perte_combinee(dictionnaire_pertes)

            # Mise à l'échelle du gradient pour float16
            self.gradscaler.scale(perte).backward()
            self.gradscaler.unscale_(self.optimiseur)
            torch.nn.utils.clip_grad_norm_(
                self.modele.parameters(),
                self.valeur_clip_gradient,
            )
            self.gradscaler.step(self.optimiseur)
            self.gradscaler.update()

        else:
            # Mode précision normale (CPU ou GPU ancien)
            dictionnaire_pertes = self.modele(images, cibles)
            perte = self.perte_combinee(dictionnaire_pertes)
            perte.backward()
            torch.nn.utils.clip_grad_norm_(
                self.modele.parameters(),
                self.valeur_clip_gradient,
            )
            self.optimiseur.step()

        return float(perte.item())

    @torch.no_grad()
    def _evaluer(self) -> Dict[str, float]:
        """
        Évalue le modèle sur le jeu de validation.

        Mask R-CNN en mode eval() retourne directement les prédictions
        (pas les pertes) → on calcule le mAP via torchmetrics.

        Returns:
            Dictionnaire de métriques de validation.
        """
        self.modele.eval()
        toutes_predictions = []
        toutes_cibles = []

        for images, cibles in self.chargeur_valid:
            images = [img.to(self.dispositif) for img in images]
            predictions = self.modele(images)
            toutes_predictions.extend([
                {k: v.cpu() for k, v in pred.items()}
                for pred in predictions
            ])
            toutes_cibles.extend(cibles)

        metriques = calculer_metriques_segmentation(
            toutes_predictions,
            toutes_cibles,
        )

        self.modele.train()
        return metriques

    def _sauvegarder_checkpoint(
        self,
        epoque: int,
        metriques: Dict[str, float],
        est_meilleur: bool = False,
    ) -> None:
        """
        Sauvegarde un checkpoint du modèle.

        Args:
            epoque : Numéro d'époque.
            metriques : Métriques de validation courantes.
            est_meilleur : Si True, sauvegarde aussi comme meilleur modèle.
        """
        etat = {
            "epoque": epoque,
            "architecture_modele": getattr(self.modele, "nom_architecture_detection", None),
            "etat_modele": self.modele.state_dict(),
            "etat_optimiseur": self.optimiseur.state_dict(),
            "etat_scheduler": self.scheduler.state_dict(),
            "metriques": metriques,
            "meilleure_map_50": self.meilleure_map_50,
            "historique": self.historique,
        }

        # Sauvegarde du dernier checkpoint
        chemin_dernier = self.dossier_sorties / NOM_DERNIER_MODELE
        torch.save(etat, chemin_dernier)
        self._fichiers_checkpoint_crees.add(chemin_dernier)

        # Sauvegarde du meilleur modèle si amélioré
        if est_meilleur:
            chemin_meilleur = self.dossier_sorties / NOM_MEILLEUR_MODELE
            torch.save(etat, chemin_meilleur)
            self._fichiers_checkpoint_crees.add(chemin_meilleur)
            console.print(
                f"  [bold green]✓ Meilleur modèle sauvegardé "
                f"(mAP@0.5 = {metriques['map_50']:.4f})[/bold green]"
            )

    def entrainer(self) -> Dict[str, List[float]]:
        """
        Lance l'entraînement complet du modèle.

        Returns:
            Dictionnaire avec l'historique des métriques par époque.
        """
        console.print(f"\n[bold cyan]{'═'*60}[/bold cyan]")
        console.print(f"[bold cyan]  DÉBUT DE L'ENTRAÎNEMENT[/bold cyan]")
        console.print(f"[bold cyan]  Dispositif : {self.dispositif}[/bold cyan]")
        console.print(f"[bold cyan]  Précision mixte : {self.precision_mixte}[/bold cyan]")
        console.print(f"[bold cyan]  Époques : {self.nombre_epoques}[/bold cyan]")
        console.print(f"[bold cyan]{'═'*60}[/bold cyan]\n")

        debut_global = time.time()

        if self.epoque_depart > self.nombre_epoques:
            console.print(
                f"[yellow]Aucune époque à entraîner : checkpoint déjà à l'époque "
                f"{self.epoque_depart - 1} et nombre d'époques demandé = {self.nombre_epoques}[/yellow]"
            )
            return self.historique

        for epoque in range(self.epoque_depart, self.nombre_epoques + 1):
            debut_epoque = time.time()

            # ── Gestion des phases de dégelage ────────────────────────────────
            self._gerer_phase_degelage(epoque)

            # ── Phase d'entraînement ──────────────────────────────────────────
            self.modele.train()
            pertes_epoque = []
            nombre_lots = len(self.chargeur_train)

            for idx_lot, (images, cibles) in enumerate(self.chargeur_train):
                perte_lot = self._etape_entrainement(images, cibles)
                pertes_epoque.append(perte_lot)

                # Affichage périodique dans la même époque
                if (idx_lot + 1) % self.frequence_affichage == 0:
                    perte_moy = sum(pertes_epoque[-self.frequence_affichage:]) / self.frequence_affichage
                    console.print(
                        f"  Époque {epoque:3d}/{self.nombre_epoques} | "
                        f"Lot {idx_lot+1:4d}/{nombre_lots} | "
                        f"Perte = {perte_moy:.4f}"
                    )

            perte_train_moy = sum(pertes_epoque) / len(pertes_epoque)
            duree_epoque = time.time() - debut_epoque

            # ── Phase de validation ───────────────────────────────────────────
            metriques_valid = self._evaluer()
            map_50_courant = metriques_valid["map_50"]

            # ── Mise à jour du scheduler ──────────────────────────────────────
            self.scheduler.step()

            # ── Enregistrement historique ─────────────────────────────────────
            self.historique["perte_train"].append(perte_train_moy)
            self.historique["map_50_valid"].append(map_50_courant)
            self.historique["map_valid"].append(metriques_valid["map"])

            # ── Vérification amélioration (early stopping) ────────────────────
            est_meilleur = map_50_courant > self.meilleure_map_50
            if est_meilleur:
                self.meilleure_map_50 = map_50_courant
                self.epoque_meilleur_modele = epoque
                self.epoques_sans_amelioration = 0
            else:
                self.epoques_sans_amelioration += 1

            # ── Affichage résumé de l'époque ──────────────────────────────────
            indicateur = "[green]↑[/green]" if est_meilleur else "[red]→[/red]"
            console.print(
                f"\n{'─'*60}\n"
                f"  Époque {epoque:3d}/{self.nombre_epoques} | "
                f"Durée : {duree_epoque:.1f}s\n"
                f"  Perte train   : {perte_train_moy:.4f}\n"
                f"  mAP@0.5 valid : {map_50_courant:.4f} {indicateur}  "
                f"(meilleur : {self.meilleure_map_50:.4f} @ ép.{self.epoque_meilleur_modele})\n"
                f"  Patience      : {self.epoques_sans_amelioration}/{self.patience_arret_precoce}\n"
                f"{'─'*60}"
            )

            # ── Sauvegarde checkpoint ─────────────────────────────────────────
            self._sauvegarder_checkpoint(epoque, metriques_valid, est_meilleur)

            # ── Early stopping ────────────────────────────────────────────────
            if self.epoques_sans_amelioration >= self.patience_arret_precoce:
                console.print(
                    f"\n[bold yellow]⚠ Arrêt anticipé à l'époque {epoque} "
                    f"(patience={self.patience_arret_precoce} atteinte)[/bold yellow]"
                )
                console.print(
                    f"  Meilleur modèle : époque {self.epoque_meilleur_modele} "
                    f"avec mAP@0.5 = {self.meilleure_map_50:.4f}"
                )
                break

        duree_totale = time.time() - debut_global
        console.print(
            f"\n[bold green]{'═'*60}[/bold green]\n"
            f"[bold green]  ENTRAÎNEMENT TERMINÉ[/bold green]\n"
            f"  Durée totale  : {duree_totale/60:.1f} minutes\n"
            f"  Meilleur mAP@0.5 : {self.meilleure_map_50:.4f} (époque {self.epoque_meilleur_modele})\n"
            f"[bold green]{'═'*60}[/bold green]"
        )

        return self.historique

    def nettoyer_sorties_interrompues(self) -> None:
        """
        Supprime les fichiers de checkpoint créés pendant l'exécution actuelle.

        Cela permet de repartir de zéro si l'entraînement a été interrompu avant
        d'être terminé, sans conserver des checkpoints partiels.
        """
        for chemin in self._fichiers_checkpoint_crees:
            try:
                if chemin.exists():
                    chemin.unlink()
                    console.print(
                        f"[yellow]  Fichier supprimé après interruption : {chemin}[/yellow]"
                    )
            except OSError:
                console.print(
                    f"[red]  Impossible de supprimer le fichier : {chemin}[/red]"
                )

    def charger_meilleur_modele(self) -> None:
        """
        Charge les poids du meilleur modèle sauvegardé.

        À appeler avant l'évaluation finale sur le jeu de test.
        """
        chemin_meilleur = self.dossier_sorties / NOM_MEILLEUR_MODELE
        if not chemin_meilleur.exists():
            console.print("[yellow]⚠ Aucun meilleur modèle trouvé. Utilisation du modèle courant.[/yellow]")
            return

        checkpoint = torch.load(chemin_meilleur, map_location=self.dispositif, weights_only=False)
        self.modele.load_state_dict(checkpoint["etat_modele"])
        epoque_sauvegardee = checkpoint.get("epoque", "?")
        map_sauvegardee = checkpoint.get("metriques", {}).get("map_50", 0.0)
        console.print(
            f"[green]✓ Meilleur modèle chargé (époque {epoque_sauvegardee}, "
            f"mAP@0.5={map_sauvegardee:.4f})[/green]"
        )
