"""
Module des fonctions de perte pour l'entraînement Mask R-CNN.

PERTES DISPONIBLES
═══════════════════

1. PerteCombineeMaskRCNN (utilisée par défaut)
   ─────────────────────────────────────────────
   Mask R-CNN (torchvision) calcule INTERNEMENT ses propres pertes
   lors du forward pass en mode entraînement. Il retourne :
       {
           "loss_classifier"  : cross-entropy classification boîtes
           "loss_box_reg"     : Smooth L1 régression boîtes
           "loss_mask"        : BCE binaire segmentation masques
           "loss_objectness"  : objectness RPN
           "loss_rpn_box_reg" : Smooth L1 régression boîtes RPN
       }
   PerteCombineeMaskRCNN agrège ces 5 pertes avec des poids ajustables.
   Recommandation : poids_masque=2.0 pour prioriser la qualité des masques.

2. PerteUnifiedFocal — UFC (alternative pour la tête masque)
   ──────────────────────────────────────────────────────────
   Unified Focal Loss (Yeung et al., Medical Image Analysis 2022)
   https://doi.org/10.1016/j.media.2022.102576

   Problème résolu :
       Les fissures occupent < 5% des pixels → déséquilibre extrême.
       La BCE interne de Mask R-CNN traite tous les pixels équitablement
       → le modèle peut apprendre à tout prédire comme "fond" (95% accuracy,
         0% de vraies fissures détectées).

   Solution UFC :
       UFC = δ · AFC + (1 - δ) · AFT

       AFC (Asymmetric Focal Cross-Entropy) :
           Pondère plus les faux négatifs (fissures manquées) que les
           faux positifs (fausses alarmes) via les paramètres α et γ.
           FBCE = -(α·y·log(p)·(1-p)^γ + (1-α)·(1-y)·log(1-p)·p^γ)

       AFT (Asymmetric Focal Tversky Loss) :
           Généralise le Dice Loss en pondérant FN vs FP séparément.
           Tversky = TP / (TP + α·FN + (1-α)·FP)
           AFT = (1 - Tversky)^γ

   Paramètres recommandés pour les fissures :
       δ = 0.6    → légèrement plus de poids sur la composante focale
       α = 0.7    → pénalise plus les faux négatifs (fissures manquées)
                    qu'en sécurité structurelle : ne pas manquer une fissure
       γ = 0.75   → atténue la contribution des exemples faciles (fond)

   Usage dans ce projet :
       UFC est prête à être utilisée si vous personnalisez la tête de masque
       de Mask R-CNN (override de loss_mask). Voir ci-dessous pour l'intégration.

COMPARAISON DES PERTES (fissures, déséquilibre <5%)
═════════════════════════════════════════════════════

    BCE seule         → Mauvais : prédit tout "fond", ignore les fissures
    Dice + BCE        → Bon : standard 2024, gère le déséquilibre
    UFC (Yeung 2022)  → Meilleur : confirmé sur imagerie médicale/infrastructure
                        (ScienceDirect 2024, Crack Segmentation of Imbalanced Data)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# PERTE COMBINÉE MASK R-CNN (utilisée par défaut dans Entraineur)
# ──────────────────────────────────────────────────────────────────────────────

class PerteCombineeMaskRCNN(nn.Module):
    """
    Agrège et pondère les 5 pertes internes de Mask R-CNN.

    Mask R-CNN retourne un dictionnaire de 5 pertes.
    Cette classe permet d'ajuster les poids relatifs de chaque perte.

    Poids par défaut :
        Tous à 1.0 = somme directe (comportement standard torchvision)
        → Ajuster si une perte domine et dégrade l'entraînement

    Conseil pour les fissures :
        Augmenter poids_masque (ex: 2.0) pour forcer le modèle à améliorer
        la qualité des masques plutôt que seulement les boîtes englobantes.

    Args:
        poids_classifieur : Poids perte classification.
        poids_reg_boite   : Poids perte régression boîtes.
        poids_masque      : Poids perte segmentation masques (×2 recommandé).
        poids_objectness  : Poids perte objectness RPN.
        poids_reg_rpn     : Poids perte régression boîtes RPN.
    """

    def __init__(
        self,
        poids_classifieur: float = 1.0,
        poids_reg_boite: float = 1.0,
        poids_masque: float = 2.0,
        poids_objectness: float = 1.0,
        poids_reg_rpn: float = 1.0,
    ) -> None:
        super().__init__()
        self.poids = {
            "loss_classifier":  poids_classifieur,
            "loss_box_reg":     poids_reg_boite,
            "loss_mask":        poids_masque,
            "loss_objectness":  poids_objectness,
            "loss_rpn_box_reg": poids_reg_rpn,
        }

    def forward(self, dictionnaire_pertes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calcule la perte pondérée totale depuis le dictionnaire Mask R-CNN.

        Args:
            dictionnaire_pertes : Dictionnaire retourné par Mask R-CNN en train.

        Returns:
            Perte scalaire totale.
        """
        perte_totale = sum(
            self.poids.get(nom, 1.0) * valeur
            for nom, valeur in dictionnaire_pertes.items()
        )
        return perte_totale


def calculer_perte_totale_masque_rcnn(
    dictionnaire_pertes: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Somme directe de toutes les pertes Mask R-CNN (poids uniformes).

    Équivalent à PerteCombineeMaskRCNN avec tous les poids à 1.0.

    Args:
        dictionnaire_pertes : Dictionnaire retourné par Mask R-CNN.

    Returns:
        Perte totale scalaire.
    """
    return sum(dictionnaire_pertes.values())


# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED FOCAL LOSS — UFC (Yeung et al., Medical Image Analysis 2022)
# ──────────────────────────────────────────────────────────────────────────────

class PerteUnifiedFocal(nn.Module):
    """
    Unified Focal Loss pour segmentation binaire avec fort déséquilibre de classes.

    Référence :
        Yeung M. et al. (2022). Unified Focal loss: Generalising Dice and
        cross entropy-based losses to handle class imbalanced medical and
        non-medical image segmentation.
        Medical Image Analysis, 75, 102576.
        https://doi.org/10.1016/j.media.2022.102576

    Architecture de la perte :
    ────────────────────────────
        UFC = δ · AFC + (1 - δ) · AFT

        Composante 1 — AFC (Asymmetric Focal Cross-Entropy) :
            Rôle : Pondère les faux négatifs (fissures manquées) plus lourdement
            que les faux positifs (fausses alarmes).

            AFC = -[α · y · log(p) · (1-p)^γ  +  (1-α) · (1-y) · log(1-p) · p^γ]

            - α · y · log(p) · (1-p)^γ  : terme positif (pixels fissure)
              → (1-p)^γ atténue la perte sur les pixels bien classifiés (p élevé)
            - (1-α) · (1-y) · log(1-p) · p^γ : terme négatif (pixels fond)
              → p^γ atténue la perte sur les pixels fond correctement rejetés

        Composante 2 — AFT (Asymmetric Focal Tversky Loss) :
            Rôle : Généralise le Dice pour contrôler FN vs FP séparément.

            Tversky = (TP + ε) / (TP + α·FN + (1-α)·FP + ε)
            AFT = (1 - Tversky)^γ

            - α > 0.5 → pénalise plus les FN (fissures manquées)
            - γ < 1   → plus d'importance aux classes difficiles à détecter

    Paramètres recommandés (fissures structurelles) :
        δ = 0.6  → légèrement plus de poids sur AFC (stabilité numérique)
        α = 0.7  → priorité aux vrais positifs / pénalité FN (sécurité civile)
        γ = 0.75 → focus modéré sur les exemples difficiles

    Entrées :
        logits : Logits bruts non normalisés [B, 1, H, W] ou [B, H, W]
        cibles : Masques binaires ground truth, mêmes dimensions, float {0., 1.}

    Args:
        delta   : Poids de la composante AFC dans UFC. (1-delta) = poids AFT.
        alpha   : Asymétrie FN/FP. α > 0.5 → pénalise plus les FN.
        gamma   : Exposant focal. Réduit l'impact des exemples faciles.
        epsilon : Stabilité numérique (évite division par zéro dans Tversky).
    """

    def __init__(
        self,
        delta: float = 0.6,
        alpha: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _asymmetric_focal_cross_entropy(
        self,
        probabilites: torch.Tensor,
        cibles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule la Asymmetric Focal Cross-Entropy (AFC).

        AFC = -[α · y · log(p) · (1-p)^γ  +  (1-α) · (1-y) · log(1-p) · p^γ]

        Args:
            probabilites : Prédictions sigmoid [B, *, H, W] dans (0, 1).
            cibles       : Masques binaires float dans {0., 1.}.

        Returns:
            Scalaire AFC.
        """
        # Clipping pour la stabilité numérique (évite log(0))
        p = torch.clamp(probabilites, self.epsilon, 1.0 - self.epsilon)

        # Terme positif (pixels de fissures) : α · y · log(p) · (1-p)^γ
        terme_positif = self.alpha * cibles * torch.log(p) * ((1.0 - p) ** self.gamma)

        # Terme négatif (pixels de fond) : (1-α) · (1-y) · log(1-p) · p^γ
        terme_negatif = (1.0 - self.alpha) * (1.0 - cibles) * torch.log(1.0 - p) * (p ** self.gamma)

        afc = -(terme_positif + terme_negatif)
        return afc.mean()

    def _asymmetric_focal_tversky(
        self,
        probabilites: torch.Tensor,
        cibles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule l'Asymmetric Focal Tversky Loss (AFT).

        Tversky = (TP + ε) / (TP + α·FN + (1-α)·FP + ε)
        AFT = (1 - Tversky)^γ

        Args:
            probabilites : Prédictions sigmoid [B, *, H, W] dans (0, 1).
            cibles       : Masques binaires float dans {0., 1.}.

        Returns:
            Scalaire AFT.
        """
        p = probabilites
        y = cibles

        # Aplatir sur les dimensions spatiales pour le calcul
        p_flat = p.reshape(p.size(0), -1)
        y_flat = y.reshape(y.size(0), -1)

        # TP : vrais positifs (prédits fissure ET sont fissure)
        tp = (p_flat * y_flat).sum(dim=1)

        # FN : faux négatifs (prédits fond MAIS sont fissure)
        fn = ((1.0 - p_flat) * y_flat).sum(dim=1)

        # FP : faux positifs (prédits fissure MAIS sont fond)
        fp = (p_flat * (1.0 - y_flat)).sum(dim=1)

        # Score Tversky (par image du batch)
        tversky = (tp + self.epsilon) / (
            tp + self.alpha * fn + (1.0 - self.alpha) * fp + self.epsilon
        )

        # Focal Tversky : (1 - Tversky)^γ
        aft = (1.0 - tversky) ** self.gamma

        return aft.mean()

    def forward(
        self,
        logits: torch.Tensor,
        cibles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule la Unified Focal Loss.

        UFC = δ · AFC + (1 - δ) · AFT

        Args:
            logits : Logits bruts [B, 1, H, W] ou [B, H, W] (non normalisés).
            cibles : Masques binaires [B, 1, H, W] ou [B, H, W] en float {0., 1.}.

        Returns:
            Perte scalaire UFC dans [0, +∞).

        Example:
            >>> perte_ufc = PerteUnifiedFocal(delta=0.6, alpha=0.7, gamma=0.75)
            >>> logits = torch.randn(4, 1, 384, 384)
            >>> masques = torch.randint(0, 2, (4, 1, 384, 384)).float()
            >>> loss = perte_ufc(logits, masques)
            >>> print(f"UFC loss : {loss.item():.4f}")
        """
        probabilites = torch.sigmoid(logits)

        # Composante 1 : AFC (Asymmetric Focal Cross-Entropy)
        afc = self._asymmetric_focal_cross_entropy(probabilites, cibles)

        # Composante 2 : AFT (Asymmetric Focal Tversky)
        aft = self._asymmetric_focal_tversky(probabilites, cibles)

        # UFC = δ · AFC + (1 - δ) · AFT
        ufc = self.delta * afc + (1.0 - self.delta) * aft

        return ufc


# ──────────────────────────────────────────────────────────────────────────────
# INTÉGRATION UFC DANS MASK R-CNN
# ──────────────────────────────────────────────────────────────────────────────

class PerteMasqueRCNNUnifiedFocal(nn.Module):
    """
    Remplacement UFC de la perte BCE interne de Mask R-CNN pour la tête masque.

    CONTEXTE :
        Par défaut, Mask R-CNN calcule loss_mask = BCE(masques_prédits, masques_vrais).
        Pour les fissures (<5% de pixels), cette BCE peut sous-entraîner les masques.

    FONCTIONNEMENT :
        Ce wrapper intercepte les logits de la tête masque (avant l'application
        de la BCE interne) et les passe à UFC.
        À utiliser avec un DataLoader retournant les logits masque bruts.

    UTILISATION RECOMMANDÉE :
        1. Exécuter un forward pass de Mask R-CNN en mode entraînement.
        2. Récupérer dict_pertes["loss_mask"] → remplacer par UFC sur les logits.
        3. Recalculer la perte totale.

    NOTE :
        Cette approche nécessite d'accéder aux logits masque intermédiaires.
        La version la plus simple est d'utiliser PerteCombineeMaskRCNN avec
        un poids élevé sur loss_mask, et d'appliquer UFC séparément si besoin.

    Args:
        poids_classifieur : Poids perte classification.
        poids_reg_boite   : Poids perte régression boîtes.
        poids_objectness  : Poids perte objectness RPN.
        poids_reg_rpn     : Poids perte RPN.
        delta_ufc         : δ UFC (poids AFC).
        alpha_ufc         : α UFC (asymétrie FN/FP).
        gamma_ufc         : γ UFC (exposant focal).
    """

    def __init__(
        self,
        poids_classifieur: float = 1.0,
        poids_reg_boite: float = 1.0,
        poids_objectness: float = 1.0,
        poids_reg_rpn: float = 1.0,
        delta_ufc: float = 0.6,
        alpha_ufc: float = 0.7,
        gamma_ufc: float = 0.75,
    ) -> None:
        super().__init__()
        self.poids = {
            "loss_classifier":  poids_classifieur,
            "loss_box_reg":     poids_reg_boite,
            "loss_objectness":  poids_objectness,
            "loss_rpn_box_reg": poids_reg_rpn,
        }
        self.ufc = PerteUnifiedFocal(
            delta=delta_ufc,
            alpha=alpha_ufc,
            gamma=gamma_ufc,
        )

    def forward(
        self,
        dictionnaire_pertes: Dict[str, torch.Tensor],
        logits_masque: Optional[torch.Tensor] = None,
        cibles_masque: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcule la perte totale avec UFC sur la tête masque.

        Si logits_masque et cibles_masque sont fournis, remplace loss_mask par UFC.
        Sinon, utilise la loss_mask interne de Mask R-CNN (mode compatibilité).

        Args:
            dictionnaire_pertes : Dict de pertes retourné par Mask R-CNN.
            logits_masque       : Logits bruts [N, 1, H, W] de la tête masque.
            cibles_masque       : Masques ground truth [N, 1, H, W] binaires.

        Returns:
            Perte totale scalaire.
        """
        # Pertes standard (tout sauf loss_mask)
        perte_totale = sum(
            self.poids.get(nom, 1.0) * valeur
            for nom, valeur in dictionnaire_pertes.items()
            if nom != "loss_mask"
        )

        # Perte masque : UFC si logits fournis, sinon fallback sur BCE interne
        if logits_masque is not None and cibles_masque is not None:
            perte_masque = self.ufc(logits_masque, cibles_masque)
        else:
            perte_masque = dictionnaire_pertes.get("loss_mask", torch.tensor(0.0))

        perte_totale = perte_totale + 2.0 * perte_masque
        return perte_totale
