"""Package utilitaires : dispositif, graine, journalisation."""
from .dispositif import detecter_dispositif, afficher_info_dispositif
from .graine import fixer_graine_aleatoire

__all__ = [
    "detecter_dispositif",
    "afficher_info_dispositif",
    "fixer_graine_aleatoire",
]
