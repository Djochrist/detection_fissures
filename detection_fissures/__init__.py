"""
Paquet d'ancrage pour exécuter le projet depuis une worktree quelconque.

Les modules historiques vivent à la racine du dépôt. On étend donc le chemin
du paquet pour que `detection_fissures.configuration`, `detection_fissures.modeles`
et les scripts racine restent importables même si le dossier local ne s'appelle
pas exactement `detection_fissures`.
"""

from pathlib import Path

_RACINE_PROJET = Path(__file__).resolve().parent.parent
__path__.append(str(_RACINE_PROJET))

