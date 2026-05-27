"""Package racine du projet detection_fissures.

Ce paquet sert de point d'entrée pour les modules répartis dans le dépôt.
Il étend __path__ pour que les packages top-level comme analyse/, configuration/
ou entrainement/ soient importables via detection_fissures.analyse,
detection_fissures.configuration, etc.
"""

from pathlib import Path

_RACINE_PROJET = Path(__file__).resolve().parent.parent
__path__.append(str(_RACINE_PROJET))

