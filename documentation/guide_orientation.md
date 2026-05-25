# Guide Orientation des Fissures

Ce guide explique comment le projet détermine l'orientation d'une fissure après
l'entraînement du modèle.

## Commande d'analyse

Après avoir entraîné Mask R-CNN, utilisez le checkpoint `.pth` avec `analyser.py`.

Commande locale :

```bash
python analyser.py \
  --modele sorties/modeles/meilleur_modele.pth \
  --images photos_test/ \
  --seuil 0.4 \
  --sortie resultats_analyse.json
```

Commande Google Colab / Drive :

```bash
python analyser.py \
  --modele /content/drive/MyDrive/detection_fissures_sorties_mask_v2/modeles/meilleur_modele.pth \
  --images /content/drive/MyDrive/images_a_tester \
  --seuil 0.5 \
  --sortie /content/drive/MyDrive/resultats_fissures.json
```

## Principe général

Le modèle entraîné ne prédit pas directement `horizontale`, `verticale` ou
`inclinée`. Il prédit d'abord un masque de segmentation pour chaque fissure.
Ensuite, le module `analyse/classificateur_fissures.py` mesure la forme du masque.

Flux complet :

```text
Image
  -> modèle entraîné
  -> masque de fissure
  -> analyse PCA des pixels du masque
  -> angle principal
  -> orientation finale
```

## Étapes de calcul

1. Le script récupère tous les pixels qui appartiennent à la fissure.
2. Ces pixels sont transformés en coordonnées `(x, y)`.
3. Une PCA est appliquée sur ces coordonnées.
4. La PCA donne l'axe principal de la fissure.
5. L'angle de cet axe est ramené entre `0°` et `90°`.
6. L'angle est comparé aux seuils configurés.

Les seuils par défaut sont :

| Orientation | Condition |
|---|---|
| Horizontale | angle `< 20°` |
| Inclinée | angle entre `20°` et `70°` |
| Verticale | angle `> 70°` |

## Pourquoi utiliser la PCA ?

Une boîte englobante peut être trompeuse si la fissure est courbée, ramifiée ou
irrégulière. La PCA regarde tous les pixels du masque et cherche la direction où
ils sont le plus étalés. Cette direction représente mieux l'orientation générale
de la fissure.

## Où modifier les seuils ?

Les seuils sont dans `analyse/classificateur_fissures.py` :

```python
SEUIL_ANGLE_HORIZONTAL = 20.0
SEUIL_ANGLE_VERTICAL = 70.0
```

Vous pouvez les ajuster si vos images ont une convention différente ou si vous
voulez une classification plus stricte.

## Résultat attendu

Dans le terminal, `analyser.py` affiche un tableau par image avec :

- l'orientation ;
- l'angle mesuré ;
- la localisation estimée ;
- la largeur moyenne ;
- l'indice de danger ;
- le score de détection.

Si `--sortie` est fourni, les mêmes informations sont aussi sauvegardées dans un
fichier JSON.
