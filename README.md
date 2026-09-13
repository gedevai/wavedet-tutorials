# WaveDet Research Tutorials

Parcours pédagogique et expérimental pour construire progressivement un projet
de recherche en vision par ordinateur autour de la détection d'objets, de
l'analyse fréquentielle et des ondelettes.

## Parcours

- T00 : environnement, Git et notebooks
- T01–T07 : méthode scientifique, image, fréquence et ondelettes
- T08–T13 : deep learning, PyTorch, CNN, détection, transformers et métriques
- T14–T16 : datasets, reproductibilité et état de l'art
- T17–T19 : contribution wavelet, intégration et expérimentations
- T20 : statistiques, analyse d'erreurs et rédaction

## Organisation

- `tutorials/` : supports pédagogiques
- `notebooks/` : exploration et visualisation
- `scripts/` : programmes et expériences exécutables
- `src/` : code réellement réutilisable
- `tests/` : tests automatiques
- `data/` : données locales et petits exemples
- `outputs/` : résultats générés
- `docs/` : documentation scientifique

## Environnement

Créer un environnement virtuel :

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Installer les dépendances pédagogiques :

```bash
python3 -m pip install --upgrade pip
python3 -m pip install .
```

Installer ensuite PyTorch avec la commande adaptée au système et au backend
de calcul indiquée par le sélecteur officiel PyTorch.

## Vérification

```bash
python3 scripts/t00/environment_check.py
```

## Notebooks

```bash
jupyter lab
```
Avant de considérer un notebook comme reproductible :

```text
Restart Kernel → Run All
```

## Tests

```bash
pytest -q
```

## Principe de progression

Le code commence dans les notebooks ou scripts lorsqu'il est pédagogique ou
exploratoire. Il ne migre vers `src/` que lorsqu'il devient réellement
réutilisable.
