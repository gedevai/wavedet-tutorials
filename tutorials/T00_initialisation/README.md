# T00 — Initialisation du parcours, environnement scientifique et notebooks

## Objectif

T00 prépare l'environnement technique utilisé dans T01–T20.

À la fin du tutoriel, l'apprenant doit être capable de :

- se déplacer dans le dépôt depuis un terminal ;
- créer et activer un environnement virtuel Python ;
- vérifier Python, NumPy et PyTorch ;
- vérifier CUDA sans en faire une condition de fonctionnement ;
- comprendre l'organisation du dépôt ;
- utiliser les commandes Git essentielles ;
- distinguer notebook, script et code réutilisable ;
- utiliser correctement JupyterLab ;
- comprendre le rôle du kernel ;
- exécuter un notebook de façon reproductible ;
- lancer les tests automatiques.

---

## 1. Organisation

Le dépôt distingue :

```text
tutorials/  documentation pédagogique
notebooks/  exploration et visualisation
scripts/    tâches exécutables
src/        logique réellement réutilisable
tests/      tests automatiques
data/       données
outputs/    résultats générés
docs/       documentation scientifique
