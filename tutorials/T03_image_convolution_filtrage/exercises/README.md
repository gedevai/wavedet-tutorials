# T03 — Exercices

Effectuer les exercices avant de consulter `solutions.md`.

---

## Exercice 1 — Shapes

Une image RGB possède :

```text
H = 480
W = 640
C = 3
```

Donner la shape :

1. en NumPy avec convention HWC ;
2. dans PyTorch avec convention CHW ;
3. dans PyTorch pour un batch de 8 images.

Expliquer ce que signifie chaque axe.

---

## Exercice 2 — Corrélation 2D manuelle

Considérons :

```text
I =
[
  [1, 2, 0],
  [3, 1, 2],
  [0, 1, 4]
]
```

et :

```text
K =
[
  [1, 0],
  [0, -1]
]
```

Avec :

```text
stride = 1
padding = 0
```

1. déterminer la taille de sortie ;
2. écrire chacune des quatre fenêtres ;
3. calculer chaque produit élément par élément ;
4. donner toute la matrice de sortie.

Il s'agit d'une **corrélation croisée** : ne pas retourner le kernel.

---

## Exercice 3 — Dimensions

Calculer la taille spatiale de sortie pour :

```text
H = 64
W = 64
K_h = K_w = 3
P_h = P_w = 1
S_h = S_w = 2
```

Utiliser :

```text
H_out =
floor((H + 2P_h - K_h) / S_h) + 1
```

et la formule analogue pour `W_out`.

Montrer toutes les étapes.

---

## Exercice 4 — Padding

Pour la ligne :

```text
[10, 20, 30]
```

expliquer qualitativement ce que fera un padding d'une valeur de type :

1. zero ;
2. reflect ;
3. replicate.

Pour chacun, préciser quel type d'information artificielle est introduit à proximité du bord.

---

## Exercice 5 — Filtre de différence

Considérons :

```text
x =
[2, 2, 2, 8, 8, 8]
```

et :

```text
k =
[-1, 1]
```

1. calculer la corrélation valide complète ;
2. identifier la position ayant la plus forte réponse ;
3. expliquer pourquoi ;
4. expliquer pourquoi une forte réponse haute fréquence ne signifie pas nécessairement « information utile ».
