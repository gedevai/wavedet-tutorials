
```

# T03 — Solutions

## Exercice 1 — Shapes

Image :

```text
H = 480
W = 640
C = 3
```

### NumPy

Convention :

```text
HWC
```

Donc :

```text
(480, 640, 3)
```

Les axes représentent :

```text
480 → hauteur
640 → largeur
3   → canaux RGB
```

### PyTorch

Convention :

```text
CHW
```

Donc :

```text
(3, 480, 640)
```

### Batch de 8

Convention :

```text
BCHW
```

Donc :

```text
(8, 3, 480, 640)
```

---

# Exercice 2 — Corrélation manuelle

Image :

```text
I =
[
  [1, 2, 0],
  [3, 1, 2],
  [0, 1, 4]
]
```

Kernel :

```text
K =
[
  [1, 0],
  [0,-1]
]
```

## 1. Taille de sortie

```text
H = 3
W = 3
K = 2
P = 0
S = 1
```

Donc :

```text
H_out
= floor((3 - 2) / 1) + 1
= 1 + 1
= 2
```

Même calcul pour `W`.

La sortie vaut donc :

```text
2 × 2
```

---

## 2. Première fenêtre

```text
[
  [1,2],
  [3,1]
]
```

Calcul :

```text
1 × 1
+ 2 × 0
+ 3 × 0
+ 1 × (-1)

= 1 - 1
= 0
```

Donc :

```text
Y[0,0] = 0
```

---

## 3. Deuxième fenêtre

```text
[
  [2,0],
  [1,2]
]
```

Calcul :

```text
2 × 1
+ 0 × 0
+ 1 × 0
+ 2 × (-1)

= 2 - 2
= 0
```

Donc :

```text
Y[0,1] = 0
```

---

## 4. Troisième fenêtre

```text
[
  [3,1],
  [0,1]
]
```

Calcul :

```text
3 × 1
+ 1 × 0
+ 0 × 0
+ 1 × (-1)

= 3 - 1
= 2
```

Donc :

```text
Y[1,0] = 2
```

---

## 5. Quatrième fenêtre

```text
[
  [1,2],
  [1,4]
]
```

Calcul :

```text
1 × 1
+ 2 × 0
+ 1 × 0
+ 4 × (-1)

= 1 - 4
= -3
```

Donc :

```text
Y[1,1] = -3
```

Résultat complet :

```text
Y =
[
  [0,  0],
  [2, -3]
]
```

---

# Exercice 3 — Dimensions

Données :

```text
H = W = 64
K = 3
P = 1
S = 2
```

Formule :

```text
H_out =
floor((H + 2P - K) / S) + 1
```

Substitution :

```text
H_out =
floor((64 + 2×1 - 3) / 2) + 1
```

Donc :

```text
H_out =
floor((64 + 2 - 3) / 2) + 1
```

```text
=
floor(63 / 2) + 1
```

```text
=
floor(31.5) + 1
```

```text
=
31 + 1
```

```text
=
32
```

Même calcul pour la largeur :

```text
W_out = 32
```

Résultat :

```text
32 × 32
```

---

# Exercice 4 — Padding

Signal :

```text
[10, 20, 30]
```

Padding d'une valeur.

## Zero

Une représentation est :

```text
[0, 10, 20, 30, 0]
```

On suppose que l'extérieur vaut zéro.

Cela peut créer une transition artificielle forte entre :

```text
0 et 10
```

ou :

```text
30 et 0
```

---

## Reflect

Avec la convention NumPy utilisée dans T03 :

```text
[20, 10, 20, 30, 20]
```

L'environnement extérieur est construit par réflexion des valeurs internes.

Il n'introduit pas directement un zéro arbitraire.

---

## Replicate

```text
[10, 10, 20, 30, 30]
```

Les valeurs des extrémités sont prolongées.

Le bord gauche est considéré comme continuant avec `10`.

Le bord droit continue avec `30`.

---

## Conclusion

Les trois stratégies inventent nécessairement quelque chose hors de l'image.

La différence réside dans ce qui est supposé :

```text
zero      → extérieur nul
reflect   → contenu réfléchi
replicate → valeur du bord répétée
```

Le résultat d'un filtre peut donc différer près des limites.

---

# Exercice 5 — Filtre de différence

Signal :

```text
x =
[2, 2, 2, 8, 8, 8]
```

Kernel :

```text
[-1, 1]
```

## Position 1

```text
[2,2]
```

```text
2 × (-1)
+ 2 × 1

= 0
```

## Position 2

```text
[2,2]
```

```text
-2 + 2
= 0
```

## Position 3

```text
[2,8]
```

```text
2 × (-1)
+ 8 × 1

= -2 + 8
= 6
```

## Position 4

```text
[8,8]
```

```text
-8 + 8
= 0
```

## Position 5

```text
[8,8]
```

```text
-8 + 8
= 0
```

Résultat :

```text
[0, 0, 6, 0, 0]
```

La réponse maximale apparaît à la transition :

```text
2 → 8
```

car le filtre mesure une différence locale.

Dans les zones constantes :

```text
2 → 2
```

ou :

```text
8 → 8
```

la différence est nulle.

Une réponse haute fréquence importante n'est cependant pas automatiquement utile.

Une variation brutale peut être :

* le contour d'un objet ;
* une texture ;
* un détail pertinent ;
* un artefact ;
* du bruit.

L'interprétation dépend donc du contexte.
