from collections.abc import Sequence

import numpy as np


Pair = tuple[int, int]


def as_pair(value: int | Sequence[int], name: str) -> Pair:
    """Convertit un entier ou une paire en tuple (height, width)."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} ne peut pas être négatif.")
        return value, value

    if len(value) != 2:
        raise ValueError(f"{name} doit contenir exactement deux valeurs.")

    first = int(value[0])
    second = int(value[1])

    if first < 0 or second < 0:
        raise ValueError(f"{name} ne peut pas contenir de valeur négative.")

    return first, second


def output_shape_2d(
    image_shape: Pair,
    kernel_shape: Pair,
    stride: int | Pair = 1,
    padding: int | Pair = 0,
) -> Pair:
    """
    Calcule la taille spatiale de sortie.

    Hypothèses :
    - dilation = 1 ;
    - padding symétrique ;
    - padding donné par côté.
    """
    h, w = image_shape
    kh, kw = kernel_shape
    sh, sw = as_pair(stride, "stride")
    ph, pw = as_pair(padding, "padding")

    if sh == 0 or sw == 0:
        raise ValueError("Le stride doit être strictement positif.")

    if kh <= 0 or kw <= 0:
        raise ValueError("Le kernel doit avoir une taille positive.")

    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            "Le kernel est trop grand pour l'image avec ce padding."
        )

    return h_out, w_out


def pad_image(
    image: np.ndarray,
    padding: int | Pair,
    mode: str = "zero",
) -> np.ndarray:
    """Ajoute un padding zero, reflect ou replicate."""
    ph, pw = as_pair(padding, "padding")

    if ph == 0 and pw == 0:
        return image

    pad_width = ((ph, ph), (pw, pw))

    if mode == "zero":
        return np.pad(
            image,
            pad_width,
            mode="constant",
            constant_values=0,
        )

    if mode == "reflect":
        return np.pad(
            image,
            pad_width,
            mode="reflect",
        )

    if mode == "replicate":
        return np.pad(
            image,
            pad_width,
            mode="edge",
        )

    raise ValueError(
        "padding_mode doit être 'zero', 'reflect' ou 'replicate'."
    )


def correlate2d(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int | Pair = 1,
    padding: int | Pair = 0,
    padding_mode: str = "zero",
    verbose: bool = False,
) -> np.ndarray:
    """
    Corrélation croisée 2D explicite.

    Le kernel est utilisé tel quel.
    """
    image = np.asarray(image)
    kernel = np.asarray(kernel)

    if image.ndim != 2:
        raise ValueError("image doit être une matrice 2D.")

    if kernel.ndim != 2:
        raise ValueError("kernel doit être une matrice 2D.")

    sh, sw = as_pair(stride, "stride")
    ph, pw = as_pair(padding, "padding")

    if sh == 0 or sw == 0:
        raise ValueError("Le stride doit être strictement positif.")

    padded = pad_image(
        image,
        padding=(ph, pw),
        mode=padding_mode,
    )

    kh, kw = kernel.shape

    h_out, w_out = output_shape_2d(
        image_shape=image.shape,
        kernel_shape=kernel.shape,
        stride=(sh, sw),
        padding=(ph, pw),
    )

    output = np.empty(
        (h_out, w_out),
        dtype=np.result_type(image.dtype, kernel.dtype, np.float64),
    )

    for out_y in range(h_out):
        for out_x in range(w_out):
            y = out_y * sh
            x = out_x * sw

            window = padded[
                y : y + kh,
                x : x + kw,
            ]

            products = window * kernel
            value = products.sum()

            output[out_y, out_x] = value

            if verbose:
                print(
                    f"\nSortie [{out_y}, {out_x}] "
                    f"— fenêtre origine ({y}, {x})"
                )
                print("Fenêtre :")
                print(window)
                print("Fenêtre × kernel :")
                print(products)
                print("Somme :", value)

    return output


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int | Pair = 1,
    padding: int | Pair = 0,
    padding_mode: str = "zero",
) -> np.ndarray:
    """
    Convolution mathématique 2D.

    La convolution est obtenue en retournant spatialement le kernel
    puis en effectuant une corrélation croisée.
    """
    flipped_kernel = np.flip(
        kernel,
        axis=(0, 1),
    )

    return correlate2d(
        image=image,
        kernel=flipped_kernel,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
    )


def main() -> None:
    image = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.float64,
    )

    kernel = np.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=np.float64,
    )

    expected_correlation = np.full(
        (3, 3),
        -5.0,
    )

    expected_convolution = np.full(
        (3, 3),
        5.0,
    )

    print("Image :")
    print(image)

    print("\nKernel :")
    print(kernel)

    print("\n=== Corrélation croisée ===")

    correlation = correlate2d(
        image,
        kernel,
        stride=1,
        padding=0,
        verbose=True,
    )

    print("\nRésultat :")
    print(correlation)

    print("\nAttendu :")
    print(expected_correlation)

    print(
        "\nnp.allclose(correlation, expected) =",
        np.allclose(correlation, expected_correlation),
    )

    print("\n=== Convolution mathématique ===")

    convolution = convolve2d(
        image,
        kernel,
        stride=1,
        padding=0,
    )

    print(convolution)

    print(
        "\nnp.allclose(convolution, expected) =",
        np.allclose(convolution, expected_convolution),
    )

    assert np.allclose(
        correlation,
        expected_correlation,
    )

    assert np.allclose(
        convolution,
        expected_convolution,
    )

    print("\nToutes les vérifications sont correctes.")


if __name__ == "__main__":
    main()
