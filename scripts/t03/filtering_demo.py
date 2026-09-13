from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from convolution_2d_numpy import correlate2d

def create_synthetic_image(
    height: int = 128,
    width: int = 128,
) -> np.ndarray:
    image = np.zeros(
        (height, width),
        dtype=np.float64,
    )

    # Rectangle clair
    image[32:96, 36:92] = 0.75

    # Petite zone plus claire
    image[48:72, 48:72] = 1.0

    # Ligne verticale fine
    image[20:108, 64] = 1.0

    # Ligne horizontale
    image[82:85, 16:112] = 0.5

    return image

def main() -> None:
    image = create_synthetic_image()

    low_pass_kernel = np.ones(
        (3, 3),
        dtype=np.float64,
    ) / 9.0

    high_pass_kernel = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ],
        dtype=np.float64,
    )

    low_pass = correlate2d(
        image=image,
        kernel=low_pass_kernel,
        stride=1,
        padding=1,
        padding_mode="reflect",
    )

    high_pass = correlate2d(
        image=image,
        kernel=high_pass_kernel,
        stride=1,
        padding=1,
        padding_mode="reflect",
    )

 # Pour la visualisation, nous affichons la magnitude de la réponse.
    high_pass_magnitude = np.abs(high_pass)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("Original")

    axes[1].imshow(
        low_pass,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Low-pass filtered")

    axes[2].imshow(
        high_pass_magnitude,
        cmap="gray",
    )
    axes[2].set_title("High-pass response")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()

    repo_root = Path(__file__).resolve().parents[2]
    output_directory = repo_root / "outputs" / "T03"
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / "filtering_demo.png"

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    print(f"Figure sauvegardée dans : {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
