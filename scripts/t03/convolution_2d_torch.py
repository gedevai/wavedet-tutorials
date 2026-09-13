import torch
import torch.nn as nn
import torch.nn.functional as F

from convolution_2d_numpy import correlate2d


def main() -> None:
    image_hw = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=torch.float32,
    )

    kernel_hw = torch.tensor(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=torch.float32,
    )

    print("Image initiale [H, W] :")
    print(image_hw)
    print("shape =", tuple(image_hw.shape))

    # [H, W] -> [1, H, W] -> [1, 1, H, W]
    image_bchw = image_hw.unsqueeze(0).unsqueeze(0)

    print("\nImage pour PyTorch [B, C_in, H, W] :")
    print("shape =", tuple(image_bchw.shape))

    # [K_h, K_w]
    # -> [1, K_h, K_w]
    # -> [1, 1, K_h, K_w]
    weight = kernel_hw.unsqueeze(0).unsqueeze(0)

    print("\nKernel initial [K_h, K_w] :")
    print(kernel_hw)

    print("\nPoids pour PyTorch [C_out, C_in, K_h, K_w] :")
    print("shape =", tuple(weight.shape))

    # ---------------------------------------------------------
    # F.conv2d
    # ---------------------------------------------------------
    output_torch = F.conv2d(
        input=image_bchw,
        weight=weight,
        bias=None,
        stride=1,
        padding=0,
    )

    print("\nRésultat F.conv2d :")
    print(output_torch)

    output_hw = output_torch.squeeze(0).squeeze(0)

    print("\nRésultat [H_out, W_out] :")
    print(output_hw)

    # ---------------------------------------------------------
    # NumPy
    # ---------------------------------------------------------
    output_numpy = correlate2d(
        image=image_hw.numpy(),
        kernel=kernel_hw.numpy(),
        stride=1,
        padding=0,
    )

    print("\nRésultat NumPy :")
    print(output_numpy)

    numpy_as_torch = torch.from_numpy(
        output_numpy,
    ).to(dtype=output_hw.dtype)

    print(
        "\ntorch.allclose(PyTorch, NumPy) =",
        torch.allclose(
            output_hw,
            numpy_as_torch,
        ),
    )

    assert torch.allclose(
        output_hw,
        numpy_as_torch,
    )

    # ---------------------------------------------------------
    # nn.Conv2d
    # ---------------------------------------------------------
    layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=(2, 2),
        stride=1,
        padding=0,
        bias=False,
    )

    # Nous ne faisons aucun apprentissage.
    # Nous imposons simplement notre kernel connu.
    with torch.no_grad():
        layer.weight.copy_(weight)

    output_module = layer(image_bchw)

    print("\nRésultat nn.Conv2d avec poids imposés :")
    print(output_module)

    assert torch.allclose(
        output_module,
        output_torch,
    )

    expected = torch.full(
        (3, 3),
        -5.0,
        dtype=torch.float32,
    )

    assert torch.allclose(
        output_hw,
        expected,
    )

    print("\nPyTorch reproduit le calcul manuel : OK")


if __name__ == "__main__":
    main()
