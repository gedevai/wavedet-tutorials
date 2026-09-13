import numpy as np
import torch

def describe_numpy(name:str, array:np.ndarray) -> None:
    print(f"\n{name}")
    print(array)
    print(f"type: {type(array)}")
    print(f"shape: {array.shape}")
    print(f"dtype : {array.dtype}")


def describe_tensor(name:str, tensor:torch.Tensor) -> None:
    print(f"\n{name}")
    print(tensor)
    print(f"type: {type(tensor)}")
    print(f"shape: {tuple(tensor.size())}")
    print(f"dtype : {tensor.dtype}")
    print(f"device: {tensor.device}")


def main() -> None:
    print("=== T03 — Image / NumPy / PyTorch ===")
    grayscale = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
        dtype=np.uint8,
    )

    describe_numpy("1. Image grayscale NumPy [H, W]", grayscale)

    rgb_hwc = np.array(
        [
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
            [
                [255, 255, 0],
                [255, 255, 255],
                [0, 0, 0],
            ],
        ],
        dtype=np.uint8,
    )

    describe_numpy("2. Image RGB NumPy [H, W, C]", rgb_hwc)

    h, w, c = rgb_hwc.shape

    print("\nDimensions RGB :")
    print(f"H = {h}")
    print(f"W = {w}")
    print(f"C = {c}")

    rgb_tensor_hwc = torch.from_numpy(rgb_hwc)

    describe_tensor(
        "3. Tensor PyTorch encore au format [H, W, C]",
        rgb_tensor_hwc,
    )

    rgb_tensor_chw = rgb_tensor_hwc.permute(2, 0, 1)

    describe_tensor(
        "4. Tensor après HWC -> CHW",
        rgb_tensor_chw,
    )

    rgb_tensor_bchw = rgb_tensor_chw.unsqueeze(0)

    describe_tensor(
        "5. Tensor après ajout du batch : BCHW",
        rgb_tensor_bchw,
    )

    rgb_float = rgb_tensor_bchw.to(torch.float32) / 255.0

    describe_tensor(
        "6. Même image en float32 dans [0, 1]",
        rgb_float,
    )

    print("\nRésumé :")
    print(f"NumPy HWC : {rgb_hwc.shape}")
    print(f"Torch CHW : {tuple(rgb_tensor_chw.shape)}")
    print(f"Torch BCHW: {tuple(rgb_tensor_bchw.shape)}")


if __name__ == "__main__":
    main()
