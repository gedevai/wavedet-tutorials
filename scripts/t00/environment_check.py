import sys

import numpy as np
import torch


def main() -> None:
    print("=== T00 environment check ===")
    print()

    print(f"Python version : {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print(f"NumPy version  : {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    cuda_available = torch.cuda.is_available()

    print(f"CUDA available : {cuda_available}")

    if cuda_available:
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA    : {torch.version.cuda}")
    else:
        print("GPU             : not available through PyTorch")
        print(f"PyTorch CUDA    : {torch.version.cuda}")

    print()
    print("Environment check completed.")


if __name__ == "__main__":
    main()
