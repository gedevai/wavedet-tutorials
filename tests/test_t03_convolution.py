from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


T03_SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "t03"
)

if str(T03_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(T03_SCRIPTS))
    from convolution_2d_numpy import (  # noqa: E402
        correlate2d,
        output_shape_2d,
    )

    IMAGE = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.float64,
    )

    KERNEL = np.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=np.float64,
    )


    def test_numpy_matches_manual_result() -> None:
        expected = np.full(
            (3, 3),
            -5.0,
        )

        result = correlate2d(
            IMAGE,
            KERNEL,
            stride=1,
            padding=0,
        )

        assert np.allclose(
            result,
            expected,
        )


    def test_numpy_and_pytorch_match() -> None:
        numpy_result = correlate2d(
            IMAGE,
            KERNEL,
        )

        image_torch = torch.tensor(
            IMAGE,
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        kernel_torch = torch.tensor(
            KERNEL,
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        torch_result = F.conv2d(
            image_torch,
            kernel_torch,
            stride=1,
            padding=0,
        )

        torch_result_numpy = (
            torch_result
            .squeeze(0)
            .squeeze(0)
            .numpy()
        )

        assert np.allclose(
            numpy_result,
            torch_result_numpy,
        )


    def test_output_dimensions() -> None:
        assert output_shape_2d(
            image_shape=(32, 32),
            kernel_shape=(3, 3),
            padding=0,
            stride=1,
        ) == (30, 30)

        assert output_shape_2d(
            image_shape=(32, 32),
            kernel_shape=(3, 3),
            padding=1,
            stride=1,
        ) == (32, 32)

        assert output_shape_2d(
            image_shape=(32, 32),
            kernel_shape=(3, 3),
            padding=1,
            stride=2,
        ) == (16, 16)


    def test_stride_two_has_expected_shape_and_values() -> None:
        result = correlate2d(
            IMAGE,
            KERNEL,
            stride=2,
            padding=0,
        )

        expected = np.full(
            (2, 2),
            -5.0,
        )

        assert result.shape == (2, 2)
        assert np.allclose(
            result,
            expected,
        )


    def test_difference_kernel_is_zero_on_constant_input() -> None:
        constant = np.full(
            (4, 5),
            7.0,
        )

        difference_kernel = np.array(
            [
                [-1, 1],
            ],
            dtype=np.float64,
        )

        result = correlate2d(
            constant,
            difference_kernel,
            stride=1,
            padding=0,
        )

        assert np.allclose(
            result,
            0.0,
        )
