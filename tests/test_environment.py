import numpy as np
import torch


def test_numpy_is_importable() -> None:
    assert np.__version__


def test_torch_is_importable() -> None:
    assert torch.__version__


def test_numpy_array_creation() -> None:
    values = np.array([1, 2, 3, 4])

    assert values.shape == (4,)
    assert values.tolist() == [1, 2, 3, 4]


def test_torch_tensor_creation() -> None:
    tensor = torch.tensor([1.0, 2.0, 3.0])

    assert tensor.shape == (3,)
    assert tensor.dtype == torch.float32


def test_numpy_simple_numeric_behavior() -> None:
    values = np.array([1, 2, 3])

    result = values * 2

    np.testing.assert_array_equal(result, np.array([2, 4, 6]))


def test_torch_simple_numeric_behavior() -> None:
    tensor = torch.tensor([1.0, 2.0, 3.0])

    result = tensor * 2

    expected = torch.tensor([2.0, 4.0, 6.0])
    assert torch.equal(result, expected)
