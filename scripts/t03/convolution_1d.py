def cross_correlate_1d_valid(
    signal: list[float],
    kernel: list[float],
    stride: int = 1,
    verbose: bool = False,
) -> list[float]:
    if stride <= 0:
        raise ValueError("stride doit être strictement positif.")

    if len(kernel) == 0:
        raise ValueError("Le kernel ne peut pas être vide.")

    if len(kernel) > len(signal):
        raise ValueError("Le kernel est plus grand que le signal.")

    output: list[float] = []

    last_start = len(signal) - len(kernel)

    for start in range(0, last_start + 1, stride):
        window = signal[start : start + len(kernel)]

        products = [
            value * weight
            for value, weight in zip(window, kernel, strict=True)
        ]

        result = sum(products)
        output.append(result)

        if verbose:
            print(f"\nPosition {start}")
            print(f"window   = {window}")
            print(f"kernel   = {kernel}")
            print(f"products = {products}")
            print(f"sum      = {result}")

    return output

def main() -> None:
    signal = [1, 2, 3, 4, 5]
    kernel = [1 / 3, 1 / 3, 1 / 3]

    print("Signal :", signal)
    print("Kernel :", kernel)

    output = cross_correlate_1d_valid(
        signal,
        kernel,
        stride=1,
        verbose=True,
    )

    print("\nRésultat :", output)

    expected = [2.0, 3.0, 4.0]

    assert all(
        abs(actual - target) < 1e-12
        for actual, target in zip(output, expected, strict=True)
    )

    print("Résultat manuel vérifié : OK")


if __name__ == "__main__":
    main()
