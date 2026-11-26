"""Reproduce the two illustrative iTrees and the exact textbook score."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Split:
    threshold: float
    left: "Split | tuple[int, ...]"
    right: "Split | tuple[int, ...]"


def average_path_size(size: int) -> Fraction:
    if size <= 1:
        return Fraction(0)
    harmonic_number = sum(Fraction(1, i) for i in range(1, size))
    return 2 * harmonic_number - Fraction(2 * (size - 1), size)


def path_length(tree: Split | tuple[int, ...], point: int) -> Fraction:
    depth = 0
    while isinstance(tree, Split):
        tree = tree.left if point < tree.threshold else tree.right
        depth += 1
    return depth + average_path_size(len(tree))


def main() -> None:
    tree_a = Split(
        6,
        Split(2.5, Split(1.5, (1,), (2,)), Split(3.5, (3,), (4,))),
        (9,),
    )
    tree_b = Split(
        3.5,
        Split(1.5, (1,), Split(2.5, (2,), (3,))),
        Split(7, (4,), (9,)),
    )
    normalizer = average_path_size(5)
    print(f"c(5) = {normalizer} = {float(normalizer):.6f}")
    print(" x   h_A   h_B   mean h   score")
    for point in (1, 2, 3, 4, 9):
        depth_a = path_length(tree_a, point)
        depth_b = path_length(tree_b, point)
        mean_depth = (depth_a + depth_b) / 2
        score = 2 ** (-float(mean_depth / normalizer))
        print(f"{point:2} {float(depth_a):5.1f} {float(depth_b):5.1f} "
              f"{float(mean_depth):8.2f} {score:7.3f}")


if __name__ == "__main__":
    main()
