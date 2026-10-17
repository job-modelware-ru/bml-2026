import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knn import classify, euclidean_distance, regress


class TestKNN(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(euclidean_distance([1, 1], [4, 5]), 5)

    def test_classification(self):
        x = [[1, 2], [2, 3], [2, 1], [6, 4], [7, 5], [6, 6]]
        y = ["A", "A", "A", "B", "B", "B"]
        names = ["A1", "A2", "A3", "B1", "B2", "B3"]

        prediction, neighbors, votes = classify(x, y, [3, 2], 3, names=names)

        self.assertEqual(prediction, "A")
        self.assertEqual([n["name"] for n in neighbors], ["A2", "A3", "A1"])
        self.assertEqual(votes["A"], 3)

    def test_regression(self):
        x = [[50], [60], [68], [74], [76], [85], [95]]
        y = [4.8, 5.7, 6.6, 7.2, 7.5, 8.4, 9.3]

        prediction, _ = regress(x, y, [72], 3)
        self.assertAlmostEqual(prediction, 7.1)


if __name__ == "__main__":
    unittest.main()
