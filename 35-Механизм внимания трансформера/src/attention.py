from math import exp, sqrt


def attention(q, k, v, causal=False):
    d_k = len(k[0])
    scores, weights, output = [], [], []
    for i, query in enumerate(q):
        row = [
            sum(a * b for a, b in zip(query, key)) / sqrt(d_k)
            if not causal or j <= i else float("-inf")
            for j, key in enumerate(k)
        ]
        peak = max(row)
        terms = [exp(score - peak) for score in row]
        total = sum(terms)
        alpha = [term / total for term in terms]
        scores.append(row)
        weights.append(alpha)
        output.append([
            sum(weight * value for weight, value in zip(alpha, column))
            for column in zip(*v)
        ])
    return scores, weights, output


if __name__ == "__main__":
    q = k = [[2, 0], [0, 2]]
    v = [[10, 0], [0, 6]]

    for causal in (False, True):
        print("causal =", causal)
        scores, weights, output = attention(q, k, v, causal=causal)
        for label, matrix in zip(("S", "A", "O"), (scores, weights, output)):
            print(label)
            for row in matrix:
                print(" ".join(f"{value:8.4f}" for value in row))
        assert all(abs(sum(row) - 1) < 1e-12 for row in weights)
        if causal:
            assert weights[0] == [1, 0] and output[0] == [10, 0]
