"""Train three anomaly detectors and evaluate on the same synthetic holdout."""

import json

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def make_dataset(
    rng: np.random.Generator, normal_count: int, anomaly_count: int
) -> tuple[np.ndarray, np.ndarray]:
    dense_count = normal_count // 2
    dense = rng.normal(loc=(-2.0, 0.0), scale=(0.25, 0.30), size=(dense_count, 2))
    loose = rng.normal(
        loc=(2.0, 0.0), scale=(0.95, 0.75), size=(normal_count - dense_count, 2)
    )
    anomalies = rng.uniform((-5.0, -4.0), (5.0, 4.0), size=(anomaly_count, 2))
    features = np.vstack((dense, loose, anomalies))
    labels = np.r_[np.zeros(normal_count, dtype=int), np.ones(anomaly_count, dtype=int)]
    order = rng.permutation(len(labels))
    return features[order], labels[order]


def main() -> None:
    rng = np.random.default_rng(42)
    x_train, _ = make_dataset(rng, normal_count=600, anomaly_count=60)
    x_test, y_test = make_dataset(rng, normal_count=400, anomaly_count=40)

    models = {
        "Isolation Forest": IsolationForest(
            n_estimators=200, max_samples=256, contamination=60 / 660, random_state=42
        ),
        "LOF": LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=60 / 660),
        "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=60 / 660),
    }
    results = {}
    for name, detector in models.items():
        model = make_pipeline(StandardScaler(), detector)
        model.fit(x_train)
        anomaly_scores = -model.decision_function(x_test)
        metrics = {
            "average_precision": round(average_precision_score(y_test, anomaly_scores), 3),
            "roc_auc": round(roc_auc_score(y_test, anomaly_scores), 3),
        }
        if name == "Isolation Forest":
            predicted_anomalies = model.predict(x_test) == -1
            tn, fp, fn, tp = confusion_matrix(
                y_test, predicted_anomalies, labels=[0, 1]
            ).ravel()
            metrics["confusion"] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
        results[name] = metrics

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
