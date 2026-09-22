"""
train_models.py
High-Performance Model Training & Serialization for Vehicle Insurance Fraud Detection
Trains and serializes:
1. Logistic Regression
2. Decision Tree Classifier
3. KNN Classifier
4. AdaBoost Classifier
5. Random Forest Classifier
6. K-Means Clustering
7. K-Medoids Clustering (PAM)
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    silhouette_score,
    average_precision_score,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class FastKMedoidsPAM:
    """
    Optimized NumPy implementation of Partitioning Around Medoids (PAM) K-Medoids Clustering.
    """
    def __init__(self, n_clusters=3, max_iter=30, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoid_indices_ = None
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        X_mat = np.asarray(X, dtype=np.float32)
        n_samples = X_mat.shape[0]
        rng = np.random.RandomState(self.random_state)

        # Initialize medoids randomly
        medoid_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)

        for _ in range(self.max_iter):
            medoids = X_mat[medoid_indices]
            # (n_samples, n_clusters)
            dists = np.linalg.norm(X_mat[:, np.newaxis, :] - medoids[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            new_medoids = []
            for k in range(self.n_clusters):
                cluster_members = np.where(labels == k)[0]
                if len(cluster_members) == 0:
                    new_medoids.append(medoid_indices[k])
                    continue
                # If cluster is large, sub-sample for fast medoid selection
                if len(cluster_members) > 300:
                    sub_idx = rng.choice(cluster_members, size=300, replace=False)
                else:
                    sub_idx = cluster_members

                sub_pts = X_mat[sub_idx]
                intra_dists = np.linalg.norm(sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :], axis=2)
                best_sub_idx = sub_idx[np.argmin(np.sum(intra_dists, axis=1))]
                new_medoids.append(best_sub_idx)

            new_medoids = np.array(new_medoids)
            if np.array_equal(np.sort(medoid_indices), np.sort(new_medoids)):
                break
            medoid_indices = new_medoids

        self.medoid_indices_ = medoid_indices
        self.cluster_centers_ = X_mat[medoid_indices]
        dists = np.linalg.norm(X_mat[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        self.labels_ = np.argmin(dists, axis=1)
        return self

    def predict(self, X):
        X_mat = np.asarray(X, dtype=np.float32)
        dists = np.linalg.norm(X_mat[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_


def train_and_export():
    print("=" * 70)
    print("STARTING VEHICLE INSURANCE ML TRAINING PIPELINE")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    # Locate dataset
    data_candidates = [
        os.path.join(base_dir, "vehicle_insurance.csv"),
        os.path.join(base_dir, "insurance_fraud_cleaned_data.csv"),
        os.path.join(base_dir, "FrontEnd", "insurance_fraud_cleaned_final.csv"),
        os.path.join(parent_dir, "insurance_fraud_cleaned_data.csv"),
    ]

    csv_path = None
    for p in data_candidates:
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        raise FileNotFoundError("Could not find cleaned dataset CSV file.")

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Save standardized copy vehicle_insurance.csv
    target_csv = os.path.join(base_dir, "vehicle_insurance.csv")
    df.to_csv(target_csv, index=False)
    print(f"Saved standardized dataset: {target_csv}")

    # Copy raw CSV if present
    raw_source = os.path.join(parent_dir, "insurance_fraud_data.csv")
    if os.path.exists(raw_source):
        df_raw = pd.read_csv(raw_source)
        df_raw.to_csv(os.path.join(base_dir, "vehicle_insurance_raw.csv"), index=False)
        print("Saved raw dataset: vehicle_insurance_raw.csv")

    # Ensure model folder exists
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Features and Target
    drop_cols = [c for c in ['claim_number', 'fraud reported'] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df['fraud reported']

    print(f"Total Feature Count: {len(feature_cols)}")
    print(f"Target Distribution: Legit={int((y==0).sum())} ({(y==0).mean()*100:.1f}%), Fraud={int((y==1).sum())} ({(y==1).mean()*100:.1f}%)")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 1. SUPERVISED MODELS TRAINING & BENCHMARKING
    # ══════════════════════════════════════════════════════════════════════════
    # Pipelines / Models
    supervised_models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7, weights='distance'))
        ]),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=8, random_state=42)
    }

    metrics_records = []
    roc_data = {}
    pr_data = {}
    confusion_matrices = {}
    feature_importances = {}
    trained_artifacts = {}

    print("\n" + "=" * 70)
    print("EVALUATING SUPERVISED MODELS")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in supervised_models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t1 = time.time()
        y_pred = model.predict(X_test)
        infer_time = time.time() - t1

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Fast 5-fold CV score calculation
        cv_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_cv_tr, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
            y_cv_tr, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_cv_tr, y_cv_tr)
            cv_scores.append(accuracy_score(y_cv_val, model.predict(X_cv_val)))
        # Refit on train set for export
        model.fit(X_train, y_train)
        cv_mean = float(np.mean(cv_scores))

        metrics_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Specificity": specificity,
            "Avg Precision": avg_prec,
            "CV Accuracy (5-Fold)": cv_mean,
            "Train Time (s)": round(train_time, 4),
            "Inference Time (ms)": round(infer_time * 1000, 2),
        })

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}

        # PR Curve
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_data[name] = {"precision": p_curve.tolist(), "recall": r_curve.tolist(), "ap": avg_prec}

        # Confusion Matrix
        confusion_matrices[name] = cm.tolist()

        # Feature importances
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            feature_importances[name] = dict(zip(feature_cols, imp.tolist()))
        elif isinstance(model, Pipeline) and hasattr(model.named_steps.get("model"), "coef_"):
            imp = np.abs(model.named_steps["model"].coef_[0])
            feature_importances[name] = dict(zip(feature_cols, imp.tolist()))
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
            feature_importances[name] = dict(zip(feature_cols, imp.tolist()))

        trained_artifacts[name] = model

        print(f"  [OK] {name:20s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | CV: {cv_mean:.4f}")

    metrics_df = pd.DataFrame(metrics_records)

    # Save individual models to model folder
    file_map = {
        "Logistic Regression": "logistic_model.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn_model.pkl",
        "AdaBoost": "adaboost_model.pkl",
        "Random Forest": "random_forest.pkl"
    }

    for name, filename in file_map.items():
        save_path = os.path.join(model_dir, filename)
        joblib.dump(trained_artifacts[name], save_path)
        print(f"Saved {name} -> {save_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. UNSUPERVISED CLUSTERING (K-Means & K-Medoids)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TRAINING UNSUPERVISED CLUSTERING MODELS")
    print("=" * 70)

    # PCA 2D & 3D for visualization
    pca_2d = PCA(n_components=2, random_state=42)
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X)
    X_pca_3d = pca_3d.fit_transform(X)

    sample_idx = np.random.RandomState(42).choice(len(X), size=min(1500, len(X)), replace=False)
    X_sub = X.iloc[sample_idx]

    elbow_kmeans = {}
    sil_kmeans = {}
    sil_kmedoids = {}

    for k in range(2, 7):
        km_test = KMeans(n_clusters=k, random_state=42, n_init=5).fit(X)
        elbow_kmeans[k] = float(km_test.inertia_)
        sil_kmeans[k] = float(silhouette_score(X_sub, km_test.predict(X_sub)))

        kmed_test = FastKMedoidsPAM(n_clusters=k, random_state=42).fit(X_sub.values)
        sil_kmedoids[k] = float(silhouette_score(X_sub, kmed_test.labels_))
        print(f"  k={k} | KMeans Inertia: {elbow_kmeans[k]:.1f} | KMeans Sil: {sil_kmeans[k]:.4f} | KMedoids Sil: {sil_kmedoids[k]:.4f}")

    # Final Clustering Models (k=3)
    best_k = 3
    kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
    kmedoids_model = FastKMedoidsPAM(n_clusters=best_k, random_state=42).fit(X.values)

    km_labels = kmeans_model.labels_
    kmed_labels = kmedoids_model.labels_

    # Save clustering models
    km_path = os.path.join(model_dir, "kmeans_model.pkl")
    kmed_path = os.path.join(model_dir, "kmedoids_model.pkl")
    joblib.dump(kmeans_model, km_path)
    joblib.dump(kmedoids_model, kmed_path)
    print(f"Saved K-Means -> {km_path}")
    print(f"Saved K-Medoids -> {kmed_path}")

    # Cluster profiling & fraud rates
    def build_cluster_profile(labels, name):
        df_c = pd.DataFrame({
            "cluster": labels,
            "fraud": y.values,
            "pca_x": X_pca_2d[:, 0],
            "pca_y": X_pca_2d[:, 1],
            "pca_z": X_pca_3d[:, 2],
            "total_claim": df["total_claim"].values,
            "vehicle_price": df["vehicle_price"].values,
            "annual_income": df["annual_income"].values,
            "age_of_driver": df["age_of_driver"].values,
        })
        profile = df_c.groupby("cluster").agg(
            total_claims=("fraud", "count"),
            fraud_cases=("fraud", "sum"),
            fraud_rate=("fraud", "mean"),
            avg_claim_amount=("total_claim", "mean"),
            avg_vehicle_price=("vehicle_price", "mean"),
            avg_income=("annual_income", "mean"),
            avg_driver_age=("age_of_driver", "mean"),
        ).reset_index()
        profile["fraud_rate_pct"] = profile["fraud_rate"] * 100
        profile["risk_tier"] = profile["fraud_rate_pct"].apply(
            lambda r: "High Risk Anomaly" if r >= 28.0 else ("Moderate Risk" if r >= 24.0 else "Low Risk Baseline")
        )
        return profile

    km_profile = build_cluster_profile(km_labels, "K-Means")
    kmed_profile = build_cluster_profile(kmed_labels, "K-Medoids")

    clustering_summary = {
        "best_k": best_k,
        "elbow_kmeans": elbow_kmeans,
        "sil_kmeans": sil_kmeans,
        "sil_kmedoids": sil_kmedoids,
        "kmeans_profile": km_profile,
        "kmedoids_profile": kmed_profile,
        "pca_variance_2d": pca_2d.explained_variance_ratio_.tolist(),
        "pca_variance_3d": pca_3d.explained_variance_ratio_.tolist(),
        "sample_points_2d": {
            "x": X_pca_2d[sample_idx, 0].tolist(),
            "y": X_pca_2d[sample_idx, 1].tolist(),
            "km_cluster": km_labels[sample_idx].tolist(),
            "kmed_cluster": kmed_labels[sample_idx].tolist(),
            "fraud": y.values[sample_idx].tolist(),
        },
        "sample_points_3d": {
            "x": X_pca_3d[sample_idx, 0].tolist(),
            "y": X_pca_3d[sample_idx, 1].tolist(),
            "z": X_pca_3d[sample_idx, 2].tolist(),
            "km_cluster": km_labels[sample_idx].tolist(),
            "kmed_cluster": kmed_labels[sample_idx].tolist(),
            "fraud": y.values[sample_idx].tolist(),
        }
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 3. EXPORT EVALUATION & COMPREHENSIVE METRICS
    # ══════════════════════════════════════════════════════════════════════════
    eval_package = {
        "metrics_df": metrics_df,
        "roc_data": roc_data,
        "pr_data": pr_data,
        "confusion_matrices": confusion_matrices,
        "feature_importances": feature_importances,
        "feature_names": feature_cols,
        "test_indices": X_test.index.tolist(),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    eval_path = os.path.join(model_dir, "evaluation_metrics.pkl")
    cluster_summary_path = os.path.join(model_dir, "clustering_summary.pkl")

    joblib.dump(eval_package, eval_path)
    joblib.dump(clustering_summary, cluster_summary_path)

    print(f"\nSaved Evaluation Metrics -> {eval_path}")
    print(f"Saved Clustering Summary -> {cluster_summary_path}")

    # Copy files to FrontEnd/ subdirectory for compatibility
    frontend_subdir = os.path.join(base_dir, "FrontEnd")
    if os.path.exists(frontend_subdir):
        fe_model_dir = os.path.join(frontend_subdir, "model")
        os.makedirs(fe_model_dir, exist_ok=True)
        for f in os.listdir(model_dir):
            src = os.path.join(model_dir, f)
            dst = os.path.join(fe_model_dir, f)
            joblib.dump(joblib.load(src), dst)
        print("Synchronized model files to FrontEnd/model/")

    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED & ARTIFACTS SERIALIZED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    train_and_export()
