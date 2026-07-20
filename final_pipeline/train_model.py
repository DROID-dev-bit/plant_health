"""
Train the plant-health model and save weights in the notebook format.

Produces weights/plant_health_model.pkl = {model, label_encoder, features},
the exact format ml_predictor.py loads. Run this locally, OR drop the
plant_health_model.pkl you exported from the Colab notebook into weights/ — the
pipeline uses whichever file is there.

    ../.venv/bin/python train_model.py
"""

import pickle

import numpy as np
import pandas as pd

import config

FEATURES = ["ndvi", "tdvi", "ndre", "gndvi", "plant_height_m"]
MODEL_OUT = config.ROOT_DIR / "weights" / "plant_health_model.pkl"
KEEP = ["Healthy", "Moderate", "Mid Stressed", "Stressed", "Unhealthy", "Critical"]


def main():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report
    from xgboost import XGBClassifier

    df = pd.read_csv(config.OUTPUT_CSV)
    feats = [c for c in FEATURES if c in df.columns]
    d = df[df["sector_label"] != "NOISE"].dropna(subset=feats + ["health_status"])
    d = d[d["health_status"].isin(KEEP)]
    print(f"[train] {len(d):,} plants, features {feats}")

    X = d[feats].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(d["health_status"])

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    model = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, eval_metric="mlogloss")
    f1 = cross_val_score(model, Xtr, ytr, cv=cv, scoring="f1_macro")
    print(f"[train] XGBoost CV f1_macro = {f1.mean():.3f} +/- {f1.std():.3f}")

    model.fit(Xtr, ytr)
    print(classification_report(yva, model.predict(Xva),
                                target_names=le.classes_))

    model.fit(X, y)   # refit on all data
    MODEL_OUT.parent.mkdir(exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le, "features": feats}, f)
    print(f"[train] saved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
