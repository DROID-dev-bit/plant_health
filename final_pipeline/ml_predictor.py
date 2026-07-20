"""
Apply a trained plant-health model to every plant.

Loads weights/plant_health_model.pkl ({model, label_encoder, features}) — the
format produced by train_model.py or the Colab notebook — and adds ML columns:

  ml_health_pred        model-predicted health class
  ml_health_confidence  predicted-class probability
  ml_agrees             does it match the rule-based health_status?

Gracefully no-ops (leaving the columns blank) if the weights file is absent.
"""

import pickle

import numpy as np


def predict_health(df, model_path, noise_label="NOISE"):
    if not model_path.exists():
        print(f"[ml_predictor] no model at {model_path} — skipping "
              "(drop your trained plant_health_model.pkl there to enable)")
        df["ml_health_pred"] = None
        df["ml_health_confidence"] = np.nan
        return df

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model, le, feats = bundle["model"], bundle["label_encoder"], bundle["features"]

    have = [f for f in feats if f in df.columns]
    ready = df[have].notna().all(axis=1) & (df["sector_label"] != noise_label) \
        if have else np.zeros(len(df), bool)

    df["ml_health_pred"] = None
    df["ml_health_confidence"] = np.nan
    if len(have) == len(feats) and ready.any():
        proba = model.predict_proba(df.loc[ready, feats].to_numpy())
        df.loc[ready, "ml_health_pred"] = le.inverse_transform(proba.argmax(1))
        df.loc[ready, "ml_health_confidence"] = np.round(proba.max(1), 3)
        n = int(ready.sum())
        if "health_status" in df.columns:
            df["ml_agrees"] = np.where(
                df["ml_health_pred"].notna(),
                df["ml_health_pred"] == df["health_status"], None)
            agree = (df.loc[ready, "ml_health_pred"]
                     == df.loc[ready, "health_status"]).mean()
            print(f"[ml_predictor] predicted {n:,} plants "
                  f"(model={type(model).__name__}); "
                  f"agreement with rule-based label = {agree*100:.1f}%")
        else:
            print(f"[ml_predictor] predicted {n:,} plants")
    else:
        missing = [f for f in feats if f not in df.columns]
        print(f"[ml_predictor] cannot predict — missing features {missing}")
    return df
