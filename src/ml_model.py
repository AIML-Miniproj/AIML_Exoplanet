# src/ml_model.py

import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from config import (TEST_SIZE, RANDOM_STATE, N_AUGMENTATIONS,
                    MODEL_PATH, OUTPUT_DIR, FEATURE_NAMES, PHASE_WIDTH)


def extract_features_from_fold(lc_flat, period, t0, duration):
    """Fold at any period and extract features."""
    try:
        folded = lc_flat.fold(period=period, epoch_time=t0)
        ph     = folded.phase.value
        fl     = folded.flux.value

        in_t   = np.abs(ph) < PHASE_WIDTH
        out_t  = ~in_t

        if np.sum(in_t) < 3 or np.sum(out_t) < 3:
            return None

        fi     = np.median(fl[in_t])
        fo     = np.median(fl[out_t])
        d      = (fo - fi) / fo
        si     = np.std(fl[in_t])
        so     = np.std(fl[out_t])
        snr_   = d / so if so > 0 else 0

        left   = fl[(ph > -PHASE_WIDTH) & (ph < 0)]
        right  = fl[(ph > 0) & (ph < PHASE_WIDTH)]
        ml     = min(len(left), len(right))
        sym    = (
            1 - np.mean(np.abs(left[:ml] - right[:ml][::-1]))
            if ml > 0 else 0
        )
        nt     = int((lc_flat.time.value[-1] - lc_flat.time.value[0]) / period)

        return {
            "depth"        : d,
            "duration"     : duration,
            "mean_flux_in" : fi,
            "std_flux_in"  : si,
            "std_flux_out" : so,
            "snr"          : snr_,
            "symmetry"     : sym,
            "sharpness"    : d / duration if duration > 0 else 0,
            "n_transits"   : nt
        }
    except Exception:
        return None


def build_dataset(true_features, lc_flat, bls_output):
    """
    Generates synthetic positive and negative training samples.
    """
    rng             = np.random.default_rng(RANDOM_STATE)
    samples, labels = [], []
    best_period     = bls_output["best_period"]
    t0              = bls_output["t0"]
    duration        = bls_output["duration"]

    # Positive samples — perturb true features
    for _ in range(N_AUGMENTATIONS):
        perturbed = {
            k: v + rng.normal(0, abs(v) * 0.05 + 1e-7)
            for k, v in true_features.items()
        }
        samples.append([perturbed[f] for f in FEATURE_NAMES])
        labels.append(1)

    # Negative samples — wrong periods
    wrong_periods = [
        best_period * 2,   best_period * 3,
        best_period * 0.5, best_period * 0.33,
        1.0, 2.0, 4.0, 7.0, 10.0, 13.0
    ]
    per_class = max(N_AUGMENTATIONS // len(wrong_periods), 1)

    for wp in wrong_periods:
        feats = extract_features_from_fold(lc_flat, wp, t0, duration)
        if feats is None:
            continue
        for _ in range(per_class):
            perturbed = {
                k: v + rng.normal(0, abs(v) * 0.05 + 1e-7)
                for k, v in feats.items()
            }
            samples.append([perturbed[f] for f in FEATURE_NAMES])
            labels.append(0)

    X = np.array(samples)
    y = np.array(labels)
    print(f"[ML] Dataset -> {np.sum(y==1)} positives | {np.sum(y==0)} negatives")
    return X, y


def train_model(X, y):
    """
    Trains XGBoost classifier, saves model + evaluation plots.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y
    )

    model = XGBClassifier(
        n_estimators  = 100,
        max_depth     = 4,
        learning_rate = 0.1,
        reg_alpha     = 0.1,
        reg_lambda    = 1.0,
        eval_metric   = "logloss",
        random_state  = RANDOM_STATE
    )

    model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = False
    )

    y_pred = model.predict(X_test)
    print("\n[ML] Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Noise", "Transit"]))

    # Confusion matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Noise", "Transit"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    plt.title("XGBoost Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
                bbox_inches="tight")
    plt.close()
    print(f"[ML] Saved confusion matrix -> {OUTPUT_DIR}confusion_matrix.png")

    # Feature importance
    importance = model.feature_importances_
    fig, ax    = plt.subplots(figsize=(8, 5))
    ax.barh(FEATURE_NAMES, importance, color="steelblue")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"),
                bbox_inches="tight")
    plt.close()
    print(f"[ML] Saved feature importance -> {OUTPUT_DIR}feature_importance.png")

    joblib.dump(model, MODEL_PATH)
    print(f"[ML] Model saved -> {MODEL_PATH}")

    return model


def predict(features_dict):
    """
    Loads saved model, predicts label and confidence for one sample.
    """
    model      = joblib.load(MODEL_PATH)
    X          = np.array([[features_dict[f] for f in FEATURE_NAMES]])
    label      = int(model.predict(X)[0])
    confidence = round(float(model.predict_proba(X)[0][label]) * 100, 2)
    return label, confidence


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    from config import STAR_NAME
    from src.fetcher import fetch_lightcurve
    from src.preprocessor import preprocess
    from src.detector import run_bls
    from src.analyzer import analyze_transit

    lc                         = fetch_lightcurve(STAR_NAME)
    _, lc_flat, _              = preprocess(lc)
    bls_output                 = run_bls(lc_flat)
    _, features, _             = analyze_transit(lc_flat, bls_output)

    X, y    = build_dataset(features, lc_flat, bls_output)
    model   = train_model(X, y)
    label, confidence = predict(features)

    verdict = "Transit Detected" if label == 1 else "Noise"
    print(f"\n[ML] Prediction  : {verdict}")
    print(f"[ML] Confidence  : {confidence}%")