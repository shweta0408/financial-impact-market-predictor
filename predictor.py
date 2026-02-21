"""
predictor.py
------------
Logistic Regression + Random Forest ensemble.
Labels: 0=Bearish  1=Neutral  2=Bullish
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Any

LABEL_MAP = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
LABEL_INV = {"Bearish": 0, "Neutral": 1, "Bullish": 2}


def compound_to_label(compound: float, threshold: float = 0.08) -> int:
    if compound >= threshold:
        return 2
    elif compound <= -threshold:
        return 0
    return 1


class MarketImpactPredictor:
    def __init__(self):
        self.scaler  = StandardScaler()
        self.lr      = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
        self.rf      = RandomForestClassifier(n_estimators=200, max_depth=8,
                                              class_weight="balanced", random_state=42)
        self._fitted = False
        self._feature_names: List[str] = []

    # ── Train ──────────────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        self._feature_names = list(X.columns)
        X_arr = X.values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y, test_size=0.2, random_state=42, stratify=y
        )
        X_tr_s = self.scaler.fit_transform(X_train)
        X_te_s = self.scaler.transform(X_test)

        self.lr.fit(X_tr_s, y_train)
        self.rf.fit(X_train, y_train)
        self._fitted = True

        lr_pred = self.lr.predict(X_te_s)
        rf_pred = self.rf.predict(X_test)

        return {
            "lr_report":    classification_report(y_test, lr_pred,
                                target_names=["Bearish","Neutral","Bullish"],
                                output_dict=True),
            "rf_report":    classification_report(y_test, rf_pred,
                                target_names=["Bearish","Neutral","Bullish"],
                                output_dict=True),
            "lr_confusion": confusion_matrix(y_test, lr_pred).tolist(),
            "rf_confusion": confusion_matrix(y_test, rf_pred).tolist(),
            "n_train": len(X_train),
            "n_test":  len(X_test),
        }

    # ── Predict ────────────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        if not self._fitted:
            return [self._heuristic(row) for _, row in X.iterrows()]

        X_arr  = X[self._feature_names].values.astype(float)
        X_s    = self.scaler.transform(X_arr)
        lr_p   = self.lr.predict_proba(X_s)
        rf_p   = self.rf.predict_proba(X_arr)
        ens    = 0.4 * lr_p + 0.6 * rf_p

        results = []
        for i in range(len(X_arr)):
            lid = int(np.argmax(ens[i]))
            results.append({
                "label":          LABEL_MAP[lid],
                "label_id":       lid,
                "confidence":     round(float(ens[i][lid]), 4),
                "lr_proba":       [round(float(v), 4) for v in lr_p[i]],
                "rf_proba":       [round(float(v), 4) for v in rf_p[i]],
                "ensemble_proba": [round(float(v), 4) for v in ens[i]],
            })
        return results

    @staticmethod
    def _heuristic(row) -> Dict[str, Any]:
        compound = float(row.get("compound", 0))
        kw       = float(row.get("kw_ratio",  0))
        combined = 0.7 * compound + 0.3 * kw
        lid      = 2 if combined >= 0.08 else (0 if combined <= -0.08 else 1)
        conf     = min(0.95, 0.5 + abs(combined) * 1.5)
        proba    = [0.0, 0.0, 0.0]
        proba[lid]             = conf
        proba[(lid + 1) % 3]   = (1 - conf) * 0.6
        proba[(lid + 2) % 3]   = (1 - conf) * 0.4
        return {"label": LABEL_MAP[lid], "label_id": lid, "confidence": round(conf, 4),
                "lr_proba": proba[:], "rf_proba": proba[:], "ensemble_proba": proba[:]}

    # ── Feature importance ─────────────────────────────────────────────────────
    def feature_importance(self) -> pd.DataFrame:
        if not self._fitted:
            return pd.DataFrame()
        return (pd.DataFrame({"feature": self._feature_names,
                               "importance": self.rf.feature_importances_})
                  .sort_values("importance", ascending=False)
                  .reset_index(drop=True))

    # ── Save / Load ────────────────────────────────────────────────────────────
    def save(self, path: str = "./saved_model"):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump({"lr": self.lr, "rf": self.rf, "scaler": self.scaler,
                         "features": self._feature_names, "fitted": self._fitted}, f)

    def load(self, path: str = "./saved_model"):
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            d = pickle.load(f)
        self.lr = d["lr"]; self.rf = d["rf"]
        self.scaler = d["scaler"]; self._feature_names = d["features"]
        self._fitted = d["fitted"]
