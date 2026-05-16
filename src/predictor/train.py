"""Train + evaluate L4 edge predictors.

Five models for the ablation table:
    1. Random        — coin flip (baseline)
    2. LR-unigram    — VADER + struct + unigram triggers
    3. LR-uni+bi     — adds bigram triggers (KEY CONTRIBUTION)
    4. LR-uni+bi+tri — adds trigram triggers
    5. XGBoost       — same features, gradient-boosted (perf ceiling)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedModel:
    name: str
    model: object
    scaler: StandardScaler | None
    feature_cols: list[str]
    test_y: np.ndarray
    test_pred_proba: np.ndarray
    metrics: dict = field(default_factory=dict)


def _precision_at_recall(y_true, y_score, recall_target: float) -> float:
    p, r, _ = precision_recall_curve(y_true, y_score)
    idx = np.where(r >= recall_target)[0]
    return float(p[idx].max()) if len(idx) > 0 else 0.0


def _precision_at_top_k(y_true, y_score, k_frac: float) -> float:
    n = len(y_true)
    k = int(round(n * k_frac))
    if k == 0:
        return 0.0
    top_idx = np.argsort(-y_score)[:k]
    return float(y_true[top_idx].mean())


def _evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        'auc_roc':        float(roc_auc_score(y_true, y_score)),
        'auc_pr':         float(average_precision_score(y_true, y_score)),
        'p_at_r20':       _precision_at_recall(y_true, y_score, 0.20),
        'p_at_r50':       _precision_at_recall(y_true, y_score, 0.50),
        'p_at_top10':     _precision_at_top_k(y_true, y_score, 0.10),
        'p_at_top20':     _precision_at_top_k(y_true, y_score, 0.20),
        'positive_rate':  float(np.mean(y_true)),
    }


def train_random(y_train: np.ndarray, y_test: np.ndarray, seed: int = 42) -> TrainedModel:
    rng = np.random.default_rng(seed)
    proba = rng.random(len(y_test))
    metrics = _evaluate(y_test, proba)
    return TrainedModel('random', None, None, [], y_test, proba, metrics)


def train_lr(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    name: str,
    feature_cols: list[str],
    *,
    C: float = 1.0,
    max_iter: int = 1000,
) -> TrainedModel:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train[feature_cols])
    Xte = scaler.transform(X_test[feature_cols])
    model = LogisticRegression(C=C, max_iter=max_iter, class_weight='balanced')
    model.fit(Xtr, y_train)
    proba = model.predict_proba(Xte)[:, 1]
    return TrainedModel(name, model, scaler, list(feature_cols), y_test, proba, _evaluate(y_test, proba))


def train_xgb(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    feature_cols: list[str],
    *,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
) -> TrainedModel:
    from xgboost import XGBClassifier
    Xtr = X_train[feature_cols].to_numpy()
    Xte = X_test[feature_cols].to_numpy()
    pos_w = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        scale_pos_weight=pos_w, eval_metric='auc',
        n_jobs=4, tree_method='hist',
    )
    model.fit(Xtr, y_train)
    proba = model.predict_proba(Xte)[:, 1]
    return TrainedModel('xgboost', model, None, list(feature_cols),
                        y_test, proba, _evaluate(y_test, proba))


def train_split(
    X: pd.DataFrame, y: np.ndarray,
    *, test_size: float = 0.2, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """80/20 stratified split. Returns X_tr, X_te, y_tr, y_te."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def lr_top_features(model: TrainedModel, k: int = 12) -> pd.DataFrame:
    """Top |coef| features for a trained LR. Returns coef + sign for interpretation."""
    if model.scaler is None or model.model is None:
        return pd.DataFrame()
    coefs = model.model.coef_[0]
    df = pd.DataFrame({'feature': model.feature_cols, 'coef': coefs})
    df['abs_coef'] = df['coef'].abs()
    return df.sort_values('abs_coef', ascending=False).head(k).reset_index(drop=True)
