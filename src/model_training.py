"""
model_training.py
Phase 6: Train and evaluate machine learning models on Lichess blunder data.

Models trained:
  1. Logistic Regression  - interpretable baseline
  2. Random Forest        - handles non-linear patterns  
  3. Gradient Boosting    - typically strongest performance

Key decisions:
  - GroupKFold splits by GAME not by move (prevents data leakage)
  - class_weight='balanced' handles the 92/8 class imbalance
  - F1 and AUC-ROC used instead of accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import GroupKFold
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

data_path    = os.path.join(project_dir, "data",    "lichess_blunder_dataset.csv")
results_path = os.path.join(project_dir, "results")
models_path  = os.path.join(project_dir, "models")
os.makedirs(results_path, exist_ok=True)
os.makedirs(models_path,  exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(data_path)

print(f"Dataset : {len(df)} moves from {df['game_id'].nunique()} games")
print(f"Blunder rate : {df['is_blunder'].mean()*100:.2f}%")
print(f"Mistake rate : {df['is_mistake'].mean()*100:.2f}%")


# ─────────────────────────────────────────────
# FEATURE SELECTION
# Only features known BEFORE the move is made
# Never include delta_cp or quality - those are the answer
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "n_legal_moves",
    "eval_before_cp",
    "eval_volatility",
    "material_balance",
    "n_pieces",
    "is_endgame",
    "king_in_check",
    "move_number",
    "player_rating",
    "prev_was_blunder",
    "prev_delta_cp",
    "time_pressure",
    "time_spent_seconds",
]

TARGET = "is_mistake"
GROUPS = "game_id"

df_clean = df[FEATURE_COLS + [TARGET, GROUPS]].dropna()
print(f"\nAfter dropping NaN : {len(df_clean)} records")
print(f"Positive rate (mistakes) : {df_clean[TARGET].mean()*100:.2f}%")

X      = df_clean[FEATURE_COLS].values
y      = df_clean[TARGET].values
groups = df_clean[GROUPS].values


# ─────────────────────────────────────────────
# CROSS VALIDATION
# GroupKFold: entire games go to train OR test
# Prevents data leakage between moves of same game
# ─────────────────────────────────────────────

n_splits = 5
gkf      = GroupKFold(n_splits=n_splits)
print(f"\nCross-validation : {n_splits}-fold GroupKFold")


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        ))
    ]),
}


# ─────────────────────────────────────────────
# TRAINING AND EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(name, pipeline, X, y, groups, gkf):
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        print(f"  Fold {fold+1} complete")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    f1  = f1_score(all_y_true, all_y_pred, zero_division=0)
    auc = roc_auc_score(all_y_true, all_y_prob)

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        all_y_true, all_y_pred,
        target_names=["Normal", "Mistake"],
        zero_division=0
    ))

    return {
        "name":     name,
        "f1":       f1,
        "auc":      auc,
        "y_true":   all_y_true,
        "y_pred":   all_y_pred,
        "y_prob":   all_y_prob,
        "pipeline": pipeline,
    }


results = {}
for name, pipeline in models.items():
    print(f"\nTraining: {name}...")
    results[name] = evaluate_model(name, pipeline, X, y, groups, gkf)


# ─────────────────────────────────────────────
# PLOT ROC AND PRECISION-RECALL CURVES
# ─────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "monospace",
    "figure.facecolor": "#0d0d0d",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#444444",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
})

COLORS = ["#4a9eff", "#e8c547", "#ff4a4a"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for (name, result), color in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(result["y_true"], result["y_prob"])
    ax1.plot(fpr, tpr, color=color, lw=2,
              label=f"{name} (AUC={result['auc']:.3f})")

ax1.plot([0,1],[0,1], "w--", alpha=0.4, label="Random baseline")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curves — Mistake Prediction")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

for (name, result), color in zip(results.items(), COLORS):
    precision, recall, _ = precision_recall_curve(result["y_true"], result["y_prob"])
    ax2.plot(recall, precision, color=color, lw=2, label=name)

baseline = y.mean()
ax2.axhline(y=baseline, color="white", linestyle="--", alpha=0.4,
             label=f"Random baseline ({baseline:.3f})")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curves — Mistake Prediction")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(results_path, "06_roc_pr_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: 06_roc_pr_curves.png")


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────

best_name   = max(results, key=lambda k: results[k]["auc"])
best_result = results[best_name]
best_clf    = best_result["pipeline"].named_steps["clf"]

print(f"\nBest model: {best_name} (AUC={best_result['auc']:.4f})")

if hasattr(best_clf, "feature_importances_"):
    importances = best_clf.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        range(len(FEATURE_COLS)),
        importances[sorted_idx],
        color="#e8c547", alpha=0.85, edgecolor="#333"
    )
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels([FEATURE_COLS[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title(
        f"Feature Importance — {best_name}\n"
        f"Which factors best predict chess mistakes?",
        fontsize=12, pad=15
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(results_path, "07_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: 07_feature_importance.png")

    print(f"\n--- Feature Importance Ranking ---")
    for i in reversed(sorted_idx):
        print(f"  {FEATURE_COLS[i]:<25}: {importances[i]:.4f}")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"FINAL MODEL COMPARISON")
print(f"{'='*50}")
print(f"{'Model':<25} {'F1':>8} {'AUC':>8}")
print("-" * 45)
for name, result in results.items():
    marker = " <-- best" if name == best_name else ""
    print(f"{name:<25} {result['f1']:>8.4f} {result['auc']:>8.4f}{marker}")