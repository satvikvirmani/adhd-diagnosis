# train_eeg_models.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.utils.validation import check_is_fitted

# -------- Optional boosters (guarded) --------
xgb_available = lgb_available = cat_available = False
try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier
    lgb_available = True
except Exception:
    pass

try:
    from catboost import CatBoostClassifier
    cat_available = True
except Exception:
    pass

# ------------------- Load --------------------
df = pd.read_csv("data/features_extracted.csv")

# Target
y = df["Class"].values

# Split features: numeric vs optional categorical 'Type'
feature_cols = [c for c in df.columns if c != "Class"]
has_type = "Type" in feature_cols and (df["Type"].dtype == "object" or str(df["Type"].dtype).startswith("category"))
if has_type:
    numeric_cols = [c for c in feature_cols if c != "Type"]
    cat_cols = ["Type"]
else:
    numeric_cols = feature_cols
    cat_cols = []

# Column transformer: pass-through numeric; one-hot on Type (if present)
from sklearn.preprocessing import OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# ---------------- Train/Test split ----------------
X_train_full, X_test_full, y_train, y_test = train_test_split(
    df[feature_cols], y, test_size=0.2, stratify=y, shuffle=True, random_state=42
)

# A small utility to build a leak-free pipeline for each model:
def build_pipe(estimator, need_scaler=True):
    steps = [
        ("prep", preprocessor),
        ("var", VarianceThreshold(0.0)),               # remove constant columns
        ("sel", SelectKBest(mutual_info_classif, k=200)),  # feature selection (tuned)
    ]
    if need_scaler:
        steps.append(("scaler", StandardScaler(with_mean=False)))  # with_mean=False for sparse safety
    steps.append(("clf", estimator))
    return Pipeline(steps)

# Figure out number of features after preprocessing
n_features = preprocessor.fit(df[feature_cols]).transform(df[feature_cols]).shape[1]

# Adjust k values to not exceed n_features
k_values = [5, 10, 15, 20, "all"]  # <= n_features
k_values = [k for k in k_values if k == "all" or (isinstance(k, int) and k <= n_features)]

# ----------------- Model spaces -------------------
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_and_params = []

# Logistic Regression (L2 + class_weight)
lr = build_pipe(LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"), need_scaler=True)
lr_space = {
    "sel__k": k_values,
    "clf__C": np.logspace(-3, 2, 10),
}
models_and_params.append(("Logistic Regression", lr, lr_space))

# SVM (RBF)
svm = build_pipe(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42), need_scaler=True)
svm_space = {
    "sel__k": k_values,
    "clf__C": np.logspace(-2, 2, 10),
    "clf__gamma": np.logspace(-4, -1, 10),
}
models_and_params.append(("SVM (RBF)", svm, svm_space))

# KNN
knn = build_pipe(KNeighborsClassifier(), need_scaler=True)
knn_space = {
    "sel__k": k_values,
    "clf__n_neighbors": [3,5,7,9,11,13],
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1,2],
}
models_and_params.append(("KNN", knn, knn_space))

# Decision Tree
dt = build_pipe(DecisionTreeClassifier(random_state=42), need_scaler=False)
dt_space = {
    "sel__k": k_values,
    "clf__max_depth": [None, 3,5,7,9,12,15],
    "clf__min_samples_split": [2,5,10,20],
    "clf__min_samples_leaf": [1,2,4,8],
}
models_and_params.append(("Decision Tree", dt, dt_space))

# Random Forest
rf = build_pipe(RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced"), need_scaler=False)
rf_space = {
    "sel__k": k_values,
    "clf__max_depth": [None, 5,10,15,20],
    "clf__min_samples_split": [2,5,10],
    "clf__min_samples_leaf": [1,2,4],
    "clf__max_features": ["sqrt", "log2", 0.2, 0.4, 0.6],
}
models_and_params.append(("Random Forest", rf, rf_space))

# Gradient Boosting
gb = build_pipe(GradientBoostingClassifier(random_state=42), need_scaler=False)
gb_space = {
    "sel__k": k_values,
    "clf__n_estimators": [200, 400, 600],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__max_depth": [2,3,4],
    "clf__subsample": [0.6, 0.8, 1.0],
    "clf__max_features": ["sqrt", "log2", None],
}
models_and_params.append(("Gradient Boosting", gb, gb_space))

# AdaBoost
ada = build_pipe(AdaBoostClassifier(random_state=42), need_scaler=False)
ada_space = {
    "sel__k": k_values,
    "clf__n_estimators": [200, 400, 600],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
}
models_and_params.append(("AdaBoost", ada, ada_space))

# Optional: XGBoost (if installed & working)
if xgb_available:
    xgb = build_pipe(
        XGBClassifier(
            n_estimators=600, objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, tree_method="hist", random_state=42
        ),
        need_scaler=False
    )
    xgb_space = {
        "sel__k": k_values,
        "clf__max_depth": [3,4,5,6],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.7, 0.9, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [0.0, 0.5, 1.0, 5.0],
    }
    models_and_params.append(("XGBoost", xgb, xgb_space))

# Optional: LightGBM
if lgb_available:
    lgbm = build_pipe(
        LGBMClassifier(n_estimators=800, objective="binary", random_state=42, verbose=-1),
        need_scaler=False
    )
    lgbm_space = {
        "sel__k": k_values,
        "clf__num_leaves": [15, 31, 63],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.7, 0.9, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_samples": [5, 10, 20],
    }
    models_and_params.append(("LightGBM", lgbm, lgbm_space))

# Optional: CatBoost (silent training)
if cat_available:
    cat = build_pipe(
        CatBoostClassifier(
            iterations=800, learning_rate=0.05, depth=6,
            loss_function="Logloss", eval_metric="AUC",
            random_seed=42, verbose=False
        ),
        need_scaler=False
    )
    cat_space = {
        "sel__k": k_values,
        "clf__depth": [4,6,8],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__l2_leaf_reg": [1,3,5,7],
        "clf__border_count": [64, 128, 254],
    }
    models_and_params.append(("CatBoost", cat, cat_space))

# --------------- Tune & Evaluate ----------------
def evaluate_model(name, pipe, param_space, X_train, y_train, X_test, y_test, n_iter=40):
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="accuracy",  # primary metric for selection
        n_jobs=-1,
        cv=inner_cv,
        refit=True,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # Evaluate on hold-out test
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # For AUC we need predict_proba or decision_function
    try:
        proba = best.predict_proba(X_test)[:, 1]
    except Exception:
        # fallback to decision function if available
        try:
            proba = best.decision_function(X_test)
        except Exception:
            proba = None
    auc = roc_auc_score(y_test, proba) if proba is not None else np.nan

    f1 = f1_score(y_test, y_pred)

    return {
        "name": name,
        "best_params": search.best_params_,
        "cv_best_score": search.best_score_,
        "test_accuracy": acc,
        "test_auc": auc,
        "test_f1": f1,
        "fitted": best,
    }

results = []
for name, pipe, space in models_and_params:
    print(f"\n=== Tuning {name} ===")
    res = evaluate_model(name, pipe, space, X_train_full, y_train, X_test_full, y_test, n_iter=35)
    print(f"{name} | CV Acc: {res['cv_best_score']:.4f} | Test Acc: {res['test_accuracy']:.4f} | "
          f"Test AUC: {res['test_auc']:.4f} | Test F1: {res['test_f1']:.4f}")
    results.append(res)

# Rank by test accuracy
summary = pd.DataFrame([{
    "Model": r["name"],
    "CV_Acc": r["cv_best_score"],
    "Test_Acc": r["test_accuracy"],
    "Test_AUC": r["test_auc"],
    "Test_F1": r["test_f1"]
} for r in results]).sort_values("Test_Acc", ascending=False).reset_index(drop=True)

print("\n===== Model Leaderboard (hold-out test) =====")
print(summary.to_string(index=False))

# ---------------- Stacking (meta-ensemble) ----------------
# Take top 3 base learners (by Test_Acc) as estimators
top3 = summary.head(3)["Model"].tolist()
estimators = []
for r in results:
    if r["name"] in top3:
        # expose the fitted base estimator (remove the final 'clf' name collision)
        estimators.append((r["name"][:15].replace(" ", "_"), r["fitted"]))

# Meta-learner: Logistic Regression on calibrated probabilities
if len(estimators) >= 2:
    print("\n=== Training Stacking Ensemble (top 2-3 models) ===")
    meta = LogisticRegression(max_iter=5000)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        passthrough=False,
        stack_method="auto",        # uses predict_proba if available
        n_jobs=-1
    )

    # Fit stacking on the training set only (no extra tuning to keep it simple)
    stack.fit(X_train_full, y_train)
    y_pred = stack.predict(X_test_full)

    try:
        proba = stack.predict_proba(X_test_full)[:, 1]
        auc = roc_auc_score(y_test, proba)
    except Exception:
        proba = None
        auc = np.nan

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Stacking | Test Acc: {acc:.4f} | Test AUC: {auc:.4f} | Test F1: {f1:.4f}")
else:
    print("\n(Not enough base models for stacking.)")