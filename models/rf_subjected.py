"""
random_forest_subject_level.py
Subject-level ADHD classification using Random Forest + channel subset selection.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Config
# -----------------------
CSV_PATH = "data/features_extracted.csv"
OUT_DIR = "rf_subject_level_outputs"
N_ESTIMATORS = 500
CV_SPLITS = 5
RANDOM_STATE = 42
TOP_K_CHANNELS = 8
BOOTSTRAP_ITERS = 100
ROWS_PER_SUBJECT = 361

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(CSV_PATH)
print("Loaded data shape:", df.shape)

# Assign subject IDs
df["Subject"] = df.index // ROWS_PER_SUBJECT
n_subjects = df["Subject"].nunique()
print("Detected subjects:", n_subjects)

# Normalize label to 0/1
df["Class"] = df["Class"].astype(int)

# Map channel numbers to EEG names
channel_map = {
    1:"Fp1",2:"Fp2",3:"F3",4:"F4",5:"C3",6:"C4",
    7:"P3",8:"P4",9:"O1",10:"O2",11:"F7",12:"F8",
    13:"T7",14:"T8",15:"P7",16:"P8",17:"Fz",18:"Cz",19:"Pz"
}

# Feature columns
feature_cols = [
    "Mean","Median","Min","Max","Std","RMS","Skewness","Kurtosis","Peak","IQR","Q1","Q2","Q3",
    "Shannon_Entropy","Renyi_Entropy","Tsallis_Entropy","Permutation_Entropy",
    "Spectral_Flatness","Log_Entropy","Higuchi_FD","Hurst_Exponent",
    "Hjorth_Activity","Hjorth_Mobility","Hjorth_Complexity"
]

# -----------------------
# Pivot to subject-level
# -----------------------
def make_col(ch, chnum, typ, mode, feat):
    if typ == "original":
        return f"{channel_map.get(chnum,str(chnum))}__{typ}__{feat}"
    else:
        return f"{channel_map.get(chnum,str(chnum))}__{typ}{mode}__{feat}"

rows = []
labels = []
for subj, block in df.groupby("Subject"):
    subj_dict = {}
    for _, r in block.iterrows():
        chnum = int(r["Channel"])
        typ = str(r["Type"]).lower()
        mode = int(r["Mode"])
        for feat in feature_cols:
            if feat in block.columns:
                colname = make_col(channel_map.get(chnum,str(chnum)), chnum, typ, mode, feat)
                subj_dict[colname] = r[feat]
    rows.append(subj_dict)
    labels.append(block["Class"].iloc[0])

X = pd.DataFrame(rows)
y = pd.Series(labels, name="label")

print("Pivoted X shape (subjects × features):", X.shape)

# Impute + scale
imp = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

# -----------------------
# Train/test split (subject-level!)
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# -----------------------
# Baseline Random Forest
# -----------------------
rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                             class_weight="balanced", n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Baseline test accuracy:", accuracy_score(y_test, y_pred))
print("Baseline test ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

# -----------------------
# Feature importance → channel importance
# -----------------------
feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feat_imp.to_csv(os.path.join(OUT_DIR,"feature_importances.csv"))

def parse_channel(col):
    return col.split("__")[0]

channel_imp = feat_imp.groupby(parse_channel).mean().sort_values(ascending=False)
channel_imp.to_csv(os.path.join(OUT_DIR,"channel_importance.csv"))

plt.figure(figsize=(10,5))
sns.barplot(x=channel_imp.values, y=channel_imp.index)
plt.title("Mean feature importance per channel")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"channel_importance.png"))
plt.close()

# -----------------------
# Top-K channels subset
# -----------------------
top_channels = channel_imp.head(TOP_K_CHANNELS).index
X_top = X_scaled[[c for c in X_scaled.columns if parse_channel(c) in top_channels]]

rf_top = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                 class_weight="balanced", n_jobs=-1)
cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scores_full = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
scores_top = cross_val_score(rf_top, X_top, y, cv=cv, scoring="accuracy")

print(f"\nCross-validated accuracy (full): {scores_full.mean():.3f} ± {scores_full.std():.3f}")
print(f"Cross-validated accuracy (top-{TOP_K_CHANNELS}): {scores_top.mean():.3f} ± {scores_top.std():.3f}")

# -----------------------
# RFECV
# -----------------------
rfecv = RFECV(estimator=rf, step=0.1, cv=cv, scoring="accuracy", n_jobs=-1)
rfecv.fit(X_scaled, y)
print("\nRFECV optimal #features:", rfecv.n_features_)
mean_scores = rfecv.cv_results_["mean_test_score"]
plt.figure()
plt.plot(range(1, len(mean_scores) + 1), mean_scores)
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation accuracy")
plt.title("RFECV performance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"rfecv_curve.png"))
plt.close()

# -----------------------
# Stability selection
# -----------------------
rng = np.random.RandomState(RANDOM_STATE)
chan_counts = Counter()
for i in range(BOOTSTRAP_ITERS):
    idx = rng.choice(len(X_scaled), size=len(X_scaled), replace=True)
    X_bs, y_bs = X_scaled.iloc[idx], y.iloc[idx]
    rf_bs = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE+i, n_jobs=-1,
                                   class_weight="balanced")
    rf_bs.fit(X_bs, y_bs)
    imp = pd.Series(rf_bs.feature_importances_, index=X_scaled.columns).nlargest(int(0.1*X_scaled.shape[1]))
    for col in imp.index:
        chan_counts[parse_channel(col)] += 1

chan_freq = pd.Series({ch: chan_counts[ch]/BOOTSTRAP_ITERS for ch in channel_imp.index}).sort_values(ascending=False)
chan_freq.to_csv(os.path.join(OUT_DIR,"channel_stability.csv"))

plt.figure(figsize=(10,5))
sns.barplot(x=chan_freq.values, y=chan_freq.index)
plt.title("Channel stability (bootstrap frequency)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"channel_stability.png"))
plt.close()

# =========================================================
# Channel/mode subset search
# =========================================================

def parse_unit(col, granularity="channel"):
    """Map feature column to grouping unit."""
    parts = col.split("__")
    if granularity == "channel":
        return parts[0]                          # Fp1
    elif granularity == "channel_mode":
        return parts[0] + "__" + parts[1]        # Fp1__emd1
    elif granularity == "channel_mode_type":
        return "__".join(parts[:2])              # full unit like Fp1__emd1
    else:
        return parts[0]

granularity = "channel_mode"   # choose "channel", "channel_mode", or "channel_mode_type"

# Aggregate importance by chosen granularity
unit_importance = feat_imp.groupby(lambda c: parse_unit(c, granularity)).mean().sort_values(ascending=False)

best_acc = 0
best_subset = None
results = []

for k in range(2, min(20, len(unit_importance))+1):   # test subset sizes
    top_units = unit_importance.head(k).index
    selected_cols = [c for c in X_scaled.columns if parse_unit(c, granularity) in top_units]
    X_sub = X_scaled[selected_cols]

    rf_sub = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                    class_weight="balanced", n_jobs=-1)
    scores = cross_val_score(rf_sub, X_sub, y, cv=cv, scoring="accuracy")
    mean_acc = scores.mean()
    results.append((k, mean_acc, top_units))

    if mean_acc > best_acc:
        best_acc = mean_acc
        best_subset = (k, top_units, selected_cols)

print("\n=== Best subset search results ===")
print(f"Best accuracy: {best_acc:.3f} with {best_subset[0]} units")
print("Units:", best_subset[1])

# Save results
pd.DataFrame(results, columns=["k","cv_accuracy","units"]).to_csv(
    os.path.join(OUT_DIR, f"subset_search_{granularity}.csv"), index=False
)

# Train final model on best subset
X_best = X_scaled[best_subset[2]]
rf_best = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                 class_weight="balanced", n_jobs=-1)
rf_best.fit(X_best, y)
joblib.dump(rf_best, os.path.join(OUT_DIR, f"rf_best_{granularity}.pkl"))

print("\nPipeline complete. Outputs in", OUT_DIR)