import os
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

CSV_PATH = "data/features_extracted.csv"
ROWS_PER_SUBJECT = 361

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
    X_scaled, y, test_size=0.2, random_state=32, stratify=y
)

# ----------------- Build ANN -----------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5)
]

# ----------------- Training -----------------
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=500,
    batch_size=4,   # smaller batch (since subject-level dataset is small)
    # callbacks=callbacks,
    verbose=1
)

# ----------------- Evaluation -----------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")
acc = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", acc)