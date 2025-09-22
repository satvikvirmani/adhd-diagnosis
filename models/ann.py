import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # pyright: ignore[reportMissingImports]

# ----------------- Load & Preprocess -----------------
df = pd.read_csv("data/features_extracted.csv")

# One-hot encode 'Type'
df_encoded = pd.get_dummies(df, columns=["Type"])

# Features & target
X = df_encoded.drop("Class", axis=1).values
y = df_encoded["Class"].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
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

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ----------------- Training -----------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# ----------------- Evaluation -----------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")
acc = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", acc)