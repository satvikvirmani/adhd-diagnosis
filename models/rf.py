import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/features_extracted.csv")

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["Type"])

# Features and target
X = df_encoded.drop("Class", axis=1)
y = df_encoded["Class"]

# ---- Normalize dataset before train-test split ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (now on normalized data)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print(X_train.shape)

# Models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosted Trees": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss", random_state=42
    )
}

# Evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Find best model
best_model = max(results, key=results.get)
print("\nBest Model:", best_model, "with Accuracy:", results[best_model])